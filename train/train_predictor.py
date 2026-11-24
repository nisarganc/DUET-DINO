# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # Respect launcher-provided mapping (e.g., Slurm gpu-bind); only fallback if unset.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
        if local_rank is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from train.data_loader import init_data
from train.transforms import make_transforms
from train.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from train.train_utils.distributed import init_distributed
from train.train_utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer, loss_metrics_to_cpu
import wandb

# --
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # when runtime distributed init falls back to world_size=1 on some tasks.
    launcher_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    pretrain_dinocheckpoint = cfgs_meta.get("pretrain_dinocheckpoint", None)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    log_freq = cfgs_meta.get("log_freq", 50)
    timing_freq = cfgs_meta.get("timing_freq", log_freq)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", True)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    cross_attn_num_heads = cfgs_model.get("cross_attn_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    dual_view_training = bool(cfgs_model.get("dual_view_training", False))
    cross_attention = bool(cfgs_model.get("cross_attention", True))
    use_dinov3_encoder = bool(cfgs_model.get("use_dinov3_encoder", False))

    # -- DATA
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])
    dataset_path = datasets
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["right_mp4_path"])
    include_roboarena = bool(cfgs_data.get("include_roboarena", False))
    primary_view = camera_views[0]
    secondary_view = camera_views[1] if len(camera_views) > 1 else 'wrist_mp4_path'
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", True)
    num_workers = cfgs_data.get("num_workers", 1)
    prefetch_factor = cfgs_data.get("prefetch_factor", 2)
    decode_num_threads = cfgs_data.get("decode_num_threads", 1)
    dataset_len_multiplier = cfgs_data.get("dataset_len_multiplier", 2)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    train_test_split = cfgs_loss.get("train_test_split", False)

    use_closs = cfgs_loss.get("use_closs", False)
    closs_coeff = float(cfgs_loss.get("closs_coeff", 0.0))
    closs_chosen = cfgs_loss.get("closs_chosen", 0.1)
    
    if use_closs:
        closs_keys = cfgs_loss.get("closs_keys", [-1.0, 0.0, 0.1, 0.2])
    else:
        closs_keys = []
    closs_maxnorm = float(cfgs_loss.get("closs_maxnorm", 0.05))
    closs_rot_std = float(cfgs_loss.get("closs_rot_std", np.pi / 30))
    closs_rot_clip = float(cfgs_loss.get("closs_rot_clip", np.pi))
    closs_gripper_clip = float(cfgs_loss.get("closs_gripper_clip", 1.0))
    
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # -- logging in wandb
    wandb_cfg = args.get("wandb", {})
    wandb_project = wandb_cfg.get('project')
    wandb_entity = wandb_cfg.get('entity')
    wandb_name = wandb_cfg.get('name')
    wandb_tags = wandb_cfg.get('tags')    
    wandb_enabled = bool(wandb_cfg.get("enabled", False))

    wandb_run = None
    wandb_resume_id = None

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("CUDA not available. Using CPU.")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        if r_file is not None:
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")
        resume_path = None

    # -- make csv_logger
    csv_columns = [
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    ]
    if use_closs:
        for key in closs_keys:
            csv_columns.append(("%.5f", f"closs_{key}"))
    csv_logger = CSVLogger(log_file, *csv_columns, mode="+a", flush_every=log_freq)
    
    # -- init model
    target_encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        predictor_num_frames=max_num_frames * tubelet_size,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        dual_view_predictor=dual_view_training,
        cross_attention=cross_attention,
        cross_attn_num_heads=cross_attn_num_heads,
        use_dinov3_encoder=use_dinov3_encoder,
    )

    target_encoder.requires_grad_(False)

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    (unsupervised_loader, unsupervised_sampler, test_loader, test_sampler) = init_data(
        data_paths=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        pair_camera_views=dual_view_training,
        include_roboarena=include_roboarena,
        decode_num_threads=decode_num_threads,
        dataset_len_multiplier=dataset_len_multiplier,
        train_test_split=train_test_split,
        rank=rank,
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"train iterations per epoch/dataset length: {ipe}/{_dlen}")

    if train_test_split:
        tpe = len(test_loader)
        logger.info(f"test iterations: {tpe}")


    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=target_encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=dtype == torch.float16,
        fused=cfgs_opt.get("fused_adamw", True) and device.type == "cuda",
        betas=betas,
        eps=eps,
    )
    # new changes for single GPU run
    use_distributed = dist.is_available() and dist.is_initialized() and world_size > 1
    if use_distributed:
        predictor = DistributedDataParallel(
            predictor, static_graph=False, find_unused_parameters=True,
            gradient_as_bucket_view=True, broadcast_buffers=False,
        )
    else:
        logger.info("Distributed process group unavailable; running without DDP wrappers.")

    if compile_model:
        logger.info("Compiling target encoder and predictor.")
        torch._dynamo.config.optimize_ddp = cfgs_model.get("optimize_ddp", True)
        if use_dinov3_encoder:
            target_encoder.forward_features = torch.compile(target_encoder.forward_features)
        else:
            target_encoder.compile()
        predictor.compile()

    # -- looad pretrained weights
    target_encoder = load_pretrained(
        r_path=pretrain_dinocheckpoint if use_dinov3_encoder else p_file,
        target_encoder=target_encoder,
        context_encoder_key=context_encoder_key,
        use_dinov3_encoder=use_dinov3_encoder,
    )

    start_epoch = 0
    # -- load training checkpoint
    if resume_path is not None:
        (
            predictor,
            optimizer,
            scaler,
            start_epoch,
            loaded_wandb_meta,
        ) = load_checkpoint(
            r_path=resume_path,
            predictor=predictor,
            opt=optimizer,
            scaler=scaler,
        )
        wandb_resume_id = loaded_wandb_meta.get("run_id")
        wandb_project = loaded_wandb_meta.get('project')
        wandb_entity = loaded_wandb_meta.get('entity')
        wandb_name = loaded_wandb_meta.get('name')
        wandb_tags = loaded_wandb_meta.get('tags')        
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    if wandb_enabled and launcher_rank == 0:
        wandb_config = {
            "seed": seed,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "ipe": ipe,
            "lr": lr,
            "weight_decay": wd,
            "model_name": model_name,
            "pred_depth": pred_depth,
            "pred_embed_dim": pred_embed_dim,
            "pred_num_heads": pred_num_heads,
            "cross_attn_num_heads": cross_attn_num_heads,
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "fps": fps,
            "crop_size": crop_size,
        }
        if use_closs:
            wandb_config.update(
                {
                    "use_closs": use_closs,
                    "closs_coeff": closs_coeff,
                    "closs_chosen": closs_chosen,
                    "closs_keys": closs_keys,
                    "closs_maxnorm": closs_maxnorm,
                    "closs_rot_std": closs_rot_std,
                    "closs_rot_clip": closs_rot_clip,
                    "closs_gripper_clip": closs_gripper_clip,
                }
            )
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            tags=wandb_tags,
            config=wandb_config,
            id=wandb_resume_id,
            resume="allow",
        )

    def _wandb_update_summary(payload):
        for key, value in payload.items():
            # Keep latest scalar values in summary (overwrite semantics).
            if isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
                wandb_run.summary[key] = value.item() if isinstance(value, np.generic) else value

    def save_checkpoint(epoch, path):
        if launcher_rank != 0:
            return
        save_dict = {
            # "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),

            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "wandb": {
                "run_id": wandb_run.id if wandb_run is not None else None,
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "tags": wandb_tags,
            },
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing train loader...")

    if train_test_split:
        logger.info("Initializing test loader...")
        test_sampler.set_epoch(start_epoch)
        test_iter = iter(test_loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def load_clips(sample):
        # clips
        clips = sample[0]
        if dual_view_training:
            clips = {k: v.to(device, non_blocking=True) for k, v in clips.items()}
        else:
            clips = clips.to(device, non_blocking=True)  # [B C T H W]
        
        # action and states
        actions = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B T-1 7]
        states = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B T 7]
        
        # extrinsics
        extrinsics = sample[3]
        if dual_view_training:
            extrinsics = {k: v.to(device, dtype=torch.float, non_blocking=True) for k, v in extrinsics.items()}
        else:
            extrinsics = extrinsics.to(device, dtype=torch.float, non_blocking=True)  # [B T 7]
        
        return (clips, actions, states, extrinsics)

    def forward_target(c):
        with torch.no_grad():
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1)
            if use_dinov3_encoder:
                target_backbone = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
                h = target_backbone.forward_features(c)["x_norm_patchtokens"]
            else:
                # vjepa2 is a video model and has extra dimension
                c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = target_encoder(c)
            h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def _normalize_rep(_z):
        if normalize_reps:
            return F.layer_norm(_z, (_z.size(-1),))
        return _z

    def _step_predictor(_z, _a, _s, _e):
        if dual_view_training:    
            primary_z, secondary_z = _z
            primary_e, secondary_e = _e
            # forward of dual predictor takes this order of inputs
            primary_out, secondary_out = predictor(primary_z, secondary_z, 
                                                   _a, _s, 
                                                   primary_e, secondary_e)
            return _normalize_rep(primary_out), _normalize_rep(secondary_out)
        else:
            return _normalize_rep(predictor(_z, _a, _s, _e))
    
    def forward_tf(z, actions, states, extrinsics):

        if dual_view_training:
            z_primary, z_secondary = z
            _a, _s = actions, states[:, :-1]
            _e_primary = extrinsics[primary_view][:, :-1]
            _e_secondary = extrinsics[secondary_view][:, :-1]
            return _step_predictor(
                (z_primary[:, :-tokens_per_frame], z_secondary[:, :-tokens_per_frame]),
                _a, _s,
                (_e_primary, _e_secondary),
            )
        else:
            _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
            return _step_predictor(_z, _a, _s, _e)

    def forward_predictions(z, actions, states, extrinsics):

        # -- one step of predictor with teacher forcing
        z_tf = forward_tf(z, actions, states, extrinsics)

        if dual_view_training:
            z_primary, z_secondary = z
            z_primary_tf, z_secondary_tf = z_tf
            z_primary_ar = torch.cat([z_primary[:, :tokens_per_frame], z_primary_tf[:, :tokens_per_frame]], dim=1)
            z_secondary_ar = torch.cat([z_secondary[:, :tokens_per_frame], z_secondary_tf[:, :tokens_per_frame]], dim=1)
            for n in range(1, auto_steps):
                _a, _s = actions[:, : n + 1], states[:, : n + 1]
                _e_primary = extrinsics[primary_view][:, : n + 1]
                _e_secondary = extrinsics[secondary_view][:, : n + 1]
                z_primary_nxt, z_secondary_nxt = _step_predictor(
                    (z_primary_ar, z_secondary_ar),
                    _a,
                    _s,
                    (_e_primary, _e_secondary),
                )
                z_primary_ar = torch.cat([z_primary_ar, z_primary_nxt[:, -tokens_per_frame:]], dim=1)
                z_secondary_ar = torch.cat([z_secondary_ar, z_secondary_nxt[:, -tokens_per_frame:]], dim=1)
            return z_tf, (z_primary_ar[:, tokens_per_frame:], z_secondary_ar[:, tokens_per_frame:])
        
        else:
            _z = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
            for n in range(1, auto_steps):
                _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
                _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                _z = torch.cat([_z, _z_nxt], dim=1)
            z_ar = _z[:, tokens_per_frame:]
            return z_tf, z_ar

    def loss_fn(z, h):
        _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

    def _sample_random_actions_like_mpc(actions_ref):

        device, dtype = actions_ref.device, actions_ref.dtype

        bsz = actions_ref.size(0)
        steps = actions_ref.size(1)
        action_dim = actions_ref.size(2)        
        mean = torch.zeros(steps, action_dim, device=device, dtype=dtype)
        std = torch.tensor(
            [
                closs_maxnorm,
                closs_maxnorm,
                closs_maxnorm,
                closs_rot_std,
                closs_rot_std,
                closs_rot_std,
                1.0,
            ],
            device=device,
            dtype=dtype,
        )
        rand_action = torch.randn(bsz, steps, action_dim, device=device, dtype=dtype) * std + mean
        rand_action[..., :3] = torch.clamp(rand_action[..., :3], min=-closs_maxnorm, max=closs_maxnorm)
        rand_action[..., 3:6] = torch.clamp(rand_action[..., 3:6], min=-closs_rot_clip, max=closs_rot_clip)
        rand_action[..., -1:] = torch.clamp(rand_action[..., -1:], min=0.0, max=closs_gripper_clip)
        return rand_action

    def controllability_loss(h, actions, states, extrinsics, jloss):

        # sample random action
        actions_rand = _sample_random_actions_like_mpc(actions)

        # -- one step of predictor
        z_tf_rand = forward_tf(h, actions_rand, states, extrinsics)

        if dual_view_training:
            z_static_rand, z_wrist_rand = z_tf_rand
            h_static, h_wrist = h
            jloss_randact = loss_fn(z_static_rand, h_static) + loss_fn(z_wrist_rand, h_wrist)
        else:
            jloss_randact = loss_fn(z_tf_rand, h)

        diff = jloss - jloss_randact
        closses = {}
        for key in closs_keys:
            if key == -1.0:
                closses[key] = F.softplus(diff)
            else:
                closses[key] = F.relu(key + diff)

        return closses

    def compute_losses(clips, actions, states, extrinsics):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            
            # -- forward target encoder
            if dual_view_training:
                h = (
                    forward_target(clips[primary_view]),
                    forward_target(clips[secondary_view]),
                )
            else:
                h = forward_target(clips)
            
            # -- forward predictor
            z_tf, z_ar = forward_predictions(h, actions, states, extrinsics)

            # -- compute losses
            view_losses = {}
            if dual_view_training:
                h_static, h_wrist = h
                z_static_tf, z_wrist_tf = z_tf
                z_static_ar, z_wrist_ar = z_ar

                jloss_static = loss_fn(z_static_tf, h_static)
                jloss_wrist = loss_fn(z_wrist_tf, h_wrist)

                sloss_static = loss_fn(z_static_ar, h_static)
                sloss_wrist = loss_fn(z_wrist_ar, h_wrist)

                jloss = jloss_static + jloss_wrist
                sloss = sloss_static + sloss_wrist
                view_losses = {
                    "jloss_static": jloss_static,
                    "jloss_wrist": jloss_wrist,
                    "sloss_static": sloss_static,
                    "sloss_wrist": sloss_wrist,
                }
            else:
                jloss = loss_fn(z_tf, h)  # teacher forcing loss
                sloss = loss_fn(z_ar, h)  # auto-regressive loss
            
            closs = {}
            if use_closs:
                closs = controllability_loss(h, actions, states, extrinsics, jloss)
                loss = jloss + sloss + (closs_coeff * closs[closs_chosen])
            else:
                loss = jloss + sloss
        
        return loss, jloss, sloss, closs, view_losses

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs): # 0-315
        logger.info("Epoch %d" % (epoch + 1))

        # Refresh sampler and iterator at epoch boundaries to avoid mid-epoch stalls.
        unsupervised_sampler.set_epoch(epoch)
        loader = iter(unsupervised_loader)

        # Optional one-time skip for the first epoch only.
        if epoch == start_epoch and skip_batches > 0:
            logger.info(f"Skip {skip_batches} batches")
            for itr in range(skip_batches):
                if itr % 10 == 0:
                    logger.info(f"Skip {itr}/{skip_batches} batches")
                try:
                    _ = next(loader)
                except StopIteration:
                    logger.warning("Loader exhausted while skipping batches; resetting iterator.")
                    loader = iter(unsupervised_loader)
                    _ = next(loader)
                except Exception:
                    logger.warning("Data loading exception while skipping batches; resetting iterator.")
                    loader = iter(unsupervised_loader)
                    _ = next(loader)

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        closs_meters = {k: AverageMeter() for k in closs_keys}
        view_loss_meters = None
        if dual_view_training:
            view_loss_meters = {
                "jloss_static": AverageMeter(),
                "jloss_wrist": AverageMeter(),
                "sloss_static": AverageMeter(),
                "sloss_wrist": AverageMeter(),
            }
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe): # 0-300
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            clips, actions, states, extrinsics = load_clips(sample)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                # Step 1. Forward
                loss, jloss, sloss, closs, view_losses = compute_losses(clips, actions, states, extrinsics)

                # Step 2. Backward & step
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (*loss_metrics_to_cpu(loss, jloss, sloss, closs, view_losses), _new_lr, _new_wd)

            (
                loss,
                jloss,
                sloss,
                closs,
                view_losses,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(
                train_step, log_timings=(itr % timing_freq == 0 or itr == ipe - 1),
            )
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            for key, val in closs.items():
                closs_meters[key].update(float(val))
            if dual_view_training:
                for key, val in view_losses.items():
                    view_loss_meters[key].update(float(val))
            iter_time_meter.update(iter_elapsed_time_ms)
            if gpu_etime_ms >= 0:
                gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms, *closs.values())
 
                if wandb_run is not None and (
                    (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss)
                ):
                    global_step = epoch * ipe + itr
                    if closs.items():
                        closs_step_payload = {f"train/step_control_loss_{k}": closs[k] for k in closs_keys}
                        closs_epoch_payload = {f"train/epoch_control_loss_{k}": closs_meters[k].avg for k in closs_keys}
                    
                    payload = {
                        "epoch": epoch + 1,
                        "itr": itr,

                        "train/step_full_loss": loss,
                        "train/step_teachforce_loss": jloss,
                        "train/step_rollout_loss": sloss,

                        "train/epoch_full_loss": loss_meter.avg,
                        "train/epoch_teachforce_loss": jloss_meter.avg,
                        "train/epoch_rollout_loss": sloss_meter.avg,

                        "train/lr": _new_lr,
                        "train/weight_decay": _new_wd,
                        "train/iter_time_ms": iter_time_meter.avg,
                        "train/gpu_time_ms": gpu_time_meter.avg,
                        "train/data_time_ms": data_elapsed_time_meter.avg,
                    }
                    if closs.items():
                        payload.update(closs_step_payload)
                        payload.update(closs_epoch_payload)
                    if dual_view_training:
                        payload.update(
                            {
                                "train/step_teachforce_loss_static": float(view_losses["jloss_static"]),
                                "train/step_teachforce_loss_wrist": float(view_losses["jloss_wrist"]),

                                "train/step_rollout_loss_static": float(view_losses["sloss_static"]),
                                "train/step_rollout_loss_wrist": float(view_losses["sloss_wrist"]),

                                "train/epoch_teachforce_loss_static": view_loss_meters["jloss_static"].avg,
                                "train/epoch_teachforce_loss_wrist": view_loss_meters["jloss_wrist"].avg,

                                "train/epoch_rollout_loss_static": view_loss_meters["sloss_static"].avg,
                                "train/epoch_rollout_loss_wrist": view_loss_meters["sloss_wrist"].avg,
                            }
                        )
                    wandb_run.log(payload, step=global_step)
                    _wandb_update_summary(payload)

                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [%.2f, %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        csv_logger.flush()
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        if train_test_split:
            # -- Eval every 10th epoch
            if (epoch + 1) % 10 == 0 and tpe is not None and tpe > 0:
                logger.info("Running eval...")
                predictor.eval()

                eval_loss_meter = AverageMeter()
                eval_jloss_meter = AverageMeter()
                eval_sloss_meter = AverageMeter()
                eval_closs_meters = {k: AverageMeter() for k in closs_keys}
                eval_view_loss_meters = None
                if dual_view_training:
                    eval_view_loss_meters = {
                        "jloss_static": AverageMeter(),
                        "jloss_wrist": AverageMeter(),
                        "sloss_static": AverageMeter(),
                        "sloss_wrist": AverageMeter(),
                    }

                test_sampler.set_epoch(epoch)
                test_iter = iter(test_loader)

                for itr in range(tpe):
                    sample = next(test_iter)

                    clips, actions, states, extrinsics = load_clips(sample)
                    with torch.no_grad():
                        loss, jloss, sloss, closs, view_losses = compute_losses(clips, actions, states, extrinsics)
                    loss, jloss, sloss, closs, view_losses = loss_metrics_to_cpu(
                        loss, jloss, sloss, closs, view_losses,
                    )

                    eval_loss_meter.update(float(loss))
                    eval_jloss_meter.update(float(jloss))
                    eval_sloss_meter.update(float(sloss))
                    for key, val in closs.items():
                        eval_closs_meters[key].update(float(val))
                    if dual_view_training:
                        for key, val in view_losses.items():
                            eval_view_loss_meters[key].update(float(val))

                    if rank == 0:
                        logger.info(
                            "[eval] epoch %d | step %d | loss: %.3f (tf %.3f, ar %.3f)"
                            % (epoch + 1, itr, loss, jloss, sloss)
                        )

                        if wandb_run is not None:
                            global_step = ((epoch + 1) * ipe) + itr
                            payload = {
                                "epoch": epoch + 1,
                                "eval/step_teachforce_loss": jloss,
                                "eval/step_rollout_loss": sloss,
                                "eval/step_full_loss": loss,
                            }
                            if dual_view_training:
                                payload.update(
                                    {
                                        "eval/step_teachforce_loss_static": float(view_losses["jloss_static"]),
                                        "eval/step_teachforce_loss_wrist": float(view_losses["jloss_wrist"]),
                                        "eval/step_rollout_loss_static": float(view_losses["sloss_static"]),
                                        "eval/step_rollout_loss_wrist": float(view_losses["sloss_wrist"]),
                                    }
                                )
                            if closs.items():
                                payload.update({f"eval/step_control_loss_{k}": closs[k] for k in closs_keys})
                            wandb_run.log(payload, step=global_step)
                            _wandb_update_summary(payload)

                if rank == 0:
                    logger.info(
                        "[eval] epoch %d | avg loss: %.3f (tf %.3f, ar %.3f)"
                        % (
                            epoch + 1,
                            eval_loss_meter.avg,
                            eval_jloss_meter.avg,
                            eval_sloss_meter.avg,
                        )
                        )

                    if wandb_run is not None:
                        global_step = ((epoch + 1) * ipe) + tpe
                        
                        payload = {
                            "epoch": epoch + 1,
                            "eval/epoch_teachforce_loss": eval_jloss_meter.avg,
                            "eval/epoch_rollout_loss": eval_sloss_meter.avg,
                            "eval/epoch_full_loss": eval_loss_meter.avg,
                        }
                        if dual_view_training:
                            payload.update(
                                {
                                    "eval/epoch_teachforce_loss_static": eval_view_loss_meters["jloss_static"].avg,
                                    "eval/epoch_teachforce_loss_wrist": eval_view_loss_meters["jloss_wrist"].avg,
                                    "eval/epoch_rollout_loss_static": eval_view_loss_meters["sloss_static"].avg,
                                    "eval/epoch_rollout_loss_wrist": eval_view_loss_meters["sloss_wrist"].avg,
                                }
                            )
                        if closs.items():
                            payload.update(
                                {
                                    f"eval/epoch_control_loss_{k}": eval_closs_meters[k].avg
                                    for k in closs_keys
                                }
                            )
                        wandb_run.log(payload, step=global_step)
                        _wandb_update_summary(payload)

                predictor.train()

    if wandb_run is not None:
        wandb_run.finish()
