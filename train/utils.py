# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import torch

from train.train_utils.checkpoint_loader import robust_checkpoint_loader
from train.train_utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _clean_state_dict_keys_vjepa(state_dict):
    # Handle checkpoints saved from DDP/module wrappers and backbone submodules.
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("backbone.", "")
        cleaned[k] = v
    return cleaned

def _clean_state_dict_keys_ddp(state_dict, target_keys):
    # Handle checkpoints saved from DDP/module wrappers and backbone submodules.
    cleaned = {}
    if any(k.startswith("module.") for k in target_keys):
        for k, v in state_dict.items():
            if not k.startswith("module."):
                k = "module." + k
            cleaned[k] = v
    else:
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v
           
    return cleaned

def load_pretrained(
    r_path,
    target_encoder=None,
    context_encoder_key="encoder",
    use_dinov3_encoder=False,
):

    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    if not use_dinov3_encoder:
        epoch = checkpoint["epoch"]
    else:
        epoch ='unknown (loaded from DINOv3 downloaded checkpoint)'

    # -- loading target_encoder
    if not use_dinov3_encoder:
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = _clean_state_dict_keys_vjepa(pretrained_dict)
        # print(f"checkpoint keys: {pretrained_dict.keys()}")
        # print(f"state dict keys: {target_encoder.state_dict().keys()}")
    else:
        pretrained_dict = checkpoint

    pretrained_dict = _clean_state_dict_keys_ddp(pretrained_dict,
                                                target_encoder.state_dict().keys())
    msg = target_encoder.load_state_dict(pretrained_dict, strict=True)
    logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return target_encoder


def load_checkpoint(
    r_path,
    predictor,
    opt=None,
    scaler=None,
):
    # -- loading predictor
    pred_checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    pretrained_dict = pred_checkpoint["predictor"]
    predictor_epoch = pred_checkpoint["epoch"]

    pretrained_dict = _clean_state_dict_keys_ddp(pretrained_dict,
                                                  predictor.state_dict().keys())
    msg = predictor.load_state_dict(pretrained_dict, strict=True)
    logger.info(f"loaded pretrained predictor from epoch {predictor_epoch} with msg: {msg}")

    # -- loading optimizer
    if opt is not None:
        opt.load_state_dict(pred_checkpoint["opt"])

    if scaler is not None:
        scaler.load_state_dict(pred_checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {predictor_epoch}")
    logger.info(f"read-path: {r_path}")
    wandb_meta = pred_checkpoint.get("wandb", {})
    del pred_checkpoint

    return (
        predictor,
        opt,
        scaler,
        predictor_epoch,
        wandb_meta,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    action_embed_dim=7,
    use_extrinsics=False,
    dual_view_predictor=False,
    cross_attention=True,
    cross_attn_num_heads=None,
    use_dinov3_encoder=False,
):
    import src_vjepa2.models_v2.ac_predictor as vit_ac_pred
    
    if use_dinov3_encoder:
        import src_dino.vision_transformer_dino as video_vit_dino
        dinov3_encoder_kwargs = dict(
            img_size=crop_size,
            patch_size=patch_size,
            embed_dim=1280,
            depth=32,
            num_heads=20,
            ffn_ratio=6.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
        # encoder = video_vit_dino.__dict__[dinov3_model_name](**dinov3_encoder_kwargs)
        encoder = video_vit_dino.DinoVisionTransformer(**dinov3_encoder_kwargs)
    else:
        import src_vjepa2.models_v2.vision_transformer as video_vit
        encoder_builder = video_vit
        encoder = encoder_builder.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=True,
            use_rope=use_rope,
        )

    predictor_kwargs = dict(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    if dual_view_predictor:
        if cross_attention:
            predictor = vit_ac_pred.__dict__["vit_ac_dual_predictor"](
                        cross_attn_num_heads=cross_attn_num_heads,
                        **predictor_kwargs,
                    )
        else:
            predictor = vit_ac_pred.__dict__["vit_ac_nocrossattn_dual_predictor"](
                        **predictor_kwargs,
                    )
    else:
        predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
            **predictor_kwargs,
        )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler("cuda") if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
