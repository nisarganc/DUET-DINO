# Copyright (c) 2026 DUET-DINO contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import struct
import socket
import pickle
import traceback
from collections import deque
from pathlib import Path
from time import perf_counter

import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DUET_DINO_ROOT = REPO_ROOT 

if str(DUET_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DUET_DINO_ROOT))

##################################### SERVER ##########################################

_HEADER = struct.Struct("!Q")

def _send_message(sock: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("DUET-DINO policy socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))

class DUETDINOPolicy:
    """Local RoboLab inference Server for the DUET-DINO world model on DroidIK envs."""

    def __init__(
        self,
        cfg_file: str | None = None,
        host: str = "0.0.0.0", 
        port: int = 8000
    ) -> None:

        self.host = host
        self.port = int(port)
        import yaml

        self.cfg_path = Path.joinpath(DUET_DINO_ROOT, 
                                      "configs",
                                      "inference",
                                      cfg_file)
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        self._env_prev_action: dict[int, object] = {}
        self._env_goal_rep: dict[int, object] = {}
        self._env_goal_rep_wrist: dict[int, object] = {}
        self._env_generators: dict[int, object] = {}
        self._env_run_idx: dict[int, int] = {}

        self._initialize_model()

    def _initialize_model(self) -> None:
        import copy
        import torch

        from train.transforms import make_transforms
        from train.utils import init_video_model, load_checkpoint, load_pretrained
        from inference.utils.world_model_wrapper import WorldModel

        cfgs_model = self.cfg.get("model", {})
        cfgs_meta = self.cfg.get("meta", {})
        cfgs_data = self.cfg.get("data", {})
        cfgs_data_aug = self.cfg.get("data_aug", {})
        cfgs_mpc_args = self.cfg.get("mpc_args", {})
        cfgs_log_args = self.cfg.get("log", {})
        cfgs_exp_args = self.cfg.get("exp", {})
        cfgs_optimization = self.cfg.get("optimization", {})
        self.base_seed = int(cfgs_exp_args.get("seed", 0))
        self.history = cfgs_mpc_args.get("history", 0)
        if not 0 <= self.history <= 7:
            raise ValueError("mpc_args.history must be an integer between 0 and 7")

        self.device = self.cfg.get("device", "cuda:0")
        self.autocast_dtype = getattr(torch, cfgs_optimization.get("precision", "bfloat16"))

        camera_views = cfgs_data.get("camera_views", ["right_mp4_path"])
        crop_size = cfgs_data.get("crop_size", 256)
        patch_size = cfgs_data.get("patch_size")
        tubelet_size = cfgs_data.get("tubelet_size")

        transform = make_transforms(
            random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
            random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
            random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
            reprob=cfgs_data_aug.get("reprob", 0.0),
            auto_augment=cfgs_data_aug.get("auto_augment", False),
            motion_shift=cfgs_data_aug.get("motion_shift", False),
            crop_size=crop_size,
        )

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
        use_dinov3_encoder = bool(cfgs_model.get("use_dinov3_encoder", False))
        use_sdpa = bool(cfgs_meta.get("use_sdpa", False))
        cross_attention = bool(cfgs_model.get("cross_attention", True))

        context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
        pretrain_checkpoint = str(REPO_ROOT / cfgs_meta.get("pretrain_checkpoint"))
        predictor_checkpoint = cfgs_model.get("predictor_checkpoint")
        pretrain_dinocheckpoint = str(REPO_ROOT / cfgs_meta.get("pretrain_dinocheckpoint"))
        if isinstance(predictor_checkpoint, list):
            predictor_checkpoint = [str(REPO_ROOT / ckpt) for ckpt in predictor_checkpoint]
        else:
            predictor_checkpoint = str(REPO_ROOT / predictor_checkpoint)

        if dual_view_training:
            inferred_mode = "duet"
        elif len(camera_views) == 1:
            inferred_mode = "wrist" if "wrist" in camera_views[0] else "side"
        elif len(camera_views) == 2:
            inferred_mode = "independent_dual"
        else:
            raise ValueError(f"Unsupported DUET_DINO camera_views configuration: {camera_views}")

        encoder, predictor = init_video_model(
            uniform_power=uniform_power,
            device=self.device,
            patch_size=patch_size,
            max_num_frames=512,
            predictor_num_frames=(self.history + cfgs_mpc_args.get("rollout_horizon", 2)) * tubelet_size,
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
            use_activation_checkpointing=False,
            dual_view_predictor=dual_view_training,
            cross_attn_num_heads=cross_attn_num_heads,
            use_dinov3_encoder=use_dinov3_encoder,
            cross_attention=cross_attention
        )

        encoder = load_pretrained(
            r_path=pretrain_dinocheckpoint if use_dinov3_encoder else pretrain_checkpoint,
            target_encoder=encoder,
            context_encoder_key=context_encoder_key,
            use_dinov3_encoder=use_dinov3_encoder,
        )

        predictor_models = {}
        if inferred_mode == "independent_dual":
            side_predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint[0],
                predictor=copy.deepcopy(predictor),
                opt=None,
                scaler=None,
            )
            wrist_predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint[1],
                predictor=copy.deepcopy(predictor),
                opt=None,
                scaler=None,
            )
            side_predictor.to(self.device).eval()
            wrist_predictor.to(self.device).eval()
            predictor_models["side"] = side_predictor
            predictor_models["wrist"] = wrist_predictor
        else:
            predictor, _, _, _, _ = load_checkpoint(
                r_path=predictor_checkpoint,
                predictor=predictor,
                opt=None,
                scaler=None,
            )
            predictor.to(self.device).eval()
            predictor_models[inferred_mode] = predictor

        encoder.to(self.device).eval()
        encoder.use_activation_checkpointing = False
        compile_args = {
            "backend": "inductor",
            "mode": cfgs_optimization.get("mode", "reduce-overhead"),
            "dynamic": cfgs_optimization.get("dynamic", False),
        }
        if use_dinov3_encoder:
            # encode() calls this method, not the encoder's forward().
            encoder.forward_features = torch.compile(encoder.forward_features, **compile_args)
        else:
            encoder = torch.compile(encoder, **compile_args)
        predictor_models = {
            name: torch.compile(model, **compile_args)
            for name, model in predictor_models.items()
        }
        tokens_per_frame = int((crop_size // patch_size) ** 2)

        self.world_model = WorldModel(
            encoder=encoder,
            predictor=predictor_models,
            inferred_mode=inferred_mode,
            use_dinov3_encoder=use_dinov3_encoder,
            tokens_per_frame=tokens_per_frame,
            mpc_args={
                "rollout": cfgs_mpc_args.get("rollout_horizon", 2),
                "samples": cfgs_mpc_args.get("samples", 25),
                "topk": cfgs_mpc_args.get("topk", 10),
                "elite_temperature": cfgs_mpc_args.get("elite_temperature", 0.25),
                "cem_steps": cfgs_mpc_args.get("cem_steps", 1),
                "momentum_mean": cfgs_mpc_args.get("momentum_mean", 0.15),
                "momentum_mean_rot": cfgs_mpc_args.get("momentum_mean_rot", 0.25),
                "momentum_mean_gripper": cfgs_mpc_args.get("momentum_mean_gripper", 0.15),
                "momentum_std": cfgs_mpc_args.get("momentum_std", 0.75),
                "momentum_std_rot": cfgs_mpc_args.get("momentum_std_rot", 0.85),
                "momentum_std_gripper": cfgs_mpc_args.get("momentum_std_gripper", 0.15),
                "maxnorm": cfgs_mpc_args.get("maxnorm", 0.075),
                "maxrotnorm": cfgs_mpc_args.get("maxrotnorm", 0.314),
                "maxgrippernorm": cfgs_mpc_args.get("maxgrippernorm", 0.75),
                "warm_starting": cfgs_exp_args.get("warm-starting", False),
            },
            normalize_reps=True,
            device=self.device,
            transform=transform,
        )
        self.step_count: dict[int, int] = {}
        self.logname = cfgs_log_args.get("modelname")

        self.output_dir = Path(cfgs_log_args.get("output_dir", "output")).expanduser()
        self.output_dir = REPO_ROOT / self.output_dir / self.logname
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_inference_times = bool(cfgs_log_args.get("log_inference_times", False))
        self.log_predictions =  bool(cfgs_log_args.get("log_predictions", False))
        if self.log_predictions:
            self._prediction_log: dict[int, list[dict]] = {}
        if self.log_inference_times:
            self._inference_times: dict[int, list[float]] = {}
        if self.history > 0:
            self._env_history: dict[int, deque] = {}
        self._warmup(crop_size)

    def _warmup(self, crop_size: int) -> None:
        import random
        import numpy as np
        import torch

        print("[DUET-DINO server] Warming up encoder and planner before accepting clients...", flush=True)
        start = perf_counter()
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        image = torch.zeros(crop_size, crop_size, 3, dtype=torch.uint8)
        history = deque(maxlen=self.history) if self.history else None
        generator = torch.Generator(device=self.device).manual_seed(self.base_seed)
        try:
            with torch.random.fork_rng(devices=[]), torch.inference_mode(), torch.autocast(
                device_type=torch.device(self.device).type, dtype=self.autocast_dtype,
            ):
                pose = torch.zeros(1, 7, device=self.device)
                goal = self.world_model.encode(image)
                for _ in range(self.history + 3):
                    self.world_model.infer_next_action(
                        pose=pose, obs=image, obs_wrist=image,
                        goal_rep=goal, goal_rep_wrist=goal.clone(),
                        generator=generator, history=history,
                        return_predictions=self.log_predictions,
                    )
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        print(f"[DUET-DINO server] Warmup finished in {perf_counter() - start:.1f}s", flush=True)

    def _save_predictions(self, env_id: int = 0) -> None:
        import torch
        run_idx = self._env_run_idx[env_id]
        path = self.output_dir / f"predictions_run{run_idx}_env{env_id}.pt"
        pending_path = path.with_suffix(".pt.tmp")
        try:
            torch.save({
                "predictions_log": self._prediction_log[env_id],
            }, pending_path)
            pending_path.replace(path)
        finally:
            pending_path.unlink(missing_ok=True)
        print(f"[DUET-DINO server] Saved predictions to {path}", flush=True)
        del self._prediction_log[env_id]

    # Call at the end of each run to clear state and save predictions.
    def reset(self, env_id: int = 0) -> None:
        import numpy as np

        if self.log_predictions and env_id in self._prediction_log:
            self._save_predictions(env_id)
        if self.log_inference_times and env_id in self._inference_times:
            run_idx = self._env_run_idx[env_id]
            np.save(self.output_dir / f"inference_times_run{run_idx}_env{env_id}.npy",
                    np.asarray(self._inference_times[env_id], 
                    dtype=np.float64)
                    )
            self._inference_times.pop(env_id, None)
        if self.history > 0:
            self._env_history.pop(env_id, None)
        self._env_prev_action.pop(env_id, None)
        self._env_goal_rep.pop(env_id, None)
        self._env_goal_rep_wrist.pop(env_id, None)
        self._env_generators.pop(env_id, None)
        self._env_run_idx.pop(env_id, None)
        self.step_count.pop(env_id, None)

    def set_goal_images(
        self,
        external_image,
        wrist_image,
        *,
        env_id: int = 0,
        instruction: str = "goal",
        run_idx: int | None = None,
    ):
        import torch

        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self.autocast_dtype,
        ):
            goal_rep = self.world_model.encode(external_image)
            goal_rep_wrist = self.world_model.encode(wrist_image)

        self._env_goal_rep[env_id] = goal_rep
        self._env_goal_rep_wrist[env_id] = goal_rep_wrist

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.base_seed + int(run_idx) * 1_000_003 + env_id)
        self._env_generators[env_id] = generator
        self._env_run_idx[env_id] = int(run_idx)
        self.world_model.reset_logs(
            goal_rep=goal_rep,
            goal_rep_wrist=goal_rep_wrist,
        )

        if env_id not in self.step_count:
            self.step_count[env_id] = 0
        if self.log_inference_times and env_id not in self._inference_times:
            self._inference_times[env_id] = []
        if self.log_predictions and env_id not in self._prediction_log:
            self._prediction_log[env_id] = []
        if self.history > 0 and env_id not in self._env_history:
            self._env_history[env_id] = deque(maxlen=self.history)

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        import torch
        import numpy as np
        from scipy.spatial.transform import Rotation

        curr_obs = {
            "external_image": obs["external_image"],
            "wrist_image": obs["wrist_image"],
            "ee_pose": obs["ee_pose"],
        }

        if env_id not in self._env_goal_rep or env_id not in self._env_goal_rep_wrist:
            raise RuntimeError(
                "DUET-DINO goal images have not been set for this environment. "
                "Provide task goal images from the eval side before calling infer()."
            )

        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self.autocast_dtype,
        ):
            pose = torch.from_numpy(curr_obs["ee_pose"]).unsqueeze(0).to(
                self.device, dtype=torch.float32, non_blocking=True
            )

            # print("EE Pose:", pose[0].cpu().numpy(), flush=True)
            # print(f"[DUET-DINO server] action step={self.step_count[env_id]}", flush=True)

            if self.log_inference_times:
                if pose.is_cuda:
                    torch.cuda.synchronize(pose.device)
                start_time = perf_counter()
            actions = self.world_model.infer_next_action(
                pose=pose,
                obs=curr_obs["external_image"],
                obs_wrist=curr_obs["wrist_image"],
                prev_action=self._env_prev_action.get(env_id),
                generator=self._env_generators.get(env_id),
                goal_rep=self._env_goal_rep[env_id],
                goal_rep_wrist=self._env_goal_rep_wrist[env_id],
                return_predictions=self.log_predictions,
                history=self._env_history.get(env_id) if self.history > 0 else None
            )
            if self.log_inference_times:
                if pose.is_cuda:
                    torch.cuda.synchronize(pose.device)
                self._inference_times[env_id].append(perf_counter() - start_time)

            if self.log_predictions:
                actions, predictions = actions
                self._prediction_log[env_id].append({
                    "env_id": env_id,
                    "step_count": self.step_count[env_id],
                    "actions": actions.detach().cpu().clone(),
                    "predictions": predictions,
                })

            all_actions = actions.detach().cpu().numpy().astype(np.float32)
            self._env_prev_action[env_id] = actions[1:] if actions.shape[0] > 1 else None
            

            self.step_count[env_id] += 1

        return_action = all_actions[0].copy()
        # isaac expects:  angle-axis / rotvec
        return_action[3:6] = Rotation.from_euler("xyz", all_actions[0][3:6], degrees=False).as_rotvec()
        # print("Return Action:", return_action, flush=True)

        return_actions = all_actions.copy()
        # isaac expects:  angle-axis / rotvec
        return_actions[:, 3:6] = Rotation.from_euler("xyz", all_actions[:, 3:6], degrees=False).as_rotvec()
        # print("All Actions:", actions, flush=True)

        return {"action": return_action, "actions": return_actions}

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            print(f"[DUET-DINO server] Listening on {self.host}:{self.port}", flush=True)

            while True:
                conn, addr = server_sock.accept()
                print(f"[DUET-DINO server] Client connected: {addr}")
                with conn:
                    while True:
                        try:
                            request = _recv_message(conn)
                            response = self._dispatch(request)
                        except ConnectionError:
                            print(f"[DUET-DINO server] Client disconnected: {addr}")
                            break
                        except Exception:
                            response = {"error": traceback.format_exc()}
                        try:
                            _send_message(conn, response)
                        except OSError:
                            print(f"[DUET-DINO server] Client disconnected: {addr}")
                            break

    def _dispatch(self, request: dict) -> dict:
        method = request.get("method", "infer")

        if method == "set_goal_images":
            self.set_goal_images(
                request["external_image"],
                request["wrist_image"],
                env_id=int(request.get("env_id", 0)),
                instruction=request.get("instruction", "goal"),
                run_idx=request.get("run_idx"),
            )
            return {"ok": True}

        if method == "reset":
            self.reset(env_id=int(request.get("env_id", 0)))
            return {"ok": True}

        if method == "infer":
            return self.infer(
                request["obs"], 
                request.get("instruction", ""), 
                env_id=int(request.get("env_id", 0))
            )
        
        if method == "metadata":
            return {"modelname": self.cfg["log"]["modelname"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load DUET-DINO once and serve inference over TCP.")
    parser.add_argument(
        "--cfg-file",
        help="DUET-DINO inference config yaml.",
        required=True
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve.")
    
    args = parser.parse_args()
    print(f"[DUET-DINO server] Loading model from {args.cfg_file}")
    policy = DUETDINOPolicy(cfg_file=args.cfg_file,
                                host=args.host, 
                                port=args.port).serve_forever()
