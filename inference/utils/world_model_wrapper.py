# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F

from .mpc_utils_single import cem as cem_single
from .mpc_utils_dual import cem as cem_dual
from .mpc_utils_dual import compute_new_pose

class WorldModel(object):

    def __init__(
        self,
        encoder,
        predictor,
        inferred_mode,
        use_dinov3_encoder,
        tokens_per_frame,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "elite_temperature": 0.25,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_mean_rot": 0.25,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.15,
            "momentum_std_rot": 0.85,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.05,
            "maxrotnorm": 0.314,
            "maxgrippernorm": 0.05,
            "warm_starting": False
        },
        normalize_reps=True,
        device="cuda:0",
        transform=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.inferred_mode = inferred_mode
        self.use_dinov3_encoder = use_dinov3_encoder

        self.normalize_reps = normalize_reps
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args
        self.transform = transform

        self.goal_rep = None
        self.goal_rep_wrist = None

    def reset_logs(self, 
                    goal_rep=None, 
                    goal_rep_wrist=None):
        self.goal_rep = goal_rep
        self.goal_rep_wrist = goal_rep_wrist

    # LATENT SPACE ENCODER
    def encode(self, clip):
        
        # [3, 720, 1280]  -> [1, 720, 1280, 3] 
        if clip.dim() == 3 and clip.shape[0] == 3:
            clip = torch.permute(clip, (1, 2, 0)).unsqueeze(0)
        elif clip.dim() == 3 and clip.shape[2] == 3:
            clip = clip.unsqueeze(0)
        elif clip.dim() == 4 and clip.shape[1] == 3:
            clip = torch.permute(clip, (0, 2, 3, 1))
        elif clip.dim() == 4 and clip.shape[3] == 3:
            pass
        else:
            raise ValueError(f"Unexpected clip shape: {clip.shape}")



        # [1, 720, 1280, 3] -> [1, 1, 720, 1408, 3]
        # [1, 3, 1, 256, 1408] i.e, [B, C, T, Patches, dim]
        clip = (self.transform(clip)[None, :]).to(
                            device=self.device, 
                            dtype=torch.float, 
                            non_blocking=True
                )
        
        # Pre-trained VJEPA 2 ENCODER: 
        # [1, 3, 1, 256, 1408] -> [1, 256, 1408]      
        B, C, T, H, W = clip.size()
        encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        with torch.no_grad():
            clip_bt = clip.permute(0, 2, 1, 3, 4).flatten(0, 1)
            
            if self.use_dinov3_encoder:
                # DINOv3 encoder expects [B, C, H, W] and uses forward_features.
                h = encoder.forward_features(clip_bt)["x_norm_patchtokens"]
            else:
                # VJEPA2 encoder expects [B*T, C, 2, H, W].
                clip_bt = clip_bt.unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = encoder(clip_bt)
            
            h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
            if self.normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
        return h

    # MODEL-BASED PLANNING WITH CEM IN LATENT SPACE
    def infer_next_action(self, 
                          pose=None, 
                          obs=None,
                          obs_wrist=None,
                          prev_action=None,
                          generator=None,
                          *,
                          goal_rep=None,
                          goal_rep_wrist=None,
                          return_predictions=False,
                          ):

        # Request-specific goals keep interleaved environments independent.
        # Retain reset_logs() goals for callers that do not supply overrides.
        goal_rep = self.goal_rep if goal_rep is None else goal_rep
        goal_rep_wrist = self.goal_rep_wrist if goal_rep_wrist is None else goal_rep_wrist

        def step_predictor(reps, actions, poses):

            # new representation prediction from predictor
            with torch.no_grad():
                
                if self.inferred_mode == "duet":
                    model = self.predictor["duet"]

                    B, T, N_T, D = reps["side"].size()

                    side_next, wrist_next = model(
                        reps["side"].flatten(1, 2),
                        reps["wrist"].flatten(1, 2),
                        actions,
                        poses,
                    )
                    side_next = side_next[:, -self.tokens_per_frame :]
                    wrist_next = wrist_next[:, -self.tokens_per_frame :]

                    if self.normalize_reps:
                        side_next = F.layer_norm(side_next, (side_next.size(-1),))
                        wrist_next = F.layer_norm(wrist_next, (wrist_next.size(-1),))

                    next_rep = {
                        "side": side_next.view(B, 1, N_T, D),
                        "wrist": wrist_next.view(B, 1, N_T, D),
                    }

                elif self.inferred_mode == "independent_dual":
                    side_model = self.predictor['side']
                    wrist_model = self.predictor['wrist']
                    B, T, N_T, D = reps["side"].size()
                    side_next = side_model(
                        reps["side"].flatten(1, 2),
                        actions,
                        poses,
                    )[:, -self.tokens_per_frame :]
                    wrist_next = wrist_model(
                        reps["wrist"].flatten(1, 2),
                        actions,
                        poses,
                    )[:, -self.tokens_per_frame :]

                    if self.normalize_reps:
                        side_next = F.layer_norm(side_next, (side_next.size(-1),))
                        wrist_next = F.layer_norm(wrist_next, (wrist_next.size(-1),))

                    next_rep = {
                        "side": side_next.view(B, 1, N_T, D),
                        "wrist": wrist_next.view(B, 1, N_T, D),
                    }

                elif self.inferred_mode == "side":
                    model = self.predictor['side']
                    B, T, N_T, D = reps.size()
                    next_rep = model(reps.flatten(1, 2), actions, poses)[:, -self.tokens_per_frame :]

                    if self.normalize_reps:
                        next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
                    next_rep = next_rep.view(B, 1, N_T, D)

                elif self.inferred_mode == "wrist":
                    model = self.predictor['wrist']
                    B, T, N_T, D = reps.size()
                    next_rep = model(reps.flatten(1, 2), actions, poses)[:, -self.tokens_per_frame :]

                    if self.normalize_reps:
                        next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
                    next_rep = next_rep.view(B, 1, N_T, D) # [25, 1, 256, 1408]

            next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
            
            return next_rep, next_pose

        if self.inferred_mode in ["independent_dual", "duet"]:
            rep = self.encode(obs)
            wrist_rep = self.encode(obs_wrist)
            
            mpc_action = cem_dual( 
                context_frame={"side": rep,"wrist": wrist_rep},
                context_pose=pose,
                goal_frame_side=goal_rep,
                goal_frame_wrist=goal_rep_wrist,
                world_model=step_predictor,
                prev_action=prev_action,
                generator=generator,
                **self.mpc_args,
            )

        else:
            rep = self.encode(obs)
            wrist_rep = self.encode(obs_wrist)
            mpc_action = cem_single( 
                context_frame=rep if self.inferred_mode == "side" else wrist_rep,
                context_pose=pose,
                goal_frame=goal_rep if self.inferred_mode == "side" else goal_rep_wrist,
                world_model=step_predictor,
                prev_action=prev_action,
                generator=generator,
                **self.mpc_args,
            )

        if not return_predictions:
            return mpc_action

        dual_view = self.inferred_mode in ("independent_dual", "duet")
        frames = {
            "side": rep.unsqueeze(1),
            "wrist": wrist_rep.unsqueeze(1),
        } if dual_view else (
            rep if self.inferred_mode == "side" else wrist_rep
        ).unsqueeze(1)

        poses = pose.unsqueeze(1)

        with torch.no_grad():
            for h in range(mpc_action.size(0)):
                next_frames, next_pose = step_predictor(
                    frames, mpc_action[None, :h + 1], poses
                )
                if dual_view:
                    for view in frames:
                        frames[view] = torch.cat([frames[view], next_frames[view]], dim=1)
                else:
                    frames = torch.cat([frames, next_frames], dim=1)
                poses = torch.cat([poses, next_pose], dim=1)

        frames = frames if dual_view else {self.inferred_mode: frames}
        return mpc_action, {
            view: rollout[:, 1:].squeeze(0).detach().cpu() for view, rollout in frames.items()
        }
