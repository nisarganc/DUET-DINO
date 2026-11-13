# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)

def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor

def round_rotation_elements(rot, threshold):
    rot_norm = rot.norm(dim=-1, keepdim=True)
    new_rot = torch.where(
        rot_norm < threshold,
        torch.zeros_like(rot),
        rot,
    )
    return new_rot

def relative_goal_loss(candidate_loss, reference_loss, eps=1e-6):
    """Normalize candidate goal loss by a fixed predicted no-op goal loss (reference_loss).
    The reference loss is computed by rolling the same world model forward for the same horizon 
    with zero actions. A value of 1.0 therefore means the candidate is predicted to be as far 
    from the goal as doing nothing, values below 1.0 are better than doing nothing, and values 
    above 1.0 are worse than doing nothing."""
    return candidate_loss / reference_loss.clamp_min(eps)

def weighted_elite_stats(actions, loss, indices, temperature=0.25):

    # Select top-k actions and corresponding losses.
    selected_actions = actions[indices]
    selected_loss = loss[indices]

    # Normalize the spread of losses within the selected top-k set.
    loss_std = selected_loss.std(
        unbiased=False
    ).clamp_min(1e-6)

    normalized_loss = (
        selected_loss - selected_loss.min()
    ) / loss_std

    weights = torch.softmax(
        -normalized_loss / temperature,
        dim=0,
    )

    view_shape = (
        weights.size(0),
    ) + (1,) * (selected_actions.dim() - 1)

    weights = weights.view(view_shape)

    mean = (
        selected_actions * weights
    ).sum(dim=0)

    var = (
        (selected_actions - mean) ** 2 * weights
    ).sum(dim=0)

    return mean, torch.sqrt(
        var.clamp_min(1e-8)
    )

def compute_new_pose(pose, action):
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
        for theta in thetas
    ]
    delta_matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
        for theta in delta_thetas
    ]
    angle_diff = [
        delta_matrices[t] @ matrices[t] for t in range(len(matrices))
    ]
    angle_diff = [
        Rotation.from_matrix(mat).as_euler("xyz", degrees=False)
        for mat in angle_diff
    ]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate(
        [new_xyz, new_angle, new_closedness],
        axis=-1,
    )
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]

def poses_to_diff(start, end):

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler(
        "xyz",
        s_thetas,
        degrees=False,
    ).as_matrix()
    e_rotation = Rotation.from_euler(
        "xyz",
        e_thetas,
        degrees=False,
    ).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(
        rotation_diff
    ).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate(
        [xyz_diff, theta_diff, gripper_diff],
        axis=0,
    )
    return torch.from_numpy(action)

def cem(
    context_obs,
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_mean_rot=0.25,
    momentum_std=0.95,
    momentum_std_rot=0.85,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    elite_temperature=0.25,
    maxnorm=0.05,
    maxrotnorm=0.314,
    maxgrippernorm=0.75,
    axis={},
    warm_starting=False,
    prev_action=None,
    generator=None,
    relative_loss_eps=1e-6,
):

    def sample_action_traj():
        """Sample several action trajectories."""
        action_traj = None
        frame_traj = context_frame
        pose_traj = context_pose

        for h in range(rollout):

            action_samples = (
                torch.randn(samples, mean.size(1), device=mean.device, generator=generator,) * std[h] + mean[h]
            )

            action_samples[:, :3] = torch.clip(action_samples[:, :3], 
                min=-maxnorm, 
                max=maxnorm,
            )
            action_samples[:, 3:6] = torch.clip(
                action_samples[:, 3:6],
                min=-maxrotnorm,
                max=maxrotnorm,
            )
            action_samples[:, -1:] = torch.clip(
                action_samples[:, -1:],
                min=-maxgrippernorm,
                max=maxgrippernorm,
            )

            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

            action_samples = action_samples[:, None]

            action_traj = (
                torch.cat(
                    [action_traj, action_samples],
                    dim=1,
                )
                if action_traj is not None
                else action_samples
            )

            next_frame, next_pose = world_model(
                frame_traj,
                action_traj,
                pose_traj,
            )

            frame_traj = torch.cat(
                [frame_traj, next_frame],
                dim=1,
            )
            pose_traj = torch.cat(
                [pose_traj, next_pose],
                dim=1,
            )

        return (
            action_traj,
            frame_traj,
        )

    def rollout_zero_action_reference(
        reference_context_frame,
        reference_context_pose,
    ):
        """Roll out the world model for the planning with zero actions."""

        action_traj = None
        frame_traj = reference_context_frame.unsqueeze(0)
        pose_traj = reference_context_pose.unsqueeze(0)

        for _ in range(rollout):

            zero_action = torch.zeros(
                (1, 1, 7),
                device=reference_context_pose.device,
                dtype=reference_context_pose.dtype,
            )

            action_traj = (
                torch.cat(
                    [action_traj, zero_action],
                    dim=1,
                )
                if action_traj is not None
                else zero_action
            )

            next_frame, next_pose = world_model(
                frame_traj,
                action_traj,
                pose_traj,
            )

            frame_traj = torch.cat(
                [frame_traj, next_frame],
                dim=1,
            )
            pose_traj = torch.cat(
                [pose_traj, next_pose],
                dim=1,
            )

        return frame_traj[:, -1]

    def select_topk_action_traj(
        reference_loss,
        final_state,
        goal_state,
        actions,
    ):
        """Select elites using scale-normalized single-view cost."""

        absolute_loss = l1(
            final_state.flatten(1),
            goal_state.flatten(1),
        )


        loss = relative_goal_loss(
            absolute_loss,
            reference_loss,
            eps=relative_loss_eps,
        )

        indices = loss.topk(
            topk,
            largest=False,
        ).indices

        # selected_actions = actions[indices]
        # return selected_actions
    
        # now obtain mean and var deom elite samples using weighted softmax
        mean, std = weighted_elite_stats(
            actions,
            loss,
            indices,
            temperature=elite_temperature,
        )

        return mean, std

    zero_final = (
        rollout_zero_action_reference(
            context_frame,
            context_pose,
        )
    )

    reference_loss = l1(
        zero_final.flatten(1),
        goal_frame.flatten(1),
    ).mean().detach()

    # ---------------------------------------------------------
    # Expand MPC inputs over candidate samples.
    # ---------------------------------------------------------

    context_obs = context_obs.repeat(
        samples,
        1,
        1,
        1,
    )

    context_frame = context_frame.repeat(
        samples,
        1,
        1,
        1,
    )

    goal_frame = goal_frame.repeat(
        samples,
        1,
        1,
        1,
    )

    context_pose = context_pose.repeat(
        samples,
        1,
        1,
    )

    # ---------------------------------------------------------
    # Initial proposal distribution.
    # ---------------------------------------------------------

    mean = torch.zeros(
        (rollout, 7),
        device=context_frame.device,
    )

    std = torch.cat(
        [
            torch.ones(
                (rollout, 3),
                device=context_frame.device,
            )
            * maxnorm,
            torch.ones(
                (rollout, 3),
                device=context_frame.device,
            )
            * maxrotnorm,
            torch.ones(
                (rollout, 1),
                device=context_frame.device,
            ) #* maxgrippernorm,
        ],
        dim=-1,
    )

    if warm_starting:
        if prev_action is not None:
            mean[:-1] = prev_action

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    # ---------------------------------------------------------
    # CEM optimization.
    # ---------------------------------------------------------

    for step in tqdm(range(cem_steps), disable=True):

        action_traj, frame_traj = (
            sample_action_traj()
        )

        mean_selected_actions, std_selected_actions = select_topk_action_traj(
            reference_loss=reference_loss,
            final_state=frame_traj[:, -1],
            goal_state=goal_frame,
            actions=action_traj,
        )

        mean = torch.cat(
            [
                (
                    mean_selected_actions[..., :3]
                    * (1.0 - momentum_mean)
                    + mean[..., :3] * momentum_mean
                ),
                (
                    mean_selected_actions[..., 3:6]
                    * (1.0 - momentum_mean_rot)
                    + mean[..., 3:6] * momentum_mean_rot
                ),
                (
                    mean_selected_actions[..., -1:]
                    * (1.0 - momentum_mean_gripper)
                    + mean[..., -1:]
                    * momentum_mean_gripper
                ),
            ],
            dim=-1,
        )

        std = torch.cat(
            [
                (
                    std_selected_actions[..., :3]
                    * (1.0 - momentum_std)
                    + std[..., :3] * momentum_std
                ),
                (
                    std_selected_actions[..., 3:6]
                    * (1.0 - momentum_std_rot)
                    + std[..., 3:6] * momentum_std_rot
                ),
                (
                    std_selected_actions[..., -1:]
                    * (1.0 - momentum_std_gripper)
                    + std[..., -1:]
                    * momentum_std_gripper
                ),
            ],
            dim=-1,
        )

    # ---------------------------------------------------------
    # Return the mean of the final proposal.
    # ---------------------------------------------------------

    new_action = torch.cat(
        [
            mean[..., :3],
            round_small_elements(mean[..., 3:6], 0.04),
            round_small_elements(mean[..., -1:], 0.15),
        ],
        dim=-1,
    )
    return new_action
