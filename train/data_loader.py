# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from logging import getLogger
from math import ceil

import io
import lmdb
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_paths,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    prefetch_factor=2,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    train_test_split=False,
    pair_camera_views=False,
    include_roboarena=False,
    decode_num_threads=1,
    dataset_len_multiplier=1,
):

    if not train_test_split:
        # -- full dataset class
        full_dataset = DROIDVideoDataset(
            data_paths=data_paths,
            frames_per_clip=frames_per_clip,
            transform=transform,
            fps=fps,
            camera_views=camera_views,
            frameskip=tubelet_size,
            camera_frame=camera_frame,
            pair_camera_views=pair_camera_views,
            include_roboarena=include_roboarena,
            decode_num_threads=decode_num_threads,
            dataset_len_multiplier=dataset_len_multiplier,
        )

        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            full_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

        # -- full dataloader
        data_loader = torch.utils.data.DataLoader(
            full_dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        return data_loader, dist_sampler, None, None

    # -- train dataset class
    train_dataset = DROIDVideoDataset(
        data_paths= [f"{data_path}/train" for data_path in data_paths],
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        pair_camera_views=pair_camera_views,
        include_roboarena=include_roboarena,
        decode_num_threads=decode_num_threads,
        dataset_len_multiplier=dataset_len_multiplier,
    )

    dist_train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # -- train dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=dist_train_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # -- test dataset class
    test_dataset = DROIDVideoDataset(
        data_paths= [f"{data_path}/test" for data_path in data_paths],
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        pair_camera_views=pair_camera_views,
        include_roboarena=include_roboarena,
        decode_num_threads=decode_num_threads,
        dataset_len_multiplier=dataset_len_multiplier,
    )

    dist_test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # -- test dataloader
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collator,
        sampler=dist_test_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return train_data_loader, dist_train_sampler, test_data_loader, dist_test_sampler


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_paths,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        pair_camera_views=False,
        include_roboarena=False,
        decode_num_threads=1,
        dataset_len_multiplier=1,
    ):
        self.data_paths = data_paths
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps # TODO: sample variable fps
        self.transform = transform
        self.camera_frame = camera_frame
        # Only require real extrinsics when model/camera-frame path uses them.
        self.use_extrinsics = camera_frame
        self.pair_camera_views = pair_camera_views
        self.include_roboarena = include_roboarena
        self.decode_num_threads = max(1, int(decode_num_threads))
        self.dataset_len_multiplier = max(1, int(dataset_len_multiplier))
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"
        self.roboarena_traj_name = "trajectory_npz"

        # samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        # self.samples = samples

        # -- lmdb paths
        # -- droid_raw
        data_path = data_paths[0]
        self.obs_lmdb_path = os.path.join(data_path, 'observations.lmdb')
        self.traj_lmdb_path = os.path.join(data_path, 'h5.lmdb')
        self.meta_lmdb_path = os.path.join(data_path, 'metadata.lmdb')
        # -- roboarena
        if self.include_roboarena:
            roboarena_dp = data_paths[1]
            self.roboarena_obs_lmdb_path = os.path.join(roboarena_dp, 'observations.lmdb')
            self.roboarena_traj_lmdb_path = os.path.join(roboarena_dp, 'traj.lmdb')
        
        self._obs_env = None
        self._traj_env = None
        self._meta_env = None
        self.roboarena_obs_env = None
        self._traj_env_roboarena = None
        self._lmdb_open_kwargs = dict(
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=4096,
        )

        # -- load meta keys for indexing
        meta_env = lmdb.open(self.meta_lmdb_path, 
                             subdir=False, 
                             readonly=True, 
                             lock=False)
        with meta_env.begin() as meta_txn:
            self.meta_keys = list(meta_txn.cursor().iternext(keys=True, values=False))
        meta_env.close()

        # -- optionally append RoboArena samples
        if self.include_roboarena:
            roboarena_traj_env = lmdb.open(self.roboarena_traj_lmdb_path, 
                                           **self._lmdb_open_kwargs)
            with roboarena_traj_env.begin() as traj_txn:
                roboarena_traj_keys = list(traj_txn.cursor().iternext(keys=True, values=False))
            roboarena_traj_env.close()

            self.meta_keys.extend(roboarena_traj_keys)
            
        self._entries = []
        roboarena_suffix = f"/{self.roboarena_traj_name}"
        for key in self.meta_keys:
            decoded = key.decode("utf-8")
            self._entries.append((key, decoded.rsplit("/", 1)[0], not decoded.endswith(roboarena_suffix)))
        self._base_len = len(self.meta_keys)

    def _get_envs(self):
        if self._obs_env is None:
            self._obs_env = lmdb.open(self.obs_lmdb_path, **self._lmdb_open_kwargs)
        if self._traj_env is None:
            self._traj_env = lmdb.open(self.traj_lmdb_path, **self._lmdb_open_kwargs)
        if self._meta_env is None:
            self._meta_env = lmdb.open(self.meta_lmdb_path, **self._lmdb_open_kwargs)
        
        if self.include_roboarena:
            if self._traj_env_roboarena is None:
                self._traj_env_roboarena = lmdb.open(
                    self.roboarena_traj_lmdb_path, **self._lmdb_open_kwargs
                )
            if self.roboarena_obs_env is None:
                self.roboarena_obs_env = lmdb.open(
                    self.roboarena_obs_lmdb_path, **self._lmdb_open_kwargs
                )

        return self._obs_env, self._traj_env, self._meta_env, \
                    self.roboarena_obs_env, self._traj_env_roboarena

    def __getitem__(self, index):
        index = int(index) % self._base_len
        json_key, path, isdroid = self._entries[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        last_error = None
        max_attempts = 10
        attempts = 0
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path, json_key, isdroid)
                loaded_video = True
            except Exception as e:
                last_error = e
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Failed to load sample after {max_attempts} attempts; last index={index}"
                    ) from last_error

                logger.info(f"error loading sample {index}: {e=}")
                index = np.random.randint(self._base_len)
                json_key, path, isdroid = self._entries[index]

        return buffer, actions, states, extrinsics, indices

    def poses_to_diffs(self, poses):
        """
        Convert per-frame absolute poses into per-timestep diffs.
        """
        xyz = poses[:, :3]  # [T, 3]
        eulers = poses[:, 3:6]  # [T, 3]

        # Batch conversion: Rotation objects support vectorized inputs.
        matrices = Rotation.from_euler("xyz", eulers, degrees=False).as_matrix()  # [T, 3, 3]
        xyz_diff = xyz[1:] - xyz[:-1]  # [T-1, 3]

        # Relative rotation between consecutive timesteps:
        #   R_rel[t] = R[t+1] @ R[t].T
        rel_mats = matrices[1:] @ np.transpose(matrices[:-1], (0, 2, 1))  # [T-1, 3, 3]
        angle_diff = Rotation.from_matrix(rel_mats).as_euler("xyz", degrees=False)  # [T-1, 3]

        closedness_delta = poses[1:, -1:] - poses[:-1, -1:]  # [T-1, 1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path, json_key, isdroid):
        obs_env, traj_env, meta_env, roboarena_obs_env, traj_env_roboarena = self._get_envs()

        if isdroid:
            with meta_env.begin() as meta_txn:
                json_data = meta_txn.get(json_key)
                if json_data is None:
                    raise Exception(f"No metadata for {path}")
                metadata = json.load(io.BytesIO(json_data))

            # -- load trajectory
            tpath = f"{path}/{self.h5_name}"
            with traj_env.begin() as traj_txn:
                traj_data = traj_txn.get(tpath.encode('utf-8'))
                if traj_data is None:
                    raise Exception(f"No trajectory data for {tpath=}")
            with h5py.File(io.BytesIO(traj_data), "r") as trajectory:
                states = np.concatenate(
                    [
                        np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                        np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
                    ],
                    axis=1,
                )  # [T, 7]

                lab_episode = "/".join(path.rstrip("/").split("/")[-2:])

                if self.pair_camera_views:
                    view_a = self.camera_views[0]
                    view_b = self.camera_views[1]

                    # Build one shared frame window valid for both views and state sequence.
                    mp4_a = metadata[view_a].split("recordings/MP4/")[-1]
                    mp4_b = metadata[view_b].split("recordings/MP4/")[-1]
                    vpath_a = f"{path}/recordings/MP4/{mp4_a}"
                    vpath_b = f"{path}/recordings/MP4/{mp4_b}"
                    with obs_env.begin() as obs_txn:
                        data_a = obs_txn.get(vpath_a.encode("utf-8"))
                        data_b = obs_txn.get(vpath_b.encode("utf-8"))

                        if data_a is None or data_b is None:
                            raise Exception(f"No video data for {lab_episode}")

                        vr_a = VideoReader(io.BytesIO(data_a), num_threads=self.decode_num_threads, ctx=cpu(0))
                        vr_b = VideoReader(io.BytesIO(data_b), num_threads=self.decode_num_threads, ctx=cpu(0))

                    vfps_a = vr_a.get_avg_fps()
                    fpc = self.frames_per_clip
                    fps = self.fps if self.fps is not None else vfps_a
                    fstp = ceil(vfps_a / fps)
                    nframes = int(fpc * fstp)
                    vlen_a = len(vr_a)
                    vlen_b = len(vr_b)
                    slen = len(states)
                    max_len = min(vlen_a, vlen_b, slen)
                    if max_len < nframes:
                        raise Exception(
                            f"Video/state pair too short: episode: {lab_episode}, exp: {nframes}, got: {max_len}"
                        )
                    max_start_pair = max_len - nframes
                    sf = np.random.randint(0, max_start_pair + 1) if max_start_pair > 0 else 0
                    indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

                    if self.use_extrinsics:
                        camera_name_a = mp4_a.split(".")[0]
                        camera_name_b = mp4_b.split(".")[0]
                        extrinsics_a = np.array(
                            trajectory["observation"]["camera_extrinsics"][f"{camera_name_a}_left"]
                        )[indices, :][:: self.frameskip]
                        extrinsics_b = np.array(
                            trajectory["observation"]["camera_extrinsics"][f"{camera_name_b}_left"]
                        )[indices, :][:: self.frameskip]
                    else:
                        extrinsics_a = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)
                        extrinsics_b = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)

                    buffer_a = vr_a.get_batch(indices).asnumpy()
                    buffer_b = vr_b.get_batch(indices).asnumpy()
                    if self.transform is not None:
                        buffer_a = self.transform(buffer_a)
                        buffer_b = self.transform(buffer_b)

                    states = states[indices, :][:: self.frameskip]
                    if self.camera_frame:
                        states = self.transform_frame(states, extrinsics_a)
                    actions = self.poses_to_diffs(states)
                    buffers = {view_a: buffer_a, view_b: buffer_b}
                    extrinsics = {view_a: extrinsics_a, view_b: extrinsics_b}
                    return buffers, actions, states, extrinsics, indices

                camera_view = self.camera_views[0]
                mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
                if self.use_extrinsics:
                    camera_name = mp4_name.split(".")[0]
                    extrinsics = np.array(trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"])

                vpath = f"{path}/recordings/MP4/{mp4_name}"
                with obs_env.begin() as obs_txn:
                    video_data = obs_txn.get(vpath.encode("utf-8"))
                    if video_data is None:
                        raise Exception(f"No video data for {lab_episode}")
                    vr = VideoReader(io.BytesIO(video_data), num_threads=self.decode_num_threads, ctx=cpu(0))

                vfps = vr.get_avg_fps()
                fpc = self.frames_per_clip
                fps = self.fps if self.fps is not None else vfps
                fstp = ceil(vfps / fps)
                nframes = int(fpc * fstp)
                vlen = len(vr)
                slen = len(states)
                max_len = min(vlen, slen)
                if max_len < nframes:
                    raise Exception(f"Video/state too short, episode: {lab_episode}, exp:{nframes}, got:{max_len}")
                max_start = max_len - nframes
                sf = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

                buffer = vr.get_batch(indices).asnumpy()
                if self.transform is not None:
                    buffer = self.transform(buffer)
                if self.use_extrinsics:
                    extrinsics = extrinsics[indices, :][:: self.frameskip]
                else:
                    extrinsics = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)

                states = states[indices, :][:: self.frameskip]
                if self.camera_frame:
                    states = self.transform_frame(states, extrinsics)
                actions = self.poses_to_diffs(states)
                return buffer, actions, states, extrinsics, indices
        
        else:
            # -- load trajectory
            with traj_env_roboarena.begin() as traj_txn:
                traj_data = traj_txn.get(json_key)
                if traj_data is None:
                    raise Exception(f"No trajectory data for {path}")
                trajectory = np.load(io.BytesIO(traj_data), allow_pickle=True)

            # TODO: load states
            source = trajectory["data"]
            if source is None:
                raise Exception(f"No valid data array in RoboArena trajectory for {path}")
            cartesian = np.stack([np.asarray(step["cartesian_position"]) for step in source], axis=0)
            gripper = np.stack([np.asarray(step["gripper_position"]) for step in source], axis=0).reshape(len(source), 1)
            states = np.concatenate([cartesian, gripper], axis=1).astype(np.float32)


            if self.pair_camera_views:
                view_a = self.camera_views[0]
                view_b = self.camera_views[1]
                vpath_a = f"{path}/{view_a}"
                vpath_b = f"{path}/{view_b}"
                with roboarena_obs_env.begin() as obs_txn:
                    data_a = obs_txn.get(vpath_a.encode("utf-8"))
                    data_b = obs_txn.get(vpath_b.encode("utf-8"))

                    if data_a is None or data_b is None:
                        raise Exception(f"No video data for {path}")

                    vr_a = VideoReader(io.BytesIO(data_a), num_threads=self.decode_num_threads, ctx=cpu(0))
                    vr_b = VideoReader(io.BytesIO(data_b), num_threads=self.decode_num_threads, ctx=cpu(0))

                vfps_a = vr_a.get_avg_fps()
                fpc = self.frames_per_clip
                fps = self.fps if self.fps is not None else vfps_a
                fstp = ceil(vfps_a / fps)
                nframes = int(fpc * fstp)
                vlen_a = len(vr_a)
                vlen_b = len(vr_b)
                slen = len(states)
                max_len = min(vlen_a, vlen_b, slen)
                if max_len < nframes:
                    raise Exception(
                        f"Video pair too short: episode: {path}, exp: {nframes}, got: {max_len}"
                    )
                max_start_pair = max_len - nframes
                sf = np.random.randint(0, max_start_pair + 1) if max_start_pair > 0 else 0
                indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

                extrinsics_a = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)
                extrinsics_b = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)

                buffer_a = vr_a.get_batch(indices).asnumpy()
                buffer_b = vr_b.get_batch(indices).asnumpy()
                if self.transform is not None:
                    buffer_a = self.transform(buffer_a)
                    buffer_b = self.transform(buffer_b)

                states = states[indices, :][:: self.frameskip]
                if self.camera_frame:
                    states = self.transform_frame(states, extrinsics_a)
                actions = self.poses_to_diffs(states)
                buffers = {view_a: buffer_a, view_b: buffer_b}
                extrinsics = {view_a: extrinsics_a, view_b: extrinsics_b}
                return buffers, actions, states, extrinsics, indices

            camera_view = self.camera_views[0]
            vpath = f"{path}/{camera_view}"
            with roboarena_obs_env.begin() as obs_txn:
                video_data = obs_txn.get(vpath.encode("utf-8"))
                if video_data is None:
                    raise Exception(f"No video data for {path}")
                vr = VideoReader(io.BytesIO(video_data), num_threads=self.decode_num_threads, ctx=cpu(0))

            vfps = vr.get_avg_fps()
            fpc = self.frames_per_clip
            fps = self.fps if self.fps is not None else vfps
            fstp = ceil(vfps / fps)
            nframes = int(fpc * fstp)
            vlen = len(vr)
            slen = len(states)
            max_len = min(vlen, slen)
            if max_len < nframes:
                raise Exception(f"Video/state too short, episode: {path}, exp:{nframes}, got:{max_len}")
            max_start = max_len - nframes
            sf = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

            buffer = vr.get_batch(indices).asnumpy()
            if self.transform is not None:
                buffer = self.transform(buffer)
            extrinsics = np.zeros((len(indices[:: self.frameskip]), 6), dtype=np.float32)

            states = states[indices, :][:: self.frameskip]
            if self.camera_frame:
                states = self.transform_frame(states, extrinsics)
            actions = self.poses_to_diffs(states)
            return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return self._base_len * self.dataset_len_multiplier
