# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint
from pathlib import Path

import yaml

from train.scaffold import main as predictor_main
from train.train_utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", 
                    type=str, 
                    help="name of config file to load", 
                    default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    help="which devices to use on local machine",
)


def process_main(rank, fname, world_size, devices):
    import os

    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        # torchrun path: force 1-GPU-per-rank mapping even if CUDA_VISIBLE_DEVICES is pre-set.
        local_rank = int(local_rank_env)
        device_index = local_rank if local_rank < len(devices) else rank
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[device_index].split(":")[-1])
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
        device_index = local_rank if local_rank < len(devices) else rank
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[device_index].split(":")[-1])

    import logging

    from train.train_utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the training with loaded config
    if params.get("train") == "predictor":
        predictor_main(args=params)
    else:
        pass


def local_process_main(rank, fname, world_size, devices):
    # Give locally spawned workers the same rank metadata as torchrun.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    process_main(rank, fname, world_size, devices)


def launch(args):
    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torchrun_rank = int(os.environ.get("RANK", "0"))
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_rank = int(os.environ.get("SLURM_PROCID", "0"))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        process_main(rank=torchrun_rank, fname=args.fname, world_size=torchrun_world_size, devices=args.devices)
    elif slurm_ntasks > 1:
        process_main(rank=slurm_rank, fname=args.fname, world_size=slurm_ntasks, devices=args.devices)
    else:
        # Local workers share one rendezvous endpoint. Preserve user overrides.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "37129")
        num_gpus = len(args.devices)
        if num_gpus == 1:
            local_process_main(0, args.fname, 1, args.devices)
            return
        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(num_gpus):
            process = ctx.Process(target=local_process_main, args=(rank, args.fname, num_gpus, args.devices))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("A local training worker failed")


if __name__ == "__main__":
    launch(parser.parse_args())
