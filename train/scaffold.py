# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 DUET-DINO contributors.
#
# Adapted from V-JEPA2 (https://github.com/facebookresearch/vjepa2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    return importlib.import_module(f"train.train_predictor").main(args=args, resume_preempt=resume_preempt)
