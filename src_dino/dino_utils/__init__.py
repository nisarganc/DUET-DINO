# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# From DINOv3 (https://github.com/facebookresearch/dinov3).
# This source code is licensed under the DINOv3 License Agreement
# found in licenses/DINOv3.txt at the root of this repository.

from .dtype import as_torch_dtype
from .utils import (
    cat_keep_shapes,
    count_parameters,
    fix_random_seeds,
    get_conda_env,
    get_sha,
    named_apply,
    named_replace,
    uncat_with_shapes,
)
