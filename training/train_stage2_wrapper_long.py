# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Long Stage 2 entry point for NeuralWrapper training."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_stage2_post_only import main


def apply_default_args(defaults):
    existing = set(sys.argv[1:])
    insert_at = 1
    for key, value in defaults.items():
        flag = f"--{key}"
        if flag not in existing:
            sys.argv[insert_at:insert_at] = [flag, str(value)]
            insert_at += 2


if __name__ == "__main__":
    apply_default_args({
        "train_scope": "post_only",
        "checkpoint_prefix": "stage2_wrapper",
        "post_as_ref": "false",
        "clip_len": "4",
        "crop_size": "128",
        "temporal_strides": "1",
        "lr": "2e-6",
        "lambda_bpp": "0.0",
        "lambda_identity": "0.1",
        "lambda_residual": "0.02",
        "lambda_core_distill": "0.2",
        "lambda_pre_delta": "0.0",
        "q_indexes": "0 21 42 63",
        "q_sample_mode": "random",
        "q_index_i_same_as_p": "true",
        "me_delta_scale": "0.02",
    })
    main()
