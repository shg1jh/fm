# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Long Stage 1B entry point: HRS plus a small safe core unfreeze."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_stage3_stable_long import main


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
        "train_scope": "stage1b_light",
        "checkpoint_prefix": "stage1b_light",
        "resume_optimizer": "false",
        "lr": "2e-7",
        "lambda_bpp": "0.002",
        "lambda_identity": "0.1",
        "lambda_distill_start": "5.0",
        "lambda_distill_end": "2.0",
        "clip_len": "6",
        "crop_size": "128",
        "temporal_strides": "1",
        "q_indexes": "0 21 42 63",
        "q_sample_mode": "random",
        "q_index_i_same_as_p": "true",
        "late_frame_gamma": "1.0",
        "detach_dpb": "true",
        "me_delta_scale": "0.02",
        "force_torch_warp": "true",
    })
    main()
