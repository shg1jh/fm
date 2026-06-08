#!/bin/bash
set -euo pipefail

# Stage 2 long training: freeze DMC_HR core and train NeuralWrapper.
# Default is post-only. Set TRAIN_SCOPE=pre_post to also train NeuralPreProcessor.

PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

CORE_CHECKPOINT="${CORE_CHECKPOINT:-results/checkpoints_stage1b_light_long_from_stage1a/stage1b_light_latest.pth.tar}"
TRAIN_SCOPE="${TRAIN_SCOPE:-post_only}"
POST_AS_REF="${POST_AS_REF:-false}"
SAVE_DIR="${SAVE_DIR:-results/checkpoints_stage2_wrapper_long_from_stage1b}"
LOG_DIR="${LOG_DIR:-results/logs_stage2_wrapper_long_from_stage1b}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_FILE="${LOG_DIR}/train_stage2_wrapper_long_${TRAIN_SCOPE}_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 2 wrapper long training..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Core checkpoint: $CORE_CHECKPOINT"
echo "Train scope: $TRAIN_SCOPE"
echo "Post as ref: $POST_AS_REF"
echo "Save dir: $SAVE_DIR"
echo "Max steps: $MAX_STEPS"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$CORE_CHECKPOINT" ] || { echo "Missing core checkpoint: $CORE_CHECKPOINT"; exit 1; }

python training/train_stage2_wrapper_long.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --core_checkpoint "$CORE_CHECKPOINT" \
  --save_dir "$SAVE_DIR" \
  --checkpoint_prefix "stage2_wrapper_${TRAIN_SCOPE}" \
  --train_scope "$TRAIN_SCOPE" \
  --post_as_ref "$POST_AS_REF" \
  --epochs 20 \
  --max_steps "$MAX_STEPS" \
  --batch_size 1 \
  --worker 0 \
  --clip_len 4 \
  --crop_size 128 \
  --temporal_strides "1" \
  --lr 2e-6 \
  --lambda_bpp 0.0 \
  --lambda_identity 0.1 \
  --lambda_residual 0.02 \
  --lambda_core_distill 0.2 \
  --lambda_pre_delta 0.01 \
  --q_indexes "0 21 42 63" \
  --q_sample_mode random \
  --q_index_i_same_as_p true \
  --me_delta_scale 0.02 \
  --rate_gop_size 8 \
  --cuda true \
  --log_interval 20 \
  --save_interval "$SAVE_INTERVAL" \
  2>&1 | tee "$LOG_FILE"

echo "Stage 2 wrapper long training finished."
