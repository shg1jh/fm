#!/bin/bash
set -euo pipefail

# Project root on the remote server.
PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

# Keep the original dataset and official I-frame checkpoint paths.
TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

# Stage 2 post-only starts from the stable Stage 1 v4 delta core.
CORE_CHECKPOINT="results/checkpoints_stage1_hr_v4_delta/stage1_hr_step2000.pth.tar"

SAVE_DIR="results/checkpoints_stage2_post_only_from_v4_step2000"
LOG_DIR="results/logs_stage2_post_only_from_v4_step2000"
LOG_FILE="${LOG_DIR}/train_stage2_post_only_from_v4_step2000_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 2 post-only training..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Core checkpoint: $CORE_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$CORE_CHECKPOINT" ] || { echo "Missing Stage 1 core checkpoint: $CORE_CHECKPOINT"; exit 1; }

python training/train_stage2_post_only.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --core_checkpoint "$CORE_CHECKPOINT" \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_steps 1000 \
  --batch_size 1 \
  --worker 0 \
  --clip_len 2 \
  --crop_size 128 \
  --lr 5e-6 \
  --lambda_identity 0.05 \
  --lambda_residual 0.01 \
  --lambda_core_distill 0.1 \
  --me_delta_scale 0.02 \
  --q_index_i 32 \
  --q_index_p 32 \
  --rate_gop_size 8 \
  --cuda true \
  --log_interval 10 \
  --save_interval 500 \
  2>&1 | tee "$LOG_FILE"

echo "Stage 2 post-only training finished."
