#!/bin/bash
set -euo pipefail

# Project root on the remote server.
PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

# Keep the original dataset and DCVC-FM official checkpoint paths.
TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
FM_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

# Stage 1 v4 HRS training outputs.
SAVE_DIR="results/checkpoints_stage1_hr_v4_delta"
LOG_DIR="results/logs_stage1_hr_v4_delta"
LOG_FILE="${LOG_DIR}/train_stage1_core_hr_v4_delta_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 1 v4 delta-only training..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "P-frame checkpoint: $FM_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Log file: $LOG_FILE"

python training/train_stage1_core_hr.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --model_path_p "$FM_CHECKPOINT" \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_steps 5000 \
  --batch_size 1 \
  --worker 0 \
  --clip_len 2 \
  --crop_size 128 \
  --lr 5e-6 \
  --lambda_bpp 0.0 \
  --lambda_mc 0.0 \
  --lambda_identity 0.1 \
  --lambda_distill 5.0 \
  --beta_balance 0.0 \
  --train_scope delta \
  --me_delta_scale 0.02 \
  --q_index_i 32 \
  --q_index_p 32 \
  --rate_gop_size 8 \
  --cuda true \
  --force_torch_warp true \
  --log_interval 10 \
  --save_interval 500 \
  2>&1 | tee "$LOG_FILE"

echo "Stage 1 HRS training finished."
