#!/bin/bash
set -euo pipefail

# Project root on the remote server.
PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

# Keep the original dataset and DCVC-FM official checkpoint paths.
TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
FM_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

# Skip Stage 2 and start from the stable Stage 1 v4 delta checkpoint.
RESUME_CHECKPOINT="results/checkpoints_stage1_hr_v4_delta/stage1_hr_step2000.pth.tar"

# Stage 3-lite probe outputs.
SAVE_DIR="results/checkpoints_stage3_lite_from_v4_step2000"
LOG_DIR="results/logs_stage3_lite_from_v4_step2000"
LOG_FILE="${LOG_DIR}/train_stage3_lite_from_v4_step2000_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 3-lite probe from v4 step2000..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Official P-frame checkpoint: $FM_CHECKPOINT"
echo "Resume checkpoint: $RESUME_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$FM_CHECKPOINT" ] || { echo "Missing official P-frame checkpoint: $FM_CHECKPOINT"; exit 1; }
[ -f "$RESUME_CHECKPOINT" ] || { echo "Missing required v4 step2000 checkpoint: $RESUME_CHECKPOINT"; exit 1; }

python training/train_stage1_core_hr.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --model_path_p "$FM_CHECKPOINT" \
  --resume "$RESUME_CHECKPOINT" \
  --resume_optimizer false \
  --required_resume_step 2000 \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_new_steps 200 \
  --batch_size 1 \
  --worker 0 \
  --clip_len 2 \
  --crop_size 128 \
  --lr 1e-7 \
  --lambda_bpp 0.005 \
  --lambda_mc 0.0 \
  --lambda_identity 0.1 \
  --lambda_distill 5.0 \
  --beta_balance 0.0 \
  --train_scope stage3_lite \
  --me_delta_scale 0.02 \
  --q_index_i 32 \
  --q_index_p 32 \
  --rate_gop_size 8 \
  --cuda true \
  --force_torch_warp true \
  --log_interval 10 \
  --save_interval 100 \
  2>&1 | tee "$LOG_FILE"

echo "Stage 3-lite probe finished."
