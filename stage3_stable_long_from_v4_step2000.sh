#!/bin/bash
set -euo pipefail

# Stable Stage 3 long-clip training from the best Stage 1 v4 delta checkpoint.
# This script intentionally saves every 100 optimizer steps. Starting from
# step2000 with --max_new_steps 300 will produce:
#   stage3_stable_step2100.pth.tar
#   stage3_stable_step2200.pth.tar
#   stage3_stable_step2300.pth.tar
# plus stage3_stable_latest.pth.tar.

PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
FM_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

RESUME_CHECKPOINT="results/checkpoints_stage1_hr_v4_delta/stage1_hr_step2000.pth.tar"

SAVE_DIR="results/checkpoints_stage3_stable_long_from_v4_step2000"
LOG_DIR="results/logs_stage3_stable_long_from_v4_step2000"
LOG_FILE="${LOG_DIR}/train_stage3_stable_long_from_v4_step2000_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 3 stable-long training..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Official P-frame checkpoint: $FM_CHECKPOINT"
echo "Resume checkpoint: $RESUME_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$FM_CHECKPOINT" ] || { echo "Missing official P-frame checkpoint: $FM_CHECKPOINT"; exit 1; }
[ -f "$RESUME_CHECKPOINT" ] || { echo "Missing resume checkpoint: $RESUME_CHECKPOINT"; exit 1; }

python training/train_stage3_stable_long.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --model_path_p "$FM_CHECKPOINT" \
  --resume "$RESUME_CHECKPOINT" \
  --resume_optimizer false \
  --required_resume_step 2000 \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_new_steps 300 \
  --batch_size 1 \
  --worker 0 \
  --clip_len 6 \
  --crop_size 128 \
  --temporal_strides "1" \
  --lr 5e-8 \
  --train_scope stable_core \
  --q_indexes "0 21 42 63" \
  --q_sample_mode random \
  --q_index_i_same_as_p true \
  --lambda_bpp 0.0 \
  --lambda_mc 0.0 \
  --lambda_identity 0.1 \
  --lambda_distill_start 2.0 \
  --lambda_distill_end 0.25 \
  --beta_balance 0.0 \
  --late_frame_gamma 1.0 \
  --detach_dpb true \
  --me_delta_scale 0.02 \
  --rate_gop_size 8 \
  --cuda true \
  --force_torch_warp true \
  --log_interval 10 \
  --save_interval 100 \
  2>&1 | tee "$LOG_FILE"

echo "Stage 3 stable-long training finished."
echo "Expected checkpoints:"
echo "  $SAVE_DIR/stage3_stable_step2100.pth.tar"
echo "  $SAVE_DIR/stage3_stable_step2200.pth.tar"
echo "  $SAVE_DIR/stage3_stable_step2300.pth.tar"
echo "  $SAVE_DIR/stage3_stable_latest.pth.tar"
