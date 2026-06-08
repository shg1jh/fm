#!/bin/bash
set -euo pipefail

# Stage 3 long training: light joint fine-tuning of HRS, feature_adaptor,
# context_fusion tail, and recon_generation tail.

PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
FM_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-results/checkpoints_stage1b_light_long_from_stage1a/stage1b_light_latest.pth.tar}"
SAVE_DIR="${SAVE_DIR:-results/checkpoints_stage3_joint_long_from_stage1b}"
LOG_DIR="${LOG_DIR:-results/logs_stage3_joint_long_from_stage1b}"
MAX_NEW_STEPS="${MAX_NEW_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_FILE="${LOG_DIR}/train_stage3_joint_long_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 3 joint long training..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Official P-frame checkpoint: $FM_CHECKPOINT"
echo "Resume checkpoint: $RESUME_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Max new steps: $MAX_NEW_STEPS"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$FM_CHECKPOINT" ] || { echo "Missing official P-frame checkpoint: $FM_CHECKPOINT"; exit 1; }
[ -f "$RESUME_CHECKPOINT" ] || { echo "Missing resume checkpoint: $RESUME_CHECKPOINT"; exit 1; }

EXTRA_ARGS=()
if [ -n "${REQUIRED_RESUME_STEP:-}" ]; then
  EXTRA_ARGS+=(--required_resume_step "$REQUIRED_RESUME_STEP")
fi

python training/train_stage3_joint_long.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --model_path_p "$FM_CHECKPOINT" \
  --resume "$RESUME_CHECKPOINT" \
  --resume_optimizer false \
  "${EXTRA_ARGS[@]}" \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_new_steps "$MAX_NEW_STEPS" \
  --batch_size 1 \
  --worker 0 \
  --clip_len 6 \
  --crop_size 128 \
  --temporal_strides "1" \
  --lr 5e-8 \
  --lambda_bpp 0.002 \
  --lambda_mc 0.0 \
  --lambda_identity 0.1 \
  --lambda_distill_start 3.0 \
  --lambda_distill_end 1.0 \
  --beta_balance 0.0 \
  --q_indexes "0 21 42 63" \
  --q_sample_mode random \
  --q_index_i_same_as_p true \
  --late_frame_gamma 1.0 \
  --detach_dpb true \
  --me_delta_scale 0.02 \
  --rate_gop_size 8 \
  --cuda true \
  --force_torch_warp true \
  --log_interval 20 \
  --save_interval "$SAVE_INTERVAL" \
  2>&1 | tee "$LOG_FILE"

echo "Stage 3 joint long training finished."
