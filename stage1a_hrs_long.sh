#!/bin/bash
set -euo pipefail

# Stage 1A RD-aware probe: train only the safest HRS delta path from official DCVC-FM.

PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

TRAIN_ROOT="/root/autodl-tmp/DCVC-DC/train"
FM_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

SAVE_DIR="${SAVE_DIR:-results/checkpoints_stage1a_hrs_rd_probe_v3_from_official}"
LOG_DIR="${LOG_DIR:-results/logs_stage1a_hrs_rd_probe_v3_from_official}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
LOG_FILE="${LOG_DIR}/train_stage1a_hrs_rd_probe_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Starting NVC-FM-HR Stage 1A RD-aware HRS probe..."
echo "Project dir: $PROJECT_DIR"
echo "Train root: $TRAIN_ROOT"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Official P-frame checkpoint: $FM_CHECKPOINT"
echo "Save dir: $SAVE_DIR"
echo "Max steps: $MAX_STEPS"
echo "Log file: $LOG_FILE"

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$FM_CHECKPOINT" ] || { echo "Missing official P-frame checkpoint: $FM_CHECKPOINT"; exit 1; }

python training/train_stage1a_hrs_long.py \
  --train_root "$TRAIN_ROOT" \
  --model_path_i "$INTRA_CHECKPOINT" \
  --model_path_p "$FM_CHECKPOINT" \
  --save_dir "$SAVE_DIR" \
  --epochs 20 \
  --max_steps "$MAX_STEPS" \
  --batch_size 1 \
  --worker 0 \
  --clip_len 6 \
  --crop_size 128 \
  --temporal_strides "1" \
  --train_scope hrs_delta_only \
  --checkpoint_prefix stage1a_hrs_rd \
  --lr 5e-7 \
  --lambda_bpp 0.003 \
  --lambda_mc 0.0 \
  --lambda_identity 0.1 \
  --lambda_distill_start 5.0 \
  --lambda_distill_end 5.0 \
  --lambda_log_bpp_distill 0.02 \
  --lambda_bit_ceiling 0.03 \
  --bit_ceiling_ratio 1.05 \
  --lambda_me_reg 0.05 \
  --lambda_feature_reg 0.0 \
  --lambda_expert_reg 0.0 \
  --beta_balance 0.0 \
  --q_indexes "0 21 42 63" \
  --q_sample_mode random \
  --q_index_i_same_as_p true \
  --late_frame_gamma 1.0 \
  --detach_dpb true \
  --me_delta_scale 0.02 \
  --max_alpha_hist 0.05 \
  --max_alpha_expert 0.03 \
  --hrs_gate_init -3.0 \
  --rate_gop_size 8 \
  --cuda true \
  --force_torch_warp true \
  --log_interval 20 \
  --save_interval "$SAVE_INTERVAL" \
  2>&1 | tee "$LOG_FILE"

echo "Stage 1A RD-aware HRS probe finished."
