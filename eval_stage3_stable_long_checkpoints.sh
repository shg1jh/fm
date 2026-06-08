#!/bin/bash
set -euo pipefail

# Project root on the remote server.
PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

# Keep the original official I-frame checkpoint path.
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

# Reference checkpoint used for comparison.
BASELINE_CHECKPOINT="results/checkpoints_stage1_hr_v4_delta/stage1_hr_step2000.pth.tar"

# Stage 3 stable-long checkpoints to evaluate.
CKPT_DIR="results/checkpoints_stage3_stable_long_from_v4_step2000"

# Test config and optional test dataset root.
TEST_CONFIG="./dataset_config_example_yuv420.json"
TEST_ROOT="${TEST_ROOT:-}"

OUT_DIR="results/eval_stage3_stable_long_from_v4_step2000"
LOG_DIR="results/logs_eval_stage3_stable_long_from_v4_step2000"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "Starting Stage 3 stable-long checkpoint evaluation..."
echo "Project dir: $PROJECT_DIR"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Baseline P-frame checkpoint: $BASELINE_CHECKPOINT"
echo "Stage 3 checkpoint dir: $CKPT_DIR"
echo "Test config: $TEST_CONFIG"
if [ -n "$TEST_ROOT" ]; then
  echo "Override test root: $TEST_ROOT"
else
  echo "Override test root: <none; use root_path in test config>"
fi

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$BASELINE_CHECKPOINT" ] || { echo "Missing baseline checkpoint: $BASELINE_CHECKPOINT"; exit 1; }
[ -f "$TEST_CONFIG" ] || { echo "Missing test config: $TEST_CONFIG"; exit 1; }

COMMON_ARGS=(
  --model_path_i "$INTRA_CHECKPOINT"
  --model_type dmc_hr
  --me_delta_scale 0.02
  --rate_num 4
  --test_config "$TEST_CONFIG"
  --cuda true
  --worker 1
  --write_stream false
  --force_intra_period 9999
  --force_frame_num 96
)

if [ -n "$TEST_ROOT" ]; then
  COMMON_ARGS+=(--force_root_path "$TEST_ROOT")
fi

run_eval() {
  local tag="$1"
  local checkpoint="$2"
  local output_json="$OUT_DIR/eval_${tag}_pframes.json"
  local log_file="$LOG_DIR/eval_${tag}_$(date +%Y%m%d_%H%M%S).txt"

  echo "============================================================"
  echo "Evaluating $tag"
  echo "P-frame checkpoint: $checkpoint"
  echo "Output JSON: $output_json"
  echo "Log file: $log_file"

  [ -f "$checkpoint" ] || { echo "Missing P-frame checkpoint: $checkpoint"; exit 1; }

  python test_video.py \
    "${COMMON_ARGS[@]}" \
    --model_path_p "$checkpoint" \
    --output_path "$output_json" \
    2>&1 | tee "$log_file"
}

run_eval "baseline_v4_step2000" "$BASELINE_CHECKPOINT"
run_eval "stage3_stable_step2100" "$CKPT_DIR/stage3_stable_step2100.pth.tar"
run_eval "stage3_stable_step2200" "$CKPT_DIR/stage3_stable_step2200.pth.tar"
run_eval "stage3_stable_step2300" "$CKPT_DIR/stage3_stable_step2300.pth.tar"

echo "All Stage 3 stable-long checkpoint evaluations finished."
