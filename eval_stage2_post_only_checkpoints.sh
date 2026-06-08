#!/bin/bash
set -euo pipefail

# Project root on the remote server.
PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

# Keep the original official I-frame checkpoint path.
INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"

# Test config and optional test dataset root.
TEST_CONFIG="./dataset_config_example_yuv420.json"
TEST_ROOT="${TEST_ROOT:-}"

CKPT_DIR="results/checkpoints_stage2_post_only_from_v4_step2000"
OUT_DIR="results/eval_stage2_post_only_from_v4_step2000"
LOG_DIR="results/logs_eval_stage2_post_only_from_v4_step2000"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "Starting Stage 2 post-only checkpoint evaluation..."
echo "Project dir: $PROJECT_DIR"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Test config: $TEST_CONFIG"
if [ -n "$TEST_ROOT" ]; then
  echo "Override test root: $TEST_ROOT"
else
  echo "Override test root: <none; use root_path in test config>"
fi

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$TEST_CONFIG" ] || { echo "Missing test config: $TEST_CONFIG"; exit 1; }

COMMON_ARGS=(
  --model_path_i "$INTRA_CHECKPOINT"
  --model_type wrapper
  --me_delta_scale 0.02
  --wrapper_use_pre false
  --wrapper_use_post true
  --wrapper_post_as_ref false
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
  local output_json="$OUT_DIR/eval_stage2_post_only_${tag}_pframes.json"
  local log_file="$LOG_DIR/eval_stage2_post_only_${tag}_$(date +%Y%m%d_%H%M%S).txt"

  echo "============================================================"
  echo "Evaluating $tag"
  echo "Wrapper checkpoint: $checkpoint"
  echo "Output JSON: $output_json"
  echo "Log file: $log_file"

  [ -f "$checkpoint" ] || { echo "Missing wrapper checkpoint: $checkpoint"; exit 1; }

  python test_video.py \
    "${COMMON_ARGS[@]}" \
    --model_path_p "$checkpoint" \
    --output_path "$output_json" \
    2>&1 | tee "$log_file"
}

run_eval "step500" "$CKPT_DIR/stage2_post_step500.pth.tar"
run_eval "step1000" "$CKPT_DIR/stage2_post_step1000.pth.tar"

echo "All Stage 2 post-only checkpoint evaluations finished."
