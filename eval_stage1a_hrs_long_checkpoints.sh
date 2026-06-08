#!/bin/bash
set -euo pipefail

# Evaluate Stage 1A HRS-only long-training checkpoints.

PROJECT_DIR="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM"
cd "$PROJECT_DIR"

INTRA_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_image.pth.tar"
OFFICIAL_P_CHECKPOINT="/root/autodl-tmp/DCVC-DC/other/new/DCVC-FM/checkpoints/cvpr2024_video.pth.tar"

CKPT_DIR="${CKPT_DIR:-results/checkpoints_stage1a_hrs_long_from_official}"
OUT_DIR="${OUT_DIR:-results/eval_stage1a_hrs_long_from_official}"
LOG_DIR="${LOG_DIR:-results/logs_eval_stage1a_hrs_long_from_official}"

TEST_CONFIG="${TEST_CONFIG:-./dataset_config_example_yuv420.json}"
TEST_ROOT="${TEST_ROOT:-}"
FORCE_FRAME_NUM="${FORCE_FRAME_NUM:-96}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "Starting Stage 1A HRS-only checkpoint evaluation..."
echo "Project dir: $PROJECT_DIR"
echo "I-frame checkpoint: $INTRA_CHECKPOINT"
echo "Official P-frame checkpoint: $OFFICIAL_P_CHECKPOINT"
echo "Stage 1A checkpoint dir: $CKPT_DIR"
echo "Output dir: $OUT_DIR"
echo "Test config: $TEST_CONFIG"
echo "Force frame num: $FORCE_FRAME_NUM"
if [ -n "$TEST_ROOT" ]; then
  echo "Override test root: $TEST_ROOT"
else
  echo "Override test root: <none; use root_path in test config>"
fi

[ -f "$INTRA_CHECKPOINT" ] || { echo "Missing I-frame checkpoint: $INTRA_CHECKPOINT"; exit 1; }
[ -f "$OFFICIAL_P_CHECKPOINT" ] || { echo "Missing official P-frame checkpoint: $OFFICIAL_P_CHECKPOINT"; exit 1; }
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
  --force_frame_num "$FORCE_FRAME_NUM"
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

run_eval "official_dmc_hr_identity" "$OFFICIAL_P_CHECKPOINT"

for step in 5000 10000 15000 20000; do
  checkpoint="$CKPT_DIR/stage1a_hrs_step${step}.pth.tar"
  if [ -f "$checkpoint" ]; then
    run_eval "stage1a_hrs_step${step}" "$checkpoint"
  else
    echo "Skip missing checkpoint: $checkpoint"
  fi
done

if [ -f "$CKPT_DIR/stage1a_hrs_latest.pth.tar" ]; then
  run_eval "stage1a_hrs_latest" "$CKPT_DIR/stage1a_hrs_latest.pth.tar"
fi

echo "All Stage 1A HRS-only checkpoint evaluations finished."
