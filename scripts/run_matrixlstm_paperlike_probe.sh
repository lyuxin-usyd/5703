#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-}"
OUT_DIR="${OUT_DIR:-results/paperlike_probe}"
LOG_DIR="${LOG_DIR:-logs/paperlike_probe}"
CURVE_DIR="${CURVE_DIR:-logs/curves/paperlike_probe}"
ADAPTER="${ADAPTER:-matrixlstm}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.001}"
VAL_WINDOWS="${VAL_WINDOWS:-20}"
VAL_STRATEGY="${VAL_STRATEGY:-block-random}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

if [[ -z "$DATA_ROOT" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_ROOT is not set.

Example:
  DATA_ROOT=/root/autodl-tmp/capstone/data/mvsec OMP_NUM_THREADS=8 BATCH_SIZE=8 \
    bash scripts/run_matrixlstm_paperlike_probe.sh
EOF
  exit 2
fi

IF1_FLOW="${IF1_FLOW:-$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_2000.npz}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CURVE_DIR"

stamp="$(date +%Y%m%d_%H%M)"
output="$OUT_DIR/matrixlstm_paperlike_${ADAPTER}_e${EPOCHS}_${VAL_STRATEGY}_${stamp}.json"
curve="$CURVE_DIR/matrixlstm_paperlike_${ADAPTER}_e${EPOCHS}_${VAL_STRATEGY}_${stamp}_curve.csv"
log="$LOG_DIR/matrixlstm_paperlike_${ADAPTER}_e${EPOCHS}_${VAL_STRATEGY}_${stamp}.log"

START_TIME=$(date +%s)
python scripts/run_matrixlstm_paperlike_probe.py \
  --adapter "$ADAPTER" \
  --train-pair "$DATA_ROOT/outdoor_day/outdoor_day1_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day1_gt_flow_full.npz" \
  --train-pair "$DATA_ROOT/outdoor_day/outdoor_day2_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day2_gt_flow_full.npz" \
  --eval-pair "$DATA_ROOT/indoor_flying1/indoor_flying1_left_events_6m.h5:$IF1_FLOW" \
  --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying2_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying2_gt_flow_full.npz" \
  --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying3_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying3_gt_flow_full.npz" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --disable-cudnn \
  --progress-every "$PROGRESS_EVERY" \
  --early-stop-patience "$PATIENCE" \
  --early-stop-min-delta "$MIN_DELTA" \
  --early-stop-val-windows "$VAL_WINDOWS" \
  --early-stop-val-strategy "$VAL_STRATEGY" \
  --curve-log "$curve" \
  --output "$output" \
  2>&1 | tee "$log"
END_TIME=$(date +%s)

echo "===== MatrixLSTM paper-like probe done ====="
echo "elapsed seconds: $((END_TIME - START_TIME))"
echo "elapsed minutes: $(((END_TIME - START_TIME) / 60))"
echo "output: $output"
echo "curve: $curve"
echo "log: $log"
