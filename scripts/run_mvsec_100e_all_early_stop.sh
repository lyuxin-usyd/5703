#!/usr/bin/env bash
set -euo pipefail

METHODS=("ergo" "est" "event_pretraining" "evrepsl" "get" "matrixlstm")

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/capstone/data/mvsec}"
OUT_DIR="${OUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"
CURVE_DIR="${CURVE_DIR:-logs/curves}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.001}"
VAL_WINDOWS="${VAL_WINDOWS:-1000}"
VAL_STRATEGY="${VAL_STRATEGY:-block-random}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$CURVE_DIR"

for method in "${METHODS[@]}"; do
  echo "===== running ${method}: max ${EPOCHS} epochs, early-stop patience ${PATIENCE}, val ${VAL_STRATEGY} ====="
  wandb_args=()
  if [[ -n "$WANDB_PROJECT" ]]; then
    wandb_args=(
      --wandb-project "$WANDB_PROJECT"
      --wandb-run-name "mvsec_${method}_e${EPOCHS}_${VAL_STRATEGY}"
      --wandb-mode "$WANDB_MODE"
    )
  fi

  python scripts/run_original_protocol.py \
    --adapter "$method" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day1_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day1_gt_flow_full.npz" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day2_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day2_gt_flow_full.npz" \
    --eval-pair "$DATA_ROOT/indoor_flying1/indoor_flying1_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_full.npz" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying2_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying2_gt_flow_full.npz" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying3_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying3_gt_flow_full.npz" \
    --epochs "$EPOCHS" \
    --batch-size 4 \
    --eval-batch-size 1 \
    --device cuda \
    --disable-cudnn \
    --progress-every "$PROGRESS_EVERY" \
    --early-stop-patience "$PATIENCE" \
    --early-stop-min-delta "$MIN_DELTA" \
    --early-stop-val-windows "$VAL_WINDOWS" \
    --early-stop-val-strategy "$VAL_STRATEGY" \
    --curve-log "$CURVE_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_curve.csv" \
    "${wandb_args[@]}" \
    --output "$OUT_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_earlystop.json" \
    2>&1 | tee "$LOG_DIR/original_od12_if123_full6m_${method}_e${EPOCHS}_${VAL_STRATEGY}_earlystop.log"

  echo "===== done ${method} ====="
done

tar -czf "/root/autodl-tmp/capstone/mvsec_original_protocol_e${EPOCHS}_earlystop_results_$(date +%Y%m%d_%H%M).tar.gz" "$OUT_DIR" "$LOG_DIR" docs README.md
echo "===== all done ====="
ls -lh /root/autodl-tmp/capstone/mvsec_original_protocol_e*_earlystop_results_*.tar.gz
