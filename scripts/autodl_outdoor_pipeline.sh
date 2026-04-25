#!/bin/bash
# Run the current formal MVSEC optical-flow protocol on AutoDL.
#
# Protocol:
#   train: outdoor_day1 + outdoor_day2
#   eval:  indoor_flying1 + indoor_flying2 + indoor_flying3
#
# This script expects preprocessed files to already exist:
#   *_left_events_6m.h5
#   *_gt_flow_full.npz
#
# Usage:
#   cd /root/autodl-tmp/capstone/5703
#   bash scripts/autodl_outdoor_pipeline.sh
#   bash scripts/autodl_outdoor_pipeline.sh est
#
# If an adapter name is provided, only that adapter runs. Otherwise all six
# runnable adapters run sequentially. OmniEvent is reported-only.

set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/capstone/5703}
DATA=${DATA:-/root/autodl-tmp/capstone/data/mvsec}
RESULTS=${RESULTS:-$REPO/results}
LOGS=${LOGS:-$REPO/logs}

cd "$REPO"
mkdir -p "$RESULTS" "$LOGS"

TRAIN_PAIRS=(
  "$DATA/outdoor_day/outdoor_day1_left_events_6m.h5:$DATA/outdoor_day/outdoor_day1_gt_flow_full.npz"
  "$DATA/outdoor_day/outdoor_day2_left_events_6m.h5:$DATA/outdoor_day/outdoor_day2_gt_flow_full.npz"
)

EVAL_PAIRS=(
  "$DATA/indoor_flying1/indoor_flying1_left_events_6m.h5:$DATA/indoor_flying/indoor_flying1_gt_flow_full.npz"
  "$DATA/indoor_flying/indoor_flying2_left_events_6m.h5:$DATA/indoor_flying/indoor_flying2_gt_flow_full.npz"
  "$DATA/indoor_flying/indoor_flying3_left_events_6m.h5:$DATA/indoor_flying/indoor_flying3_gt_flow_full.npz"
)

require_pair_files() {
  local pair path
  for pair in "$@"; do
    IFS=: read -r h5 flow <<< "$pair"
    for path in "$h5" "$flow"; do
      if [ ! -f "$path" ]; then
        echo "Missing required file: $path" >&2
        exit 1
      fi
    done
  done
}

require_pair_files "${TRAIN_PAIRS[@]}" "${EVAL_PAIRS[@]}"

ADAPTERS=(ergo est event_pretraining evrepsl get matrixlstm)
if [ "${1:-}" != "" ]; then
  ADAPTERS=("$1")
fi

for adapter in "${ADAPTERS[@]}"; do
  echo "===== running $adapter ====="
  python scripts/run_original_protocol.py \
    --adapter "$adapter" \
    --train-pair "${TRAIN_PAIRS[0]}" \
    --train-pair "${TRAIN_PAIRS[1]}" \
    --eval-pair "${EVAL_PAIRS[0]}" \
    --eval-pair "${EVAL_PAIRS[1]}" \
    --eval-pair "${EVAL_PAIRS[2]}" \
    --epochs 1 \
    --batch-size 4 \
    --eval-batch-size 1 \
    --device cuda \
    --disable-cudnn \
    --progress-every 100 \
    --output "$RESULTS/original_od12_if123_full6m_${adapter}_e1_stream.json" \
    2>&1 | tee "$LOGS/original_od12_if123_full6m_${adapter}_e1_stream.log"
  echo "===== done $adapter ====="
done

echo "===== summary ====="
python - <<'PY'
import glob
import json

files = sorted(glob.glob("results/original_od12_if123_full6m_*_e1_stream.json"))
print(f"found {len(files)} result files")
print("method,aee,outlier_percent,train_windows,eval_windows,valid_count,file")
for path in files:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    print(
        f"{result['adapter_name']},"
        f"{result['aee']:.6f},"
        f"{result['outlier_percent']:.6f},"
        f"{result['train_windows']},"
        f"{result['eval_windows']},"
        f"{result['valid_count']},"
        f"{path}"
    )
PY
