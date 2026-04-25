#!/bin/bash
# Full MVSEC outdoor pipeline on AutoDL.
# Run this after all bags are downloaded.
#
# Usage:
#   cd /root/autodl-tmp/capstone/5703
#   bash scripts/autodl_outdoor_pipeline.sh [adapter]
#
# If adapter is given (e.g. "est"), only that adapter runs.
# If omitted, all 6 adapters run sequentially.

set -e

REPO=/root/autodl-tmp/capstone/5703
DATA=/root/autodl-tmp/capstone/data/mvsec
RESULTS=$REPO/results

cd $REPO

# --- Step 1: Convert bags to HDF5 ---
echo "=== Step 1: Convert event bags to HDF5 ==="

mkdir -p $DATA/outdoor_day
mkdir -p $DATA/indoor_flying

# outdoor_day1 (train)
if [ ! -f "$DATA/outdoor_day/outdoor_day1_left_events.h5" ]; then
  echo "Converting outdoor_day1_data.bag ..."
  python scripts/convert_mvsec_bag_events.py \
    $DATA/outdoor_day/outdoor_day1_data.bag \
    --output $DATA/outdoor_day/outdoor_day1_left_events.h5
else
  echo "outdoor_day1_left_events.h5 already exists, skipping."
fi

# outdoor_day2 (train)
if [ ! -f "$DATA/outdoor_day/outdoor_day2_left_events.h5" ]; then
  echo "Converting outdoor_day2_data.bag ..."
  python scripts/convert_mvsec_bag_events.py \
    $DATA/outdoor_day/outdoor_day2_data.bag \
    --output $DATA/outdoor_day/outdoor_day2_left_events.h5
else
  echo "outdoor_day2_left_events.h5 already exists, skipping."
fi

# indoor_flying2 (eval)
if [ ! -f "$DATA/indoor_flying/indoor_flying2_left_events.h5" ]; then
  echo "Converting indoor_flying2_data.bag ..."
  python scripts/convert_mvsec_bag_events.py \
    $DATA/indoor_flying/indoor_flying2_data.bag \
    --output $DATA/indoor_flying/indoor_flying2_left_events.h5
else
  echo "indoor_flying2_left_events.h5 already exists, skipping."
fi

# indoor_flying3 (eval)
if [ ! -f "$DATA/indoor_flying/indoor_flying3_left_events.h5" ]; then
  echo "Converting indoor_flying3_data.bag ..."
  python scripts/convert_mvsec_bag_events.py \
    $DATA/indoor_flying/indoor_flying3_data.bag \
    --output $DATA/indoor_flying/indoor_flying3_left_events.h5
else
  echo "indoor_flying3_left_events.h5 already exists, skipping."
fi

echo "=== Step 1 Done ==="

# --- Step 2: Generate GT flow NPZ ---
echo "=== Step 2: Generate GT flow ==="

CALIB=$DATA/indoor_flying/indoor_flying_calib.zip

if [ ! -f "$DATA/outdoor_day/outdoor_day1_gt_flow.npz" ]; then
  echo "Generating outdoor_day1 GT flow ..."
  python scripts/generate_mvsec_flow_from_gt_bag.py \
    --gt-bag $DATA/outdoor_day/outdoor_day1_gt.bag \
    --calib $CALIB \
    --output $DATA/outdoor_day/outdoor_day1_gt_flow.npz
else
  echo "outdoor_day1_gt_flow.npz already exists, skipping."
fi

if [ ! -f "$DATA/outdoor_day/outdoor_day2_gt_flow.npz" ]; then
  echo "Generating outdoor_day2 GT flow ..."
  python scripts/generate_mvsec_flow_from_gt_bag.py \
    --gt-bag $DATA/outdoor_day/outdoor_day2_gt.bag \
    --calib $CALIB \
    --output $DATA/outdoor_day/outdoor_day2_gt_flow.npz
else
  echo "outdoor_day2_gt_flow.npz already exists, skipping."
fi

if [ ! -f "$DATA/indoor_flying/indoor_flying2_gt_flow.npz" ]; then
  echo "Generating indoor_flying2 GT flow ..."
  python scripts/generate_mvsec_flow_from_gt_bag.py \
    --gt-bag $DATA/indoor_flying/indoor_flying2_gt.bag \
    --calib $CALIB \
    --output $DATA/indoor_flying/indoor_flying2_gt_flow.npz
else
  echo "indoor_flying2_gt_flow.npz already exists, skipping."
fi

if [ ! -f "$DATA/indoor_flying/indoor_flying3_gt_flow.npz" ]; then
  echo "Generating indoor_flying3 GT flow ..."
  python scripts/generate_mvsec_flow_from_gt_bag.py \
    --gt-bag $DATA/indoor_flying/indoor_flying3_gt.bag \
    --calib $CALIB \
    --output $DATA/indoor_flying/indoor_flying3_gt_flow.npz
else
  echo "indoor_flying3_gt_flow.npz already exists, skipping."
fi

# indoor_flying1 GT (already generated as _20.npz during smoke test, regenerate full)
if [ ! -f "$DATA/indoor_flying/indoor_flying1_gt_flow.npz" ]; then
  echo "Generating indoor_flying1 GT flow (full) ..."
  python scripts/generate_mvsec_flow_from_gt_bag.py \
    --gt-bag $DATA/indoor_flying/indoor_flying1_gt.bag \
    --calib $CALIB \
    --output $DATA/indoor_flying/indoor_flying1_gt_flow.npz
else
  echo "indoor_flying1_gt_flow.npz already exists, skipping."
fi

echo "=== Step 2 Done ==="

# --- Step 3: Run outdoor benchmark ---
echo "=== Step 3: Run outdoor benchmark ==="

mkdir -p $RESULTS

ADAPTERS="est ergo event_pretraining get matrixlstm evrepsl"
if [ -n "$1" ]; then
  ADAPTERS="$1"
fi

for ADAPTER in $ADAPTERS; do
  echo ""
  echo "--- Running adapter: $ADAPTER ---"
  python scripts/run_outdoor_suite.py \
    --adapter $ADAPTER \
    --train-h5 \
      $DATA/outdoor_day/outdoor_day1_left_events.h5 \
      $DATA/outdoor_day/outdoor_day2_left_events.h5 \
    --train-flow \
      $DATA/outdoor_day/outdoor_day1_gt_flow.npz \
      $DATA/outdoor_day/outdoor_day2_gt_flow.npz \
    --eval-h5 \
      $DATA/indoor_flying/indoor_flying1_left_events.h5 \
      $DATA/indoor_flying/indoor_flying2_left_events.h5 \
      $DATA/indoor_flying/indoor_flying3_left_events.h5 \
    --eval-flow \
      $DATA/indoor_flying/indoor_flying1_gt_flow.npz \
      $DATA/indoor_flying/indoor_flying2_gt_flow.npz \
      $DATA/indoor_flying/indoor_flying3_gt_flow.npz \
    --epochs 50 \
    --device cuda \
    --disable-cudnn \
    --output $RESULTS/outdoor_${ADAPTER}.json
  echo "--- Done: $ADAPTER ---"
done

echo ""
echo "=== All done. Results in $RESULTS/ ==="
ls -lh $RESULTS/outdoor_*.json
