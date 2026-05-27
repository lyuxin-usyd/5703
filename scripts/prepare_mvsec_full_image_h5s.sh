#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-}"
MAX_IMAGES="${MAX_IMAGES:-}"

if [[ -z "$DATA_ROOT" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_ROOT is not set.

Example:
  DATA_ROOT=/root/autodl-tmp/capstone/data/mvsec bash scripts/prepare_mvsec_full_image_h5s.sh
EOF
  exit 2
fi

convert_one() {
  local bag="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    echo "[skip] $output"
    return
  fi
  if [[ ! -f "$bag" ]]; then
    echo "[missing] $bag" >&2
    return 1
  fi
  mkdir -p "$(dirname "$output")"
  local args=("$bag" "--output" "$output" "--topic" "/davis/left/image_raw")
  if [[ -n "$MAX_IMAGES" ]]; then
    args+=("--max-images" "$MAX_IMAGES")
  fi
  echo "[convert] $bag -> $output"
  python scripts/convert_mvsec_bag_images.py "${args[@]}"
}

convert_one "$DATA_ROOT/outdoor_day/outdoor_day1_data.bag" \
  "$DATA_ROOT/outdoor_day/outdoor_day1_left_images_full.h5"
convert_one "$DATA_ROOT/outdoor_day/outdoor_day2_data.bag" \
  "$DATA_ROOT/outdoor_day/outdoor_day2_left_images_full.h5"
convert_one "$DATA_ROOT/indoor_flying1/indoor_flying1_data.bag" \
  "$DATA_ROOT/indoor_flying1/indoor_flying1_left_images_full.h5"
convert_one "$DATA_ROOT/indoor_flying/indoor_flying2_data.bag" \
  "$DATA_ROOT/indoor_flying/indoor_flying2_left_images_full.h5"
convert_one "$DATA_ROOT/indoor_flying/indoor_flying3_data.bag" \
  "$DATA_ROOT/indoor_flying/indoor_flying3_left_images_full.h5"

echo "Image H5 preparation finished."
