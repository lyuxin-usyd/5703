#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
LR="${LR:-3e-4}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
MODEL_VARIANT="${MODEL_VARIANT:-evflownet_multiscale}"
TRAINING_OBJECTIVE="${TRAINING_OBJECTIVE:-self_supervised}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.001}"
VAL_WINDOWS="${VAL_WINDOWS:-100}"
VAL_STRATEGY="${VAL_STRATEGY:-block-random}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
PHOTOMETRIC_WEIGHT="${PHOTOMETRIC_WEIGHT:-1.0}"
SMOOTHNESS_WEIGHT="${SMOOTHNESS_WEIGHT:-0.5}"
PHOTOMETRIC_LOSS="${PHOTOMETRIC_LOSS:-evflownet}"
PHOTOMETRIC_SSIM_WEIGHT="${PHOTOMETRIC_SSIM_WEIGHT:-0.0}"
PHOTOMETRIC_CHARBONNIER_EPSILON="${PHOTOMETRIC_CHARBONNIER_EPSILON:-0.001}"
PHOTOMETRIC_CHARBONNIER_ALPHA="${PHOTOMETRIC_CHARBONNIER_ALPHA:-0.45}"
PHOTOMETRIC_VALID_MASK="${PHOTOMETRIC_VALID_MASK:-0}"
SMOOTHNESS_MODE="${SMOOTHNESS_MODE:-evflownet_8conn}"
SMOOTHNESS_EDGE_WEIGHT="${SMOOTHNESS_EDGE_WEIGHT:-10.0}"
LR_SCHEDULE="${LR_SCHEDULE:-evflownet}"
LR_DECAY="${LR_DECAY:-0.9}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
MIN_LR="${MIN_LR:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-}"
MODEL_BATCH_NORM="${MODEL_BATCH_NORM:-1}"
PAPER_CROP_SIZE="${PAPER_CROP_SIZE:-256}"
PAPER_TRAIN_RANDOM_CROP="${PAPER_TRAIN_RANDOM_CROP:-1}"
PAPER_RANDOM_FLIP="${PAPER_RANDOM_FLIP:-1}"
PAPER_RANDOM_ROTATION_DEGREES="${PAPER_RANDOM_ROTATION_DEGREES:-30}"
IMAGE_STRIDE="${IMAGE_STRIDE:-1}"
TRAIN_IMAGE_STRIDE_MIN="${TRAIN_IMAGE_STRIDE_MIN:-1}"
TRAIN_IMAGE_STRIDE_MAX="${TRAIN_IMAGE_STRIDE_MAX:-5}"
IMAGE_SCALE="${IMAGE_SCALE:-raw255}"
METRIC_SCOPE="${METRIC_SCOPE:-evflownet_official}"
RANDOM_TRAIN_STRIDE_PER_EPOCH="${RANDOM_TRAIN_STRIDE_PER_EPOCH:-1}"
LAZY_RANDOM_TRAIN_STRIDE="${LAZY_RANDOM_TRAIN_STRIDE:-1}"
INFERENCE_SAMPLE_INDICES="${INFERENCE_SAMPLE_INDICES:-0 120 360}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_WINDOWS_PER_SET="${MAX_TRAIN_WINDOWS_PER_SET:-}"
MAX_EVAL_WINDOWS_PER_SET="${MAX_EVAL_WINDOWS_PER_SET:-}"
IF1_EVENTS="${IF1_EVENTS:-}"
IF1_IMAGES="${IF1_IMAGES:-}"
IF1_FLOW="${IF1_FLOW:-}"
RUN_ROOT="${RUN_ROOT:-}"

if [[ -z "$DATA_ROOT" ]]; then
  cat >&2 <<'EOF'
ERROR: DATA_ROOT is not set.

Example:
  DATA_ROOT=/root/autodl-tmp/capstone/data/mvsec OMP_NUM_THREADS=8 \
    bash scripts/run_photometric_full_6methods_e100.sh
EOF
  exit 2
fi

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="results/v9_$(date +%Y%m%d_%H%M)"
fi
if [[ -z "$IF1_FLOW" ]]; then
  if [[ -f "$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_full.npz" ]]; then
    IF1_FLOW="$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_full.npz"
  elif [[ -f "$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_2000.npz" ]]; then
    IF1_FLOW="$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_2000.npz"
  else
    IF1_FLOW="$DATA_ROOT/indoor_flying/indoor_flying1_gt_flow_full.npz"
  fi
fi
if [[ -z "$IF1_EVENTS" ]]; then
  if [[ -f "$DATA_ROOT/indoor_flying/indoor_flying1_left_events_6m.h5" ]]; then
    IF1_EVENTS="$DATA_ROOT/indoor_flying/indoor_flying1_left_events_6m.h5"
  elif [[ -f "$DATA_ROOT/indoor_flying1/indoor_flying1_left_events_6m.h5" ]]; then
    IF1_EVENTS="$DATA_ROOT/indoor_flying1/indoor_flying1_left_events_6m.h5"
  else
    IF1_EVENTS="$DATA_ROOT/indoor_flying/indoor_flying1_left_events_6m.h5"
  fi
fi
if [[ -z "$IF1_IMAGES" ]]; then
  if [[ -f "$DATA_ROOT/indoor_flying/indoor_flying1_left_images_full.h5" ]]; then
    IF1_IMAGES="$DATA_ROOT/indoor_flying/indoor_flying1_left_images_full.h5"
  elif [[ -f "$DATA_ROOT/indoor_flying1/indoor_flying1_left_images_full.h5" ]]; then
    IF1_IMAGES="$DATA_ROOT/indoor_flying1/indoor_flying1_left_images_full.h5"
  else
    IF1_IMAGES="$DATA_ROOT/indoor_flying/indoor_flying1_left_images_full.h5"
  fi
fi

METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(matrixlstm)
fi

required_files=(
  "$DATA_ROOT/outdoor_day/outdoor_day1_left_events_6m.h5"
  "$DATA_ROOT/outdoor_day/outdoor_day1_gt_flow_full.npz"
  "$DATA_ROOT/outdoor_day/outdoor_day1_left_images_full.h5"
  "$DATA_ROOT/outdoor_day/outdoor_day2_left_events_6m.h5"
  "$DATA_ROOT/outdoor_day/outdoor_day2_gt_flow_full.npz"
  "$DATA_ROOT/outdoor_day/outdoor_day2_left_images_full.h5"
  "$IF1_EVENTS"
  "$IF1_IMAGES"
  "$IF1_FLOW"
  "$DATA_ROOT/indoor_flying/indoor_flying2_left_events_6m.h5"
  "$DATA_ROOT/indoor_flying/indoor_flying2_gt_flow_full.npz"
  "$DATA_ROOT/indoor_flying/indoor_flying2_left_images_full.h5"
  "$DATA_ROOT/indoor_flying/indoor_flying3_left_events_6m.h5"
  "$DATA_ROOT/indoor_flying/indoor_flying3_gt_flow_full.npz"
  "$DATA_ROOT/indoor_flying/indoor_flying3_left_images_full.h5"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "If image H5 files are missing, run:" >&2
    echo "  DATA_ROOT=$DATA_ROOT bash scripts/prepare_mvsec_full_image_h5s.sh" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/curves" "$RUN_ROOT/logs" "$RUN_ROOT/artifacts" "$RUN_ROOT/figures"

window_limit_args=()
if [[ -n "$MAX_TRAIN_WINDOWS_PER_SET" ]]; then
  window_limit_args+=("--max-train-windows-per-set" "$MAX_TRAIN_WINDOWS_PER_SET")
fi
if [[ -n "$MAX_EVAL_WINDOWS_PER_SET" ]]; then
  window_limit_args+=("--max-eval-windows-per-set" "$MAX_EVAL_WINDOWS_PER_SET")
fi
photometric_mask_args=()
case "${PHOTOMETRIC_VALID_MASK,,}" in
  1|true|yes|on)
    photometric_mask_args+=("--photometric-valid-mask")
    ;;
esac
gradient_clip_args=()
if [[ -n "$GRADIENT_CLIP_NORM" ]]; then
  gradient_clip_args+=("--gradient-clip-norm" "$GRADIENT_CLIP_NORM")
fi
batch_norm_args=()
case "${MODEL_BATCH_NORM,,}" in
  1|true|yes|on)
    batch_norm_args+=("--model-batch-norm")
    ;;
esac
crop_args=()
if [[ "$PAPER_CROP_SIZE" != "0" ]]; then
  crop_args+=("--paper-crop-size" "$PAPER_CROP_SIZE")
fi
case "${PAPER_TRAIN_RANDOM_CROP,,}" in
  1|true|yes|on)
    crop_args+=("--paper-train-random-crop")
    ;;
esac
augmentation_args=()
case "${PAPER_RANDOM_FLIP,,}" in
  1|true|yes|on)
    augmentation_args+=("--paper-random-flip")
    ;;
esac
if [[ "$PAPER_RANDOM_ROTATION_DEGREES" != "0" ]]; then
  augmentation_args+=("--paper-random-rotation-degrees" "$PAPER_RANDOM_ROTATION_DEGREES")
fi
random_stride_args=()
case "${RANDOM_TRAIN_STRIDE_PER_EPOCH,,}" in
  1|true|yes|on)
    random_stride_args+=("--random-train-stride-per-epoch")
    ;;
esac
lazy_random_stride_args=()
case "${LAZY_RANDOM_TRAIN_STRIDE,,}" in
  1|true|yes|on)
    lazy_random_stride_args+=("--lazy-random-train-stride")
    ;;
esac
inference_args=()
for sample_index in $INFERENCE_SAMPLE_INDICES; do
  inference_args+=("--inference-sample-index" "$sample_index")
done

echo "Run root: $RUN_ROOT"
echo "Methods: ${METHODS[*]}"
echo "Train: outdoor_day1 + outdoor_day2"
echo "Eval: indoor_flying1 + indoor_flying2 + indoor_flying3"
echo "V9 MatrixLSTM-2bin lazy-stride official-loader mode: $MODEL_VARIANT + $TRAINING_OBJECTIVE"
echo "Loss: supervised_weight=$SUPERVISED_WEIGHT + photometric($PHOTOMETRIC_WEIGHT,$PHOTOMETRIC_LOSS,alpha=$PHOTOMETRIC_CHARBONNIER_ALPHA,mask=$PHOTOMETRIC_VALID_MASK) + smoothness($SMOOTHNESS_WEIGHT,$SMOOTHNESS_MODE)"
echo "Schedule: lr=$LR base_channels=$BASE_CHANNELS lr_schedule=$LR_SCHEDULE lr_decay=$LR_DECAY warmup_epochs=$WARMUP_EPOCHS min_lr=$MIN_LR weight_decay=$WEIGHT_DECAY gradient_clip=$GRADIENT_CLIP_NORM"
echo "Paper-like extras: model_batch_norm=$MODEL_BATCH_NORM paper_crop_size=$PAPER_CROP_SIZE paper_train_random_crop=$PAPER_TRAIN_RANDOM_CROP paper_random_flip=$PAPER_RANDOM_FLIP paper_random_rotation_degrees=$PAPER_RANDOM_ROTATION_DEGREES"
echo "Official loader extras: image_scale=$IMAGE_SCALE train_image_strides=${TRAIN_IMAGE_STRIDE_MIN}..${TRAIN_IMAGE_STRIDE_MAX} random_train_stride_per_epoch=$RANDOM_TRAIN_STRIDE_PER_EPOCH lazy_random_train_stride=$LAZY_RANDOM_TRAIN_STRIDE eval_image_stride=$IMAGE_STRIDE metric_scope=$METRIC_SCOPE"
echo "Artifact export: checkpoint + inference sample indices=${INFERENCE_SAMPLE_INDICES}"
echo "Device: $DEVICE"

for method in "${METHODS[@]}"; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  name="v9_${method}_${stamp}"
  output="$RUN_ROOT/results/${name}.json"
  curve="$RUN_ROOT/curves/${name}.csv"
  log="$RUN_ROOT/logs/${name}.log"
  artifact_dir="$RUN_ROOT/artifacts/${method}"

  echo "===== running $method at $(date -Iseconds) =====" | tee "$log"
  start_time="$(date +%s)"
  python scripts/run_matrixlstm_paperlike_probe.py \
    --adapter "$method" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day1_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day1_gt_flow_full.npz" \
    --train-image-h5 "$DATA_ROOT/outdoor_day/outdoor_day1_left_images_full.h5" \
    --train-pair "$DATA_ROOT/outdoor_day/outdoor_day2_left_events_6m.h5:$DATA_ROOT/outdoor_day/outdoor_day2_gt_flow_full.npz" \
    --train-image-h5 "$DATA_ROOT/outdoor_day/outdoor_day2_left_images_full.h5" \
    --eval-pair "$IF1_EVENTS:$IF1_FLOW" \
    --eval-image-h5 "$IF1_IMAGES" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying2_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying2_gt_flow_full.npz" \
    --eval-image-h5 "$DATA_ROOT/indoor_flying/indoor_flying2_left_images_full.h5" \
    --eval-pair "$DATA_ROOT/indoor_flying/indoor_flying3_left_events_6m.h5:$DATA_ROOT/indoor_flying/indoor_flying3_gt_flow_full.npz" \
    --eval-image-h5 "$DATA_ROOT/indoor_flying/indoor_flying3_left_images_full.h5" \
    --image-stride "$IMAGE_STRIDE" \
    --train-image-stride-min "$TRAIN_IMAGE_STRIDE_MIN" \
    --train-image-stride-max "$TRAIN_IMAGE_STRIDE_MAX" \
    --image-scale "$IMAGE_SCALE" \
    "${window_limit_args[@]}" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --base-channels "$BASE_CHANNELS" \
    --model-variant "$MODEL_VARIANT" \
    --training-objective "$TRAINING_OBJECTIVE" \
    --supervised-weight "$SUPERVISED_WEIGHT" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --disable-cudnn \
    --progress-every "$PROGRESS_EVERY" \
    --early-stop-patience "$PATIENCE" \
    --early-stop-min-delta "$MIN_DELTA" \
    --early-stop-val-windows "$VAL_WINDOWS" \
    --early-stop-val-strategy "$VAL_STRATEGY" \
    --photometric-weight "$PHOTOMETRIC_WEIGHT" \
    --smoothness-weight "$SMOOTHNESS_WEIGHT" \
    --photometric-loss "$PHOTOMETRIC_LOSS" \
    --photometric-ssim-weight "$PHOTOMETRIC_SSIM_WEIGHT" \
    --photometric-charbonnier-epsilon "$PHOTOMETRIC_CHARBONNIER_EPSILON" \
    --photometric-charbonnier-alpha "$PHOTOMETRIC_CHARBONNIER_ALPHA" \
    "${photometric_mask_args[@]}" \
    --smoothness-mode "$SMOOTHNESS_MODE" \
    --smoothness-edge-weight "$SMOOTHNESS_EDGE_WEIGHT" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-decay "$LR_DECAY" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    "${gradient_clip_args[@]}" \
    "${batch_norm_args[@]}" \
    "${crop_args[@]}" \
    "${augmentation_args[@]}" \
    "${random_stride_args[@]}" \
    "${lazy_random_stride_args[@]}" \
    --metric-scope "$METRIC_SCOPE" \
    --curve-log "$curve" \
    --artifact-dir "$artifact_dir" \
    "${inference_args[@]}" \
    --output "$output" \
    2>&1 | tee -a "$log"
  end_time="$(date +%s)"
  echo "===== finished $method elapsed_seconds=$((end_time - start_time)) =====" | tee -a "$log"
done

python scripts/build_photometric_run_outputs.py --run-root "$RUN_ROOT"
date -Iseconds > "$RUN_ROOT/DONE.txt"

echo "===== full photometric run done ====="
echo "Run root: $RUN_ROOT"
echo "Summary: $RUN_ROOT/summary.csv"
echo "Markdown summary: $RUN_ROOT/summary.md"
