# Optical Flow Folder Guide

This folder is the optical-flow part of the benchmark group work. The current
formal optical-flow package has only two V9 result parts:

```text
results/learning-base/
results/baseline/
```

`learning-base` contains the six learned representation methods. `baseline`
contains the traditional representations run under the same V9 protocol.

## Main Method Code

```text
scripts/
src/mvsec_benchmark/
configs/envs/
```

Use `README.md` for method details and `results/learning-base/summary.md` for
the learned-method result notes.

## Current Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- image input: aligned image H5 files
- decoder: EV-FlowNet-style multi-scale decoder
- objective: photometric warping + smoothness
- training: max 100 epochs, batch size 8, patience 10
- metrics: AEE / KITTI-style Outlier over valid GT flow pixels

## Latest Result Files

```text
results/learning-base/summary.csv
results/learning-base/per_sequence_summary.csv
results/learning-base/validation_aee_curves.png

results/baseline/summary.csv
results/baseline/per_sequence_summary.csv
results/baseline/validation_aee_curves.png
```

## Most Important Entry Points

- `README.md`
- `scripts/run_photometric_full_6methods_e100.sh`
- `src/mvsec_benchmark/`
- `results/learning-base/`
- `results/baseline/`

Raw MVSEC data and processed training files are not committed to GitHub.
