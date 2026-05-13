# Optical Flow Folder Guide

This folder is the optical-flow part of the benchmark group work. It contains
the MVSEC adapted reproduction benchmark, result summaries, figures, and the
shared downstream pipeline used for the six runnable event representations.

## What Is Inside

- `src/`: benchmark package code, including MVSEC loading, representation
  adapters, metrics, and the shared optical-flow training/evaluation pipeline.
- `scripts/`: end-to-end scripts for smoke tests, MVSEC conversion, GT flow
  generation, and the outdoor-train / indoor-test benchmark runner.
- `docs/`: status notes, result reports, environment notes, adapter status, and
  scientific limitations.
- `results/`: current main result table and presentation-ready SVG figures.
- `artifacts/e100_earlystop_20260501/`: per-method JSON results, logs, and CSV
  learning curves from the completed main experiment.
- `configs/envs/`: per-method dependency lists.

## Current Main Result

The checked-in result package is the completed MVSEC optical-flow adapted
benchmark. The runner has now been updated for the next rerun:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- training: max 100 epochs, batch size 8, early-stop patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- runnable methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`

This is a unified downstream optical-flow benchmark / adapted reproduction, not
a paper-identical rerun of every original optical-flow codebase.

Before rerunning on AutoDL, make sure `indoor_flying1_gt_flow_full.npz` uses the
corrected 1398-frame file copied from `indoor_flying1_gt_flow_2000.npz`.

| Method | AEE | Outlier % | Epochs | Best epoch | Best val AEE |
|---|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 20 | 10 | 2.6049 |
| est | 2.8654 | 37.04 | 11 | 1 | 2.6380 |
| event_pretraining | 2.9653 | 38.19 | 20 | 10 | 2.6004 |
| evrepsl | 3.0180 | 39.06 | 15 | 5 | 2.6243 |
| get | 2.9619 | 38.34 | 22 | 12 | 2.6007 |
| matrixlstm | 3.0138 | 38.97 | 22 | 12 | 2.6071 |
| OmniEvent✳ | 0.9900 | 3.24 | paper | paper | paper |

✳ OmniEvent is a paper-reported reference row. Its displayed values are simple
averages from the OmniEvent paper's MVSEC `indoor_flying1/2/3` results, not a
local run in this repository. Source:
[arXiv:2508.01842](https://arxiv.org/abs/2508.01842), Table 2.

## Result Locations

- Table CSV: `results/summary/mvsec_e100_earlystop_summary.csv`
- Table Markdown: `results/summary/mvsec_e100_earlystop_summary.md`
- AEE figure: `results/figures/mvsec_e100_earlystop_aee.svg`
- Outlier figure: `results/figures/mvsec_e100_earlystop_outlier.svg`
- Train-loss curve: `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- Validation-AEE curve: `results/figures/mvsec_e100_earlystop_val_curve.svg`
- Raw per-method curves: `artifacts/e100_earlystop_20260501/logs/curves/`
- Raw per-method JSON results: `artifacts/e100_earlystop_20260501/results/`
- Post-run table/figure builder: `scripts/build_mvsec_e100_outputs.py`

## Most Important Entry Points

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/MVSEC_E100_EARLYSTOP_RESULTS_20260501.md`
- `docs/MVSEC_E100_EARLYSTOP_REPORT_CN_20260501.md`
- `docs/AUTODL_RERUN_INSTRUCTIONS.md`
- `results/summary/mvsec_e100_earlystop_summary.md`
- `scripts/run_original_protocol.py`
- `scripts/build_mvsec_e100_outputs.py`

## Wording To Use

Use:

```text
MVSEC unified downstream optical-flow benchmark / adapted reproduction.
Six runnable event representations are compared under the same EVFlowNet-like
decoder and the same outdoor-train / indoor-test protocol.
Event windows use timestamp-aligned intervals when flow timestamps are
available.
```

Avoid:

```text
Fully reproduced every paper's original optical-flow decoder/head.
Official paper-number comparison.
Treating OmniEvent✳ as a local run.
```
