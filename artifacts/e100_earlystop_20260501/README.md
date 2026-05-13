# MVSEC Optical Flow E100 Early-Stop Artifact

This artifact folder contains the completed MVSEC optical-flow adapted benchmark
results generated on 2026-05-01.

This is a unified downstream optical-flow benchmark / adapted reproduction. Six
local runnable event representations are compared under the same MVSEC split and
the same EVFlowNet-like downstream decoder. It is not a paper-identical
reproduction of each method's original optical-flow decoder/head.

## Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- early stopping: patience 10
- validation: block-random validation sampled from outdoor training windows
- metrics: AEE/EPE and KITTI-style outlier percentage
- train windows: 16329
- eval windows: 3583

## Main Result

Lower is better for AEE and Outlier %.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE |
|---|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper |

Among the six local runnable methods, EST has the best test AEE in this run.
The local method gaps are small, so this result should be reported as a
consistent downstream comparison rather than a strong ranking claim.

✳ OmniEvent is a paper-reported reference row, not a local run in this pipeline.
The displayed values are simple averages from the OmniEvent paper's MVSEC
`indoor_flying1/2/3` results. Source:
[arXiv:2508.01842](https://arxiv.org/abs/2508.01842), Table 2.

## Files In This Artifact

- `results/*.json`: per-method final result JSONs
- `logs/*.log`: per-method training logs
- `logs/curves/*.csv`: per-method training-loss and validation-AEE curves
- `docs/`: documentation snapshots for this result package

Presentation-ready summary tables and figures are kept at the optical-flow
folder level:

- `results/summary/mvsec_e100_earlystop_summary.csv`
- `results/summary/mvsec_e100_earlystop_summary.md`
- `results/figures/mvsec_e100_earlystop_aee.svg`
- `results/figures/mvsec_e100_earlystop_outlier.svg`
- `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- `results/figures/mvsec_e100_earlystop_val_curve.svg`

## Reporting Notes

- Do not describe this as a full reproduction of each paper's original
  optical-flow decoder or downstream training stack.
- This artifact folder preserves the earlier checked-in result package. The
  active runner now uses timestamp-aligned event intervals when flow timestamps
  are available.
- `OmniEvent✳` is reported-only and should not be ranked as a local result.
- Use this E100 early-stop package as the current main result.
