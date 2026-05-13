# MVSEC Optical Flow Task Review

This snapshot belongs to the completed E100 early-stop artifact and records the
current formal optical-flow result package.

## What Is Complete

The optical-flow benchmark has a closed runnable path for six local methods:

- `est`
- `ergo`
- `event_pretraining`
- `evrepsl`
- `get`
- `matrixlstm`

`OmniEvent✳` is not reproduced locally. It is included only as a paper-reported
reference row.

## Current Formal Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event cap: 6M events per sequence HDF5
- flow GT: full generated `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- early stopping: patience 10
- validation: block-random outdoor validation
- metrics: AEE/EPE and KITTI-style outlier percentage

## Current Result Table

| Method | AEE | Outlier % | Train windows | Eval windows | Valid count |
|---|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 16329 | 3583 | 322326680 |
| est | 2.8654 | 37.04 | 16329 | 3583 | 322326680 |
| event_pretraining | 2.9653 | 38.19 | 16329 | 3583 | 322326680 |
| evrepsl | 3.0180 | 39.06 | 16329 | 3583 | 322326680 |
| get | 2.9619 | 38.34 | 16329 | 3583 | 322326680 |
| matrixlstm | 3.0138 | 38.97 | 16329 | 3583 | 322326680 |
| OmniEvent✳ | 0.9900 | 3.24 | paper | paper | paper |

✳ OmniEvent values are simple averages from the OmniEvent paper's MVSEC
`indoor_flying1/2/3` results. Source:
[arXiv:2508.01842](https://arxiv.org/abs/2508.01842), Table 2.

## Reporting Boundary

The current result is an adapted reproduction / unified downstream benchmark.
It should not be presented as a paper-identical reproduction of every original
optical-flow training stack.

Known limitations:

- shared EVFlowNet-like decoder rather than each paper's original downstream
  decoder/head
- this artifact folder preserves the earlier checked-in result package; the
  active runner now uses timestamp-aligned event intervals when flow timestamps
  are available
- OmniEvent reported-only, not locally reproduced
