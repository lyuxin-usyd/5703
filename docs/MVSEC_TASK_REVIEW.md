# MVSEC Optical Flow Task Review

This document records the current optical-flow benchmark status for the
completed E100 early-stop result package.

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
- batch size: 8
- early stopping: patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
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
- timestamp-aligned event/flow pairing is closer to the MVSEC optical-flow
  setup, but the shared downstream decoder still makes this an adapted
  reproduction rather than a paper-identical rerun
- OmniEvent reported-only, not locally reproduced

## Reproducible Interface

The benchmark uses preprocessed MVSEC event and flow files:

```text
*_left_events_6m.h5
*_gt_flow_full.npz
```

The raw data files are not committed to GitHub. The repository keeps the
preprocessing scripts and the exact run script:

- `scripts/convert_mvsec_bag_events.py`
- `scripts/generate_mvsec_flow_from_gt_bag.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/build_mvsec_e100_outputs.py`
