# MVSEC Optical Flow Task Review

This document records the current optical-flow benchmark state after removing
the old result package.

## What Is Complete

The repository contains a runnable unified downstream benchmark path for six
local methods:

- `est`
- `ergo`
- `event_pretraining`
- `evrepsl`
- `get`
- `matrixlstm`

`OmniEvent✳` is not reproduced locally. It is included only as a paper-reported
reference when needed for comparison discussion.

## Current Formal Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event cap: 6M events per sequence HDF5
- event timestamp precision: float64 required for Unix-time event timestamps
- flow GT: generated flow files with timestamps
- corrected `indoor_flying1` GT: `indoor_flying1_gt_flow_2000.npz`, 1398 frames
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- batch size: 8
- early stopping: patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- metrics: AEE/EPE and KITTI-style outlier percentage

## Result State

No current formal result table is committed. The previous table, figures, JSON
logs, and curve artifacts were removed because they came from the pre-correction
state and should not be treated as final.

The older processed event tar files may also be invalid if their HDF5 timestamp
column was written as float32. Those files cannot recover sub-second timing
after the fact; regenerate event HDF5 files from raw bags with the current
converter before accepting new results.

The next accepted result should include:

- new per-method JSON outputs under `results/`
- new logs under `logs/`
- new curve CSV files under `logs/curves/`
- rebuilt summary CSV and Markdown under `results/summary/`
- rebuilt SVG figures under `results/figures/`

## Reporting Boundary

The current project should be described as an adapted reproduction / unified
downstream benchmark. It should not be presented as a paper-identical
reproduction of every original optical-flow training stack.

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
*_gt_flow_full.npz or corrected generated flow npz
```

The raw data files are not committed to GitHub. The repository keeps the
preprocessing scripts, sanity checks, and exact run scripts:

- `scripts/convert_mvsec_bag_events.py`
- `scripts/generate_mvsec_flow_from_gt_bag.py`
- `scripts/check_mvsec_alignment.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/build_mvsec_e100_outputs.py`
