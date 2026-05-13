# MVSEC Optical Flow Project Status

Last updated: 2026-05-13

This file is the current handoff note for the optical-flow part of the COMP5703
benchmark. The current checked-in result is the completed 100-epoch-maximum
early-stop run below.

Update for the next rerun: the runner now uses timestamp-aligned event/flow
pairing when flow timestamps are available, batch size 8 by default, and the
corrected 1398-frame `indoor_flying1` GT file.

## Current Main Result

The MVSEC optical-flow adapted benchmark has completed for all six runnable
event-representation methods.

Protocol:

- train: `outdoor_day1 + outdoor_day2`
- evaluate: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event window: 6M extracted left-camera events per sequence
- flow GT: full generated `*_gt_flow_full.npz` files
- decoder: shared `src/mvsec_benchmark/models/evflownet_like.py`
- max epochs: 100
- batch size: 8
- early stopping: patience 10
- validation: block-random validation sampled from outdoor training windows
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- metrics: AEE/EPE and outlier percentage
- train windows: 16329
- eval windows: 3583

Results:

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE |
|---|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper |

Among the six local runnable methods, the best test AEE is EST at `2.8654`,
with `37.04%` outliers. The six local methods are close, so the safest claim is
that the benchmark provides a consistent comparison under one downstream
protocol.

✳ OmniEvent is a paper-reported reference row. Its displayed values are simple
averages from the OmniEvent paper's MVSEC `indoor_flying1/2/3` results, not a
local run in this repository. Source:
[arXiv:2508.01842](https://arxiv.org/abs/2508.01842), Table 2.

## Result Files

Current summarized deliverables:

- `results/summary/mvsec_e100_earlystop_summary.csv`
- `results/summary/mvsec_e100_earlystop_summary.md`
- `results/figures/mvsec_e100_earlystop_aee.svg`
- `results/figures/mvsec_e100_earlystop_outlier.svg`
- `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- `results/figures/mvsec_e100_earlystop_val_curve.svg`

Raw per-method artifacts:

- `artifacts/e100_earlystop_20260501/results/*.json`
- `artifacts/e100_earlystop_20260501/logs/*.log`
- `artifacts/e100_earlystop_20260501/logs/curves/*.csv`
- `scripts/build_mvsec_e100_outputs.py` rebuilds the summary CSV, summary
  Markdown, and SVG figures from a new AutoDL run.

## Interpretation

Use this wording:

```text
This is a unified downstream optical-flow benchmark / adapted reproduction.
Six runnable event representations are connected to the same EVFlowNet-like
decoder and compared under the same MVSEC train/eval protocol.
```

Important limits:

- This is not an exact reimplementation of each paper's original optical-flow
  decoder or downstream training stack.
- Event windows are paired to flow frames by timestamp intervals when flow
  timestamps are available. This is closer to the MVSEC optical-flow setup than
  the previous pairing approach, while still using the shared downstream
  decoder.
- The numbers are suitable for the group benchmark comparison, but should not be
  presented as official-paper reproduction numbers.
- `OmniEvent✳` is included only as reported-only context. Do not rank it as a
  local result from this pipeline.
- W&B support exists in the code. The current completed deliverable uses local
  CSV curves and SVG figures.

## Code Entry Points

- main runner: `scripts/run_original_protocol.py`
- shared training/eval path: `src/mvsec_benchmark/pipeline.py`
- shared decoder: `src/mvsec_benchmark/models/evflownet_like.py`
- adapters: `src/mvsec_benchmark/adapters/`
- metrics: `src/mvsec_benchmark/utils/flow_metrics.py`

## AutoDL Notes

The completed main run came from the AutoDL result package:

```text
mvsec_e100_earlystop_results_20260501_0141.tar.gz
```

Expected AutoDL paths if a rerun is ever needed:

```text
/root/autodl-tmp/capstone/5703
/root/autodl-tmp/capstone/data/mvsec
```

Known path detail:

- `indoor_flying1_left_events_6m.h5` is under `indoor_flying1/`.
- `indoor_flying2/3` event files and all indoor `*_gt_flow_full.npz` files are
  under `indoor_flying/`.
- Use the corrected 1398-frame `indoor_flying1_gt_flow_2000.npz` as the
  `indoor_flying1` GT file before rerunning.

Recommended rerun commands:

```bash
cd /root/autodl-tmp/capstone/5703
OMP_NUM_THREADS=8 BATCH_SIZE=8 bash scripts/run_mvsec_100e_all_early_stop.sh
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures
```

## Local Archive Notes

Older smoke and indoor-only runs are useful for debugging but are not the
current main result.

Local archives include:

- `mvsec_smoke_results_20260425.tar.gz`
- `mvsec_indoor_100f_2m_results_20260425.tar.gz`
- `mvsec_indoor_full_6m_results_20260425.tar.gz`
