# Optical Flow Folder Guide

This folder is the optical-flow part of the benchmark group work. It contains
the MVSEC adapted reproduction benchmark code, the shared downstream pipeline,
and the scripts needed to rerun the six local event-representation methods.

## What Is Inside

- `src/`: benchmark package code, including MVSEC loading, representation
  adapters, metrics, and the shared optical-flow training/evaluation pipeline.
- `scripts/`: MVSEC conversion helpers, alignment checks, the main benchmark
  runner, and post-run table/figure generation.
- `docs/`: project status, environment notes, adapter status, and rerun notes.
- `configs/envs/`: per-method dependency lists.
- `tests/`: synthetic tests for the pipeline and data handling.

The old result tables, figures, and archived JSON/log artifacts were removed
after a data/alignment issue was found. New result artifacts should be committed
only after the corrected rerun is complete.

## Current Formal Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated flow files with timestamps
- important data check: `indoor_flying1_gt_flow_2000.npz` should have 1398
  flow frames
- decoder: shared `EVFlowNetLike`
- training: max 100 epochs, batch size 8, early-stop patience 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps
- runnable methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`

This is a unified downstream optical-flow benchmark / adapted reproduction, not
a paper-identical rerun of every original optical-flow codebase.

## Rerun Checks

Run the alignment checker before the long benchmark:

```bash
python scripts/check_mvsec_alignment.py --data-root /path/to/processed/mvsec
```

The checker should confirm that flow timestamps are covered by the event
timestamp range. It should also make it obvious if `indoor_flying1` has the old
20-frame GT file instead of the corrected 1398-frame file.

Main run:

```bash
DATA_ROOT=/path/to/processed/mvsec \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

Post-run table and figure generation:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures
```

## Most Important Entry Points

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/GPU_RERUN_INSTRUCTIONS.md`
- `scripts/check_mvsec_alignment.py`
- `scripts/run_original_protocol.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/build_mvsec_e100_outputs.py`

## Wording To Use

Use:

```text
MVSEC unified downstream optical-flow benchmark / adapted reproduction.
Six runnable event representations are compared under the same EVFlowNet-like
decoder and the same outdoor-train / indoor-test protocol.
Event windows use timestamp-aligned intervals from flow GT timestamps.
```

Avoid:

```text
Fully reproduced every paper's original optical-flow decoder/head.
Official paper-number comparison.
Treating OmniEvent✳ as a local run.
```
