# MVSEC Optical Flow Benchmark

This folder contains the optical-flow part of the COMP5703 benchmark group
work. It implements a unified downstream benchmark on MVSEC for six runnable
event representations, all evaluated with the same EVFlowNet-like decoder.

This is an adapted reproduction benchmark. It is not a bit-for-bit rerun of
each paper's original optical-flow decoder, training stack, or private
downstream head.

## Current Status

- Dataset: MVSEC optical-flow sequences.
- Train split: `outdoor_day1 + outdoor_day2`.
- Test split: `indoor_flying1/2/3`.
- Runnable methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`.
- Reported-only method: OmniEvent.
- Decoder: shared `EVFlowNetLike`.
- Training protocol: max 100 epochs, batch size 8, early-stop patience 10.
- Validation: block-random validation sampled from outdoor training windows.
- Metrics: AEE/EPE and KITTI-style outlier percentage.
- Event/flow pairing: timestamp-aligned event intervals from flow GT
  timestamps.
- Current result state: previous result artifacts were removed after an
  alignment/data issue was found. New results should be added only after the
  corrected rerun finishes.

## Rerun Protocol

The next formal run should use:

- train: `outdoor_day1 + outdoor_day2`
- evaluate: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated flow files with timestamps
- `indoor_flying1` GT: corrected 1398-frame
  `indoor_flying1_gt_flow_2000.npz`
- shared decoder: `src/mvsec_benchmark/models/evflownet_like.py`
- method list: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`,
  `matrixlstm`

Before launching the long run, use `scripts/check_mvsec_alignment.py` to check
that each event file covers the corresponding flow timestamps. This is the main
guard against repeating the earlier wrong-result problem.

```bash
python scripts/check_mvsec_alignment.py --data-root /path/to/processed/mvsec
```

Then run:

```bash
DATA_ROOT=/path/to/processed/mvsec \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
bash scripts/run_mvsec_100e_all_early_stop.sh
```

After the run finishes, rebuild the table and figures:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures
```

## Reporting Notes

- Describe this as a unified downstream optical-flow benchmark / adapted
  reproduction.
- The six runnable event representations use the same EVFlowNet-like decoder
  and the same train/eval split.
- Do not directly compare the future local numbers as official-paper
  reproduction numbers, because several papers do not release the same
  optical-flow downstream code.
- Treat `OmniEvent✳` as reported-only context, not as a local run in this
  pipeline.
- Raw MVSEC data and processed `.h5` / `.npz` data are not committed to GitHub.

## Code Layout

```text
optical-flow/
├── configs/
│   └── envs/
├── docs/
├── scripts/
├── src/
│   └── mvsec_benchmark/
└── tests/
```

Key entry points:

- `scripts/run_original_protocol.py`
- `scripts/run_mvsec_100e_all_early_stop.sh`
- `scripts/check_mvsec_alignment.py`
- `scripts/build_mvsec_e100_outputs.py`
- `src/mvsec_benchmark/pipeline.py`
- `src/mvsec_benchmark/models/evflownet_like.py`
- `docs/PROJECT_STATUS.md`
- `docs/GPU_RERUN_INSTRUCTIONS.md`

## Local Smoke Test

The smoke tests use synthetic/mock data and do not require MVSEC downloads.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
python scripts/run_linear_suite.py
```
