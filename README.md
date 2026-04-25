# MVSEC Benchmark

Unified benchmark scaffold for COMP5703 optical-flow work on MVSEC.

This repository is intentionally staged:

1. build a common task interface
2. attach seven paper-specific representation adapters
3. verify small synthetic smoke tests locally
4. run the adapted MVSEC optical-flow protocol on real data

Current status:

- common event and metric utilities
- six runnable method adapters plus OmniEvent as reported-only
- per-method environment requirement files
- synthetic smoke test that runs without MVSEC downloads
- minimal MVSEC-style loader and a CPU-friendly linear flow benchmark loop
- real MVSEC smoke, indoor-only controlled runs, and the formal original-style
  AutoDL run are archived and documented
- AutoDL handoff and current experiment state in `docs/CLAUDE_CODE_HANDOFF.md`
- review of remaining scientific limitations in
  `docs/MVSEC_TASK_REVIEW_20260426.md`

This is **not** an exact paper-faithful reproduction of all seven methods. It is
a unified adapted MVSEC benchmark: six runnable event representations are
compared with the same shared EV-FlowNet-like decoder, while OmniEvent remains
reported-only because its current public code path is not runnable for this
downstream task.

For continuing or auditing the AutoDL work, start from
`docs/CLAUDE_CODE_HANDOFF.md`. That file records which runs are already done,
which files exist on the data disk, and the exact command for the completed
outdoor-train / indoor-test protocol.

## Current formal result

The main result currently available is the original-style MVSEC split:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- epochs: 1
- GPU: RTX 4090

| Method | AEE | Outlier % |
|---|---:|---:|
| ERGO | 3.007065 | 38.588946 |
| EST | 2.935838 | 38.963377 |
| Event Pre-training | 2.948537 | 38.059867 |
| EvRepSL | 3.032907 | 39.112478 |
| GET | 3.016356 | 38.631755 |
| MatrixLSTM | 3.059037 | 39.388230 |

The archive is stored outside Git under
`results/autodl_archives/20260426/mvsec_original_protocol_results_20260426.tar.gz`
and mirrored in the Obsidian artifact folder.

## Layout

```text
mvsec-benchmark/
├── configs/
│   └── envs/
├── docs/
├── refs/              # optional local upstream repos for reference only
├── src/
│   └── mvsec_benchmark/
└── tests/
```

`refs/` is not required for the local smoke tests or the tiny linear benchmark
loop, and it is excluded from Git by default.

## Quick local smoke test

```bash
cd D:\event-benchmark\mvsec-benchmark
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_smoke.py
python scripts/run_linear_suite.py
```

The smoke test does not need MVSEC files. It uses synthetic event streams and
checks that:

- each adapter can build a finite representation
- a dummy flow head returns a finite flow map
- AEE / outlier metrics are computed successfully
- `scripts/run_smoke.py` prints a small JSON result table for the current
  adapters
- `scripts/run_linear_suite.py` creates a tiny mock MVSEC pair, trains a
  CPU-friendly linear flow head, and evaluates the six runnable methods

## Tiny end-to-end benchmark loop

This repository now includes a very small end-to-end path that does not require
full MVSEC downloads:

1. `scripts/make_mock_mvsec.py` creates a tiny HDF5 + NPZ pair
2. `src/mvsec_benchmark/data/mvsec.py` slices event windows from that pair
3. one adapter builds a representation tensor
4. `LinearFlowRegressor` fits a per-pixel linear flow head
5. the benchmark reports AEE / outlier on held-out windows

Run one method:

```bash
cd D:\event-benchmark\mvsec-benchmark
python scripts/run_linear_benchmark.py --adapter est --use-mock
```

Run all six runnable methods:

```bash
cd D:\event-benchmark\mvsec-benchmark
python scripts/run_linear_suite.py
```

## Shared learned decoder path

The repository now also has a shared learned optical-flow decoder path for the
real benchmark direction:

- model: `src/mvsec_benchmark/models/evflownet_like.py`
- pipeline entry: `run_torch_benchmark(...)`
- script: `scripts/run_torch_benchmark.py`

This path is intended to become the pre-rental and post-rental shared decoder
for real MVSEC experiments.

Example:

```bash
cd D:\event-benchmark\mvsec-benchmark
.venv\Scripts\python.exe scripts\run_torch_benchmark.py --adapter est --use-mock --epochs 5
```

Run all six runnable methods through the shared learned decoder:

```bash
cd D:\event-benchmark\mvsec-benchmark
.venv\Scripts\python.exe scripts\run_torch_suite.py --epochs 3
```

## Recommended real-data plan

### AutoDL / UPenn ROS bag route

On AutoDL, direct Google Drive downloads for the official MVSEC HDF5 and
`*_gt_flow_dist.npz` files can be unreliable. The UPenn-hosted ROS bag files are
reachable and can be downloaded with `aria2`:

```bash
mkdir -p /root/autodl-tmp/capstone/data/mvsec/indoor_flying1
cd /root/autodl-tmp/capstone/data/mvsec/indoor_flying1
aria2c -x 16 -s 16 -k 1M --file-allocation=none \
  -o indoor_flying1_data.bag \
  https://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_data.bag
```

Inspect the bag and export left-camera events into a lightweight HDF5 file:

```bash
python scripts/inspect_rosbag.py /root/autodl-tmp/capstone/data/mvsec/indoor_flying1/indoor_flying1_data.bag
python scripts/convert_mvsec_bag_events.py \
  /root/autodl-tmp/capstone/data/mvsec/indoor_flying1/indoor_flying1_data.bag \
  --topic /davis/left/events \
  --max-events 200000 \
  --output /root/autodl-tmp/capstone/data/mvsec/indoor_flying1/indoor_flying1_left_events_200k.h5
```

This route verifies real MVSEC events without a ROS installation. It still needs
real flow ground truth before producing paper-comparable AEE numbers.

### Unified flow head

For the real MVSEC benchmark, the recommended choice is a single shared
EV-FlowNet-like decoder for all six runnable methods.

Why:

- it keeps the benchmark focused on representation quality instead of changing
  both the representation and the downstream optical-flow network at once
- it is easier to explain in the final report
- it is a much fairer comparison than letting each method pick a different
  decoder or training stack

The current `LinearFlowRegressor` is only the local bring-up head. It exists to
prove that the full `data -> adapter -> train -> evaluate` loop already works
before renting GPUs.

### Completed real-data protocol

The tiny real-data smoke tests, indoor-only controlled experiments, and the
formal outdoor-train / indoor-test run have all completed. The current formal
run command is captured in `scripts/autodl_outdoor_pipeline.sh`.

Use this only if a rerun is needed:

```bash
cd /root/autodl-tmp/capstone/5703
bash scripts/autodl_outdoor_pipeline.sh
```

This route uses the full original-style split:

1. training: `outdoor_day1 + outdoor_day2`
2. testing: `indoor_flying1/2/3`
3. methods: `EST`, `ERGO`, `Event Pre-training`, `GET`, `MatrixLSTM`, `EvRepSL`
4. keep the same flow head and the same metrics

## Next implementation steps

1. Use the formal table above in the report as an adapted reproduction result.
2. State the limitations clearly: shared decoder, one epoch, and index-based
   event-window / flow-frame pairing.
3. If stronger numeric claims are needed, rerun the exact same protocol for more
   epochs without changing the data split or decoder.
4. Only after that, consider paper-specific decoder/backbone work.
