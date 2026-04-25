# MVSEC Benchmark

Unified benchmark scaffold for COMP5703 optical-flow work on MVSEC.

This repository is intentionally staged:

1. build a common task interface
2. attach seven paper-specific representation adapters
3. verify small synthetic smoke tests locally
4. later plug in real MVSEC files and GPU training

Current status:

- common event and metric utilities
- six runnable method adapters plus OmniEvent as reported-only
- per-method environment requirement files
- synthetic smoke test that runs without MVSEC downloads
- minimal MVSEC-style loader and a CPU-friendly linear flow benchmark loop
- AutoDL handoff and current experiment state in `docs/CLAUDE_CODE_HANDOFF.md`

This is **not** yet a paper-faithful full reproduction of all seven methods.
It is the benchmark skeleton that lets the team implement each method in a
consistent place.

For continuing the AutoDL work, start from
`docs/CLAUDE_CODE_HANDOFF.md`. That file records which runs are already done,
which files exist on the data disk, and the exact next command for the
outdoor-train / indoor-test protocol.

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

### Minimal real-data protocol before renting more time

Before running the full benchmark on a rented GPU machine, fix a tiny real-data
smoke-test protocol and do not change it casually:

1. dataset: `MVSEC indoor_flying1`
2. metrics: `AEE + Outlier`
3. methods: start with `EST` and `Event Pre-training`
4. decoder: one shared EV-FlowNet-like head
5. goal: verify the real MVSEC training and evaluation path, not paper numbers

After that smoke test is stable, move to the full MVSEC protocol:

1. training: `outdoor_day1 + outdoor_day2`
2. testing: `indoor_flying1/2/3`
3. methods: `EST`, `ERGO`, `Event Pre-training`, `GET`, `MatrixLSTM`, `EvRepSL`
4. keep the same flow head and the same metrics

## Next implementation steps

1. run the shared EV-FlowNet-like path on `indoor_flying1`
2. stabilize one real-data smoke test with `EST` and `Event Pre-training`
3. progressively upgrade first-pass adapters toward paper-faithful code paths
4. expand to the full MVSEC train/test split once the smoke test is stable
