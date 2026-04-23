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

This is **not** yet a paper-faithful full reproduction of all seven methods.
It is the benchmark skeleton that lets the team implement each method in a
consistent place.

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

## Next implementation steps

1. replace synthetic data with real MVSEC loader
2. replace the linear flow head with EV-FlowNet or another unified decoder
3. progressively upgrade first-pass adapters toward paper-faithful code paths
4. add small real-data tests once `indoor_flying1` is available
