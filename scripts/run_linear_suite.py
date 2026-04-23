from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import load_mvsec_windows, write_mock_mvsec_pair
from mvsec_benchmark.pipeline import run_linear_benchmark


METHODS = [
    "est",
    "ergo",
    "event_pretraining",
    "get",
    "matrixlstm",
    "evrepsl",
]


def main() -> None:
    outdir = ROOT / "examples" / "mock_mvsec"
    h5_path, flow_path = write_mock_mvsec_pair(outdir)
    samples = load_mvsec_windows(
        h5_path=h5_path,
        flow_path=flow_path,
        window_size=200,
        stride=200,
        max_windows=6,
    )

    results = {}
    for method in METHODS:
        results[method] = run_linear_benchmark(
            samples,
            adapter_name=method,
            train_windows=4,
        ).__dict__
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
