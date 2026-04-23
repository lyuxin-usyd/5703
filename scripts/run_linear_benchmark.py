from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import load_mvsec_windows, write_mock_mvsec_pair
from mvsec_benchmark.pipeline import run_linear_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny train/eval loop for one adapter.")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--flow", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--use-mock", action="store_true")
    args = parser.parse_args()

    if args.use_mock or args.h5 is None or args.flow is None:
        outdir = ROOT / "examples" / "mock_mvsec"
        h5_path, flow_path = write_mock_mvsec_pair(outdir)
    else:
        h5_path, flow_path = args.h5, args.flow

    samples = load_mvsec_windows(
        h5_path=h5_path,
        flow_path=flow_path,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    result = run_linear_benchmark(
        samples,
        adapter_name=args.adapter,
        train_windows=args.train_windows,
        ridge=args.ridge,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
