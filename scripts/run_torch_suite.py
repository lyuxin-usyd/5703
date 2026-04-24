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
from mvsec_benchmark.pipeline import run_torch_benchmark


METHODS = [
    "est",
    "ergo",
    "event_pretraining",
    "get",
    "matrixlstm",
    "evrepsl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared EV-FlowNet-like path for all six runnable methods.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN for GPUs with unsupported conv kernels.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for saving the JSON benchmark results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--train-windows", type=int, default=4)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch

        torch.backends.cudnn.enabled = False

    outdir = ROOT / "examples" / "mock_mvsec"
    h5_path, flow_path = write_mock_mvsec_pair(outdir)
    samples = load_mvsec_windows(
        h5_path=h5_path,
        flow_path=flow_path,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
    )

    results = {}
    for method in METHODS:
        results[method] = run_torch_benchmark(
            samples,
            adapter_name=method,
            train_windows=args.train_windows,
            epochs=args.epochs,
            base_channels=args.base_channels,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ).__dict__
    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
