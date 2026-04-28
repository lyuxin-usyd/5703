from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import load_mvsec_windows
from mvsec_benchmark.pipeline import run_torch_train_eval_benchmark


def _load_sets(
    pairs: list[str],
    *,
    window_size: int,
    stride: int,
    max_windows_per_set: int | None,
    label: str,
) -> list:
    samples = []
    for idx, item in enumerate(pairs, start=1):
        try:
            h5_raw, flow_raw = item.split(":", 1)
        except ValueError as exc:
            raise SystemExit(f"Expected H5:FLOW pair, got: {item}") from exc
        print(f"[load:{label}] pair {idx}/{len(pairs)} h5={h5_raw}", flush=True)
        print(f"[load:{label}] pair {idx}/{len(pairs)} flow={flow_raw}", flush=True)
        loaded = load_mvsec_windows(
            h5_path=Path(h5_raw),
            flow_path=Path(flow_raw),
            window_size=window_size,
            stride=stride,
            max_windows=max_windows_per_set,
        )
        print(f"[load:{label}] pair {idx}/{len(pairs)} windows={len(loaded)}", flush=True)
        for sample in loaded:
            sample.meta["source_h5"] = h5_raw
            sample.meta["source_flow"] = flow_raw
        samples.extend(loaded)
    print(f"[load:{label}] total_windows={len(samples)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run original-style MVSEC protocol: train on outdoor, evaluate on indoor."
    )
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument(
        "--train-pair",
        action="append",
        required=True,
        help="Training pair formatted as /path/events.h5:/path/flow.npz. Repeat for outdoor_day1/2.",
    )
    parser.add_argument(
        "--eval-pair",
        action="append",
        required=True,
        help="Evaluation pair formatted as /path/events.h5:/path/flow.npz. Repeat for indoor_flying1/2/3.",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-train-windows-per-set", type=int, default=None)
    parser.add_argument("--max-eval-windows-per-set", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-metrics", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop when outdoor validation AEE has not improved for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation AEE improvement required to reset early-stop patience.",
    )
    parser.add_argument(
        "--early-stop-val-windows",
        type=int,
        default=0,
        help="Hold out this many outdoor training windows for early-stop validation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch

        torch.backends.cudnn.enabled = False

    train_samples = _load_sets(
        args.train_pair,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_set=args.max_train_windows_per_set,
        label="train",
    )
    eval_samples = _load_sets(
        args.eval_pair,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_set=args.max_eval_windows_per_set,
        label="eval",
    )

    result = run_torch_train_eval_benchmark(
        train_samples,
        eval_samples,
        adapter_name=args.adapter,
        epochs=args.epochs,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        seed=args.seed,
        return_window_metrics=args.window_metrics,
        progress_every=args.progress_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_val_windows=args.early_stop_val_windows,
    )
    result_dict = {key: value for key, value in result.__dict__.items() if value is not None}
    result_dict["train_sets"] = args.train_pair
    result_dict["eval_sets"] = args.eval_pair
    result_dict["window_size"] = args.window_size
    result_dict["stride"] = args.stride
    result_dict["max_train_windows_per_set"] = args.max_train_windows_per_set
    result_dict["max_eval_windows_per_set"] = args.max_eval_windows_per_set
    result_dict["progress_every"] = args.progress_every
    result_dict["early_stop_patience"] = args.early_stop_patience
    result_dict["early_stop_min_delta"] = args.early_stop_min_delta
    result_dict["early_stop_val_windows_requested"] = args.early_stop_val_windows

    payload = json.dumps(result_dict, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
