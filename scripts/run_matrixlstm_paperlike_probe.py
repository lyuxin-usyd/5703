from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.paperlike import load_matrixlstm_image_pair_windows, load_matrixlstm_paperlike_windows
from mvsec_benchmark.pipeline import run_torch_train_eval_benchmark


def _split_pair(item: str) -> tuple[str, str]:
    if "::" in item:
        return tuple(item.split("::", 1))  # type: ignore[return-value]
    for marker in (".h5:", ".hdf5:"):
        pos = item.lower().find(marker)
        if pos >= 0:
            split_at = pos + len(marker) - 1
            return item[:split_at], item[split_at + 1:]
    try:
        return tuple(item.split(":", 1))  # type: ignore[return-value]
    except ValueError as exc:
        raise SystemExit(f"Expected H5:FLOW pair, got: {item}") from exc


def _load_sets(
    pairs: list[str],
    *,
    image_h5s: list[str] | None,
    image_stride: int,
    max_windows_per_set: int | None,
    label: str,
) -> list:
    if image_h5s is not None and len(image_h5s) != len(pairs):
        raise SystemExit(f"--{label}-image-h5 count must match --{label}-pair count.")
    samples = []
    for idx, item in enumerate(pairs, start=1):
        h5_raw, flow_raw = _split_pair(item)
        print(f"[load:{label}] pair {idx}/{len(pairs)} h5={h5_raw}", flush=True)
        print(f"[load:{label}] pair {idx}/{len(pairs)} flow={flow_raw}", flush=True)
        if image_h5s is None:
            loaded = load_matrixlstm_paperlike_windows(
                h5_path=Path(h5_raw),
                flow_path=Path(flow_raw),
                max_windows=max_windows_per_set,
            )
        else:
            image_h5 = image_h5s[idx - 1]
            print(f"[load:{label}] pair {idx}/{len(pairs)} image_h5={image_h5}", flush=True)
            loaded = load_matrixlstm_image_pair_windows(
                h5_path=Path(h5_raw),
                flow_path=Path(flow_raw),
                image_h5_path=Path(image_h5),
                image_stride=image_stride,
                max_windows=max_windows_per_set,
            )
        print(f"[load:{label}] pair {idx}/{len(pairs)} windows={len(loaded)}", flush=True)
        for sample in loaded:
            sample.meta["source_h5"] = h5_raw
            sample.meta["source_flow"] = flow_raw
            if image_h5s is not None:
                sample.meta["source_image_h5"] = image_h5s[idx - 1]
        samples.extend(loaded)
    print(f"[load:{label}] total_windows={len(samples)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental MatrixLSTM/EV-FlowNet paper-like probe: timestamp GT propagation "
            "plus event-mask AEE/outlier evaluation."
        )
    )
    parser.add_argument("--adapter", type=str, default="matrixlstm")
    parser.add_argument("--train-pair", action="append", required=True)
    parser.add_argument("--eval-pair", action="append", required=True)
    parser.add_argument("--train-image-h5", action="append", default=None)
    parser.add_argument("--eval-image-h5", action="append", default=None)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--max-train-windows-per-set", type=int, default=None)
    parser.add_argument("--max-eval-windows-per-set", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-metrics", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-val-windows", type=int, default=20)
    parser.add_argument(
        "--early-stop-val-strategy",
        choices=["tail", "block-random"],
        default="block-random",
    )
    parser.add_argument("--curve-log", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--photometric-weight", type=float, default=0.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch

        torch.backends.cudnn.enabled = False

    train_samples = _load_sets(
        args.train_pair,
        image_h5s=args.train_image_h5,
        image_stride=args.image_stride,
        max_windows_per_set=args.max_train_windows_per_set,
        label="train",
    )
    eval_samples = _load_sets(
        args.eval_pair,
        image_h5s=args.eval_image_h5,
        image_stride=args.image_stride,
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
        early_stop_val_strategy=args.early_stop_val_strategy,
        curve_log_path=args.curve_log,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        metric_scope="matrixlstm_paperlike",
        photometric_weight=args.photometric_weight,
        smoothness_weight=args.smoothness_weight,
    )
    result_dict = {key: value for key, value in result.__dict__.items() if value is not None}
    has_images = args.train_image_h5 is not None or args.eval_image_h5 is not None
    has_extra_losses = bool(args.photometric_weight or args.smoothness_weight)
    result_dict["probe_scope"] = (
        "matrixlstm_image_pair_photometric_smoothness"
        if has_images or has_extra_losses
        else "matrixlstm_gt_propagation_event_mask"
    )
    result_dict["not_full_original_reproduction"] = True
    result_dict["included_losses"] = ["supervised_smooth_l1"]
    if args.photometric_weight:
        result_dict["included_losses"].append("image_pair_photometric_warping")
    if args.smoothness_weight:
        result_dict["included_losses"].append("flow_smoothness")
    result_dict["not_yet_included"] = ["original TensorFlow EV-FlowNet architecture"]
    if not args.photometric_weight:
        result_dict["not_yet_included"].append("image-pair photometric warping loss")
    result_dict["not_yet_included"].append("TFRecord loader based on grayscale frame pairs")
    result_dict["train_sets"] = args.train_pair
    result_dict["eval_sets"] = args.eval_pair
    result_dict["train_image_h5s"] = args.train_image_h5
    result_dict["eval_image_h5s"] = args.eval_image_h5
    result_dict["image_stride"] = args.image_stride
    result_dict["photometric_weight"] = args.photometric_weight
    result_dict["smoothness_weight"] = args.smoothness_weight
    result_dict["max_train_windows_per_set"] = args.max_train_windows_per_set
    result_dict["max_eval_windows_per_set"] = args.max_eval_windows_per_set
    result_dict["progress_every"] = args.progress_every
    result_dict["early_stop_patience"] = args.early_stop_patience
    result_dict["early_stop_min_delta"] = args.early_stop_min_delta
    result_dict["early_stop_val_windows_requested"] = args.early_stop_val_windows
    result_dict["early_stop_val_strategy_requested"] = args.early_stop_val_strategy
    result_dict["curve_log_requested"] = str(args.curve_log) if args.curve_log is not None else None

    payload = json.dumps(result_dict, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
