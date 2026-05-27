from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.paperlike import (
    load_matrixlstm_image_pair_windows,
    load_matrixlstm_lazy_random_image_pair_windows,
    load_matrixlstm_paperlike_windows,
)
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
    image_strides: list[int],
    image_scale: str,
    max_windows_per_set: int | None,
    lazy_random_stride: bool = False,
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
            if lazy_random_stride:
                stride_min = min(image_strides)
                stride_max = max(image_strides)
                loaded = load_matrixlstm_lazy_random_image_pair_windows(
                    h5_path=Path(h5_raw),
                    flow_path=Path(flow_raw),
                    image_h5_path=Path(image_h5),
                    image_stride_min=stride_min,
                    image_stride_max=stride_max,
                    image_scale=image_scale,
                    max_windows=max_windows_per_set,
                )
                print(
                    f"[load:{label}] pair {idx}/{len(pairs)} lazy_stride={stride_min}..{stride_max} windows={len(loaded)}",
                    flush=True,
                )
            else:
                loaded = []
                for stride in image_strides:
                    stride_loaded = load_matrixlstm_image_pair_windows(
                        h5_path=Path(h5_raw),
                        flow_path=Path(flow_raw),
                        image_h5_path=Path(image_h5),
                        image_stride=stride,
                        image_scale=image_scale,
                        max_windows=max_windows_per_set,
                    )
                    print(
                        f"[load:{label}] pair {idx}/{len(pairs)} stride={stride} windows={len(stride_loaded)}",
                        flush=True,
                    )
                    loaded.extend(stride_loaded)
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
    parser.add_argument("--train-image-stride-min", type=int, default=None)
    parser.add_argument("--train-image-stride-max", type=int, default=None)
    parser.add_argument("--image-scale", choices=["unit", "raw255"], default="unit")
    parser.add_argument("--max-train-windows-per-set", type=int, default=None)
    parser.add_argument("--max-eval-windows-per-set", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--model-variant",
        choices=["lite", "evflownet_multiscale"],
        default="lite",
    )
    parser.add_argument(
        "--training-objective",
        choices=["supervised_regularized", "self_supervised"],
        default="supervised_regularized",
    )
    parser.add_argument("--supervised-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--photometric-loss",
        choices=["l1", "charbonnier", "ssim", "charbonnier_ssim", "evflownet"],
        default="l1",
    )
    parser.add_argument("--photometric-ssim-weight", type=float, default=0.0)
    parser.add_argument("--photometric-charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--photometric-charbonnier-alpha", type=float, default=0.45)
    parser.add_argument("--photometric-valid-mask", action="store_true")
    parser.add_argument(
        "--smoothness-mode",
        choices=["first_order", "edge_aware", "evflownet_8conn"],
        default="first_order",
    )
    parser.add_argument("--smoothness-edge-weight", type=float, default=10.0)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "step", "evflownet"],
        default="constant",
    )
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--model-batch-norm", action="store_true")
    parser.add_argument("--paper-crop-size", type=int, default=0)
    parser.add_argument("--paper-train-random-crop", action="store_true")
    parser.add_argument("--paper-random-flip", action="store_true")
    parser.add_argument("--paper-random-rotation-degrees", type=float, default=0.0)
    parser.add_argument("--random-train-stride-per-epoch", action="store_true")
    parser.add_argument("--lazy-random-train-stride", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--inference-dir", type=Path, default=None)
    parser.add_argument("--inference-manifest", type=Path, default=None)
    parser.add_argument("--inference-sample-index", action="append", type=int, default=None)
    parser.add_argument(
        "--metric-scope",
        choices=["matrixlstm_paperlike", "evflownet_official"],
        default="matrixlstm_paperlike",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch

        torch.backends.cudnn.enabled = False

    train_stride_min = args.train_image_stride_min if args.train_image_stride_min is not None else args.image_stride
    train_stride_max = args.train_image_stride_max if args.train_image_stride_max is not None else args.image_stride
    if train_stride_min < 1 or train_stride_max < train_stride_min:
        raise SystemExit("--train-image-stride-min/max must define a positive inclusive range.")
    train_image_strides = list(range(train_stride_min, train_stride_max + 1))
    eval_image_strides = [args.image_stride]

    train_samples = _load_sets(
        args.train_pair,
        image_h5s=args.train_image_h5,
        image_strides=train_image_strides,
        image_scale=args.image_scale,
        max_windows_per_set=args.max_train_windows_per_set,
        lazy_random_stride=args.lazy_random_train_stride,
        label="train",
    )
    eval_samples = _load_sets(
        args.eval_pair,
        image_h5s=args.eval_image_h5,
        image_strides=eval_image_strides,
        image_scale=args.image_scale,
        max_windows_per_set=args.max_eval_windows_per_set,
        lazy_random_stride=False,
        label="eval",
    )
    checkpoint_output = args.checkpoint_output
    inference_dir = args.inference_dir
    inference_manifest = args.inference_manifest
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_output = checkpoint_output or args.artifact_dir / "best_checkpoint.pt"
        inference_dir = inference_dir or args.artifact_dir / "inference"
        inference_manifest = inference_manifest or args.artifact_dir / "sample_manifest.csv"
    inference_sample_indices = args.inference_sample_index
    if inference_sample_indices is None and inference_dir is not None:
        inference_sample_indices = [0]

    result = run_torch_train_eval_benchmark(
        train_samples,
        eval_samples,
        adapter_name=args.adapter,
        epochs=args.epochs,
        learning_rate=args.lr,
        base_channels=args.base_channels,
        model_variant=args.model_variant,
        training_objective=args.training_objective,
        supervised_weight=args.supervised_weight,
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
        metric_scope=args.metric_scope,
        photometric_weight=args.photometric_weight,
        smoothness_weight=args.smoothness_weight,
        photometric_loss=args.photometric_loss,
        photometric_ssim_weight=args.photometric_ssim_weight,
        photometric_charbonnier_epsilon=args.photometric_charbonnier_epsilon,
        photometric_charbonnier_alpha=args.photometric_charbonnier_alpha,
        photometric_use_valid_mask=args.photometric_valid_mask,
        smoothness_mode=args.smoothness_mode,
        smoothness_edge_weight=args.smoothness_edge_weight,
        lr_schedule=args.lr_schedule,
        lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        model_batch_norm=args.model_batch_norm,
        paper_crop_size=args.paper_crop_size,
        paper_train_random_crop=args.paper_train_random_crop,
        paper_random_flip=args.paper_random_flip,
        paper_random_rotation_degrees=args.paper_random_rotation_degrees,
        random_train_stride_per_epoch=args.random_train_stride_per_epoch,
        checkpoint_path=checkpoint_output,
        inference_dir=inference_dir,
        inference_sample_indices=inference_sample_indices,
        inference_manifest_path=inference_manifest,
    )
    result_dict = {key: value for key, value in result.__dict__.items() if value is not None}
    has_images = args.train_image_h5 is not None or args.eval_image_h5 is not None
    has_extra_losses = bool(args.photometric_weight or args.smoothness_weight)
    result_dict["probe_scope"] = (
        "v9_optical_flow_protocol"
        if has_images or has_extra_losses
        else "matrixlstm_gt_propagation_event_mask"
    )
    result_dict["not_full_original_reproduction"] = True
    result_dict["included_losses"] = []
    if args.training_objective != "self_supervised" and args.supervised_weight:
        result_dict["included_losses"].append(f"supervised_smooth_l1_weight_{args.supervised_weight}")
    if args.photometric_weight:
        result_dict["included_losses"].append(f"image_pair_photometric_warping_{args.photometric_loss}")
        if args.photometric_valid_mask:
            result_dict["included_losses"].append("photometric_in_bounds_valid_mask")
    if args.smoothness_weight:
        result_dict["included_losses"].append(f"flow_smoothness_{args.smoothness_mode}")
    result_dict["not_yet_included"] = []
    if args.model_variant != "evflownet_multiscale":
        result_dict["not_yet_included"].append("EV-FlowNet multi-scale decoder")
    result_dict["not_yet_included"].append("original TensorFlow runtime/checkpoint")
    if not args.photometric_weight:
        result_dict["not_yet_included"].append("image-pair photometric warping loss")
    if args.photometric_loss != "ssim":
        result_dict["not_yet_included"].append("census transform photometric loss")
    result_dict["not_yet_included"].append("TFRecord loader based on grayscale frame pairs")
    result_dict["train_sets"] = args.train_pair
    result_dict["eval_sets"] = args.eval_pair
    result_dict["train_image_h5s"] = args.train_image_h5
    result_dict["eval_image_h5s"] = args.eval_image_h5
    result_dict["image_stride"] = args.image_stride
    result_dict["train_image_strides"] = train_image_strides
    result_dict["image_scale"] = args.image_scale
    result_dict["metric_scope_requested"] = args.metric_scope
    result_dict["model_variant"] = args.model_variant
    result_dict["training_objective"] = args.training_objective
    result_dict["supervised_weight"] = args.supervised_weight
    result_dict["photometric_weight"] = args.photometric_weight
    result_dict["smoothness_weight"] = args.smoothness_weight
    result_dict["photometric_loss"] = args.photometric_loss
    result_dict["photometric_ssim_weight"] = args.photometric_ssim_weight
    result_dict["photometric_charbonnier_epsilon"] = args.photometric_charbonnier_epsilon
    result_dict["photometric_charbonnier_alpha"] = args.photometric_charbonnier_alpha
    result_dict["photometric_valid_mask"] = args.photometric_valid_mask
    result_dict["smoothness_mode"] = args.smoothness_mode
    result_dict["smoothness_edge_weight"] = args.smoothness_edge_weight
    result_dict["lr_schedule"] = args.lr_schedule
    result_dict["lr_decay"] = args.lr_decay
    result_dict["warmup_epochs"] = args.warmup_epochs
    result_dict["min_lr"] = args.min_lr
    result_dict["weight_decay"] = args.weight_decay
    result_dict["gradient_clip_norm"] = args.gradient_clip_norm
    result_dict["model_batch_norm"] = args.model_batch_norm
    result_dict["paper_crop_size"] = args.paper_crop_size
    result_dict["paper_train_random_crop"] = args.paper_train_random_crop
    result_dict["paper_random_flip"] = args.paper_random_flip
    result_dict["paper_random_rotation_degrees"] = args.paper_random_rotation_degrees
    result_dict["random_train_stride_per_epoch"] = args.random_train_stride_per_epoch
    result_dict["lazy_random_train_stride"] = args.lazy_random_train_stride
    result_dict["max_train_windows_per_set"] = args.max_train_windows_per_set
    result_dict["max_eval_windows_per_set"] = args.max_eval_windows_per_set
    result_dict["progress_every"] = args.progress_every
    result_dict["early_stop_patience"] = args.early_stop_patience
    result_dict["early_stop_min_delta"] = args.early_stop_min_delta
    result_dict["early_stop_val_windows_requested"] = args.early_stop_val_windows
    result_dict["early_stop_val_strategy_requested"] = args.early_stop_val_strategy
    result_dict["curve_log_requested"] = str(args.curve_log) if args.curve_log is not None else None
    result_dict["artifact_dir"] = str(args.artifact_dir) if args.artifact_dir is not None else None
    result_dict["inference_sample_indices_requested"] = inference_sample_indices

    payload = json.dumps(result_dict, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    if args.artifact_dir is not None:
        (args.artifact_dir / "metrics.json").write_text(payload + "\n", encoding="utf-8")
        config_keys = [
            "adapter",
            "train_pair",
            "eval_pair",
            "train_image_h5",
            "eval_image_h5",
            "image_stride",
            "train_image_stride_min",
            "train_image_stride_max",
            "image_scale",
            "epochs",
            "lr",
            "base_channels",
            "model_variant",
            "training_objective",
            "supervised_weight",
            "batch_size",
            "eval_batch_size",
            "device",
            "seed",
            "early_stop_patience",
            "early_stop_min_delta",
            "early_stop_val_windows",
            "early_stop_val_strategy",
            "photometric_weight",
            "smoothness_weight",
            "photometric_loss",
            "photometric_ssim_weight",
            "photometric_charbonnier_epsilon",
            "photometric_charbonnier_alpha",
            "photometric_valid_mask",
            "smoothness_mode",
            "smoothness_edge_weight",
            "lr_schedule",
            "lr_decay",
            "warmup_epochs",
            "min_lr",
            "weight_decay",
            "gradient_clip_norm",
            "model_batch_norm",
            "paper_crop_size",
            "paper_train_random_crop",
            "paper_random_flip",
            "paper_random_rotation_degrees",
            "random_train_stride_per_epoch",
            "lazy_random_train_stride",
            "metric_scope",
        ]
        config_payload = {key: getattr(args, key) for key in config_keys}
        (args.artifact_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
