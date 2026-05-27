from __future__ import annotations

from dataclasses import dataclass
import csv
import copy
import json
import math
from pathlib import Path
import re

import numpy as np

from .adapters import build_adapters
from .data.mvsec import FlowWindowSample
from .models.linear_flow import LinearFlowRegressor
from .paperlike import make_matrixlstm_image_pair_sample_from_arrays
from .utils.flow_metrics import FlowMetrics, compute_flow_metrics, event_gt_valid_mask


@dataclass(frozen=True)
class BenchmarkResult:
    adapter_name: str
    train_windows: int
    eval_windows: int
    channels: int
    aee: float
    outlier_percent: float
    valid_count: int
    metric_scope: str = "full_gt_valid"
    window_metrics: list[dict[str, float | int | str]] | None = None
    epochs_completed: int | None = None
    early_stopped: bool | None = None
    best_epoch: int | None = None
    best_val_aee: float | None = None
    early_stop_val_windows: int | None = None
    early_stop_val_strategy: str | None = None
    early_stop_val_source_counts: dict[str, int] | None = None
    curve_log_path: str | None = None
    per_sequence_metrics: dict[str, dict[str, float | int]] | None = None
    checkpoint_path: str | None = None
    inference_dir: str | None = None
    inference_manifest_path: str | None = None


def _split_samples(samples: list[FlowWindowSample], train_windows: int) -> tuple[list[FlowWindowSample], list[FlowWindowSample]]:
    if len(samples) < 2:
        raise ValueError("At least two windows are required for train/eval.")
    split = min(max(train_windows, 1), len(samples) - 1)
    return samples[:split], samples[split:]


def _build_adapter_representations(
    samples: list[FlowWindowSample],
    *,
    adapter_name: str,
) -> tuple[object, list[np.ndarray]]:
    if adapter_name == "omnievent":
        raise ValueError("OmniEvent is reported-only in the current benchmark workflow.")

    adapters = build_adapters()
    if adapter_name not in adapters:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    adapter = adapters[adapter_name]
    reps = [adapter.build(s.events, s.sensor_size) for s in samples]
    return adapter, reps


def _source_key(sample: FlowWindowSample) -> str:
    source = sample.meta.get("source_h5") or sample.meta.get("source_flow") or "unknown"
    return str(source)


def _random_stride_group_key(sample: FlowWindowSample) -> str:
    image_end_index = sample.meta.get("image_end_index")
    if image_end_index is None:
        return f"{_source_key(sample)}|window:{sample.meta.get('window_index', id(sample))}"
    return f"{_source_key(sample)}|image_end:{image_end_index}"


def _representative_group_sample(group: list[FlowWindowSample]) -> FlowWindowSample:
    return sorted(group, key=lambda sample: int(sample.meta.get("image_stride") or 1))[0]


_KNOWN_SEQUENCES = (
    "outdoor_day1",
    "outdoor_day2",
    "indoor_flying1",
    "indoor_flying2",
    "indoor_flying3",
)


def _sequence_key(sample: FlowWindowSample) -> str:
    explicit = sample.meta.get("sequence")
    if explicit:
        return str(explicit)

    for key_name in ("source_h5", "source_flow", "source_image_h5"):
        source = sample.meta.get(key_name)
        if not source:
            continue
        lowered = str(source).replace("\\", "/").lower()
        for sequence in _KNOWN_SEQUENCES:
            if sequence in lowered:
                return sequence
    source = _source_key(sample)
    if source == "unknown":
        return "unknown"
    return Path(source).stem


def _safe_sample_meta(sample: FlowWindowSample) -> dict[str, int | float | str | bool]:
    safe: dict[str, int | float | str | bool] = {}
    for key, value in sample.meta.items():
        if str(key).startswith("lazy_"):
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        elif isinstance(value, np.integer):
            safe[str(key)] = int(value)
        elif isinstance(value, np.floating):
            safe[str(key)] = float(value)
        elif isinstance(value, np.bool_):
            safe[str(key)] = bool(value)
    return safe


def _sample_trace_row(sample: FlowWindowSample, *, eval_index: int) -> dict[str, int | float | str | bool]:
    meta = _safe_sample_meta(sample)
    row: dict[str, int | float | str | bool] = {
        "eval_index": int(eval_index),
        "sequence": _sequence_key(sample),
        "sensor_height": int(sample.sensor_size[0]),
        "sensor_width": int(sample.sensor_size[1]),
    }
    for key in (
        "alignment",
        "window_index",
        "flow_index",
        "image_start_index",
        "image_end_index",
        "image_stride",
        "event_start",
        "event_end",
        "event_start_time",
        "event_end_time",
        "flow_timestamp",
        "source_h5",
        "source_flow",
        "source_image_h5",
    ):
        if key in meta:
            row[key] = meta[key]
    return row


def _sample_artifact_id(sample: FlowWindowSample, *, eval_index: int) -> str:
    row = _sample_trace_row(sample, eval_index=eval_index)
    parts = [
        f"sample_{eval_index:04d}",
        str(row.get("sequence", "unknown")),
        f"window_{row.get('window_index', eval_index)}",
    ]
    if "image_end_index" in row:
        parts.append(f"image_{row['image_end_index']}")
    if "image_stride" in row:
        parts.append(f"stride_{row['image_stride']}")
    raw = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _count_sources(samples: list[FlowWindowSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = _source_key(sample)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _summarize_metrics_by_sequence(
    samples: list[FlowWindowSample],
    metrics: list[FlowMetrics],
) -> dict[str, dict[str, float | int]]:
    if len(samples) != len(metrics):
        raise ValueError("samples and metrics must have the same length.")

    grouped: dict[str, list[FlowMetrics]] = {}
    for sample, metric in zip(samples, metrics):
        grouped.setdefault(_sequence_key(sample), []).append(metric)

    summary: dict[str, dict[str, float | int]] = {}
    for sequence, group in sorted(grouped.items()):
        group = [
            metric
            for metric in group
            if metric.valid_count > 0 and math.isfinite(metric.aee) and math.isfinite(metric.outlier_percent)
        ]
        if not group:
            continue
        summary[sequence] = {
            "windows": int(len(group)),
            "aee": float(sum(metric.aee for metric in group) / len(group)),
            "outlier_percent": float(sum(metric.outlier_percent for metric in group) / len(group)),
            "valid_count": int(sum(metric.valid_count for metric in group)),
            "outlier_count": int(sum(metric.outlier_count for metric in group)),
        }
    return summary


def _compute_benchmark_metrics(
    pred_flow: np.ndarray,
    sample: FlowWindowSample,
    *,
    metric_scope: str = "full_gt_valid",
) -> FlowMetrics:
    pred_hw2 = np.asarray(pred_flow)
    target_hw = (int(pred_hw2.shape[0]), int(pred_hw2.shape[1]))
    gt_flow = sample.gt_flow
    outlier_mode = "kitti"
    if metric_scope == "full_gt_valid":
        valid_mask = None
    elif metric_scope in {"event_valid", "matrixlstm_paperlike", "evflownet_official"}:
        valid_mask = event_gt_valid_mask(sample.events, gt_flow, sample.sensor_size)
    else:
        raise ValueError("metric_scope must be 'full_gt_valid', 'event_valid', 'matrixlstm_paperlike', or 'evflownet_official'.")
    if tuple(gt_flow.shape[:2]) != target_hw:
        gt_flow = _center_crop_or_pad_np(gt_flow, target_hw, fill_value=0)
        if valid_mask is not None:
            valid_mask = _center_crop_or_pad_np(valid_mask, target_hw, fill_value=False)
    if metric_scope == "evflownet_official":
        outlier_mode = "px"
        if valid_mask is not None and "outdoor" in _sequence_key(sample):
            valid_mask = np.asarray(valid_mask, dtype=bool).copy()
            valid_mask[190:, :] = False
    return compute_flow_metrics(
        pred_hw2,
        gt_flow,
        valid_mask=valid_mask,
        outlier_mode=outlier_mode,
    )


def _center_crop_or_pad_np(array: np.ndarray, target_hw: tuple[int, int], *, fill_value: int | float | bool = 0) -> np.ndarray:
    arr = np.asarray(array)
    target_h, target_w = target_hw
    source_h, source_w = arr.shape[:2]
    start_y = max((source_h - target_h) // 2, 0)
    start_x = max((source_w - target_w) // 2, 0)
    cropped = arr[start_y:start_y + min(source_h, target_h), start_x:start_x + min(source_w, target_w), ...]
    pad_top = max((target_h - source_h) // 2, 0)
    pad_left = max((target_w - source_w) // 2, 0)
    output_shape = (target_h, target_w) + arr.shape[2:]
    output = np.full(output_shape, fill_value, dtype=arr.dtype)
    output[
        pad_top:pad_top + cropped.shape[0],
        pad_left:pad_left + cropped.shape[1],
        ...,
    ] = cropped
    return output


def _allocate_val_counts(group_sizes: dict[str, int], requested: int) -> dict[str, int]:
    capacities = {key: max(size - 1, 0) for key, size in group_sizes.items()}
    requested = min(requested, sum(capacities.values()))
    if requested <= 0:
        return {key: 0 for key in group_sizes}

    total_size = sum(group_sizes.values())
    raw = {
        key: requested * (size / total_size)
        for key, size in group_sizes.items()
    }
    counts = {
        key: min(int(np.floor(value)), capacities[key])
        for key, value in raw.items()
    }
    remaining = requested - sum(counts.values())
    order = sorted(
        group_sizes,
        key=lambda key: (raw[key] - np.floor(raw[key]), group_sizes[key]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for key in order:
            if counts[key] < capacities[key]:
                counts[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return counts


def _split_early_stop_samples(
    samples: list[FlowWindowSample],
    *,
    val_windows: int,
    strategy: str,
    seed: int,
) -> tuple[list[FlowWindowSample], list[FlowWindowSample], dict[str, int]]:
    if val_windows == 0:
        return samples, [], {}
    if strategy == "tail":
        val_samples = samples[-val_windows:]
        return samples[:-val_windows], val_samples, _count_sources(val_samples)
    if strategy != "block-random":
        raise ValueError("early_stop_val_strategy must be 'tail' or 'block-random'.")

    grouped: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(_source_key(sample), []).append(idx)
    counts = _allocate_val_counts({key: len(indices) for key, indices in grouped.items()}, val_windows)
    if sum(counts.values()) == 0:
        raise ValueError("early_stop_val_windows must leave at least one training window.")

    rng = np.random.default_rng(seed)
    val_indices: set[int] = set()
    for key, count in counts.items():
        if count <= 0:
            continue
        indices = grouped[key]
        start = int(rng.integers(0, len(indices) - count + 1))
        val_indices.update(indices[start:start + count])

    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples, _count_sources(val_samples)


def _split_early_stop_sample_groups(
    samples: list[FlowWindowSample],
    *,
    val_windows: int,
    strategy: str,
    seed: int,
) -> tuple[list[list[FlowWindowSample]], list[FlowWindowSample], dict[str, int]]:
    grouped_by_key: dict[str, list[FlowWindowSample]] = {}
    for sample in samples:
        grouped_by_key.setdefault(_random_stride_group_key(sample), []).append(sample)
    groups = sorted(
        grouped_by_key.values(),
        key=lambda group: (
            _source_key(_representative_group_sample(group)),
            int(_representative_group_sample(group).meta.get("image_end_index") or 0),
            int(_representative_group_sample(group).meta.get("window_index") or 0),
        ),
    )
    if val_windows == 0:
        return groups, [], {}
    if val_windows >= len(groups):
        raise ValueError("early_stop_val_windows must leave at least one training group.")

    representatives = [_representative_group_sample(group) for group in groups]
    if strategy == "tail":
        val_indices = set(range(len(groups) - val_windows, len(groups)))
    elif strategy == "block-random":
        grouped_indices: dict[str, list[int]] = {}
        for idx, sample in enumerate(representatives):
            grouped_indices.setdefault(_source_key(sample), []).append(idx)
        counts = _allocate_val_counts({key: len(indices) for key, indices in grouped_indices.items()}, val_windows)
        rng = np.random.default_rng(seed)
        val_indices = set()
        for key, count in counts.items():
            if count <= 0:
                continue
            indices = grouped_indices[key]
            start = int(rng.integers(0, len(indices) - count + 1))
            val_indices.update(indices[start:start + count])
    else:
        raise ValueError("early_stop_val_strategy must be 'tail' or 'block-random'.")

    train_groups = [group for idx, group in enumerate(groups) if idx not in val_indices]
    val_samples = [representatives[idx] for idx in sorted(val_indices)]
    return train_groups, val_samples, _count_sources(val_samples)


def run_linear_benchmark(
    samples: list[FlowWindowSample],
    *,
    adapter_name: str,
    train_windows: int = 4,
    ridge: float = 1e-3,
) -> BenchmarkResult:
    if adapter_name == "omnievent":
        raise ValueError("OmniEvent is reported-only in the current benchmark workflow.")
    train_samples, eval_samples = _split_samples(samples, train_windows)
    adapter, train_reps = _build_adapter_representations(train_samples, adapter_name=adapter_name)
    train_flows = [s.gt_flow for s in train_samples]

    model = LinearFlowRegressor(ridge=ridge).fit(train_reps, train_flows)

    metrics: list[FlowMetrics] = []
    first_channels = int(train_reps[0].shape[0])
    for sample in eval_samples:
        rep = adapter.build(sample.events, sample.sensor_size)
        pred = model.predict(rep)
        metrics.append(_compute_benchmark_metrics(pred, sample))

    mean_aee = sum(m.aee for m in metrics) / len(metrics)
    mean_outlier = sum(m.outlier_percent for m in metrics) / len(metrics)
    valid_count = sum(m.valid_count for m in metrics)
    return BenchmarkResult(
        adapter_name=adapter_name,
        train_windows=len(train_samples),
        eval_windows=len(eval_samples),
        channels=first_channels,
        aee=float(mean_aee),
        outlier_percent=float(mean_outlier),
        valid_count=int(valid_count),
    )


def run_torch_benchmark(
    samples: list[FlowWindowSample],
    *,
    adapter_name: str,
    train_windows: int = 4,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
    batch_size: int = 2,
    eval_batch_size: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    return_window_metrics: bool = False,
) -> BenchmarkResult:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("run_torch_benchmark requires torch to be installed") from exc

    from .models.evflownet_like import EVFlowNetLike

    torch.manual_seed(seed)
    train_samples, eval_samples = _split_samples(samples, train_windows)
    adapter, train_reps = _build_adapter_representations(train_samples, adapter_name=adapter_name)
    eval_reps = [adapter.build(s.events, s.sensor_size) for s in eval_samples]

    x_train = torch.from_numpy(np.stack(train_reps, axis=0)).float().to(device)
    y_train = torch.from_numpy(np.stack([np.moveaxis(s.gt_flow, -1, 0) for s in train_samples], axis=0)).float().to(device)
    x_eval = torch.from_numpy(np.stack(eval_reps, axis=0)).float().to(device)

    model = EVFlowNetLike(in_channels=int(x_train.shape[1]), base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_train = int(x_train.shape[0])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(x_train[idx])
            loss = F.smooth_l1_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics: list[FlowMetrics] = []
    window_metrics: list[dict[str, float | int]] = []
    eval_batch = int(eval_batch_size or batch_size)
    if eval_batch < 1:
        raise ValueError("eval_batch_size must be >= 1")
    model.eval()
    with torch.no_grad():
        for start in range(0, int(x_eval.shape[0]), eval_batch):
            pred_batch = model(x_eval[start:start + eval_batch]).detach().cpu().numpy()
            for offset, pred in enumerate(pred_batch):
                eval_index = start + offset
                sample = eval_samples[eval_index]
                pred_hw2 = np.moveaxis(pred, 0, -1)
                metric = _compute_benchmark_metrics(pred_hw2, sample)
                metrics.append(metric)
                if return_window_metrics:
                    window_metrics.append(
                        {
                            "sample_index": int(len(train_samples) + eval_index),
                            "eval_index": int(eval_index),
                            "aee": float(metric.aee),
                            "outlier_percent": float(metric.outlier_percent),
                            "valid_count": int(metric.valid_count),
                            "outlier_count": int(metric.outlier_count),
                        }
                    )

    mean_aee = sum(m.aee for m in metrics) / len(metrics)
    mean_outlier = sum(m.outlier_percent for m in metrics) / len(metrics)
    valid_count = sum(m.valid_count for m in metrics)
    return BenchmarkResult(
        adapter_name=adapter_name,
        train_windows=len(train_samples),
        eval_windows=len(eval_samples),
        channels=int(x_train.shape[1]),
        aee=float(mean_aee),
        outlier_percent=float(mean_outlier),
        valid_count=int(valid_count),
        window_metrics=window_metrics if return_window_metrics else None,
    )


def run_torch_train_eval_benchmark(
    train_samples: list[FlowWindowSample],
    eval_samples: list[FlowWindowSample],
    *,
    adapter_name: str,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
    model_variant: str = "lite",
    training_objective: str = "supervised_regularized",
    supervised_weight: float = 1.0,
    batch_size: int = 2,
    eval_batch_size: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    return_window_metrics: bool = False,
    progress_every: int = 100,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    early_stop_val_windows: int = 0,
    early_stop_val_strategy: str = "tail",
    curve_log_path: str | Path | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
    metric_scope: str = "full_gt_valid",
    photometric_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    photometric_loss: str = "l1",
    photometric_ssim_weight: float = 0.0,
    photometric_charbonnier_epsilon: float = 1e-3,
    photometric_charbonnier_alpha: float = 0.45,
    photometric_use_valid_mask: bool = False,
    smoothness_mode: str = "first_order",
    smoothness_edge_weight: float = 10.0,
    lr_schedule: str = "constant",
    lr_decay: float = 0.9,
    warmup_epochs: int = 0,
    min_learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    model_batch_norm: bool = False,
    paper_crop_size: int = 0,
    paper_train_random_crop: bool = False,
    paper_random_flip: bool = False,
    paper_random_rotation_degrees: float = 0.0,
    random_train_stride_per_epoch: bool = False,
    checkpoint_path: str | Path | None = None,
    inference_dir: str | Path | None = None,
    inference_sample_indices: list[int] | None = None,
    inference_manifest_path: str | Path | None = None,
) -> BenchmarkResult:
    """Train on one set of MVSEC windows and evaluate on a separate set.

    This is the path used for the original-style MVSEC protocol:
    outdoor_day1/2 for training and indoor_flying1/2/3 for evaluation.
    Batches are moved to the target device lazily to avoid holding all outdoor
    representations in GPU memory at once.
    """
    if not train_samples:
        raise ValueError("At least one training window is required.")
    if not eval_samples:
        raise ValueError("At least one evaluation window is required.")
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1.")
    if early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be >= 0.")
    if early_stop_val_windows < 0:
        raise ValueError("early_stop_val_windows must be >= 0.")
    if early_stop_patience is not None and early_stop_val_windows == 0:
        raise ValueError("early_stop_val_windows must be > 0 when early stopping is enabled.")
    if early_stop_val_windows and early_stop_val_windows >= len(train_samples):
        raise ValueError("early_stop_val_windows must leave at least one training window.")
    if metric_scope not in {"full_gt_valid", "event_valid", "matrixlstm_paperlike", "evflownet_official"}:
        raise ValueError("metric_scope must be 'full_gt_valid', 'event_valid', 'matrixlstm_paperlike', or 'evflownet_official'.")
    if photometric_weight < 0:
        raise ValueError("photometric_weight must be >= 0.")
    if smoothness_weight < 0:
        raise ValueError("smoothness_weight must be >= 0.")
    photometric_loss = photometric_loss.lower()
    if photometric_loss not in {"l1", "charbonnier", "ssim", "charbonnier_ssim", "evflownet"}:
        raise ValueError("photometric_loss must be 'l1', 'charbonnier', 'ssim', 'charbonnier_ssim', or 'evflownet'.")
    if photometric_ssim_weight < 0 or photometric_ssim_weight > 1:
        raise ValueError("photometric_ssim_weight must be in [0, 1].")
    if photometric_charbonnier_epsilon <= 0:
        raise ValueError("photometric_charbonnier_epsilon must be > 0.")
    if photometric_charbonnier_alpha <= 0:
        raise ValueError("photometric_charbonnier_alpha must be > 0.")
    smoothness_mode = smoothness_mode.lower()
    if smoothness_mode not in {"first_order", "edge_aware", "evflownet_8conn"}:
        raise ValueError("smoothness_mode must be 'first_order', 'edge_aware', or 'evflownet_8conn'.")
    if smoothness_edge_weight < 0:
        raise ValueError("smoothness_edge_weight must be >= 0.")
    lr_schedule = lr_schedule.lower()
    if lr_schedule not in {"constant", "cosine", "step", "evflownet"}:
        raise ValueError("lr_schedule must be 'constant', 'cosine', 'step', or 'evflownet'.")
    if lr_decay <= 0:
        raise ValueError("lr_decay must be > 0.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0.")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be >= 0.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be > 0 when set.")
    if paper_crop_size < 0:
        raise ValueError("paper_crop_size must be >= 0.")
    if paper_random_rotation_degrees < 0:
        raise ValueError("paper_random_rotation_degrees must be >= 0.")
    model_variant = model_variant.lower()
    if model_variant not in {"lite", "evflownet_multiscale"}:
        raise ValueError("model_variant must be 'lite' or 'evflownet_multiscale'.")
    training_objective = training_objective.lower()
    if training_objective not in {"supervised_regularized", "self_supervised"}:
        raise ValueError("training_objective must be 'supervised_regularized' or 'self_supervised'.")
    if supervised_weight < 0:
        raise ValueError("supervised_weight must be >= 0.")
    if training_objective == "self_supervised" and photometric_weight <= 0:
        raise ValueError("self_supervised training requires photometric_weight > 0.")
    if training_objective == "supervised_regularized" and supervised_weight == 0 and not (photometric_weight or smoothness_weight):
        raise ValueError("supervised_regularized training needs supervised_weight or an auxiliary loss.")
    if inference_sample_indices is not None:
        for sample_index in inference_sample_indices:
            if sample_index < 0:
                raise ValueError("inference_sample_indices must be non-negative.")

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("run_torch_train_eval_benchmark requires torch to be installed") from exc

    from .models.evflownet_like import EVFlowNetLike, EVFlowNetMultiScale

    torch.manual_seed(seed)
    if adapter_name == "omnievent":
        raise ValueError("OmniEvent is reported-only in the current benchmark workflow.")

    adapters = build_adapters()
    if adapter_name not in adapters:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    adapter = adapters[adapter_name]

    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    def _progress(message: str) -> None:
        if progress_every:
            print(message, flush=True)

    train_sample_groups: list[list[FlowWindowSample]] | None = None
    if random_train_stride_per_epoch:
        train_sample_groups, val_samples, val_source_counts = _split_early_stop_sample_groups(
            train_samples,
            val_windows=early_stop_val_windows,
            strategy=early_stop_val_strategy,
            seed=seed,
        )
        effective_train_samples = [_representative_group_sample(group) for group in train_sample_groups]
    else:
        effective_train_samples, val_samples, val_source_counts = _split_early_stop_samples(
            train_samples,
            val_windows=early_stop_val_windows,
            strategy=early_stop_val_strategy,
            seed=seed,
        )

    curve_path = Path(curve_log_path) if curve_log_path is not None else None
    if curve_path is not None:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_aee",
                    "best_val_aee",
                    "is_best",
                    "stale_epochs",
                    "early_stopped",
                ],
            )
            writer.writeheader()

    wandb_run = None
    if wandb_project:
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("wandb_project was set, but wandb is not installed.") from exc
        wandb_kwargs: dict[str, object] = {
            "project": wandb_project,
            "name": wandb_run_name,
            "config": {
                "adapter_name": adapter_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "base_channels": base_channels,
                "model_variant": model_variant,
                "training_objective": training_objective,
                "supervised_weight": supervised_weight,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_delta": early_stop_min_delta,
                "early_stop_val_windows": early_stop_val_windows,
                "early_stop_val_strategy": early_stop_val_strategy,
                "metric_scope": metric_scope,
                "photometric_weight": photometric_weight,
                "smoothness_weight": smoothness_weight,
                "photometric_loss": photometric_loss,
                "photometric_ssim_weight": photometric_ssim_weight,
                "photometric_charbonnier_alpha": photometric_charbonnier_alpha,
                "photometric_use_valid_mask": photometric_use_valid_mask,
                "smoothness_mode": smoothness_mode,
                "lr_schedule": lr_schedule,
                "lr_decay": lr_decay,
                "warmup_epochs": warmup_epochs,
                "min_learning_rate": min_learning_rate,
                "weight_decay": weight_decay,
                "gradient_clip_norm": gradient_clip_norm,
                "model_batch_norm": model_batch_norm,
                "paper_crop_size": paper_crop_size,
                "paper_train_random_crop": paper_train_random_crop,
                "paper_random_flip": paper_random_flip,
                "paper_random_rotation_degrees": paper_random_rotation_degrees,
                "random_train_stride_per_epoch": random_train_stride_per_epoch,
                "train_windows": len(effective_train_samples),
                "val_windows": len(val_samples),
                "eval_windows": len(eval_samples),
            },
        }
        if wandb_mode:
            wandb_kwargs["mode"] = wandb_mode
        wandb_run = wandb.init(**wandb_kwargs)

    _progress(
        f"[setup] adapter={adapter_name} train_windows={len(effective_train_samples)} "
        f"val_windows={len(val_samples)} eval_windows={len(eval_samples)} "
        f"val_strategy={early_stop_val_strategy if val_samples else 'none'} "
        f"metric_scope={metric_scope} photometric_weight={photometric_weight} "
        f"smoothness_weight={smoothness_weight} photometric_loss={photometric_loss} "
        f"smoothness_mode={smoothness_mode} lr_schedule={lr_schedule} "
        f"model_variant={model_variant} training_objective={training_objective} "
        f"batch_norm={model_batch_norm} crop_size={paper_crop_size} "
        f"random_crop={paper_train_random_crop} random_flip={paper_random_flip} "
        f"random_rotation_degrees={paper_random_rotation_degrees} "
        f"random_train_stride_per_epoch={random_train_stride_per_epoch}"
    )
    _progress("[setup] building first representation")
    first_rep = adapter.build(effective_train_samples[0].events, effective_train_samples[0].sensor_size)
    first_rep_sample = effective_train_samples[0]
    channels = int(first_rep.shape[0])
    if model_variant == "lite":
        model = EVFlowNetLike(in_channels=channels, base_channels=base_channels).to(device)
    else:
        model = EVFlowNetMultiScale(
            in_channels=channels,
            base_channels=base_channels,
            batch_norm=model_batch_norm,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _epoch_learning_rate(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return float(learning_rate * (epoch_index + 1) / warmup_epochs)
        if lr_schedule == "constant":
            return float(learning_rate)
        if lr_schedule == "evflownet":
            schedule_index = max(epoch_index - warmup_epochs, 0)
            return float(max(min_learning_rate, learning_rate * (lr_decay ** (schedule_index // 4))))
        schedule_index = max(epoch_index - warmup_epochs, 0)
        schedule_epochs = max(epochs - warmup_epochs, 1)
        if lr_schedule == "cosine":
            if schedule_epochs == 1:
                progress = 1.0
            else:
                progress = min(schedule_index / max(schedule_epochs - 1, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_learning_rate + (learning_rate - min_learning_rate) * cosine)
        step_size = max(schedule_epochs // 3, 1)
        return float(max(min_learning_rate, learning_rate * (0.5 ** (schedule_index // step_size))))

    def _set_epoch_learning_rate(epoch_index: int) -> float:
        lr = _epoch_learning_rate(epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _flow_predictions(prediction: object) -> list[object]:
        if isinstance(prediction, (list, tuple)):
            return list(prediction)
        return [prediction]

    def _final_flow(prediction: object) -> object:
        return _flow_predictions(prediction)[-1]

    def _image_chw(image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=np.float32)
        if image_arr.ndim == 2:
            image_arr = image_arr[None, :, :]
        elif image_arr.ndim == 3:
            image_arr = np.moveaxis(image_arr, -1, 0)
            if image_arr.shape[0] != 1:
                image_arr = image_arr.mean(axis=0, keepdims=True)
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {image_arr.shape}.")
        return image_arr

    def _crop_or_pad_batch_tensor(batch: object | None, *, top: int, left: int, target_size: int) -> object | None:
        if batch is None:
            return None
        height, width = batch.shape[-2:]
        pad_h = max(target_size - int(height), 0)
        pad_w = max(target_size - int(width), 0)
        if pad_h or pad_w:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            batch = F.pad(batch, (pad_left, pad_right, pad_top, pad_bottom))
        height, width = batch.shape[-2:]
        crop_top = min(max(top, 0), max(int(height) - target_size, 0))
        crop_left = min(max(left, 0), max(int(width) - target_size, 0))
        return batch[..., crop_top:crop_top + target_size, crop_left:crop_left + target_size]

    def _maybe_apply_paper_crop(
        x_batch: object,
        y_batch: object,
        prev_batch: object | None,
        next_batch: object | None,
        *,
        phase: str,
    ) -> tuple[object, object, object | None, object | None]:
        if paper_crop_size <= 0:
            return x_batch, y_batch, prev_batch, next_batch
        target_size = int(paper_crop_size)
        height, width = x_batch.shape[-2:]
        padded_height = max(int(height), target_size)
        padded_width = max(int(width), target_size)
        max_top = padded_height - target_size
        max_left = padded_width - target_size
        if phase == "train" and paper_train_random_crop:
            top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
            left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
        else:
            top = max_top // 2
            left = max_left // 2
        x_batch = _crop_or_pad_batch_tensor(x_batch, top=top, left=left, target_size=target_size)
        y_batch = _crop_or_pad_batch_tensor(y_batch, top=top, left=left, target_size=target_size)
        prev_batch = _crop_or_pad_batch_tensor(prev_batch, top=top, left=left, target_size=target_size)
        next_batch = _crop_or_pad_batch_tensor(next_batch, top=top, left=left, target_size=target_size)
        return x_batch, y_batch, prev_batch, next_batch

    def _grid_sample_if_present(batch: object | None, grid: object, *, mode: str = "bilinear") -> object | None:
        if batch is None:
            return None
        return F.grid_sample(batch, grid, mode=mode, padding_mode="zeros", align_corners=True)

    def _maybe_apply_paper_augmentation(
        x_batch: object,
        y_batch: object,
        prev_batch: object | None,
        next_batch: object | None,
        *,
        phase: str,
    ) -> tuple[object, object, object | None, object | None]:
        if phase != "train":
            return x_batch, y_batch, prev_batch, next_batch

        if paper_random_flip and bool((torch.rand((), device=x_batch.device) < 0.5).item()):
            x_batch = torch.flip(x_batch, dims=(-1,))
            y_batch = torch.flip(y_batch, dims=(-1,))
            y_batch[:, 0:1] = -y_batch[:, 0:1]
            if prev_batch is not None:
                prev_batch = torch.flip(prev_batch, dims=(-1,))
            if next_batch is not None:
                next_batch = torch.flip(next_batch, dims=(-1,))

        if paper_random_rotation_degrees > 0:
            max_radians = math.radians(float(paper_random_rotation_degrees))
            angle = (torch.rand((), device=x_batch.device, dtype=x_batch.dtype) * 2.0 - 1.0) * max_radians
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            batch_size = int(x_batch.shape[0])
            theta = torch.zeros((batch_size, 2, 3), device=x_batch.device, dtype=x_batch.dtype)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            grid = F.affine_grid(theta, x_batch.size(), align_corners=True)
            x_batch = F.grid_sample(x_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            y_batch = F.grid_sample(y_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            flow_x = y_batch[:, 0:1].clone()
            flow_y = y_batch[:, 1:2].clone()
            y_batch[:, 0:1] = cos_a * flow_x - sin_a * flow_y
            y_batch[:, 1:2] = sin_a * flow_x + cos_a * flow_y
            prev_batch = _grid_sample_if_present(prev_batch, grid)
            next_batch = _grid_sample_if_present(next_batch, grid)

        return x_batch, y_batch, prev_batch, next_batch

    def _warp_next_to_prev(next_image: object, flow: object) -> tuple[object, object]:
        batch, _, height, width = flow.shape
        ys, xs = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=flow.dtype),
            torch.arange(width, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        sample_x = xs.unsqueeze(0) + flow[:, 0]
        sample_y = ys.unsqueeze(0) + flow[:, 1]
        grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        valid = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
        warped = F.grid_sample(next_image, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped, valid.unsqueeze(1)

    def _masked_mean(values: object, mask: object | None) -> object:
        if mask is None:
            return values.mean()
        expanded = mask.expand_as(values)
        selected = values[expanded]
        if selected.numel() == 0:
            return values.mean()
        return selected.mean()

    def _charbonnier(residual: object) -> object:
        eps = float(photometric_charbonnier_epsilon)
        return torch.pow(residual * residual + eps * eps, float(photometric_charbonnier_alpha))

    def _ssim_distance(left: object, right: object) -> object:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_left = F.avg_pool2d(left, kernel_size=3, stride=1, padding=1)
        mu_right = F.avg_pool2d(right, kernel_size=3, stride=1, padding=1)
        sigma_left = F.avg_pool2d(left * left, kernel_size=3, stride=1, padding=1) - mu_left * mu_left
        sigma_right = F.avg_pool2d(right * right, kernel_size=3, stride=1, padding=1) - mu_right * mu_right
        sigma_lr = F.avg_pool2d(left * right, kernel_size=3, stride=1, padding=1) - mu_left * mu_right
        numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * sigma_lr + c2)
        denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (sigma_left + sigma_right + c2)
        ssim = numerator / torch.clamp(denominator, min=1e-6)
        return torch.clamp((1.0 - ssim) * 0.5, min=0.0, max=1.0)

    def _photometric_warp_loss(pred_flow: object, prev_image: object | None, next_image: object | None) -> object:
        if prev_image is None or next_image is None:
            raise ValueError("photometric_weight requires samples with prev_image and next_image.")
        if prev_image.shape[-2:] != pred_flow.shape[-2:]:
            target_size = pred_flow.shape[-2:]
            prev_image = F.interpolate(prev_image, size=target_size, mode="bilinear", align_corners=False)
            next_image = F.interpolate(next_image, size=target_size, mode="bilinear", align_corners=False)
        warped_next, valid_mask = _warp_next_to_prev(next_image, pred_flow)
        residual = warped_next - prev_image
        robust = torch.abs(residual) if photometric_loss == "l1" else _charbonnier(residual)

        if photometric_loss == "ssim":
            loss_map = _ssim_distance(warped_next, prev_image)
        elif photometric_loss == "charbonnier_ssim":
            weight = photometric_ssim_weight if photometric_ssim_weight > 0 else 0.15
            loss_map = (1.0 - weight) * robust + weight * _ssim_distance(warped_next, prev_image)
        elif photometric_ssim_weight > 0:
            weight = photometric_ssim_weight
            loss_map = (1.0 - weight) * robust + weight * _ssim_distance(warped_next, prev_image)
        else:
            loss_map = robust
        mask = valid_mask if photometric_use_valid_mask else None
        return _masked_mean(loss_map, mask)

    def _flow_smoothness_loss(pred_flow: object, image_batch: object | None) -> object:
        if smoothness_mode == "evflownet_8conn":
            horizontal = _charbonnier(pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]).mean()
            vertical = _charbonnier(pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]).mean()
            diag_down = _charbonnier(pred_flow[:, :, 1:, 1:] - pred_flow[:, :, :-1, :-1]).mean()
            diag_up = _charbonnier(pred_flow[:, :, :-1, 1:] - pred_flow[:, :, 1:, :-1]).mean()
            return (horizontal + vertical + diag_down + diag_up) / 4.0
        dx = torch.abs(pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1])
        dy = torch.abs(pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :])
        if smoothness_mode == "edge_aware":
            if image_batch is None:
                raise ValueError("edge_aware smoothness requires image-pair samples.")
            if image_batch.shape[-2:] != pred_flow.shape[-2:]:
                image_batch = F.interpolate(image_batch, size=pred_flow.shape[-2:], mode="bilinear", align_corners=False)
            image_dx = torch.mean(torch.abs(image_batch[:, :, :, 1:] - image_batch[:, :, :, :-1]), dim=1, keepdim=True)
            image_dy = torch.mean(torch.abs(image_batch[:, :, 1:, :] - image_batch[:, :, :-1, :]), dim=1, keepdim=True)
            dx = dx * torch.exp(-float(smoothness_edge_weight) * image_dx)
            dy = dy * torch.exp(-float(smoothness_edge_weight) * image_dy)
        dx = dx.mean()
        dy = dy.mean()
        return dx + dy

    def _multi_scale_photometric_loss(prediction: object, prev_image: object | None, next_image: object | None) -> object:
        losses = [
            _photometric_warp_loss(flow, prev_image, next_image)
            for flow in _flow_predictions(prediction)
        ]
        return torch.stack(losses).mean()

    def _multi_scale_smoothness_loss(prediction: object, image_batch: object | None) -> object:
        losses = [
            _flow_smoothness_loss(flow, image_batch)
            for flow in _flow_predictions(prediction)
        ]
        return torch.stack(losses).mean()

    def _make_batch(
        samples: list[FlowWindowSample],
        indices: object,
        *,
        phase: str,
        total: int,
    ) -> tuple[object, object, object | None, object | None]:
        reps: list[np.ndarray] = []
        prev_images: list[np.ndarray] = []
        next_images: list[np.ndarray] = []
        has_images = True
        for raw_idx in indices:
            idx = int(raw_idx)
            if phase == "train" and samples[idx] is first_rep_sample:
                rep = first_rep
            else:
                rep = adapter.build(samples[idx].events, samples[idx].sensor_size)
            reps.append(rep)
            if samples[idx].prev_image is None or samples[idx].next_image is None:
                has_images = False
            else:
                prev_images.append(_image_chw(samples[idx].prev_image))
                next_images.append(_image_chw(samples[idx].next_image))
            current = idx + 1
            if progress_every and (current == 1 or current == total or current % progress_every == 0):
                _progress(f"[{phase}] built representation {current}/{total}")
        x_np = np.stack(reps, axis=0)
        y_np = np.stack([np.moveaxis(samples[int(i)].gt_flow, -1, 0) for i in indices], axis=0)
        prev_batch = None
        next_batch = None
        if has_images:
            prev_batch = torch.from_numpy(np.stack(prev_images, axis=0)).float().to(device)
            next_batch = torch.from_numpy(np.stack(next_images, axis=0)).float().to(device)
        x_batch = torch.from_numpy(x_np).float().to(device)
        y_batch = torch.from_numpy(y_np).float().to(device)
        x_batch, y_batch, prev_batch, next_batch = _maybe_apply_paper_augmentation(
            x_batch,
            y_batch,
            prev_batch,
            next_batch,
            phase=phase,
        )
        return _maybe_apply_paper_crop(
            x_batch,
            y_batch,
            prev_batch,
            next_batch,
            phase=phase,
        )

    def _evaluate_samples(
        samples: list[FlowWindowSample],
        *,
        collect_window_metrics: bool,
        phase: str,
    ) -> tuple[list[FlowMetrics], list[dict[str, float | int | str]]]:
        metrics: list[FlowMetrics] = []
        window_metrics: list[dict[str, float | int | str]] = []
        eval_batch = int(eval_batch_size or batch_size)
        if eval_batch < 1:
            raise ValueError("eval_batch_size must be >= 1")

        model.eval()
        _progress(f"[{phase}] batches={(len(samples) + eval_batch - 1) // eval_batch}")
        with torch.no_grad():
            for start in range(0, len(samples), eval_batch):
                stop = min(start + eval_batch, len(samples))
                idx = list(range(start, stop))
                x_batch, _, _, _ = _make_batch(samples, idx, phase=phase, total=len(samples))
                pred_batch = _final_flow(model(x_batch)).detach().cpu().numpy()
                for offset, pred in enumerate(pred_batch):
                    eval_index = start + offset
                    sample = samples[eval_index]
                    pred_hw2 = np.moveaxis(pred, 0, -1)
                    metric = _compute_benchmark_metrics(pred_hw2, sample, metric_scope=metric_scope)
                    metrics.append(metric)
                    if collect_window_metrics:
                        window_metrics.append(
                            {
                                "sample_index": int(eval_index),
                                "eval_index": int(eval_index),
                                "aee": float(metric.aee),
                                "outlier_percent": float(metric.outlier_percent),
                                "valid_count": int(metric.valid_count),
                                "outlier_count": int(metric.outlier_count),
                                "sequence": _sequence_key(sample),
                                "source": _source_key(sample),
                            }
                        )
                batch_no = start // eval_batch + 1
                if progress_every and (batch_no == 1 or stop >= len(samples) or batch_no % progress_every == 0):
                    _progress(f"[{phase}] batch {batch_no}")
        return metrics, window_metrics

    def _valid_mask_for_export(sample: FlowWindowSample, target_hw: tuple[int, int]) -> np.ndarray:
        gt = sample.gt_flow
        if metric_scope == "full_gt_valid":
            cropped_gt = _center_crop_or_pad_np(gt, target_hw, fill_value=0)
            return np.isfinite(cropped_gt[..., 0]) & np.isfinite(cropped_gt[..., 1])
        valid = event_gt_valid_mask(sample.events, gt, sample.sensor_size)
        valid = _center_crop_or_pad_np(valid, target_hw, fill_value=False).astype(bool)
        if metric_scope == "evflownet_official" and "outdoor" in _sequence_key(sample):
            valid = valid.copy()
            valid[190:, :] = False
        return valid

    def _image_to_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.size == 0:
            return arr.astype(np.uint8)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.nan_to_num(arr, nan=0.0, posinf=float(finite.max()), neginf=float(finite.min()))
        max_value = float(arr.max())
        min_value = float(arr.min())
        if max_value > 1.5:
            arr = arr / 255.0
        elif max_value > min_value:
            arr = (arr - min_value) / max(max_value - min_value, 1e-6)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def _flow_to_rgb(flow_hw2: np.ndarray, *, clip_magnitude: float | None = None) -> np.ndarray:
        from matplotlib.colors import hsv_to_rgb

        flow = np.asarray(flow_hw2, dtype=np.float32)
        u = np.nan_to_num(flow[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(flow[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
        magnitude = np.sqrt(u * u + v * v)
        if clip_magnitude is None:
            finite_mag = magnitude[np.isfinite(magnitude)]
            clip_magnitude = float(np.percentile(finite_mag, 99.0)) if finite_mag.size else 1.0
            clip_magnitude = max(clip_magnitude, 1e-6)
        angle = np.arctan2(v, u)
        hue = (angle + math.pi) / (2.0 * math.pi)
        saturation = np.clip(magnitude / clip_magnitude, 0.0, 1.0)
        value = np.ones_like(saturation)
        rgb = hsv_to_rgb(np.stack([hue, saturation, value], axis=-1))
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _save_png(path: Path, image: np.ndarray, *, cmap: str | None = None, vmin: float | None = None, vmax: float | None = None) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        if image.ndim == 2:
            plt.imsave(path, image, cmap=cmap or "gray", vmin=vmin, vmax=vmax)
        else:
            plt.imsave(path, image)

    def _export_inference_artifacts() -> tuple[str | None, str | None]:
        if inference_dir is None or not inference_sample_indices:
            return None, None
        export_root = Path(inference_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        manifest_path = Path(inference_manifest_path) if inference_manifest_path is not None else export_root / "sample_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        model.eval()
        for eval_index in inference_sample_indices:
            if eval_index >= len(eval_samples):
                continue
            sample = eval_samples[eval_index]
            artifact_id = _sample_artifact_id(sample, eval_index=eval_index)
            sample_dir = export_root / artifact_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                x_batch, y_batch, prev_batch, next_batch = _make_batch([sample], [0], phase="inference", total=1)
                pred = _final_flow(model(x_batch)).detach().cpu().numpy()[0]
            pred_hw2 = np.moveaxis(pred, 0, -1).astype(np.float32, copy=False)
            gt_hw2 = np.moveaxis(y_batch.detach().cpu().numpy()[0], 0, -1).astype(np.float32, copy=False)
            target_hw = (int(pred_hw2.shape[0]), int(pred_hw2.shape[1]))
            valid_mask = _valid_mask_for_export(sample, target_hw)
            epe = np.linalg.norm(pred_hw2 - gt_hw2, axis=-1).astype(np.float32, copy=False)
            metric = compute_flow_metrics(
                pred_hw2,
                gt_hw2,
                valid_mask=valid_mask,
                outlier_mode="px" if metric_scope == "evflownet_official" else "kitti",
            )
            flow_clip = max(
                float(np.percentile(np.linalg.norm(gt_hw2, axis=-1), 99.0)),
                float(np.percentile(np.linalg.norm(pred_hw2, axis=-1), 99.0)),
                1e-6,
            )
            pred_png = sample_dir / "pred_flow.png"
            gt_png = sample_dir / "gt_flow.png"
            error_png = sample_dir / "error_map.png"
            _save_png(pred_png, _flow_to_rgb(pred_hw2, clip_magnitude=flow_clip))
            _save_png(gt_png, _flow_to_rgb(gt_hw2, clip_magnitude=flow_clip))
            error_vmax = max(float(np.percentile(epe[np.isfinite(epe)], 95.0)) if np.any(np.isfinite(epe)) else 1.0, 1e-6)
            _save_png(error_png, epe, cmap="magma", vmin=0.0, vmax=error_vmax)
            prev_png = ""
            next_png = ""
            if prev_batch is not None:
                prev_png_path = sample_dir / "prev_image.png"
                _save_png(prev_png_path, _image_to_uint8(prev_batch.detach().cpu().numpy()[0]))
                prev_png = str(prev_png_path)
            if next_batch is not None:
                next_png_path = sample_dir / "next_image.png"
                _save_png(next_png_path, _image_to_uint8(next_batch.detach().cpu().numpy()[0]))
                next_png = str(next_png_path)

            trace = _sample_trace_row(sample, eval_index=eval_index)
            trace.update(
                {
                    "artifact_id": artifact_id,
                    "method": adapter_name,
                    "target_height": target_hw[0],
                    "target_width": target_hw[1],
                    "metric_scope": metric_scope,
                    "aee": float(metric.aee),
                    "outlier_percent": float(metric.outlier_percent),
                    "valid_count": int(metric.valid_count),
                    "outlier_count": int(metric.outlier_count),
                    "pred_flow_npz": str(sample_dir / "pred_flow.npz"),
                    "pred_flow_png": str(pred_png),
                    "gt_flow_png": str(gt_png),
                    "error_map_png": str(error_png),
                    "prev_image_png": prev_png,
                    "next_image_png": next_png,
                }
            )
            (sample_dir / "sample_meta.json").write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
            np.savez_compressed(
                sample_dir / "pred_flow.npz",
                pred_flow=pred_hw2,
                gt_flow=gt_hw2,
                epe=epe,
                valid_mask=valid_mask.astype(np.uint8),
                sample_meta_json=np.asarray(json.dumps(trace, sort_keys=True)),
            )
            rows.append(trace)

        if rows:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        return str(export_root), str(manifest_path)

    def _mean_aee(metrics: list[FlowMetrics]) -> float:
        usable = [m for m in metrics if m.valid_count > 0 and math.isfinite(m.aee)]
        if not usable:
            raise ValueError("No valid metrics were available for validation.")
        return float(sum(m.aee for m in usable) / len(usable))

    def _materialize_lazy_stride_sample(sample: FlowWindowSample, rng: np.random.Generator) -> FlowWindowSample:
        if not sample.meta.get("lazy_random_stride"):
            return sample
        stride_min = int(sample.meta.get("lazy_stride_min", 1))
        stride_max = int(sample.meta.get("lazy_stride_max", stride_min))
        stride = int(rng.integers(stride_min, stride_max + 1))
        image_idx = int(sample.meta["image_end_index"])
        materialized = make_matrixlstm_image_pair_sample_from_arrays(
            sample.meta["lazy_events_arr"],  # type: ignore[arg-type]
            sample.meta["lazy_flow"],  # type: ignore[arg-type]
            sample.meta["lazy_flow_timestamps"],  # type: ignore[arg-type]
            sample.meta["lazy_image_timestamps"],  # type: ignore[arg-type]
            sample.meta["lazy_images"],  # type: ignore[arg-type]
            image_idx=image_idx,
            image_stride=stride,
            sensor_size=sample.sensor_size,
            image_scale=str(sample.meta.get("lazy_image_scale", "unit")),
            window_index=int(sample.meta.get("window_index", 0)),
            boundary_eps=float(sample.meta.get("lazy_boundary_eps", 1e-9)),
        )
        if materialized is None:
            return sample
        for key in ("source_h5", "source_flow", "source_image_h5", "sequence"):
            if key in sample.meta:
                materialized.meta[key] = sample.meta[key]
        return materialized

    num_train = len(effective_train_samples)

    def _epoch_train_samples(epoch_index: int) -> list[FlowWindowSample]:
        rng = np.random.default_rng(seed + epoch_index)
        if train_sample_groups is None:
            return [_materialize_lazy_stride_sample(sample, rng) for sample in effective_train_samples]
        return [
            _materialize_lazy_stride_sample(group[int(rng.integers(0, len(group)))], rng)
            for group in train_sample_groups
        ]

    best_val_aee: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, object] | None = None
    stale_epochs = 0
    epochs_completed = 0
    early_stopped = False

    for epoch in range(epochs):
        current_lr = _set_epoch_learning_rate(epoch)
        epoch_samples = _epoch_train_samples(epoch)
        model.train()
        perm = torch.randperm(num_train)
        _progress(
            f"[train] epoch {epoch + 1}/{epochs} batches={(num_train + batch_size - 1) // batch_size} "
            f"lr={current_lr:.6g}"
        )
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size].tolist()
            x_batch, y_batch, prev_batch, next_batch = _make_batch(
                epoch_samples,
                idx,
                phase="train",
                total=num_train,
            )
            prediction = model(x_batch)
            pred = _final_flow(prediction)
            if training_objective == "self_supervised":
                loss = pred.sum() * 0.0
            else:
                loss = float(supervised_weight) * F.smooth_l1_loss(pred, y_batch)
            if photometric_weight:
                loss = loss + float(photometric_weight) * _multi_scale_photometric_loss(prediction, prev_batch, next_batch)
            if smoothness_weight:
                loss = loss + float(smoothness_weight) * _multi_scale_smoothness_loss(prediction, prev_batch)
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_batches += 1
            batch_no = start // batch_size + 1
            if progress_every and (batch_no == 1 or start + batch_size >= num_train or batch_no % progress_every == 0):
                _progress(f"[train] epoch {epoch + 1}/{epochs} batch {batch_no}")
        epochs_completed = epoch + 1
        avg_loss = epoch_loss / max(epoch_batches, 1)
        _progress(f"[train] epoch {epoch + 1}/{epochs} mean_loss={avg_loss:.6f}")

        if early_stop_patience is not None:
            val_metrics, _ = _evaluate_samples(val_samples, collect_window_metrics=False, phase="val")
            val_aee = _mean_aee(val_metrics)
            improved = best_val_aee is None or val_aee < best_val_aee - early_stop_min_delta
            if improved:
                best_val_aee = val_aee
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
                _progress(f"[early-stop] epoch {epoch + 1}: val_aee={val_aee:.6f} best")
            else:
                stale_epochs += 1
                _progress(
                    f"[early-stop] epoch {epoch + 1}: val_aee={val_aee:.6f} "
                    f"best={best_val_aee:.6f} stale={stale_epochs}/{early_stop_patience}"
                )
                if stale_epochs >= early_stop_patience:
                    early_stopped = True
                    _progress(f"[early-stop] stopping at epoch {epoch + 1}; best_epoch={best_epoch}")
            if curve_path is not None:
                with curve_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "train_loss",
                            "val_aee",
                            "best_val_aee",
                            "is_best",
                            "stale_epochs",
                            "early_stopped",
                        ],
                    )
                    writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "val_aee": val_aee,
                            "best_val_aee": best_val_aee,
                            "is_best": improved,
                            "stale_epochs": stale_epochs,
                            "early_stopped": early_stopped,
                        }
                    )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "val/aee": val_aee,
                        "val/best_aee": best_val_aee,
                        "early_stop/stale_epochs": stale_epochs,
                        "early_stop/is_best": improved,
                        "train/lr": current_lr,
                    },
                    step=epoch + 1,
                )
            if early_stopped:
                break
        else:
            if curve_path is not None:
                with curve_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "train_loss",
                            "val_aee",
                            "best_val_aee",
                            "is_best",
                            "stale_epochs",
                            "early_stopped",
                        ],
                    )
                    writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "val_aee": "",
                            "best_val_aee": "",
                            "is_best": "",
                            "stale_epochs": "",
                            "early_stopped": "",
                        }
                    )
            if wandb_run is not None:
                wandb_run.log({"train/loss": avg_loss, "train/lr": current_lr}, step=epoch + 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, window_metrics = _evaluate_samples(
        eval_samples,
        collect_window_metrics=return_window_metrics,
        phase="eval",
    )

    usable_metrics = [
        metric
        for metric in metrics
        if metric.valid_count > 0 and math.isfinite(metric.aee) and math.isfinite(metric.outlier_percent)
    ]
    if not usable_metrics:
        raise ValueError("No valid metrics were available for final evaluation.")
    mean_aee = sum(m.aee for m in usable_metrics) / len(usable_metrics)
    mean_outlier = sum(m.outlier_percent for m in usable_metrics) / len(usable_metrics)
    valid_count = sum(m.valid_count for m in usable_metrics)
    per_sequence_metrics = _summarize_metrics_by_sequence(eval_samples, metrics)
    checkpoint_path_value: str | None = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "format_version": 1,
            "adapter_name": adapter_name,
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_aee": float(best_val_aee) if best_val_aee is not None else None,
            "epochs_completed": int(epochs_completed),
            "early_stopped": bool(early_stopped),
            "metric_scope": metric_scope,
            "final_eval": {
                "aee": float(mean_aee),
                "outlier_percent": float(mean_outlier),
                "valid_count": int(valid_count),
                "per_sequence_metrics": per_sequence_metrics,
            },
            "model_config": {
                "model_variant": model_variant,
                "base_channels": int(base_channels),
                "model_batch_norm": bool(model_batch_norm),
                "in_channels": int(channels),
            },
            "training_config": {
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "eval_batch_size": int(eval_batch_size or batch_size),
                "seed": int(seed),
                "training_objective": training_objective,
                "supervised_weight": float(supervised_weight),
                "photometric_weight": float(photometric_weight),
                "smoothness_weight": float(smoothness_weight),
                "photometric_loss": photometric_loss,
                "photometric_ssim_weight": float(photometric_ssim_weight),
                "photometric_charbonnier_epsilon": float(photometric_charbonnier_epsilon),
                "photometric_charbonnier_alpha": float(photometric_charbonnier_alpha),
                "photometric_use_valid_mask": bool(photometric_use_valid_mask),
                "smoothness_mode": smoothness_mode,
                "smoothness_edge_weight": float(smoothness_edge_weight),
                "lr_schedule": lr_schedule,
                "lr_decay": float(lr_decay),
                "warmup_epochs": int(warmup_epochs),
                "min_learning_rate": float(min_learning_rate),
                "weight_decay": float(weight_decay),
                "gradient_clip_norm": float(gradient_clip_norm) if gradient_clip_norm is not None else None,
                "paper_crop_size": int(paper_crop_size),
                "paper_train_random_crop": bool(paper_train_random_crop),
                "paper_random_flip": bool(paper_random_flip),
                "paper_random_rotation_degrees": float(paper_random_rotation_degrees),
                "random_train_stride_per_epoch": bool(random_train_stride_per_epoch),
                "early_stop_patience": int(early_stop_patience) if early_stop_patience is not None else None,
                "early_stop_min_delta": float(early_stop_min_delta),
                "early_stop_val_windows": int(len(val_samples)) if val_samples else 0,
                "early_stop_val_strategy": early_stop_val_strategy if val_samples else None,
            },
        }
        torch.save(checkpoint_payload, checkpoint_file)
        checkpoint_path_value = str(checkpoint_file)

    inference_dir_value, inference_manifest_path_value = _export_inference_artifacts()
    if wandb_run is not None:
        eval_log: dict[str, float | int | str] = {
            "eval/aee": float(mean_aee),
            "eval/outlier_percent": float(mean_outlier),
            "eval/valid_count": int(valid_count),
            "eval/metric_scope": metric_scope,
        }
        for sequence, values in per_sequence_metrics.items():
            eval_log[f"eval/{sequence}_aee"] = float(values["aee"])
            eval_log[f"eval/{sequence}_outlier_percent"] = float(values["outlier_percent"])
        wandb_run.log(eval_log, step=int(epochs_completed))
        wandb_run.finish()
    return BenchmarkResult(
        adapter_name=adapter_name,
        train_windows=len(effective_train_samples),
        eval_windows=len(eval_samples),
        channels=channels,
        aee=float(mean_aee),
        outlier_percent=float(mean_outlier),
        valid_count=int(valid_count),
        metric_scope=metric_scope,
        window_metrics=window_metrics if return_window_metrics else None,
        epochs_completed=int(epochs_completed),
        early_stopped=bool(early_stopped) if early_stop_patience is not None else None,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        best_val_aee=float(best_val_aee) if best_val_aee is not None else None,
        early_stop_val_windows=int(len(val_samples)) if val_samples else None,
        early_stop_val_strategy=early_stop_val_strategy if val_samples else None,
        early_stop_val_source_counts=val_source_counts if val_source_counts else None,
        curve_log_path=str(curve_path) if curve_path is not None else None,
        per_sequence_metrics=per_sequence_metrics,
        checkpoint_path=checkpoint_path_value,
        inference_dir=inference_dir_value,
        inference_manifest_path=inference_manifest_path_value,
    )
