from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import build_adapters
from .data.mvsec import FlowWindowSample
from .models.linear_flow import LinearFlowRegressor
from .utils.flow_metrics import FlowMetrics, compute_flow_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    adapter_name: str
    train_windows: int
    eval_windows: int
    channels: int
    aee: float
    outlier_percent: float
    valid_count: int
    window_metrics: list[dict[str, float | int]] | None = None


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
        metrics.append(compute_flow_metrics(pred, sample.gt_flow))

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
                metric = compute_flow_metrics(pred_hw2, sample.gt_flow)
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
    batch_size: int = 2,
    eval_batch_size: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    return_window_metrics: bool = False,
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

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("run_torch_train_eval_benchmark requires torch to be installed") from exc

    from .models.evflownet_like import EVFlowNetLike

    torch.manual_seed(seed)
    adapter, train_reps = _build_adapter_representations(train_samples, adapter_name=adapter_name)
    eval_reps = [adapter.build(s.events, s.sensor_size) for s in eval_samples]

    channels = int(train_reps[0].shape[0])
    model = EVFlowNetLike(in_channels=channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _make_batch(reps: list[np.ndarray], samples: list[FlowWindowSample], indices: object) -> tuple[object, object]:
        x_np = np.stack([reps[int(i)] for i in indices], axis=0)
        y_np = np.stack([np.moveaxis(samples[int(i)].gt_flow, -1, 0) for i in indices], axis=0)
        return torch.from_numpy(x_np).float().to(device), torch.from_numpy(y_np).float().to(device)

    num_train = len(train_samples)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(num_train)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size].tolist()
            x_batch, y_batch = _make_batch(train_reps, train_samples, idx)
            pred = model(x_batch)
            loss = F.smooth_l1_loss(pred, y_batch)
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
        for start in range(0, len(eval_samples), eval_batch):
            stop = min(start + eval_batch, len(eval_samples))
            x_np = np.stack(eval_reps[start:stop], axis=0)
            x_batch = torch.from_numpy(x_np).float().to(device)
            pred_batch = model(x_batch).detach().cpu().numpy()
            for offset, pred in enumerate(pred_batch):
                eval_index = start + offset
                sample = eval_samples[eval_index]
                pred_hw2 = np.moveaxis(pred, 0, -1)
                metric = compute_flow_metrics(pred_hw2, sample.gt_flow)
                metrics.append(metric)
                if return_window_metrics:
                    window_metrics.append(
                        {
                            "sample_index": int(eval_index),
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
        channels=channels,
        aee=float(mean_aee),
        outlier_percent=float(mean_outlier),
        valid_count=int(valid_count),
        window_metrics=window_metrics if return_window_metrics else None,
    )
