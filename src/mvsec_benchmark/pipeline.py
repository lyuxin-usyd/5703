from __future__ import annotations

from dataclasses import dataclass

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


def run_linear_benchmark(
    samples: list[FlowWindowSample],
    *,
    adapter_name: str,
    train_windows: int = 4,
    ridge: float = 1e-3,
) -> BenchmarkResult:
    if adapter_name == "omnievent":
        raise ValueError("OmniEvent is reported-only in the current benchmark workflow.")
    if len(samples) < 2:
        raise ValueError("At least two windows are required for train/eval.")

    adapters = build_adapters()
    if adapter_name not in adapters:
        raise KeyError(f"Unknown adapter: {adapter_name}")
    adapter = adapters[adapter_name]

    split = min(max(train_windows, 1), len(samples) - 1)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    train_reps = [adapter.build(s.events, s.sensor_size) for s in train_samples]
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
