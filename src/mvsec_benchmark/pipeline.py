from __future__ import annotations

from dataclasses import dataclass
import copy

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
    epochs_completed: int | None = None
    early_stopped: bool | None = None
    best_epoch: int | None = None
    best_val_aee: float | None = None
    early_stop_val_windows: int | None = None


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
    progress_every: int = 100,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    early_stop_val_windows: int = 0,
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

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("run_torch_train_eval_benchmark requires torch to be installed") from exc

    from .models.evflownet_like import EVFlowNetLike

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

    val_samples: list[FlowWindowSample] = []
    effective_train_samples = train_samples
    if early_stop_val_windows:
        val_samples = train_samples[-early_stop_val_windows:]
        effective_train_samples = train_samples[:-early_stop_val_windows]

    _progress(
        f"[setup] adapter={adapter_name} train_windows={len(effective_train_samples)} "
        f"val_windows={len(val_samples)} eval_windows={len(eval_samples)}"
    )
    _progress("[setup] building first representation")
    first_rep = adapter.build(effective_train_samples[0].events, effective_train_samples[0].sensor_size)
    channels = int(first_rep.shape[0])
    model = EVFlowNetLike(in_channels=channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _make_batch(samples: list[FlowWindowSample], indices: object, *, phase: str, total: int) -> tuple[object, object]:
        reps: list[np.ndarray] = []
        for raw_idx in indices:
            idx = int(raw_idx)
            if phase == "train" and idx == 0:
                rep = first_rep
            else:
                rep = adapter.build(samples[idx].events, samples[idx].sensor_size)
            reps.append(rep)
            current = idx + 1
            if progress_every and (current == 1 or current == total or current % progress_every == 0):
                _progress(f"[{phase}] built representation {current}/{total}")
        x_np = np.stack(reps, axis=0)
        y_np = np.stack([np.moveaxis(samples[int(i)].gt_flow, -1, 0) for i in indices], axis=0)
        return torch.from_numpy(x_np).float().to(device), torch.from_numpy(y_np).float().to(device)

    def _evaluate_samples(
        samples: list[FlowWindowSample],
        *,
        collect_window_metrics: bool,
        phase: str,
    ) -> tuple[list[FlowMetrics], list[dict[str, float | int]]]:
        metrics: list[FlowMetrics] = []
        window_metrics: list[dict[str, float | int]] = []
        eval_batch = int(eval_batch_size or batch_size)
        if eval_batch < 1:
            raise ValueError("eval_batch_size must be >= 1")

        model.eval()
        _progress(f"[{phase}] batches={(len(samples) + eval_batch - 1) // eval_batch}")
        with torch.no_grad():
            for start in range(0, len(samples), eval_batch):
                stop = min(start + eval_batch, len(samples))
                idx = list(range(start, stop))
                x_batch, _ = _make_batch(samples, idx, phase=phase, total=len(samples))
                pred_batch = model(x_batch).detach().cpu().numpy()
                for offset, pred in enumerate(pred_batch):
                    eval_index = start + offset
                    sample = samples[eval_index]
                    pred_hw2 = np.moveaxis(pred, 0, -1)
                    metric = compute_flow_metrics(pred_hw2, sample.gt_flow)
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
                            }
                        )
                batch_no = start // eval_batch + 1
                if progress_every and (batch_no == 1 or stop >= len(samples) or batch_no % progress_every == 0):
                    _progress(f"[{phase}] batch {batch_no}")
        return metrics, window_metrics

    def _mean_aee(metrics: list[FlowMetrics]) -> float:
        return float(sum(m.aee for m in metrics) / len(metrics))

    num_train = len(effective_train_samples)
    best_val_aee: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, object] | None = None
    stale_epochs = 0
    epochs_completed = 0
    early_stopped = False

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train)
        _progress(f"[train] epoch {epoch + 1}/{epochs} batches={(num_train + batch_size - 1) // batch_size}")
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size].tolist()
            x_batch, y_batch = _make_batch(effective_train_samples, idx, phase="train", total=num_train)
            pred = model(x_batch)
            loss = F.smooth_l1_loss(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
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
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, window_metrics = _evaluate_samples(
        eval_samples,
        collect_window_metrics=return_window_metrics,
        phase="eval",
    )

    mean_aee = sum(m.aee for m in metrics) / len(metrics)
    mean_outlier = sum(m.outlier_percent for m in metrics) / len(metrics)
    valid_count = sum(m.valid_count for m in metrics)
    return BenchmarkResult(
        adapter_name=adapter_name,
        train_windows=len(effective_train_samples),
        eval_windows=len(eval_samples),
        channels=channels,
        aee=float(mean_aee),
        outlier_percent=float(mean_outlier),
        valid_count=int(valid_count),
        window_metrics=window_metrics if return_window_metrics else None,
        epochs_completed=int(epochs_completed),
        early_stopped=bool(early_stopped) if early_stop_patience is not None else None,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        best_val_aee=float(best_val_aee) if best_val_aee is not None else None,
        early_stop_val_windows=int(len(val_samples)) if val_samples else None,
    )
