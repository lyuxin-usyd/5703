from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .data.mvsec import FlowWindowSample, infer_sensor_size, load_mvsec_events, load_mvsec_flow_data
from .utils.flow_metrics import FlowMetrics, compute_flow_metrics, event_gt_valid_mask


def _as_flow_sequence(flow: np.ndarray) -> np.ndarray:
    arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 2:
        return np.moveaxis(arr, 1, -1)
    raise ValueError(f"Expected flow shape (N,H,W,2), got {arr.shape}")


def _median_positive_dt(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError("Flow timestamps must contain at least one positive step.")
    return float(np.median(positive))


def _sample_flow_nearest(flow: np.ndarray, x_pos: np.ndarray, y_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    x_idx = np.rint(x_pos).astype(np.int64)
    y_idx = np.rint(y_pos).astype(np.int64)
    in_bounds = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    sampled = np.zeros((*x_pos.shape, 2), dtype=np.float32)
    if np.any(in_bounds):
        sampled[in_bounds] = flow[y_idx[in_bounds], x_idx[in_bounds]]
    return sampled, in_bounds


def estimate_corresponding_gt_flow(
    flow_sequence: np.ndarray,
    flow_timestamps: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Propagate dense GT flow across an arbitrary time interval.

    This ports the important part of the MatrixLSTM/EV-FlowNet evaluation
    protocol: the requested image/event interval does not need to land exactly
    on a stored MVSEC GT-flow timestamp.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time.")

    flow = _as_flow_sequence(flow_sequence)
    timestamps = np.asarray(flow_timestamps, dtype=np.float64)
    if len(timestamps) != flow.shape[0]:
        raise ValueError(
            f"Flow timestamp count {len(timestamps)} does not match flow frame count {flow.shape[0]}."
        )
    if len(timestamps) < 2:
        raise ValueError("At least two flow timestamps are required for propagation.")

    median_dt = _median_positive_dt(timestamps)
    height, width = flow.shape[1:3]
    x0, y0 = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x_pos = x0.copy()
    y_pos = y0.copy()
    valid = np.ones((height, width), dtype=bool)

    first_idx = max(0, int(np.searchsorted(timestamps, start_time, side="right") - 1))
    last_idx = min(flow.shape[0] - 1, int(np.searchsorted(timestamps, end_time, side="left")))
    for flow_idx in range(first_idx, last_idx + 1):
        segment_end = float(timestamps[flow_idx])
        segment_start = float(timestamps[flow_idx - 1]) if flow_idx > 0 else segment_end - median_dt
        segment_dt = segment_end - segment_start
        if segment_dt <= 0:
            continue
        overlap_start = max(float(start_time), segment_start)
        overlap_end = min(float(end_time), segment_end)
        if overlap_end <= overlap_start:
            continue

        sampled, in_bounds = _sample_flow_nearest(flow[flow_idx], x_pos, y_pos)
        valid &= in_bounds
        scale = float((overlap_end - overlap_start) / segment_dt)
        x_pos = x_pos + sampled[..., 0] * scale
        y_pos = y_pos + sampled[..., 1] * scale

    propagated = np.stack([x_pos - x0, y_pos - y0], axis=-1).astype(np.float32, copy=False)
    propagated[~valid] = 0.0
    return propagated


def compute_matrixlstm_paperlike_metrics(
    pred_flow: np.ndarray,
    sample: FlowWindowSample,
) -> FlowMetrics:
    valid_mask = event_gt_valid_mask(sample.events, sample.gt_flow, sample.sensor_size)
    return compute_flow_metrics(
        pred_flow,
        sample.gt_flow,
        valid_mask=valid_mask,
        outlier_mode="kitti",
    )


def iter_matrixlstm_paperlike_windows(
    events: np.ndarray,
    gt_flow: np.ndarray,
    flow_timestamps: np.ndarray,
    *,
    sensor_size: tuple[int, int] | None = None,
    max_windows: int | None = None,
) -> Iterator[FlowWindowSample]:
    """Yield MatrixLSTM/EV-FlowNet-style timestamp windows.

    Each sample uses events inside one GT-flow timestamp interval and propagates
    the dense GT flow over exactly that interval.
    """
    flow = _as_flow_sequence(gt_flow)
    timestamps = np.asarray(flow_timestamps, dtype=np.float64)
    if len(timestamps) != flow.shape[0]:
        raise ValueError(
            f"Flow timestamp count {len(timestamps)} does not match flow frame count {flow.shape[0]}."
        )
    if len(timestamps) < 2:
        return

    sensor_size = infer_sensor_size(events) if sensor_size is None else sensor_size
    events_arr = np.asarray(events, dtype=np.float64)
    event_t = events_arr[:, 2]
    if np.any(np.diff(event_t) < 0):
        order = np.argsort(event_t, kind="stable")
        events_arr = events_arr[order]
        event_t = event_t[order]

    median_dt = _median_positive_dt(timestamps)
    boundary_eps = max(1e-9, median_dt * 1e-9)
    n_yielded = 0
    for flow_idx, end_t_raw in enumerate(timestamps):
        end_t = float(end_t_raw)
        start_t = float(timestamps[flow_idx - 1]) if flow_idx > 0 else end_t - median_dt
        start = int(np.searchsorted(event_t, start_t + boundary_eps, side="right"))
        end = int(np.searchsorted(event_t, end_t + boundary_eps, side="right"))
        if end <= start:
            continue

        gt = estimate_corresponding_gt_flow(flow, timestamps, start_t, end_t)
        yield FlowWindowSample(
            events=events_arr[start:end].astype(np.float64, copy=False),
            gt_flow=gt,
            sensor_size=sensor_size,
            meta={
                "alignment": "matrixlstm_paperlike",
                "window_index": n_yielded,
                "flow_index": int(flow_idx),
                "event_start": start,
                "event_end": end,
                "event_start_time": float(start_t),
                "event_end_time": float(end_t),
                "flow_timestamp": float(end_t),
            },
        )
        n_yielded += 1
        if max_windows is not None and n_yielded >= max_windows:
            break


def load_matrixlstm_paperlike_windows(
    h5_path: str | Path,
    flow_path: str | Path,
    *,
    sensor_size: tuple[int, int] | None = None,
    max_windows: int | None = None,
) -> list[FlowWindowSample]:
    events = load_mvsec_events(h5_path)
    flow_data = load_mvsec_flow_data(flow_path)
    if flow_data.timestamps is None:
        raise ValueError("MatrixLSTM paper-like windows require flow timestamps.")
    gt_flow = flow_data.flow
    if sensor_size is None:
        sensor_size = gt_flow.shape[:2] if gt_flow.ndim == 3 else gt_flow.shape[1:3]
    return list(
        iter_matrixlstm_paperlike_windows(
            events,
            gt_flow,
            flow_data.timestamps,
            sensor_size=sensor_size,
            max_windows=max_windows,
        )
    )
