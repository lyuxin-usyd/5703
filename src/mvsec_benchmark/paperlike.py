from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
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


def _load_image_sequence(image_h5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(image_h5_path, "r") as h5:
        if "timestamps" not in h5 or "images" not in h5:
            raise KeyError("Image HDF5 must contain 'timestamps' and 'images' datasets.")
        timestamps = np.asarray(h5["timestamps"], dtype=np.float64)
        images = np.asarray(h5["images"])
    if images.ndim not in {3, 4}:
        raise ValueError(f"Expected image dataset shape (N,H,W) or (N,H,W,C), got {images.shape}.")
    if len(timestamps) != images.shape[0]:
        raise ValueError(
            f"Image timestamp count {len(timestamps)} does not match image count {images.shape[0]}."
        )
    order = np.argsort(timestamps, kind="stable")
    if np.any(order != np.arange(len(timestamps))):
        timestamps = timestamps[order]
        images = images[order]
    return timestamps, images


def _as_float_image(image: np.ndarray, *, image_scale: str = "unit") -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] == 3:
            arr = arr.mean(axis=-1)
        else:
            raise ValueError(f"Unsupported image channel shape: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {arr.shape}")
    image_scale = image_scale.lower()
    if image_scale not in {"unit", "raw255"}:
        raise ValueError("image_scale must be 'unit' or 'raw255'.")
    if image_scale == "unit" and arr.size and float(arr.max()) > 1.5:
        arr = arr / 255.0
    return arr.astype(np.float32, copy=False)


def make_matrixlstm_image_pair_sample_from_arrays(
    events_arr: np.ndarray,
    flow: np.ndarray,
    flow_timestamps: np.ndarray,
    image_timestamps: np.ndarray,
    images: np.ndarray,
    *,
    image_idx: int,
    image_stride: int,
    sensor_size: tuple[int, int],
    image_scale: str = "unit",
    window_index: int = 0,
    boundary_eps: float | None = None,
) -> FlowWindowSample | None:
    if image_stride < 1:
        raise ValueError("image_stride must be >= 1.")
    if image_idx < image_stride or image_idx >= len(image_timestamps):
        return None

    event_t = events_arr[:, 2]
    start_t = float(image_timestamps[image_idx - image_stride])
    end_t = float(image_timestamps[image_idx])
    if end_t <= start_t:
        return None
    if boundary_eps is None:
        boundary_eps = max(1e-9, _median_positive_dt(flow_timestamps) * 1e-9)

    start = int(np.searchsorted(event_t, start_t + boundary_eps, side="right"))
    end = int(np.searchsorted(event_t, end_t + boundary_eps, side="right"))
    if end <= start:
        return None

    gt = estimate_corresponding_gt_flow(flow, flow_timestamps, start_t, end_t)
    window_events = events_arr[start:end].astype(np.float64, copy=False)
    if not np.any(event_gt_valid_mask(window_events, gt, sensor_size)):
        return None

    return FlowWindowSample(
        events=window_events,
        gt_flow=gt,
        sensor_size=sensor_size,
        meta={
            "alignment": "matrixlstm_image_pair",
            "window_index": int(window_index),
            "image_start_index": int(image_idx - image_stride),
            "image_end_index": int(image_idx),
            "image_stride": int(image_stride),
            "event_start": start,
            "event_end": end,
            "event_start_time": float(start_t),
            "event_end_time": float(end_t),
        },
        prev_image=_as_float_image(images[image_idx - image_stride], image_scale=image_scale),
        next_image=_as_float_image(images[image_idx], image_scale=image_scale),
    )


def iter_matrixlstm_image_pair_windows(
    events: np.ndarray,
    gt_flow: np.ndarray,
    flow_timestamps: np.ndarray,
    image_timestamps: np.ndarray,
    images: np.ndarray,
    *,
    sensor_size: tuple[int, int] | None = None,
    max_windows: int | None = None,
    image_stride: int = 1,
    image_scale: str = "unit",
) -> Iterator[FlowWindowSample]:
    """Yield image-pair windows for the MatrixLSTM/EV-FlowNet loss probe.

    The original MatrixLSTM training path associates events with pairs of
    grayscale frames. This helper keeps that image-pair structure while still
    using the existing dense GT-flow propagation for supervised comparison.
    """
    if image_stride < 1:
        raise ValueError("image_stride must be >= 1.")
    flow = _as_flow_sequence(gt_flow)
    timestamps = np.asarray(flow_timestamps, dtype=np.float64)
    image_t = np.asarray(image_timestamps, dtype=np.float64)
    if len(image_t) != images.shape[0]:
        raise ValueError("Image timestamps must match image count.")
    if len(image_t) <= image_stride:
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
    for image_idx in range(image_stride, len(image_t)):
        sample = make_matrixlstm_image_pair_sample_from_arrays(
            events_arr,
            flow,
            timestamps,
            image_t,
            images,
            image_idx=image_idx,
            image_stride=image_stride,
            sensor_size=sensor_size,
            image_scale=image_scale,
            window_index=n_yielded,
            boundary_eps=boundary_eps,
        )
        if sample is None:
            continue
        yield sample
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


def load_matrixlstm_image_pair_windows(
    h5_path: str | Path,
    flow_path: str | Path,
    image_h5_path: str | Path,
    *,
    sensor_size: tuple[int, int] | None = None,
    max_windows: int | None = None,
    image_stride: int = 1,
    image_scale: str = "unit",
) -> list[FlowWindowSample]:
    events = load_mvsec_events(h5_path)
    flow_data = load_mvsec_flow_data(flow_path)
    if flow_data.timestamps is None:
        raise ValueError("Image-pair paper-like windows require flow timestamps.")
    image_timestamps, images = _load_image_sequence(image_h5_path)
    gt_flow = flow_data.flow
    if sensor_size is None:
        sensor_size = gt_flow.shape[:2] if gt_flow.ndim == 3 else gt_flow.shape[1:3]
    return list(
        iter_matrixlstm_image_pair_windows(
            events,
            gt_flow,
            flow_data.timestamps,
            image_timestamps,
            images,
            sensor_size=sensor_size,
            max_windows=max_windows,
            image_stride=image_stride,
            image_scale=image_scale,
        )
    )


def load_matrixlstm_lazy_random_image_pair_windows(
    h5_path: str | Path,
    flow_path: str | Path,
    image_h5_path: str | Path,
    *,
    sensor_size: tuple[int, int] | None = None,
    max_windows: int | None = None,
    image_stride_min: int = 1,
    image_stride_max: int = 5,
    image_scale: str = "unit",
) -> list[FlowWindowSample]:
    if image_stride_min < 1 or image_stride_max < image_stride_min:
        raise ValueError("image_stride_min/max must define a positive inclusive range.")
    events = load_mvsec_events(h5_path)
    flow_data = load_mvsec_flow_data(flow_path)
    if flow_data.timestamps is None:
        raise ValueError("Image-pair paper-like windows require flow timestamps.")
    image_timestamps, images = _load_image_sequence(image_h5_path)
    flow = _as_flow_sequence(flow_data.flow)
    timestamps = np.asarray(flow_data.timestamps, dtype=np.float64)
    if sensor_size is None:
        sensor_size = flow.shape[:2] if flow.ndim == 3 else flow.shape[1:3]

    events_arr = np.asarray(events, dtype=np.float64)
    event_t = events_arr[:, 2]
    if np.any(np.diff(event_t) < 0):
        order = np.argsort(event_t, kind="stable")
        events_arr = events_arr[order]
        event_t = event_t[order]

    boundary_eps = max(1e-9, _median_positive_dt(timestamps) * 1e-9)
    samples: list[FlowWindowSample] = []
    for image_idx in range(image_stride_max, len(image_timestamps)):
        sample = make_matrixlstm_image_pair_sample_from_arrays(
            events_arr,
            flow,
            timestamps,
            image_timestamps,
            images,
            image_idx=image_idx,
            image_stride=1,
            sensor_size=sensor_size,
            image_scale=image_scale,
            window_index=len(samples),
            boundary_eps=boundary_eps,
        )
        if sample is None:
            continue
        sample.meta.update(
            {
                "lazy_random_stride": 1,
                "lazy_stride_min": int(image_stride_min),
                "lazy_stride_max": int(image_stride_max),
                "lazy_events_arr": events_arr,
                "lazy_flow": flow,
                "lazy_flow_timestamps": timestamps,
                "lazy_image_timestamps": image_timestamps,
                "lazy_images": images,
                "lazy_image_scale": image_scale,
                "lazy_boundary_eps": float(boundary_eps),
            }
        )
        samples.append(sample)
        if max_windows is not None and len(samples) >= max_windows:
            break
    return samples
