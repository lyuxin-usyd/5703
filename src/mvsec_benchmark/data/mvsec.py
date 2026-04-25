from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class FlowWindowSample:
    events: np.ndarray
    gt_flow: np.ndarray
    sensor_size: tuple[int, int]
    meta: dict[str, int | float | str] = field(default_factory=dict)


def _find_events_dataset(h5: h5py.File) -> np.ndarray:
    candidate_paths = [
        "/events",
        "events",
        "/davis/left/events",
        "davis/left/events",
    ]
    for key in candidate_paths:
        if key in h5:
            arr = np.asarray(h5[key])
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr[:, :4].astype(np.float32, copy=False)

    def _visit(name: str, obj: h5py.Dataset) -> np.ndarray | None:
        if not isinstance(obj, h5py.Dataset):
            return None
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape[1] >= 4:
            return arr[:, :4].astype(np.float32, copy=False)
        return None

    found: np.ndarray | None = None
    for _, obj in h5.items():
        if isinstance(obj, h5py.Dataset):
            found = _visit("", obj)
            if found is not None:
                return found
        else:
            for _, sub in obj.items():
                if isinstance(sub, h5py.Dataset):
                    found = _visit("", sub)
                    if found is not None:
                        return found

    # Some files store events as separate x/y/t/p arrays inside a group.
    for group_key in ["events", "davis/left", "/davis/left"]:
        if group_key in h5:
            group = h5[group_key]
            if all(k in group for k in ["x", "y", "t", "p"]):
                return np.stack(
                    [
                        np.asarray(group["x"], dtype=np.float32),
                        np.asarray(group["y"], dtype=np.float32),
                        np.asarray(group["t"], dtype=np.float32),
                        np.asarray(group["p"], dtype=np.float32),
                    ],
                    axis=1,
                )
    raise KeyError("Could not find an MVSEC-style events dataset in the HDF5 file.")


def load_mvsec_events(h5_path: str | Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        return _find_events_dataset(h5)


def load_mvsec_flow(flow_path: str | Path) -> np.ndarray:
    data = np.load(flow_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "x_flow_dist" in data and "y_flow_dist" in data:
            arr = np.stack(
                [
                    np.asarray(data["x_flow_dist"], dtype=np.float32),
                    np.asarray(data["y_flow_dist"], dtype=np.float32),
                ],
                axis=-1,
            )
        elif "flow" in data:
            arr = data["flow"]
        else:
            first_key = list(data.keys())[0]
            arr = data[first_key]
    else:
        arr = np.asarray(data)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 2:
        return np.moveaxis(arr, 0, -1)
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 2:
        return np.moveaxis(arr, 1, -1)
    raise ValueError(f"Unsupported flow array shape: {arr.shape}")


def infer_sensor_size(events: np.ndarray) -> tuple[int, int]:
    if events.size == 0:
        raise ValueError("Cannot infer sensor size from empty events.")
    width = int(events[:, 0].max()) + 1
    height = int(events[:, 1].max()) + 1
    return height, width


def iter_event_windows(
    events: np.ndarray,
    gt_flow: np.ndarray,
    sensor_size: tuple[int, int] | None = None,
    window_size: int = 200,
    stride: int | None = None,
    max_windows: int | None = None,
) -> Iterator[FlowWindowSample]:
    stride = window_size if stride is None else stride
    sensor_size = infer_sensor_size(events) if sensor_size is None else sensor_size
    total = len(events)
    if total == 0:
        return

    flow_limit = gt_flow.shape[0] if gt_flow.ndim == 4 else None
    n_yielded = 0
    for start in range(0, max(total - window_size + 1, 1), stride):
        end = min(start + window_size, total)
        if end <= start:
            continue

        if gt_flow.ndim == 4:
            if flow_limit is not None and n_yielded >= flow_limit:
                break
            flow_idx = n_yielded
            flow = gt_flow[flow_idx]
        else:
            flow = gt_flow

        yield FlowWindowSample(
            events=events[start:end].astype(np.float32, copy=False),
            gt_flow=np.asarray(flow, dtype=np.float32),
            sensor_size=sensor_size,
            meta={"window_index": n_yielded, "event_start": start, "event_end": end},
        )
        n_yielded += 1
        if max_windows is not None and n_yielded >= max_windows:
            break


def load_mvsec_windows(
    h5_path: str | Path,
    flow_path: str | Path,
    sensor_size: tuple[int, int] | None = None,
    window_size: int = 200,
    stride: int | None = None,
    max_windows: int | None = None,
) -> list[FlowWindowSample]:
    events = load_mvsec_events(h5_path)
    gt_flow = load_mvsec_flow(flow_path)
    if sensor_size is None:
        sensor_size = gt_flow.shape[:2] if gt_flow.ndim == 3 else gt_flow.shape[1:3]
    return list(
        iter_event_windows(
            events=events,
            gt_flow=gt_flow,
            sensor_size=sensor_size,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
        )
    )
