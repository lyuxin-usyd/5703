from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class SequencePair:
    name: str
    events_h5: Path
    flow_npz: Path


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
                return arr[:, :4]

    for _, obj in h5.items():
        if isinstance(obj, h5py.Dataset):
            arr = np.asarray(obj)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr[:, :4]
        elif isinstance(obj, h5py.Group):
            for _, sub in obj.items():
                if isinstance(sub, h5py.Dataset):
                    arr = np.asarray(sub)
                    if arr.ndim == 2 and arr.shape[1] >= 4:
                        return arr[:, :4]

    for group_key in ["events", "davis/left", "/davis/left"]:
        if group_key in h5:
            group = h5[group_key]
            if all(k in group for k in ["x", "y", "t", "p"]):
                return np.stack(
                    [
                        np.asarray(group["x"]),
                        np.asarray(group["y"]),
                        np.asarray(group["t"]),
                        np.asarray(group["p"]),
                    ],
                    axis=1,
                )
    raise KeyError("Could not find an MVSEC-style events dataset.")


def _load_events(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return _find_events_dataset(h5)


def _count_index_windows(num_events: int, flow_frames: int, window_size: int, stride: int) -> int:
    if num_events <= 0:
        return 0
    raw = len(range(0, max(num_events - window_size + 1, 1), stride))
    return min(raw, flow_frames)


def _count_events_before(events_t: np.ndarray, timestamp: float) -> int:
    return int(np.searchsorted(events_t, timestamp, side="right"))


def _timestamp_protocol_stats(events_t: np.ndarray, timestamps: np.ndarray) -> tuple[int, int, float]:
    if len(timestamps) == 0:
        return 0, 0, 0.0
    diffs = np.diff(timestamps)
    positive = diffs[diffs > 0]
    first_dt = float(np.median(positive)) if positive.size else 0.0
    used_indices: set[int] = set()
    windows = 0
    for idx, end_t in enumerate(timestamps):
        if idx == 0:
            start_t = max(float(events_t[0]), float(end_t) - first_dt) if first_dt > 0 else float(events_t[0])
        else:
            start_t = float(timestamps[idx - 1])
        eps = max(1e-6, abs(float(end_t)) * 1e-9)
        start = int(np.searchsorted(events_t, start_t + eps, side="right"))
        end = int(np.searchsorted(events_t, float(end_t) + eps, side="right"))
        if end <= start:
            continue
        windows += 1
        used_indices.update(range(start, end))
    used = len(used_indices)
    return windows, used, 100.0 * used / max(len(events_t), 1)


def _summarize_pair(pair: SequencePair, window_size: int, stride: int) -> None:
    print(f"\n=== {pair.name} ===")
    print(f"events: {pair.events_h5}")
    print(f"flow:   {pair.flow_npz}")

    if not pair.events_h5.exists():
        print("ERROR: events file missing")
        return
    if not pair.flow_npz.exists():
        print("ERROR: flow file missing")
        return

    events = _load_events(pair.events_h5)
    if events.size == 0:
        print("ERROR: empty events")
        return

    event_t = np.asarray(events[:, 2], dtype=np.float64)
    order_ok = bool(np.all(np.diff(event_t) >= 0))
    if not order_ok:
        event_t = np.sort(event_t)

    data = np.load(pair.flow_npz)
    keys = list(data.keys())
    timestamps = np.asarray(data["timestamps"], dtype=np.float64) if "timestamps" in data else None
    if "x_flow_dist" in data:
        flow_frames = int(data["x_flow_dist"].shape[0])
        flow_shape = tuple(data["x_flow_dist"].shape)
    elif "flow" in data:
        flow_frames = int(data["flow"].shape[0]) if data["flow"].ndim == 4 else 1
        flow_shape = tuple(data["flow"].shape)
    else:
        first_key = keys[0]
        arr = data[first_key]
        flow_frames = int(arr.shape[0]) if arr.ndim == 4 else 1
        flow_shape = tuple(arr.shape)

    index_windows = _count_index_windows(len(events), flow_frames, window_size, stride)
    index_event_end = min(index_windows * stride + max(window_size - stride, 0), len(events))
    index_event_fraction = 100.0 * index_event_end / max(len(events), 1)

    print(f"event rows: {len(events):,}")
    print(f"event time: {event_t[0]:.6f} -> {event_t[-1]:.6f}  sorted={order_ok}")
    print(f"npz keys: {keys}")
    print(f"flow shape: {flow_shape}, frames={flow_frames:,}")
    if timestamps is None:
        print("flow timestamps: MISSING")
    else:
        ts_sorted = bool(np.all(np.diff(timestamps) >= 0))
        in_event_range = int(((timestamps >= event_t[0]) & (timestamps <= event_t[-1])).sum())
        before_first = _count_events_before(event_t, float(timestamps[0]))
        before_last = _count_events_before(event_t, float(timestamps[-1]))
        print(f"flow timestamps: {len(timestamps):,}  sorted={ts_sorted}")
        print(f"flow time: {timestamps[0]:.6f} -> {timestamps[-1]:.6f}")
        print(f"timestamps inside event range: {in_event_range:,}/{len(timestamps):,}")
        print(f"events before first/last flow timestamp: {before_first:,} / {before_last:,}")
        ts_windows, ts_events, ts_fraction = _timestamp_protocol_stats(event_t, timestamps)
        print(f"timestamp protocol windows: {ts_windows:,}")
        print(f"events consumed by timestamp protocol: about {ts_events:,}/{len(events):,} ({ts_fraction:.2f}%)")
        if len(timestamps) != flow_frames:
            print("WARNING: timestamp count does not match flow frame count")
        if in_event_range < max(1, int(0.95 * len(timestamps))):
            print("WARNING: many flow timestamps are outside the event time range")

    print(f"current index protocol windows: {index_windows:,}")
    print(f"events consumed by current index protocol: about {index_event_end:,}/{len(events):,} ({index_event_fraction:.2f}%)")
    if flow_frames <= 50:
        print("WARNING: very few flow frames. This sequence is probably incomplete or generated with a small max-frames setting.")
    if index_event_fraction < 50.0:
        print("WARNING: current index protocol uses less than half of the extracted events.")


def _default_pairs(data_root: Path) -> list[SequencePair]:
    return [
        SequencePair(
            "outdoor_day1",
            data_root / "outdoor_day" / "outdoor_day1_left_events_6m.h5",
            data_root / "outdoor_day" / "outdoor_day1_gt_flow_full.npz",
        ),
        SequencePair(
            "outdoor_day2",
            data_root / "outdoor_day" / "outdoor_day2_left_events_6m.h5",
            data_root / "outdoor_day" / "outdoor_day2_gt_flow_full.npz",
        ),
        SequencePair(
            "indoor_flying1",
            data_root / "indoor_flying1" / "indoor_flying1_left_events_6m.h5",
            data_root / "indoor_flying" / "indoor_flying1_gt_flow_full.npz",
        ),
        SequencePair(
            "indoor_flying2",
            data_root / "indoor_flying" / "indoor_flying2_left_events_6m.h5",
            data_root / "indoor_flying" / "indoor_flying2_gt_flow_full.npz",
        ),
        SequencePair(
            "indoor_flying3",
            data_root / "indoor_flying" / "indoor_flying3_left_events_6m.h5",
            data_root / "indoor_flying" / "indoor_flying3_gt_flow_full.npz",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MVSEC event/flow timestamp alignment.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else None,
        help="Processed MVSEC folder. Can also be provided via DATA_ROOT.",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    args = parser.parse_args()

    if args.data_root is None:
        raise SystemExit("Set --data-root or DATA_ROOT to the processed MVSEC folder.")

    print(f"data_root={args.data_root}")
    print(f"index protocol window_size={args.window_size}, stride={args.stride}")
    for pair in _default_pairs(args.data_root):
        _summarize_pair(pair, args.window_size, args.stride)


if __name__ == "__main__":
    main()
