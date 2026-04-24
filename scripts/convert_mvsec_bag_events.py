from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


EVENT_ARRAY_MSG = """\
std_msgs/Header header
uint32 height
uint32 width
dvs_msgs/Event[] events
"""

EVENT_MSG = """\
uint16 x
uint16 y
time ts
bool polarity
"""

HEADER_MSG = """\
uint32 seq
time stamp
string frame_id
"""


def _time_to_seconds(value: object) -> float:
    sec = getattr(value, "sec", getattr(value, "secs", 0))
    nsec = getattr(value, "nanosec", getattr(value, "nsec", 0))
    return float(sec) + float(nsec) * 1e-9


def _register_bag_types(connections: list[object]) -> object:
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    typestore = get_typestore(Stores.EMPTY)
    types = {}
    for conn in connections:
        if conn.msgtype == "dvs_msgs/msg/EventArray":
            types.update(get_types_from_msg(HEADER_MSG, "std_msgs/msg/Header"))
            types.update(get_types_from_msg(EVENT_MSG, "dvs_msgs/msg/Event"))
            types.update(get_types_from_msg(EVENT_ARRAY_MSG, "dvs_msgs/msg/EventArray"))
        elif conn.msgdef:
            types.update(get_types_from_msg(conn.msgdef, conn.msgtype))
    typestore.register(types)
    return typestore


def _select_event_connections(reader: object, topic: str | None) -> list[object]:
    if topic:
        return [conn for conn in reader.connections if conn.topic == topic]

    candidates = [
        conn
        for conn in reader.connections
        if "EventArray" in conn.msgtype or "events" in conn.topic.lower()
    ]
    if not candidates:
        raise ValueError("Could not find an event topic. Run scripts/inspect_rosbag.py first and pass --topic.")
    return candidates


def _event_batch_to_array(message: object) -> np.ndarray:
    events = getattr(message, "events", None)
    if events is None:
        raise ValueError("Selected topic does not contain an 'events' field.")

    arr = np.empty((len(events), 4), dtype=np.float32)
    for idx, event in enumerate(events):
        arr[idx, 0] = float(event.x)
        arr[idx, 1] = float(event.y)
        arr[idx, 2] = _time_to_seconds(event.ts)
        arr[idx, 3] = 1.0 if bool(event.polarity) else 0.0
    return arr


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if values.size == 0:
        return
    start = int(dataset.shape[0])
    dataset.resize((start + values.shape[0], 4))
    dataset[start:start + values.shape[0]] = values


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MVSEC event messages from a ROS1 bag into HDF5.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", type=str, default=None, help="Event topic. Auto-detected when omitted.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for quick smoke conversions.")
    args = parser.parse_args()

    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: python -m pip install rosbags") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Reader(args.bag) as reader:
        connections = _select_event_connections(reader, args.topic)
        typestore = _register_bag_types(connections)
        print("event topics:")
        for conn in connections:
            print(f"  {conn.topic} ({conn.msgtype}, {conn.msgcount} messages)")

        written = 0
        with h5py.File(args.output, "w") as h5:
            dataset = h5.create_dataset(
                "events",
                shape=(0, 4),
                maxshape=(None, 4),
                chunks=(min(1_000_000, args.max_events or 1_000_000), 4),
                dtype=np.float32,
            )
            for conn, _, rawdata in reader.messages(connections=connections):
                message = typestore.deserialize_ros1(rawdata, conn.msgtype)
                batch = _event_batch_to_array(message)
                if args.max_events is not None:
                    remaining = args.max_events - written
                    if remaining <= 0:
                        break
                    batch = batch[:remaining]
                _append(dataset, batch)
                written += int(batch.shape[0])
                if written and written % 1_000_000 < batch.shape[0]:
                    print(f"written_events: {written}")
                if args.max_events is not None and written >= args.max_events:
                    break

            h5.attrs["source_bag"] = str(args.bag)
            h5.attrs["event_count"] = written
        print(f"saved: {args.output}")
        print(f"event_count: {written}")


if __name__ == "__main__":
    main()
