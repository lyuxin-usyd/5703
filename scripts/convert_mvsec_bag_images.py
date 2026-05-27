from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _time_to_seconds(value: object) -> float:
    sec = getattr(value, "sec", getattr(value, "secs", 0))
    nsec = getattr(value, "nanosec", getattr(value, "nsec", 0))
    return float(sec) + float(nsec) * 1e-9


def _msgdef_text(conn: object) -> str | None:
    msgdef = getattr(conn, "msgdef", None)
    if msgdef is None:
        return None
    data = getattr(msgdef, "data", None)
    return data if data is not None else str(msgdef)


def _register_bag_types(connections: list[object]) -> object:
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    typestore = get_typestore(Stores.EMPTY)
    types = {}
    for conn in connections:
        text = _msgdef_text(conn)
        if text:
            types.update(get_types_from_msg(text, conn.msgtype))
    typestore.register(types)
    return typestore


def _select_image_connections(reader: object, topic: str | None) -> list[object]:
    if topic:
        selected = [conn for conn in reader.connections if conn.topic == topic]
        if not selected:
            raise ValueError(f"Could not find image topic: {topic}")
        return selected

    candidates = [
        conn
        for conn in reader.connections
        if conn.msgtype == "sensor_msgs/msg/Image" or "image_raw" in conn.topic.lower()
    ]
    if not candidates:
        raise ValueError("Could not find an image topic. Run scripts/inspect_rosbag.py first and pass --topic.")
    left = [conn for conn in candidates if "/left/" in conn.topic]
    return left or candidates[:1]


def _image_to_array(message: object) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()

    if encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"Only mono8/8UC1 images are supported for this probe, got {message.encoding!r}.")

    raw = getattr(message, "data")
    data = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, dtype=np.uint8)
    if data.size < height * step:
        raise ValueError(f"Image payload too small: got {data.size}, expected at least {height * step}.")

    rows = data[: height * step].reshape(height, step)
    return rows[:, :width].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MVSEC grayscale image_raw messages from a ROS1 bag into HDF5.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", type=str, default="/davis/left/image_raw")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick smoke conversions.")
    args = parser.parse_args()

    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: python -m pip install rosbags") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Reader(args.bag) as reader:
        connections = _select_image_connections(reader, args.topic)
        typestore = _register_bag_types(connections)
        print("image topics:")
        for conn in connections:
            print(f"  {conn.topic} ({conn.msgtype}, {conn.msgcount} messages)")

        timestamps: list[float] = []
        written = 0
        images_ds = None
        with h5py.File(args.output, "w") as h5:
            for conn, _, rawdata in reader.messages(connections=connections):
                if args.max_images is not None and written >= args.max_images:
                    break
                message = typestore.deserialize_ros1(rawdata, conn.msgtype)
                image = _image_to_array(message)
                if images_ds is None:
                    images_ds = h5.create_dataset(
                        "images",
                        shape=(0, image.shape[0], image.shape[1]),
                        maxshape=(None, image.shape[0], image.shape[1]),
                        chunks=(1, image.shape[0], image.shape[1]),
                        dtype=np.uint8,
                    )
                start = int(images_ds.shape[0])
                images_ds.resize((start + 1, image.shape[0], image.shape[1]))
                images_ds[start] = image
                timestamps.append(_time_to_seconds(message.header.stamp))
                written += 1
                if written == 1 or written % 500 == 0:
                    print(f"written_images: {written}")

            if images_ds is None:
                raise ValueError("No images were written.")
            h5.create_dataset("timestamps", data=np.asarray(timestamps, dtype=np.float64))
            h5.attrs["source_bag"] = str(args.bag)
            h5.attrs["topic"] = connections[0].topic
            h5.attrs["image_count"] = written
        print(f"saved: {args.output}")
        print(f"image_count: {written}")


if __name__ == "__main__":
    main()
