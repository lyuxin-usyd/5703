from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List topics in a ROS1 bag without requiring a full ROS install.")
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()

    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: python -m pip install rosbags") from exc

    with Reader(args.bag) as reader:
        counts = Counter()
        rows = []
        for conn in reader.connections:
            counts[(conn.topic, conn.msgtype)] += int(conn.msgcount)
        for (topic, msgtype), msgcount in sorted(counts.items()):
            rows.append((topic, msgtype, msgcount))

        print(f"bag: {args.bag}")
        print(f"messages: {reader.message_count}")
        print(f"duration_sec: {(reader.end_time - reader.start_time) / 1e9:.3f}")
        print("topics:")
        for topic, msgtype, msgcount in rows:
            print(f"  {msgcount:>10}  {msgtype:<35}  {topic}")


if __name__ == "__main__":
    main()
