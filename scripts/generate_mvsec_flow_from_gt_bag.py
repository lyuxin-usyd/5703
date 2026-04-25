from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np


HEADER_MSG = """\
uint32 seq
time stamp
string frame_id
"""

IMAGE_MSG = """\
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
"""

POINT_MSG = """\
float64 x
float64 y
float64 z
"""

QUATERNION_MSG = """\
float64 x
float64 y
float64 z
float64 w
"""

POSE_MSG = """\
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
"""

POSE_STAMPED_MSG = """\
std_msgs/Header header
geometry_msgs/Pose pose
"""


def _time_to_seconds(value: object) -> float:
    sec = getattr(value, "sec", getattr(value, "secs", 0))
    nsec = getattr(value, "nanosec", getattr(value, "nsec", 0))
    return float(sec) + float(nsec) * 1e-9


def _read_calibration(calib_zip: Path) -> dict:
    import yaml

    with zipfile.ZipFile(calib_zip) as zf:
        yaml_names = [name for name in zf.namelist() if name.endswith(".yaml")]
        if not yaml_names:
            raise ValueError(f"No calibration yaml found in {calib_zip}")
        with zf.open(yaml_names[0]) as f:
            return yaml.safe_load(f)


def _left_camera_params(calibration: dict) -> tuple[float, float, float, float, tuple[int, int]]:
    cam0 = calibration["cam0"]
    projection = cam0["projection_matrix"]
    fx = float(projection[0][0])
    fy = float(projection[1][1])
    px = float(projection[0][2])
    py = float(projection[1][2])
    width, height = [int(v) for v in cam0["resolution"]]
    return fx, fy, px, py, (height, width)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quaternion_to_matrix(quat_wxyz)
    out[:3, 3] = pos
    return out


def _skew_to_omega(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=np.float64)


def _matrix_log_rotation(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-9:
        return np.zeros((3, 3), dtype=np.float64)
    return theta / (2.0 * math.sin(theta)) * (rotation - rotation.T)


def _relative_velocity(prev: tuple[float, np.ndarray, np.ndarray], cur: tuple[float, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    t0, p0, q0 = prev
    t1, p1, q1 = cur
    dt = float(t1 - t0)
    if dt <= 0:
        return np.zeros(3), np.zeros(3), 0.0
    h0 = _pose_to_matrix(p0, q0)
    h1 = _pose_to_matrix(p1, q1)
    h01 = np.linalg.inv(h0) @ h1
    linear = h01[:3, 3] / dt
    angular = _skew_to_omega(_matrix_log_rotation(h01[:3, :3]) / dt)
    return linear, angular, dt


def _flow_grid(height: int, width: int, fx: float, fy: float, px: float, py: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_inds, y_inds = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x_map = (x_inds - px) / fx
    y_map = (y_inds - py) / fy
    omega = np.zeros((height * width, 2, 3), dtype=np.float64)
    flat_x = x_map.reshape(-1)
    flat_y = y_map.reshape(-1)
    omega[:, 0, 0] = flat_x * flat_y
    omega[:, 1, 0] = 1 + np.square(flat_y)
    omega[:, 0, 1] = -(1 + np.square(flat_x))
    omega[:, 1, 1] = -(flat_x * flat_y)
    omega[:, 0, 2] = flat_y
    omega[:, 1, 2] = -flat_x
    return flat_x, flat_y, omega


def _compute_flow(depth: np.ndarray, linear: np.ndarray, angular: np.ndarray, dt: float, fx: float, fy: float, grid: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flat_depth = depth.astype(np.float64, copy=False).reshape(-1)
    flat_x, flat_y, omega = grid
    mask = np.isfinite(flat_depth) & (flat_depth > 0)
    x_flow = np.zeros_like(flat_depth, dtype=np.float64)
    y_flow = np.zeros_like(flat_depth, dtype=np.float64)
    inv_depth = 1.0 / flat_depth[mask]
    x_flow[mask] = inv_depth * (flat_x[mask] * linear[2] - linear[0])
    x_flow[mask] += omega[mask, 0, :] @ angular
    y_flow[mask] = inv_depth * (flat_y[mask] * linear[2] - linear[1])
    y_flow[mask] += omega[mask, 1, :] @ angular
    return (x_flow.reshape(depth.shape) * dt * fx).astype(np.float32), (y_flow.reshape(depth.shape) * dt * fy).astype(np.float32)


def _register_types(connections: list[object]) -> object:
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    typestore = get_typestore(Stores.EMPTY)
    typs = {}
    for conn in connections:
        if conn.msgtype == "sensor_msgs/msg/Image":
            typs.update(get_types_from_msg(HEADER_MSG, "std_msgs/msg/Header"))
            typs.update(get_types_from_msg(IMAGE_MSG, "sensor_msgs/msg/Image"))
        elif conn.msgtype == "geometry_msgs/msg/PoseStamped":
            typs.update(get_types_from_msg(HEADER_MSG, "std_msgs/msg/Header"))
            typs.update(get_types_from_msg(POINT_MSG, "geometry_msgs/msg/Point"))
            typs.update(get_types_from_msg(QUATERNION_MSG, "geometry_msgs/msg/Quaternion"))
            typs.update(get_types_from_msg(POSE_MSG, "geometry_msgs/msg/Pose"))
            typs.update(get_types_from_msg(POSE_STAMPED_MSG, "geometry_msgs/msg/PoseStamped"))
        elif conn.msgdef:
            typs.update(get_types_from_msg(conn.msgdef, conn.msgtype))
    typestore.register(typs)
    return typestore


def _image_to_array(message: object) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding)
    raw = bytes(message.data)
    if encoding in {"32FC1", "TYPE_32FC1"}:
        return np.frombuffer(raw, dtype="<f4").reshape(height, width).copy()
    if encoding in {"16UC1", "mono16"}:
        return np.frombuffer(raw, dtype="<u2").reshape(height, width).astype(np.float32)
    raise ValueError(f"Unsupported depth image encoding: {encoding}")


def _pose_from_message(message: object) -> tuple[float, np.ndarray, np.ndarray]:
    pose = message.pose
    return (
        _time_to_seconds(message.header.stamp),
        np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
        np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z], dtype=np.float64),
    )


def _find_connection(reader: object, topic: str) -> object:
    matches = [conn for conn in reader.connections if conn.topic == topic]
    if not matches:
        raise ValueError(f"Topic not found: {topic}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small MVSEC flow GT npz from a ground-truth ROS1 bag.")
    parser.add_argument("--gt-bag", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-topic", type=str, default="/davis/left/depth_image_rect")
    parser.add_argument("--odom-topic", type=str, default="/davis/left/odometry")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: python -m pip install rosbags") from exc

    fx, fy, px, py, (height, width) = _left_camera_params(_read_calibration(args.calib))
    grid = _flow_grid(height, width, fx, fy, px, py)

    with Reader(args.gt_bag) as reader:
        depth_conn = _find_connection(reader, args.depth_topic)
        odom_conn = _find_connection(reader, args.odom_topic)
        typestore = _register_types([depth_conn, odom_conn])

        depths: list[np.ndarray] = []
        poses: list[tuple[float, np.ndarray, np.ndarray]] = []
        for conn, _, rawdata in reader.messages(connections=[depth_conn, odom_conn]):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            if conn.topic == args.depth_topic:
                depths.append(_image_to_array(msg))
            elif conn.topic == args.odom_topic:
                poses.append(_pose_from_message(msg))
            if (
                args.max_frames is not None
                and len(depths) >= args.start_index + args.max_frames
                and len(poses) >= args.start_index + args.max_frames
            ):
                break

    stop = (
        min(len(depths), len(poses))
        if args.max_frames is None
        else min(len(depths), len(poses), args.start_index + args.max_frames)
    )
    if stop <= args.start_index:
        raise ValueError("Not enough synchronized depth/odometry messages for requested range.")

    x_flows = []
    y_flows = []
    timestamps = []
    for idx in range(args.start_index, stop):
        linear, angular, dt = _relative_velocity(poses[idx - 1], poses[idx])
        x_flow, y_flow = _compute_flow(depths[idx], linear, angular, dt, fx, fy, grid)
        x_flows.append(x_flow)
        y_flows.append(y_flow)
        timestamps.append(poses[idx][0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        x_flow_dist=np.stack(x_flows, axis=0).astype(np.float32),
        y_flow_dist=np.stack(y_flows, axis=0).astype(np.float32),
    )
    print(f"saved: {args.output}")
    print(f"frames: {len(timestamps)}")
    print(f"shape: {x_flows[0].shape if x_flows else None}")


if __name__ == "__main__":
    main()
