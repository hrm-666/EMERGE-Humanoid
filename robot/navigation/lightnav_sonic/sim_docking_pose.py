"""Privileged, read-only simulation robot pose for station docking.

This source is separate from policy RGB/yaw readers. Object coordinates and
images are never consumed. No pose is published to the Agent or to a policy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


def finite_number(value: Any, name: str) -> float:
    """Return a finite numeric value, rejecting booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SimRobotPose:
    """One timestamped robot-root pose; quaternion convention is wxyz."""

    x: float
    y: float
    z: float
    yaw: float
    tilt: float
    wall_time_s: float
    sim_time_s: float
    support_enabled: bool

    def fresh(self, now: float, max_age_s: float) -> bool:
        return -0.05 <= now - self.wall_time_s <= max_age_s


def decode_sim_robot_pose(raw: bytes) -> SimRobotPose:
    """Decode only robot diagnostics; reject absent or malformed evidence."""
    import msgpack

    if not isinstance(raw, bytes) or len(raw) > 16 * 1024 * 1024:
        raise ValueError("invalid observation size")
    payload = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    diagnostics = payload["diagnostics"]
    xyz = diagnostics["robot_base_xyz"]
    quat = payload["base_quat"]
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError("robot_base_xyz must contain three numbers")
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        raise ValueError("base_quat must contain four wxyz numbers")
    x, y, z = (finite_number(v, "robot_base_xyz") for v in xyz)
    w, qx, qy, qz = (finite_number(v, "base_quat") for v in quat)
    norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if abs(norm - 1.0) > 0.01:
        raise ValueError("base_quat must be approximately normalized")
    w, qx, qy, qz = (v / norm for v in (w, qx, qy, qz))
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy))))
    wall = finite_number(payload["wall_time_s"], "wall_time_s")
    sim = finite_number(payload["sim_time_s"], "sim_time_s")
    support = diagnostics["support_enabled"]
    if type(support) is not bool or sim < 0:
        raise ValueError("invalid support flag or simulation time")
    return SimRobotPose(x, y, z, yaw, tilt, wall, sim, support)


class SimDockingPoseSource:
    """Read the existing owner stream through a conflated SUB connection."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        if not (endpoint.startswith("ipc:///") or
                endpoint.startswith("tcp://127.0.0.1:")):
            raise ValueError("simulation pose endpoint must be local")
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._pose: SimRobotPose | None = None
        self._last_stamp: tuple[float, float] | None = None
        self.error = "waiting_for_pose"
        self._faulted = False

    def latest(self, max_age_s: float) -> SimRobotPose | None:
        if self._faulted:
            return None
        try:
            raw = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass
        else:
            try:
                pose = decode_sim_robot_pose(raw)
                stamp = (pose.wall_time_s, pose.sim_time_s)
                if self._last_stamp is not None:
                    wall, sim = self._last_stamp
                    if stamp != self._last_stamp and (
                            pose.wall_time_s <= wall or pose.sim_time_s <= sim):
                        raise ValueError("nonmonotonic simulation observation")
                self._last_stamp = stamp
                self._pose = pose
                self.error = ""
            except Exception as exc:
                # A reset/protocol failure must not disappear on a later frame.
                self._pose = None
                self._faulted = True
                self.error = f"invalid_pose:{type(exc).__name__}"
        if self._pose is not None and self._pose.fresh(time.time(), max_age_s):
            return self._pose
        self.error = self.error or "stale_pose"
        return None

    def close(self) -> None:
        self._socket.close()
        self._context.term()
