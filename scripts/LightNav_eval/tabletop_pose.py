"""Read-only robot pose feedback from the LightNav-owned tabletop wrapper."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any
from typing import Mapping


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class RobotPose:
    """A timestamped world pose exported by the tabletop wrapper."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    wall_time_s: float
    sim_time_s: float
    qpos: tuple[float, ...] | None = None
    start_config: dict | None = None

    @property
    def yaw_rad(self) -> float:
        w, x, y, z = self.quaternion_wxyz
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def as_dict(self) -> dict[str, object]:
        return {
            "position_m": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "yaw_rad": self.yaw_rad,
            "wall_time_s": self.wall_time_s,
            "sim_time_s": self.sim_time_s,
            "start_config": self.start_config,
        }


class TabletopPoseSubscriber:
    """Conflated SUB socket for wrapper-owned world pose telemetry."""

    def __init__(self, endpoint: str) -> None:
        import msgpack
        import zmq

        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._pose: RobotPose | None = None
        self.error = "waiting for pose"

    def latest(self, max_age_s: float) -> RobotPose | None:
        try:
            raw = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            raw = None
        if raw is not None:
            try:
                payload = self._msgpack.unpackb(raw,
                                                raw=False,
                                                strict_map_key=False)
                if not isinstance(payload, Mapping):
                    raise ValueError("pose payload must be an object")
                wall = float(payload["wall_time_s"])
                sim = float(payload["sim_time_s"])
                if not math.isfinite(wall) or not math.isfinite(sim):
                    raise ValueError("pose timestamps must be finite")
                position = _finite_vector(payload["position_m"], 3,
                                          "position_m")
                quaternion = _finite_vector(payload["quaternion_wxyz"], 4,
                                            "quaternion_wxyz")
                if math.sqrt(sum(item * item for item in quaternion)) < 1e-8:
                    raise ValueError("pose quaternion has zero norm")
                full_qpos = payload.get("qpos")
                if full_qpos is not None:
                    full_qpos = _finite_vector(full_qpos, len(full_qpos),
                                               "qpos")
                config = payload.get("start_config")
                if config is not None and not isinstance(config, dict):
                    raise ValueError("start_config must be an object")
                self._pose = RobotPose(position, quaternion, wall, sim,
                                       full_qpos, config)
                self.error = ""
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self.error = f"invalid pose: {exc}"
        if self._pose is not None and -0.25 <= time.time(
        ) - self._pose.wall_time_s <= max_age_s:
            return self._pose
        return None

    def close(self) -> None:
        self._socket.close()
        self._context.term()
