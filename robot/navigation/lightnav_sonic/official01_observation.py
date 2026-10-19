"""Read the existing P5 simulation RGB stream without importing GR00T/MuJoCo.

Only the ego image is decoded from the msgpack-numpy envelope. Scene metadata,
object positions and manipulation state never enter the navigation policy.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from PIL import Image

from .feedback import yaw_from_wxyz


@dataclass(frozen=True)
class ObservationFrame:
    jpeg: bytes
    wall_time_s: float
    sim_time_s: float
    yaw_rad: float | None = None

    def fresh(self, max_age_s: float) -> bool:
        # All namespace processes share the workstation's wall clock.
        return -0.25 <= time.time() - self.wall_time_s <= max_age_s


def decode_observation(raw: bytes) -> ObservationFrame:
    """Decode P5's numeric ndarray wire format; never deserialize pickle."""
    payload = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    image: Any = payload["observation"]["video"]["ego_view"]
    if isinstance(image, dict) and image.get(b"nd") is True:
        # The current MsgSerializer uses msgpack_numpy's byte-keyed envelope.
        if image.get(b"kind") not in (b"", "") or image.get(b"type") not in (
                "|u1", "uint8"):
            raise ValueError("ego_view must be a uint8 ndarray")
        if tuple(image[b"shape"]) != (1, 1, 480, 640, 3):
            raise ValueError("ego_view must have shape (1,1,480,640,3)")
        data = image[b"data"]
        if not isinstance(data, bytes) or len(data) != 480 * 640 * 3:
            raise ValueError("ego_view byte count does not match its shape")
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
    elif isinstance(image, dict) and image.get("__ndarray_class__"):
        # Compatibility with the older non-pickle .npy envelope.
        rgb = np.load(io.BytesIO(image["as_npy"]), allow_pickle=False)
        if rgb.dtype != np.uint8 or rgb.shape != (1, 1, 480, 640, 3):
            raise ValueError("ego_view must be (1,1,480,640,3)/uint8")
        rgb = rgb[0, 0]
    else:
        raise ValueError("ego_view has an unsupported ndarray envelope")
    wall_time = float(payload["wall_time_s"])
    sim_time = float(payload["sim_time_s"])
    if not math.isfinite(wall_time) or not math.isfinite(sim_time):
        raise ValueError("observation timestamps must be finite")
    base_quat = payload.get("base_quat")
    yaw_rad = None if base_quat is None else yaw_from_wxyz(base_quat)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    return ObservationFrame(buffer.getvalue(), wall_time, sim_time, yaw_rad)


class Official01Subscriber:
    """Conflated, read-only SUB connection to the original simulation owner."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._frame: ObservationFrame | None = None
        self.error = "waiting for observation"

    def latest(self, max_age_s: float) -> ObservationFrame | None:
        try:
            raw = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass
        else:
            try:
                self._frame = decode_observation(raw)
                self.error = ""
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self._frame = None
                self.error = f"invalid observation: {exc}"
        if self._frame is not None and self._frame.fresh(max_age_s):
            return self._frame
        return None

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class Official01YawSubscriber:
    """Read root yaw from the host-visible Official01 observation stream."""

    def __init__(self, endpoint: str, _topic: str = "") -> None:
        self._subscriber = Official01Subscriber(endpoint)

    def latest(self, max_age_s: float) -> float | None:
        frame = self._subscriber.latest(max_age_s)
        return frame.yaw_rad if frame is not None else None

    def close(self) -> None:
        self._subscriber.close()
