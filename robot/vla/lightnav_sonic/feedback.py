"""Passive SONIC camera and heading subscriptions."""

from __future__ import annotations

import base64
import binascii
import math
import time
from numbers import Real
from typing import Any
from typing import Mapping
from typing import Sequence


def extract_jpeg_from_camera_message(payload: Mapping[str, Any],
                                     camera_name: str) -> bytes:
    """Extract JPEG bytes from SONIC's camera-publisher msgpack payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("camera payload must be an object")
    candidate = payload.get(camera_name)
    if candidate is None:
        images = payload.get("images")
        if isinstance(images, Mapping):
            candidate = images.get(camera_name)
    if isinstance(candidate, bytes):
        if not candidate:
            raise ValueError(f"camera {camera_name!r} published empty bytes")
        return candidate
    if not isinstance(candidate, str) or not candidate:
        raise ValueError(
            f"camera payload does not contain non-empty {camera_name!r} JPEG")
    try:
        return base64.b64decode(candidate, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            f"camera {camera_name!r} JPEG is not valid base64") from exc


def yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    """Return SONIC/MuJoCo world yaw from a finite wxyz quaternion."""

    if isinstance(quaternion, (str, bytes)) or len(quaternion) != 4:
        raise ValueError("base_quat must contain four wxyz values")
    values: list[float] = []
    for value in quaternion:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("base_quat values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("base_quat values must be finite numbers")
        values.append(number)
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("base_quat has zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SonicCameraSubscriber:
    """Conflated subscriber for SONIC's msgpack camera publisher."""

    def __init__(self, endpoint: str, camera_name: str) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("runner requires pyzmq and msgpack") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._camera_name = camera_name
        self._jpeg: bytes | None = None
        self._received_at: float | None = None

    def poll(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            payload = self._msgpack.unpackb(raw,
                                            raw=False,
                                            strict_map_key=False)
            try:
                self._jpeg = extract_jpeg_from_camera_message(
                    payload, self._camera_name)
                self._received_at = time.monotonic()
            except (TypeError, ValueError):
                # A malformed camera frame is discarded; its existing value is
                # allowed to expire naturally and then produces SONIC IDLE.
                continue

    def latest(self, max_age_s: float) -> bytes | None:
        self.poll()
        if self._jpeg is None or self._received_at is None:
            return None
        if time.monotonic() - self._received_at > max_age_s:
            return None
        return self._jpeg

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class SonicYawSubscriber:
    """Conflated subscriber for SONIC's topic-prefixed msgpack feedback."""

    def __init__(self, endpoint: str, topic: str) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("runner requires pyzmq and msgpack") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._topic = topic.encode("utf-8")
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(endpoint)
        self._yaw: float | None = None
        self._received_at: float | None = None

    def poll(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            if not raw.startswith(self._topic):
                continue
            payload = self._msgpack.unpackb(raw[len(self._topic):],
                                            raw=False,
                                            strict_map_key=False)
            if not isinstance(payload, Mapping):
                continue
            try:
                self._yaw = yaw_from_wxyz(payload.get("base_quat"))
                self._received_at = time.monotonic()
            except (TypeError, ValueError):
                continue

    def latest(self, max_age_s: float) -> float | None:
        self.poll()
        if self._yaw is None or self._received_at is None:
            return None
        if time.monotonic() - self._received_at > max_age_s:
            return None
        return self._yaw

    def close(self) -> None:
        self._socket.close()
        self._context.term()
