"""Per-run recording of Official01 first- and third-person simulation views."""

from __future__ import annotations

import io
import shutil
import subprocess
import threading
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from Emerge.runtime.storage import atomic_json

FRAME_SHAPE = (480, 640, 3)


def _mapping_value(mapping: dict, key: str) -> Any:
    if key in mapping:
        return mapping[key]
    byte_key = key.encode()
    if byte_key in mapping:
        return mapping[byte_key]
    raise KeyError(key)


def _decode_ndarray(value: Any) -> np.ndarray:
    """Decode the safe ndarray envelopes emitted by GR00T's MsgSerializer."""

    if isinstance(value, dict):
        try:
            if _mapping_value(value, "nd") is True:
                dtype_name = _mapping_value(value, "type")
                if isinstance(dtype_name, bytes):
                    dtype_name = dtype_name.decode("ascii")
                dtype = np.dtype(dtype_name)
                shape = tuple(
                    int(item) for item in _mapping_value(value, "shape"))
                data = _mapping_value(value, "data")
                if not isinstance(data, bytes):
                    raise ValueError("ndarray data must be bytes")
                return np.frombuffer(data, dtype=dtype).reshape(shape)
        except KeyError:
            pass
        try:
            if _mapping_value(value, "__ndarray_class__"):
                return np.load(
                    io.BytesIO(_mapping_value(value, "as_npy")),
                    allow_pickle=False,
                )
        except KeyError:
            pass
    if isinstance(value, np.ndarray):
        return value
    raise ValueError("unsupported ndarray envelope")


def decode_recording_frames(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB ego and whole-robot frames from one mirrored P5 payload."""

    payload = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    observation = _mapping_value(payload, "observation")
    video = _mapping_value(observation, "video")
    ego = _decode_ndarray(_mapping_value(video, "ego_view"))
    if ego.shape == (1, 1, *FRAME_SHAPE):
        ego = ego[0, 0]
    third = _decode_ndarray(_mapping_value(payload, "third_person_rgb"))
    for name, frame in (("ego_view", ego), ("third_person_rgb", third)):
        if frame.shape != FRAME_SHAPE or frame.dtype != np.uint8:
            raise ValueError(
                f"{name} must be {FRAME_SHAPE}/uint8, got {frame.shape}/{frame.dtype}"
            )
    return np.ascontiguousarray(ego), np.ascontiguousarray(third)


class _FFmpegVideoWriter:

    def __init__(self, path: Path, *, fps: float, ffmpeg_binary: str) -> None:
        self.path = path
        self.frames = 0
        self._log_path = path.with_suffix(".ffmpeg.log")
        self._log = self._log_path.open("wb")
        self._process = subprocess.Popen(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "640x480",
                "-framerate",
                f"{fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        self._process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> str | None:
        error = None
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                return_code = self._process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    return_code = self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    return_code = self._process.wait(timeout=3.0)
            if return_code != 0:
                error = f"ffmpeg exited with status {return_code}; see {self._log_path.name}"
        finally:
            self._log.close()
        return error


class RunVideoRecorder:
    """Subscribe to P5's host mirror and write videos into one run directory."""

    def __init__(
        self,
        directory: Path,
        *,
        endpoint: str,
        fps: float,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self.directory = directory
        self.endpoint = endpoint
        self.fps = float(fps)
        self.ffmpeg_binary = ffmpeg_binary
        self.manifest_path = directory / "recording.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writers: dict[str, _FFmpegVideoWriter] = {}
        self._error: str | None = None
        self._last_frame_error: str | None = None
        self._started_at = datetime.now(timezone.utc).isoformat()

    def _manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": "Emerge.run_recording.v1",
            "status": status,
            "endpoint": self.endpoint,
            "fps": self.fps,
            "started_at": self._started_at,
            "finished_at": (datetime.now(timezone.utc).isoformat()
                            if status != "recording" else None),
            "videos": {
                "first_person": {
                    "path":
                        "first_person.mp4",
                    "frames":
                        self._writers.get("first_person").frames
                        if "first_person" in self._writers else 0,
                },
                "third_person": {
                    "path":
                        "third_person.mp4",
                    "frames":
                        self._writers.get("third_person").frames
                        if "third_person" in self._writers else 0,
                },
            },
            "error": self._error,
            "last_frame_error": self._last_frame_error,
        }

    def start(self) -> dict[str, Any]:
        binary = shutil.which(self.ffmpeg_binary)
        if binary is None:
            self._error = f"ffmpeg binary not found: {self.ffmpeg_binary}"
            manifest = self._manifest("failed")
            atomic_json(self.manifest_path, manifest)
            return manifest
        self.ffmpeg_binary = binary
        self._thread = threading.Thread(
            target=self._record_loop,
            name="emerge-run-video-recorder",
            daemon=True,
        )
        self._thread.start()
        manifest = self._manifest("recording")
        atomic_json(self.manifest_path, manifest)
        return manifest

    def _writer(self, name: str) -> _FFmpegVideoWriter:
        writer = self._writers.get(name)
        if writer is None:
            writer = _FFmpegVideoWriter(
                self.directory / f"{name}.mp4",
                fps=self.fps,
                ffmpeg_binary=self.ffmpeg_binary,
            )
            self._writers[name] = writer
        return writer

    def _record_loop(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            socket.connect(self.endpoint)
            while not self._stop.is_set():
                if not socket.poll(timeout=100):
                    continue
                raw = socket.recv()
                try:
                    ego, third = decode_recording_frames(raw)
                except Exception as exc:
                    # A recorder can briefly overlap an older publisher during
                    # a stack restart.  Keep waiting for the new dual-view
                    # schema rather than killing the entire run recording.
                    self._last_frame_error = f"{type(exc).__name__}: {exc}"
                    continue
                self._writer("first_person").write(ego)
                self._writer("third_person").write(third)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            socket.close(linger=0)
            context.term()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive() and self._error is None:
                self._error = "recording thread did not stop within 5 seconds"
        close_errors = [writer.close() for writer in self._writers.values()]
        close_errors = [error for error in close_errors if error]
        if close_errors:
            suffix = "; ".join(close_errors)
            self._error = f"{self._error}; {suffix}" if self._error else suffix
        frame_counts = [writer.frames for writer in self._writers.values()]
        if not frame_counts and self._error is None:
            detail = f"; last frame error: {self._last_frame_error}" if self._last_frame_error else ""
            self._error = f"no dual-view simulation frames received{detail}"
        elif len(frame_counts) < 2 and self._error is None:
            self._error = "one required camera stream was missing"
        status = "completed" if self._error is None else "failed"
        manifest = self._manifest(status)
        atomic_json(self.manifest_path, manifest)
        return manifest
