"""Optional RoboCasa cameras and calibrated RGB snapshots, without file I/O.

Camera names are MuJoCo model names, without the Gym ``video.`` prefix.
``existing`` cameras may specify width/height; omitted dimensions use the
native image size when available, otherwise 256. ``fixed`` cameras require
world-frame ``pos`` (metres) and ``quat`` (WXYZ), with optional ``fovy`` in
degrees (default 62). Only fixed cameras are injected into model XML.

Native images are already upright. Direct MuJoCo renders are flipped once to
match them. Calibration uses optical camera axes: +X right, +Y down, +Z forward.
This module does not import RoboCasa/robosuite, step physics, or write artifacts.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RoboCasaCameraFrame:
    """One RGB/calibration snapshot; array contents must be treated as read-only.

    ``T_world_camera`` maps optical camera coordinates into simulation world
    coordinates. The arrays are owned by this snapshot, not by native Gym obs.
    """

    name: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@dataclass(frozen=True)
class _CameraConfig:
    name: str
    mode: str
    width: int | None
    height: int | None
    pos: tuple[float, ...] | None = None
    quat: tuple[float, ...] | None = None
    fovy: float | None = None


class RoboCasaCameraManager:
    """Inject fixed cameras on model rebuild and cache configured camera views.

    Register with ``install()`` before the manager's first explicit Gym reset.
    The XML callback then runs on subsequent model rebuilds too. Existing
    camera names are never overwritten. IDs and poses are read from the current
    simulation on every refresh, since hard resets replace the compiled model.
    """

    def __init__(self, camera_config: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        if camera_config is not None and not isinstance(camera_config, Mapping):
            raise ValueError("camera_config must be a mapping of camera names to settings")
        self._configs = tuple(
            self._parse_config(name, settings)
            for name, settings in (camera_config or {}).items()
        )
        self._frames: dict[str, RoboCasaCameraFrame] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(config.name for config in self._configs)

    @property
    def frames(self) -> dict[str, RoboCasaCameraFrame]:
        """Latest complete frame set; empty after invalidation or failed capture."""
        return dict(self._frames)

    def has_camera(self, name: str) -> bool:
        return any(config.name == name for config in self._configs)

    def get_frame(self, name: str) -> RoboCasaCameraFrame:
        if not self.has_camera(name):
            raise KeyError(f"RoboCasa camera is not configured: {name!r}")
        if name not in self._frames:
            raise RuntimeError("RoboCasa camera frame is unavailable; call refresh_cameras()")
        return self._frames[name]

    def install(self, environment: Any) -> None:
        """Attach one XML processor to a newly created Gym environment."""
        if any(config.mode == "fixed" for config in self._configs):
            environment.unwrapped.set_xml_processor(self.process_model_xml)

    def process_model_xml(self, xml: str) -> str:
        """Add world-fixed cameras without changing existing model elements."""
        fixed = [config for config in self._configs if config.mode == "fixed"]
        if not fixed:
            return xml
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MuJoCo XML is missing worldbody; cannot add RoboCasa cameras")
        existing_names = {node.get("name") for node in root.iter("camera")}
        for config in fixed:
            if config.name in existing_names:
                raise ValueError(
                    f"fixed camera {config.name!r} collides with an existing model camera"
                )
            ET.SubElement(
                worldbody,
                "camera",
                name=config.name,
                mode="fixed",
                pos=" ".join(format(value, ".17g") for value in config.pos),
                quat=" ".join(format(value, ".17g") for value in config.quat),
                fovy=format(config.fovy, ".17g"),
            )
        return ET.tostring(root, encoding="unicode")

    def refresh(self, sim: Any, native_obs: Mapping[str, Any]) -> dict[str, RoboCasaCameraFrame]:
        """Capture a complete set at the current physics state, without stepping.

        Native RGB is reused only at its actual dimensions. A requested size
        change, or a camera absent from native obs, uses a direct sim render.
        Any capture failure invalidates the whole frame set to avoid mixed ages.
        """
        self.clear()
        frames = {}
        for config in self._configs:
            camera_id = int(sim.model.camera_name2id(config.name))
            if camera_id < 0:
                raise ValueError(f"RoboCasa model is missing camera {config.name!r}")
            native = native_obs.get(f"video.{config.name}")
            if native is not None:
                native = self._rgb_array(native, config.name)
            width = config.width or (native.shape[1] if native is not None else 256)
            height = config.height or (native.shape[0] if native is not None else 256)
            if native is not None and native.shape[:2] == (height, width):
                rgb = native.copy(order="C")
            else:
                rendered = sim.render(
                    camera_name=config.name, width=width, height=height, depth=False
                )
                rgb = self._rgb_array(rendered, config.name)[::-1].copy(order="C")
                if rgb.shape[:2] != (height, width):
                    raise ValueError(f"camera {config.name!r} rendered an unexpected image size")
            fovy = float(sim.model.cam_fovy[camera_id])
            if not np.isfinite(fovy) or not 0.0 < fovy < 180.0:
                raise ValueError(f"camera {config.name!r} requires a perspective vertical FOV")
            focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
            intrinsics = np.array(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
            )
            transform = np.eye(4)
            transform[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3) @ np.diag(
                [1.0, -1.0, -1.0]
            )
            transform[:3, 3] = sim.data.cam_xpos[camera_id]
            if not np.all(np.isfinite(transform)):
                raise ValueError(f"camera {config.name!r} has a non-finite world pose")
            for array in (rgb, intrinsics, transform):
                array.setflags(write=False)
            frames[config.name] = RoboCasaCameraFrame(config.name, rgb, intrinsics, transform)
        self._frames = frames
        return self.frames

    def clear(self) -> None:
        """Invalidate cached images and calibration without changing camera configuration."""
        self._frames.clear()

    def close(self) -> None:
        """Discard snapshots; rendering resources are owned by the environment."""
        self.clear()

    @staticmethod
    def _rgb_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if (
            array.ndim != 3
            or array.shape[2] != 3
            or array.dtype != np.uint8
            or min(array.shape[:2]) < 1
        ):
            raise ValueError(f"camera {name!r} must provide a non-empty HWC uint8 RGB image")
        return array

    @classmethod
    def _parse_config(cls, name: str, settings: Mapping[str, Any]) -> _CameraConfig:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name.startswith("video.")
        ):
            raise ValueError("camera names must be non-empty MuJoCo names without a video. prefix")
        if not isinstance(settings, Mapping):
            raise ValueError(f"camera {name!r} settings must be a mapping")
        unknown = set(settings) - {"mode", "width", "height", "pos", "quat", "fovy"}
        if unknown:
            raise ValueError(
                f"camera {name!r} has unsupported settings: {', '.join(sorted(unknown))}"
            )
        mode = settings.get("mode", "existing")
        if not isinstance(mode, str) or mode not in {"existing", "fixed"}:
            raise ValueError(
                f"camera {name!r} mode must be existing or fixed; auto is not implemented"
            )
        dimensions = []
        for key in ("width", "height"):
            value = settings.get(key)
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"camera {name!r} {key} must be a positive integer")
            dimensions.append(None if value is None else int(value))
        if mode == "existing":
            if {"pos", "quat", "fovy"}.intersection(settings):
                raise ValueError(f"existing camera {name!r} cannot override model pose or FOV")
            return _CameraConfig(name, mode, *dimensions)
        pos = cls._vector(settings.get("pos"), 3, f"{name}.pos")
        quat = cls._vector(settings.get("quat"), 4, f"{name}.quat (WXYZ)")
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"camera {name!r} requires a nonzero finite quaternion")
        quat = tuple(value / norm for value in quat)
        raw_fovy = settings.get("fovy", 62.0)
        if isinstance(raw_fovy, (bool, np.bool_)) or not np.isscalar(raw_fovy):
            raise ValueError(f"camera {name!r} fovy must be a finite number between 0 and 180")
        fovy = float(raw_fovy)
        if not np.isfinite(fovy) or not 0.0 < fovy < 180.0:
            raise ValueError(f"camera {name!r} fovy must be a finite number between 0 and 180")
        return _CameraConfig(name, mode, *dimensions, pos, quat, fovy)

    @staticmethod
    def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
        array = np.asarray(value)
        if (
            array.shape != (length,)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"{label} must contain {length} finite real values")
        return tuple(float(item) for item in array)
