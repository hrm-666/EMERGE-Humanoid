"""Update current RoboCasa images and calibration using LIBERO's file lifecycle.

Each camera has a fixed PNG path, replaced through a temporary file. The shared
observation.json is replaced after the images, just as in LIBERO. Historical
action evidence is owned by EMERGE's existing visual memory, not this writer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from robot.robocasa.cameras import RoboCasaCameraFrame


class RoboCasaObservationWriter:
    """Save selected camera frames without accessing or advancing simulation.

    Camera optical axes are +X right, +Y down, +Z forward. T_world_camera maps
    those coordinates into the simulation world, in metres. start_episode()
    removes the current manifest before reset; close of the owning environment
    leaves the current images and manifest available for final visual inspection.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        camera_names: Sequence[str],
        reference_view: str,
    ) -> None:
        if isinstance(camera_names, str) or not isinstance(camera_names, Sequence):
            raise ValueError("observation camera_names must be a sequence of names")
        names = tuple(camera_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("observation requires at least one non-empty camera name")
        if len(set(names)) != len(names):
            raise ValueError("observation camera_names must be unique")
        if reference_view not in names:
            raise ValueError("observation reference_view must be one of camera_names")
        self.workspace = Path(workspace).expanduser().resolve()
        self.camera_names = names
        self.reference_view = reference_view
        self.manifest_path = self.workspace / "artifacts/observations/observation.json"
        self._image_root = self.workspace / "artifacts/cameras/robocasa"
        self._episode_id = uuid4().hex
        self._revision = 0

    def invalidate(self) -> None:
        """Remove the current manifest before resetting or invalidating a scene."""
        self.manifest_path.unlink(missing_ok=True)

    def start_episode(self) -> None:
        """Invalidate the previous scene before reset and assign a fresh identity."""
        self.invalidate()
        self._episode_id = uuid4().hex

    def write(self, frames: Mapping[str, RoboCasaCameraFrame]) -> Path:
        """Overwrite current images and then publish their calibration manifest."""
        selected = []
        for name in self.camera_names:
            if name not in frames:
                raise KeyError(f"observation is missing configured camera {name!r}")
            frame = frames[name]
            self._validate_frame(name, frame)
            selected.append(frame)

        from PIL import Image

        self._revision += 1
        self._image_root.mkdir(parents=True, exist_ok=True)
        views = []
        for index, frame in enumerate(selected):
            # Stable numeric names also support MuJoCo names containing slashes.
            image_path = self._image_root / f"{index:03d}.png"
            temporary_image = image_path.with_name(f".{image_path.stem}.{os.getpid()}.tmp.png")
            try:
                Image.fromarray(frame.rgb).save(temporary_image, format="PNG")
                os.replace(temporary_image, image_path)
            finally:
                temporary_image.unlink(missing_ok=True)
            views.append({
                "name": frame.name,
                "image_path": image_path.relative_to(self.workspace).as_posix(),
                "width": frame.width,
                "height": frame.height,
                "intrinsics": frame.intrinsics.tolist(),
                "T_world_camera": frame.T_world_camera.tolist(),
            })
        payload = {
            "schema_version": "Emerge.robocasa_observation.v1",
            "revision": self._revision,
            "episode_id": self._episode_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_frame": "world",
            "reference_view": self.reference_view,
            "rgbd": False,
            "views": views,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = self.manifest_path.with_name(f".observation.{os.getpid()}.tmp.json")
        try:
            temporary_manifest.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, self.manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        return self.manifest_path

    @staticmethod
    def _validate_frame(name: str, frame: RoboCasaCameraFrame) -> None:
        if frame.name != name:
            raise ValueError(f"camera frame name does not match observation key {name!r}")
        rgb, intrinsics, transform = frame.rgb, frame.intrinsics, frame.T_world_camera
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
            raise ValueError(f"camera {name!r} requires a non-empty HWC uint8 RGB image")
        if (
            intrinsics.shape != (3, 3)
            or not np.all(np.isfinite(intrinsics))
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-8)
        ):
            raise ValueError(f"camera {name!r} requires valid finite 3x3 intrinsics")
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        ):
            raise ValueError(f"camera {name!r} requires a finite homogeneous T_world_camera")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-4
        ):
            raise ValueError(f"camera {name!r} requires a proper camera rotation")
