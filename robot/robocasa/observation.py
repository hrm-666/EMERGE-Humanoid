"""Update current RoboCasa images and calibration using LIBERO's file lifecycle.

Each camera has a fixed PNG path, replaced through a temporary file. The shared
observation.json is replaced after the images, just as in LIBERO. Historical
action evidence is owned by EMERGE's existing visual memory, not this writer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from Emerge.runtime.storage import atomic_json
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
        names = tuple(camera_names)
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
        selected = [frames[name] for name in self.camera_names]

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
        atomic_json(self.manifest_path, payload)
        return self.manifest_path
