"""Publish current RGB(-D) frames and calibrated observation manifests."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from Emerge.runtime.storage import atomic_json
from robot.robocasa.cameras import RoboCasaCameraFrame


class RoboCasaObservationWriter:
    def __init__(
        self, workspace: str | Path, *, camera_names: Sequence[str],
        reference_view: str, verification_camera_names: Sequence[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.camera_names = tuple(camera_names)
        self.verification_camera_names = tuple(verification_camera_names or camera_names)
        if reference_view not in self.camera_names:
            raise ValueError("observation reference_view must be one of camera_names")
        self.reference_view = reference_view
        self.manifest_path = self.workspace / "artifacts/observations/observation.json"
        self.verification_manifest_path = self.workspace / "artifacts/verification_observations/observation.json"
        self._image_root = self.workspace / "artifacts/cameras/robocasa"
        self._episode_id = uuid4().hex
        self._revision = 0

    def invalidate(self) -> None:
        self.manifest_path.unlink(missing_ok=True)
        self.verification_manifest_path.unlink(missing_ok=True)

    def start_episode(self) -> None:
        self.invalidate()
        self._episode_id = uuid4().hex

    def write(self, frames: Mapping[str, RoboCasaCameraFrame]) -> Path:
        from PIL import Image

        selected = [frames[name] for name in dict.fromkeys(
            self.camera_names + self.verification_camera_names
        )]
        self._image_root.mkdir(parents=True, exist_ok=True)
        views = {}
        for index, frame in enumerate(selected):
            view = {
                "name": frame.name, "width": frame.width, "height": frame.height,
                "intrinsics": frame.intrinsics.tolist(),
                "T_world_camera": frame.T_world_camera.tolist(),
            }
            # PNG and NPY share the same temporary-file replacement lifecycle.
            for key, suffix, array in (
                ("image_path", ".png", frame.rgb),
                ("depth_path", "_depth.npy", frame.depth_m),
            ):
                if array is None:
                    continue
                path = self._image_root / f"{index:03d}{suffix}"
                temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
                try:
                    if key == "image_path":
                        Image.fromarray(array).save(temporary, format="PNG")
                    else:
                        np.save(temporary, array, allow_pickle=False)
                        view["depth_unit"] = "meter"
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                view[key] = path.relative_to(self.workspace).as_posix()
            views[frame.name] = view

        self._revision += 1
        publications = []
        if self.verification_camera_names != self.camera_names:
            publications.append((self.verification_manifest_path, self.verification_camera_names))
        else:
            self.verification_manifest_path.unlink(missing_ok=True)
        publications.append((self.manifest_path, self.camera_names))
        for path, names in publications:
            atomic_json(path, {
                "schema_version": "Emerge.robocasa_observation.v1",
                "revision": self._revision, "episode_id": self._episode_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "coordinate_frame": "world",
                "reference_view": self.reference_view if self.reference_view in names else names[0],
                "rgbd": all(frames[name].depth_m is not None for name in names),
                "views": [views[name] for name in names],
            })
        return self.manifest_path
