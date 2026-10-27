"""Publish calibrated RoboCasa RGB snapshots for EMERGE's visual readers.

Each publication owns a unique image directory. Only after every PNG is saved
is observation.json atomically replaced. Previously published images remain
available to readers that already loaded an older manifest. Episode/workspace
cleanup belongs to the runner, not to this synchronous writer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from uuid import uuid4

import numpy as np

from robot.robocasa.cameras import RoboCasaCameraFrame

logger = logging.getLogger(__name__)


class RoboCasaObservationWriter:
    """Save selected camera frames without accessing or advancing simulation.

    Camera optical axes are +X right, +Y down, +Z forward. T_world_camera maps
    those coordinates into the simulation world, in metres. start_episode()
    removes the current manifest before reset; close of the owning environment
    leaves the last complete publication available for final visual inspection.
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
        """Remove the current manifest, retaining images used by existing readers."""
        self.manifest_path.unlink(missing_ok=True)

    def start_episode(self) -> None:
        """Invalidate the previous scene before reset and assign a fresh identity."""
        self.invalidate()
        self._episode_id = uuid4().hex

    def write(self, frames: Mapping[str, RoboCasaCameraFrame]) -> Path:
        """Publish one complete snapshot; failures before commit keep the prior manifest.

        An interrupted replacement may already have published the new snapshot.
        Preserve its images and revision in that case. If its outcome cannot be
        read, retain the images and reserve the revision rather than risk deleting
        referenced files or reusing an observed revision. Inputs must belong to
        one synchronous capture; this method neither renders nor resizes them.
        """
        captured_at = datetime.now(timezone.utc).isoformat()
        selected = []
        for name in self.camera_names:
            if name not in frames:
                raise KeyError(f"observation is missing configured camera {name!r}")
            frame = frames[name]
            self._validate_frame(name, frame)
            selected.append(frame)

        from PIL import Image

        revision = self._revision + 1
        episode_root = self._image_root / self._episode_id
        episode_root.mkdir(parents=True, exist_ok=True)
        image_directory = Path(mkdtemp(prefix=f"{revision:06d}-", dir=episode_root))
        temporary_manifest: Path | None = None
        commit_attempted = False
        try:
            views = []
            for index, frame in enumerate(selected):
                # Use numeric filenames: model camera names need not be safe paths.
                image_path = image_directory / f"{index:03d}.png"
                Image.fromarray(frame.rgb).save(image_path, format="PNG")
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
                "revision": revision,
                "episode_id": self._episode_id,
                "captured_at": captured_at,
                "coordinate_frame": "world",
                "reference_view": self.reference_view,
                "rgbd": False,
                "views": views,
            }
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.manifest_path.parent,
                prefix=".observation.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_manifest = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
            commit_attempted = True
            os.replace(temporary_manifest, self.manifest_path)
            self._revision = revision
            temporary_manifest = None
        except BaseException:
            remove_images = True
            if commit_attempted:
                # A signal can arrive after rename succeeds but before the next
                # Python assignment. Reconcile against the actual publication.
                self._revision = revision
                try:
                    current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    pass
                except (OSError, ValueError):
                    # Unknown outcome: these files may already be referenced.
                    remove_images = False
                else:
                    remove_images = current != payload
                if remove_images:
                    self._revision = revision - 1
            if remove_images:
                shutil.rmtree(image_directory, ignore_errors=True)
            raise
        finally:
            if temporary_manifest is not None:
                try:
                    temporary_manifest.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary observation manifest", exc_info=True)
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
