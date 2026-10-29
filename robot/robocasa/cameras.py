"""Optional RoboCasa cameras and calibrated RGB snapshots, without file I/O.

Camera names are MuJoCo model names, without the Gym ``video.`` prefix.
``existing`` cameras may specify width/height; omitted dimensions use the
native image size when available, otherwise 256. ``fixed`` cameras require
world-frame ``pos`` (metres) and ``quat`` (WXYZ), with optional ``fovy`` in
degrees (default 62). ``auto`` injects LIBERO's five perception camera names
and uses the shared VGGT rig around legal initial item placement regions after
every reset, falling back to task fixtures when no item regions are available.

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

from robot.mujoco_simulation.vggt_camera_rig import VggtCameraRig


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
    """Inject fixed/auto cameras and cache configured calibrated camera views.

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
        auto_names = {c.name for c in self._configs if c.mode == "auto"}
        if auto_names and auto_names != set(VggtCameraRig._LAYOUT):
            raise ValueError("auto cameras must declare all five VGGT perception views")
        self._rig_region: dict[str, Any] | None = None

    @property
    def rig_region(self) -> dict[str, Any] | None:
        """Region used for this reset's fixed rig, for diagnostic readers."""
        from copy import deepcopy

        return deepcopy(self._rig_region)

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
        if any(config.mode in {"fixed", "auto"} for config in self._configs):
            environment.unwrapped.set_xml_processor(self.process_model_xml)

    def process_model_xml(self, xml: str) -> str:
        """Add world-fixed cameras without changing existing model elements."""
        fixed = [config for config in self._configs if config.mode in {"fixed", "auto"}]
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
                # Auto placeholders are replaced after native reset selects
                # the actual kitchen. They are not native Gym policy cameras.
                pos=" ".join(format(value, ".17g") for value in (config.pos or (0, 0, 2))),
                quat=" ".join(format(value, ".17g") for value in (config.quat or (1, 0, 0, 0))),
                fovy=format(config.fovy, ".17g"),
            )
        return ET.tostring(root, encoding="unicode")

    def apply_auto_rig(self, environment: Any) -> None:
        """Plan once after reset; forward kinematics without advancing physics.

        Region extraction reads already selected fixtures only. It neither
        reads movable-object poses nor invokes random fixture selection.
        """
        self._rig_region = None
        auto = [config for config in self._configs if config.mode == "auto"]
        if not auto:
            return
        self.clear()
        native = environment.unwrapped
        kitchen = native.env
        sim = native.sim
        region = self._observation_region(kitchen, sim)
        poses = VggtCameraRig().plan_from_bounds(
            bounds_min=np.asarray(region["bounds_min_m"]),
            bounds_max=np.asarray(region["bounds_max_m"]),
            look_at=np.asarray(region["look_at_m"]),
            azimuth_offset_deg=region["azimuth_offset_deg"],
            camera_settings={c.name: {"width": c.width, "height": c.height, "fovy": c.fovy} for c in auto},
        )
        for config in auto:
            camera_id = int(sim.model.camera_name2id(config.name))
            if camera_id < 0:
                raise ValueError(f"RoboCasa model is missing auto camera {config.name!r}")
            sim.model.cam_pos[camera_id] = poses[config.name].position_m
            sim.model.cam_quat[camera_id] = poses[config.name].quaternion_wxyz
            sim.model.cam_fovy[camera_id] = config.fovy
        sim.forward()
        self._rig_region = region

    @staticmethod
    def _observation_region(kitchen: Any, sim: Any) -> dict[str, Any]:
        """Adapt fixture placement regions to the shared, world-frame rig API."""
        state = vars(kitchen)
        fixtures = state.get("fixtures", {})
        selected: dict[str, Any] = {}

        def add(value: Any) -> Any:
            if isinstance(value, tuple):
                value = value[0]
            if isinstance(value, str):
                value = fixtures.get(value)
            if value is not None and callable(getattr(value, "get_ext_sites", None)):
                selected[value.name] = value
                return value
            return None

        anchor = add(state.get("init_robot_base_ref"))
        for role, value in state.get("fixture_refs", {}).items():
            if not role.startswith("distr"):
                add(value)
        # Several atomic tasks (coffee/blender, for example) do not register
        # fixture_refs, but expose their already selected fixtures as attributes.
        for role in ("counter", "cab", "cabinet", "drawer", "sink", "stove",
                     "coffee_machine", "blender", "toaster", "toaster_oven",
                     "fridge", "microwave", "stand_mixer", "electric_kettle", "dishwasher"):
            add(state.get(role))
        for cfg in state.get("object_cfgs", ()):
            if not str(cfg.get("name", "")).startswith("distr"):
                add(cfg.get("placement", {}).get("fixture"))
        if not selected:
            raise ValueError("auto camera rig cannot find this task's manipulation fixtures")
        anchor = anchor if anchor is not None else next(iter(selected.values()))

        # A movable appliance's adjacent lid/object may lie on its parent
        # counter even when the task has no counter attribute or object_cfgs.
        if not any("Counter" in type(f).__name__ for f in selected.values()):
            counters = [f for f in fixtures.values() if "Counter" in type(f).__name__]
            if counters:
                def counter_distance(fixture):
                    points = np.asarray(fixture.get_ext_sites(all_points=True, relative=False))
                    delta = np.maximum(np.maximum(points.min(0)[:2]-anchor.pos[:2],
                                                  anchor.pos[:2]-points.max(0)[:2]), 0)
                    return float(np.linalg.norm(delta))
                nearest = min(counters, key=counter_distance)
                if counter_distance(nearest) <= 0.25:
                    add(nearest)

        # Use legal placement ranges, analogous to LIBERO BDDL regions, rather
        # than aiming at an entire long counter. Read existing samplers only;
        # never call sample() or inspect sampled movable-object coordinates.
        sampling_points: dict[str, list[np.ndarray]] = {}
        samplers = getattr(state.get("placement_initializer"), "samplers", {})
        for cfg in state.get("object_cfgs", ()):
            name = str(cfg.get("name", ""))
            if name.startswith("distr"):
                continue
            sampler = samplers.get(f"{name}_Sampler")
            fixture = fixtures.get(getattr(sampler, "reference_object", None))
            if sampler is None or fixture is None:
                continue
            yaw = float(sampler.reference_rot)
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            region = cfg.get("reset_region") or {}
            height = max(0.12, float(region.get("height") or 0.32))
            lower = [min(sampler.x_range), min(sampler.y_range), 0]
            upper = [max(sampler.x_range), max(sampler.y_range), height]
            corners = (VggtCameraRig._box_corners(np.asarray(lower), np.asarray(upper))
                       @ rotation.T + np.asarray(sampler.reference_pos))
            # Match the native sliding-region displacement, using fixture
            # geometry, not the placed object's resulting position.
            region_name = region.get("name")
            original = getattr(fixture, "_regions", {}).get(region_name, {})
            elem = original.get("elem")
            if elem is not None:
                geom_name = f"{fixture.naming_prefix}reg_{region_name}"
                try:
                    geom_id = int(sim.model.geom_name2id(geom_name))
                except (ValueError, KeyError):
                    geom_id = -1
                if geom_id >= 0:
                    local_center = np.fromstring(elem.get("pos", "0 0 0"), sep=" ")
                    original_center = local_center @ rotation.T + fixture.pos
                    corners += sim.data.geom_xpos[geom_id] - original_center
            sampling_points.setdefault(fixture.name, []).extend(corners)

        points = []
        for fixture in selected.values():
            if fixture.name in sampling_points:
                points.extend(sampling_points[fixture.name])
                continue
            # As in LIBERO's placement regions, prioritize legal initial item
            # locations. A tall destination cabinet must not pull a counter
            # object's rig upward. Fixtures remain the fallback for tasks
            # without item placement samplers (e.g. opening a cabinet).
            if sampling_points:
                continue
            regions = fixture.get_reset_regions(env=kitchen)
            if not regions:
                points.extend(fixture.get_ext_sites(all_points=True, relative=False))
                continue
            yaw = float(fixture.rot)
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for name, region in regions.items():
                # Sliding interiors move relative to fixture.pos. Read their
                # compiled region geom transform instead of the closed pose.
                geom_name = f"{fixture.naming_prefix}reg_{name}"
                try:
                    geom_id = int(sim.model.geom_name2id(geom_name))
                except (ValueError, KeyError):
                    geom_id = -1
                if geom_id >= 0:
                    center = np.asarray(sim.data.geom_xpos[geom_id])
                    basis = np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3)
                    half = np.asarray(sim.model.geom_size[geom_id])
                    corners = VggtCameraRig._box_corners(-half, half) @ basis.T + center
                else:
                    offset = np.asarray(region["offset"], dtype=float)
                    size = np.asarray(region["size"], dtype=float)
                    height = max(0.12, float(region.get("height") or 0.32))
                    lower = offset - np.array([size[0]/2, size[1]/2, 0])
                    upper = offset + np.array([size[0]/2, size[1]/2, height])
                    corners = VggtCameraRig._box_corners(lower, upper) @ rotation.T + fixture.pos
                points.extend(corners)
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("task fixtures have invalid observation bounds")
        minimum = points.min(0) - [0.10, 0.10, 0.02]
        maximum = points.max(0) + [0.10, 0.10, 0.05]
        return {
            "fixture_names": list(selected),
            "placement_region_fixtures": list(sampling_points),
            "bounds_min_m": minimum.tolist(),
            "bounds_max_m": maximum.tolist(),
            "look_at_m": ((minimum+maximum)/2).tolist(),
            # Fixture fronts point along local -Y. Rotate the original +X rig
            # to that accessible side; preserve all five relative angles.
            "azimuth_offset_deg": float(np.rad2deg(anchor.rot)-90.0),
        }

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
        self._rig_region = None

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
        if not isinstance(mode, str) or mode not in {"existing", "fixed", "auto"}:
            raise ValueError(
                f"camera {name!r} mode must be existing, fixed or auto"
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
        pos = quat = None
        if mode == "auto":
            if {"pos", "quat"}.intersection(settings):
                raise ValueError(f"auto camera {name!r} pose is determined by the shared rig")
            if any(value is None for value in dimensions):
                raise ValueError(f"auto camera {name!r} requires width and height")
        else:
            pos = cls._vector(settings.get("pos"), 3, f"{name}.pos")
            quat = cls._vector(settings.get("quat"), 4, f"{name}.quat (WXYZ)")
        if quat is not None:
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
