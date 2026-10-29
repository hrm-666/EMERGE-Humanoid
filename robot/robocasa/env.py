"""RoboCasa Gym lifecycle, native observations and action12 stepping.

This module imports RoboCasa only when an environment is created. Run the
manager in the RoboCasa Python environment; importing it does not load a VLA
model or install LIBERO's robosuite compatibility patches.
"""

from __future__ import annotations

import importlib
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robot.robocasa.cameras import RoboCasaCameraFrame, RoboCasaCameraManager
from robot.robocasa.observation import RoboCasaObservationWriter

logger = logging.getLogger(__name__)

StepResult = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]


class RoboCasaEnvManager:
    """Own one PandaOmron Gym environment without benchmark stop policies.

    ``create()`` follows the existing OpenPI RoboCasa evaluation entrypoint:
    ``gym.make(..., seed=seed)`` followed by one explicit ``reset()``. The Gym
    wrapper also resets internally during construction. No settling actions or
    repeated seeding are added here.

    Native terminated/truncated flags are preserved. Task success alone does
    not stop this debug-capable base class. Horizon enforcement, latched episode
    success and evaluation status files belong in RoboCasaEvalEnvManager in
    env_eval.py. Driver-level adaptation to EMERGE belongs in the driver.

    Optional existing/fixed/auto camera configurations add calibrated frame
    caches. They never add keys to latest_obs or alter native policy images.
    File output is off by default. To enable it, supply workspace and
    observation_config={"enabled": True, "reference_view": "<configured name>"}.
    Optional camera_names selects a subset of configured cameras to publish.
    """

    def __init__(
        self,
        *,
        env_name: str,
        split: str | None = "target",
        seed: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
        mujoco_gl: str = "egl",
        camera_config: dict[str, dict[str, Any]] | None = None,
        workspace: str | Path | None = None,
        observation_config: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(env_name, str) or not env_name.strip():
            raise ValueError("env_name must be a non-empty RoboCasa task name")
        if split not in {None, "target", "pretrain", "all"}:
            raise ValueError("split must be target, pretrain, all or None")
        self._validate_seed(seed)
        self.env_name = env_name.strip()
        self.split = split
        self.seed = None if seed is None else int(seed)
        self.env_kwargs = deepcopy(env_kwargs or {})
        reserved = {"env_name", "split", "seed"}.intersection(self.env_kwargs)
        if reserved:
            raise ValueError(
                f"pass {', '.join(sorted(reserved))} as constructor arguments, not env_kwargs"
            )
        self.mujoco_gl = mujoco_gl
        self._camera_manager = RoboCasaCameraManager(camera_config)
        self._observation_writer = self._make_observation_writer(workspace, observation_config)
        self._environment: Any | None = None
        self._convert_action: Callable[[np.ndarray], dict[str, Any]] | None = None
        self._ready = False
        self._action_steps = 0
        self._clear_observation()

    @property
    def latest_obs(self) -> dict[str, Any]:
        """Latest native Gym observation; arrays must be treated as read-only."""
        return self._latest_obs

    @property
    def instruction(self) -> str:
        return str(self._latest_obs.get("annotation.human.task_description", ""))

    @property
    def action_steps(self) -> int:
        """Successful Gym step calls since reset; close() clears this count.

        No budget is applied here. The evaluation subclass retains the final
        episode's count after close().
        """
        return self._action_steps

    @property
    def last_gripper_close(self) -> bool | None:
        """Last applied Gym open/close command, not evidence of a grasp.

        None means no reliable command since reset, close or a physics error.
        The Gym wrapper closes at action12[6] >= 0.5 and opens below it.
        """
        return self._last_gripper_close

    @property
    def sim(self) -> Any:
        """Underlying MuJoCo simulation for later calibration/debugging tools."""
        return self._require_environment().unwrapped.sim

    @property
    def cameras(self) -> dict[str, RoboCasaCameraFrame]:
        """Latest configured RGB/calibration snapshots; empty when invalidated."""
        return self._camera_manager.frames

    @property
    def observation_manifest_path(self) -> Path | None:
        """Configured output path; its existence indicates a complete publication."""
        return None if self._observation_writer is None else self._observation_writer.manifest_path

    def create(self) -> None:
        """Replace any owned environment and initialize its first episode."""
        self.close()
        self._invalidate_published_observation()
        # Rendering backends must be selected before importing MuJoCo/robosuite.
        os.environ.setdefault("MUJOCO_GL", self.mujoco_gl)
        importlib.import_module("robocasa")  # Registers robocasa/<task> with Gym.
        gym = importlib.import_module("gymnasium")
        env_utils = importlib.import_module("robocasa.utils.env_utils")
        self._environment = gym.make(
            f"robocasa/{self.env_name}",
            split=self.split,
            seed=self.seed,
            **deepcopy(self.env_kwargs),
        )
        self._convert_action = env_utils.convert_action
        try:
            # Register before the existing explicit reset, which rebuilds the
            # model. This adds no resets or simulator steps to the native flow.
            self._camera_manager.install(self._environment)
            self.reset()
        except BaseException:
            try:
                self.close()
            except BaseException:
                # Keep the initialization error as the caller-visible failure.
                logger.exception("Could not close RoboCasa environment after initialization failed")
            raise

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset once; only an explicitly supplied seed reseeds the Gym env."""
        environment = self._require_environment()
        self._validate_seed(seed)
        self._clear_observation()
        self._action_steps = 0
        if self._observation_writer is not None:
            self._observation_writer.start_episode()
        if seed is None:
            obs, info = environment.reset()
        else:
            obs, info = environment.reset(seed=int(seed))
        self._store_result(obs, 0.0, False, False, info)
        try:
            # A reset may select another kitchen. Plan once for its fixtures,
            # then retain those world-fixed perception views for this episode.
            self._camera_manager.apply_auto_rig(environment)
            self.refresh_cameras()
        except BaseException:
            self._clear_observation()
            raise
        return self._latest_obs, dict(self._latest_info)

    def step(self, action12: Any, *, publish_observation: bool = True) -> StepResult:
        """Execute one policy-format action without clipping or re-encoding it.

        Ordering is xyz(3), rotation(3), gripper(1), base(3), torso(1), mode(1).
        RoboCasa's converter and Gym wrapper retain responsibility for gripper
        and control-mode thresholds. This method returns Gym's five values.

        publish_observation refreshes camera caches and, if enabled, writes a
        snapshot. False invalidates the caches while retaining the last published
        files until refresh_cameras(). Capture/write errors occur after the action
        was applied and counted; callers must not retry it as if it never executed.
        """
        environment = self._require_ready()
        if self._latest_terminated or self._latest_truncated:
            raise RuntimeError("RoboCasa environment ended; reset before stepping again")
        action = np.asarray(action12)
        if (
            action.shape != (12,)
            or not np.issubdtype(action.dtype, np.number)
            or np.iscomplexobj(action)
            or not np.all(np.isfinite(action))
        ):
            raise ValueError("RoboCasa action must be a vector of exactly 12 finite real values")
        if self._convert_action is None:
            raise RuntimeError("RoboCasa action converter is not initialized")
        gripper_close = bool(action[6] >= 0.5)
        action_dict = self._convert_action(action.copy())
        self._camera_manager.clear()
        try:
            obs, reward, terminated, truncated, info = environment.step(action_dict)
        except BaseException:
            # A failed simulator step may already have changed physics state.
            # Require reset rather than exposing the preceding observation as fresh.
            self._clear_observation()
            try:
                self._invalidate_published_observation()
            except Exception:
                logger.exception("Could not invalidate observation after simulator step failed")
            raise
        self._action_steps += 1
        # Commit alongside the completed Gym step, before observation/status I/O.
        # VLA and future geometric actions therefore share one command history.
        self._last_gripper_close = gripper_close
        self._store_result(obs, reward, terminated, truncated, info)
        if publish_observation:
            self.refresh_cameras()
        return (
            self._latest_obs,
            self._latest_reward,
            self._latest_terminated,
            self._latest_truncated,
            dict(self._latest_info),
        )

    def check_success(self) -> bool:
        """Latest Gym success signal, cleared on reset/close and never latched.

        The evaluation subclass instead latches success for the episode.
        """
        return bool(self._latest_info.get("success", False))

    def get_eef_pose_relative(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stock base-relative position and XYZW quaternion.

        Preserve RoboCasa's legacy observation convention: position is from the
        EEF site, while orientation is from the EEF body. This pair must not be
        treated as a calibrated site transform for future geometric controllers.
        """
        return (
            self._observation_vector("state.end_effector_position_relative", 3),
            self._observation_vector("state.end_effector_rotation_relative", 4),
        )

    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the mobile base center's world position and XYZW quaternion."""
        return (
            self._observation_vector("state.base_position", 3),
            self._observation_vector("state.base_rotation", 4),
        )

    def get_eef_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the grip site's world position and XYZW quaternion, as copies.

        Both components come from the same current MuJoCo site. Resolve the
        robot on each call: a hard reset can replace robots, controllers and IDs.
        This read neither updates controller goals nor advances simulation.
        """
        site_id = self._right_robot().eef_site_id["right"]
        if (
            isinstance(site_id, (bool, np.bool_))
            or not isinstance(site_id, (int, np.integer))
            or not 0 <= site_id < len(self.sim.data.site_xpos)
        ):
            raise RuntimeError("EEF grip site ID is invalid in the current model")
        transform = self._pose_transform(
            self.sim.data.site_xpos[site_id],
            np.asarray(self.sim.data.site_xmat[site_id]).reshape(3, 3),
            "EEF grip site",
        )
        return transform[:3, 3].copy(), self._rotation_to_quaternion(transform[:3, :3])

    def get_arm_control_frame(self) -> np.ndarray:
        """Return T_world_control, the right arm's base reference in world.

        This part-specific frame need not equal the mobile base center returned
        by get_base_pose(). It is read from current simulation data using the
        composite controller's own API, without calling update_state().
        """
        composite = self._right_robot().composite_controller
        position, rotation = composite.get_controller_base_pose("right")
        return self._pose_transform(position, rotation, "right arm control frame")

    def get_arm_control_spec(self) -> dict[str, Any]:
        """Return detached JSON-compatible limits for fixed-impedance delta OSC.

        Validate on demand so unsupported geometric control modes do not alter
        native environment construction or VLA stepping. Limits are per-axis
        controller values, not assumed 0.05 m / 0.5 rad defaults. The control
        frame describes the base origin even when input_ref_frame is 'world'.
        """
        robot = self._right_robot()
        controller = robot.composite_controller.part_controllers["right"]
        if (
            controller.name != "OSC_POSE"
            or controller.control_dim != 6
            or controller.impedance_mode != "fixed"
            or controller.input_type != "delta"
            or controller.input_ref_frame not in {"base", "world"}
        ):
            raise RuntimeError(
                "geometric control requires a 6-D fixed-impedance delta OSC_POSE "
                "controller with base or world inputs"
            )
        site_name = controller.ref_name
        if (
            not isinstance(site_name, str)
            or self.sim.model.site_name2id(site_name) != robot.eef_site_id["right"]
        ):
            raise RuntimeError("right arm controller reference must match the EEF grip site")
        limits = {}
        for key in ("input_min", "input_max", "output_min", "output_max"):
            value = np.asarray(getattr(controller, key), dtype=np.float64)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError(f"right arm controller {key} must contain 6 finite values")
            limits[key] = value
        for prefix in ("input", "output"):
            if np.any(limits[f"{prefix}_min"] >= limits[f"{prefix}_max"]):
                raise ValueError(f"right arm controller {prefix} ranges must be strictly increasing")
        return {
            "controller_type": controller.name,
            "input_type": controller.input_type,
            "input_ref_frame": controller.input_ref_frame,
            "impedance_mode": controller.impedance_mode,
            "control_dim": int(controller.control_dim),
            "eef_site_name": site_name,
            **{key: value.tolist() for key, value in limits.items()},
        }

    def get_gripper_qpos(self) -> np.ndarray:
        return self._observation_vector("state.gripper_qpos", 2)

    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Return a configured camera's RGB, or an unconfigured native Gym image.

        Configured views use their requested capture dimensions. Native policy
        images in latest_obs retain their original dimensions and pixel values.
        """
        self._require_ready()
        name = camera_name.removeprefix("video.")
        if self._camera_manager.has_camera(name):
            return self._camera_manager.get_frame(name).rgb
        key = camera_name if camera_name.startswith("video.") else f"video.{camera_name}"
        if key not in self._latest_obs:
            raise KeyError(f"RoboCasa observation is missing camera {key!r}")
        return np.asarray(self._latest_obs[key])

    def get_camera_frame(self, camera_name: str) -> RoboCasaCameraFrame:
        """Return one configured RGB/calibration snapshot without rendering."""
        self._require_ready()
        return self._camera_manager.get_frame(camera_name.removeprefix("video."))

    def refresh_cameras(self) -> dict[str, RoboCasaCameraFrame]:
        """Capture configured views and calibration without advancing physics.

        The controller must call this synchronously with respect to reset/step.
        With output enabled, publish the complete cache through the writer.
        Publication errors preserve the previous complete files for this episode.
        """
        self._require_ready()
        # Keep the unconfigured path independent of simulator camera APIs.
        if not self._camera_manager.names:
            return {}
        frames = self._camera_manager.refresh(self.sim, self._latest_obs)
        if self._observation_writer is not None:
            self._observation_writer.write(frames)
        return frames

    def render(self) -> np.ndarray:
        """Return Gym's cached replay image without advancing simulation."""
        return np.asarray(self._require_ready().render())

    def get_robot_state(self) -> dict[str, Any]:
        """Return an environment-side, JSON-serializable robot state summary.

        Here, done is the native terminated-or-truncated flag. The evaluation
        subclass exposes episode completion as done, while preserving both
        native flags. EMERGE schema adaptation belongs in the driver.
        """
        position, quaternion = self.get_eef_pose_relative()
        base_position, base_quaternion = self.get_base_pose()
        world_position, world_quaternion = self.get_eef_pose_world()
        return {
            "type": "panda_omron",
            "task": {
                "env_name": self.env_name,
                "split": self.split,
                "instruction": self.instruction,
            },
            "eef_pose_relative": {
                "frame": "robot_base",
                "position_m": position.tolist(),
                "orientation_quat_xyzw": quaternion.tolist(),
                "orientation_reference": "eef_body",
            },
            "eef_pose_world": {
                "frame": "world",
                "reference": "grip_site",
                "position_m": world_position.tolist(),
                "orientation_quat_xyzw": world_quaternion.tolist(),
            },
            "base_pose": {
                "frame": "world",
                "position_m": base_position.tolist(),
                "orientation_quat_xyzw": base_quaternion.tolist(),
            },
            "gripper_qpos": self.get_gripper_qpos().tolist(),
            "last_gripper_close": self.last_gripper_close,
            "action_steps": self._action_steps,
            "reward": self._latest_reward,
            "terminated": self._latest_terminated,
            "truncated": self._latest_truncated,
            "done": self._latest_terminated or self._latest_truncated,
            "success": self.check_success(),
        }

    def is_connected(self) -> bool:
        return self._environment is not None and self._ready

    def close(self) -> None:
        """Release resources, retaining the last published files for inspection."""
        environment, self._environment = self._environment, None
        self._camera_manager.close()
        self._convert_action = None
        self._action_steps = 0
        self._clear_observation()
        if environment is not None:
            environment.close()

    def _make_observation_writer(
        self, workspace: str | Path | None, config: dict[str, Any] | None
    ) -> RoboCasaObservationWriter | None:
        if config is not None and not isinstance(config, dict):
            raise ValueError("observation_config must be a mapping")
        config = dict(config or {})
        unknown = set(config) - {"enabled", "reference_view", "camera_names"}
        if unknown:
            raise ValueError(f"unsupported observation settings: {', '.join(sorted(unknown))}")
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("observation enabled must be a boolean")
        if not enabled:
            return None
        if workspace is None:
            raise ValueError("workspace is required when observation output is enabled")
        writer = RoboCasaObservationWriter(
            workspace,
            camera_names=config.get("camera_names", self._camera_manager.names),
            reference_view=config.get("reference_view", ""),
        )
        unknown_cameras = set(writer.camera_names) - set(self._camera_manager.names)
        if unknown_cameras:
            raise ValueError(f"observation cameras are not configured: {', '.join(sorted(unknown_cameras))}")
        return writer

    def _invalidate_published_observation(self) -> None:
        """Remove stale scene publication before rebuilding or after invalid physics."""
        if self._observation_writer is not None:
            self._observation_writer.invalidate()

    def _clear_observation(self) -> None:
        self._camera_manager.clear()
        self._ready = False
        self._latest_obs: dict[str, Any] = {}
        self._latest_reward = 0.0
        self._latest_terminated = False
        self._latest_truncated = False
        self._latest_info: dict[str, Any] = {}
        self._last_gripper_close: bool | None = None

    def _right_robot(self) -> Any:
        """Resolve the current single-robot PandaOmron adapter, without caching."""
        robots = self._require_ready().unwrapped.robots
        if len(robots) != 1 or "right" not in robots[0].arms:
            raise RuntimeError("RoboCasa geometry access requires one robot with a right arm")
        return robots[0]

    @staticmethod
    def _pose_transform(position: Any, rotation: Any, source: str) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{source} must have a finite 3-D position")
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{source} must have a proper 3x3 rotation")
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        """Convert a validated rotation to XYZW, including rotations near pi.

        The principal eigenvector avoids a trace-only division at 180 degrees.
        Keep this NumPy-only so reading geometry does not import model packages.
        """
        (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = rotation
        matrix = np.array([
            [xx - yy - zz, xy + yx, xz + zx, zy - yz],
            [xy + yx, yy - xx - zz, yz + zy, xz - zx],
            [xz + zx, yz + zy, zz - xx - yy, yx - xy],
            [zy - yz, xz - zx, yx - xy, xx + yy + zz],
        ]) / 3.0
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        quaternion = eigenvectors[:, np.argmax(eigenvalues)].copy()
        if quaternion[3] < 0:
            quaternion = -quaternion
        return quaternion

    def _store_result(
        self,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._latest_obs = dict(obs)
        self._latest_reward = float(reward)
        self._latest_terminated = bool(terminated)
        self._latest_truncated = bool(truncated)
        self._latest_info = dict(info)
        self._ready = True

    def _require_environment(self) -> Any:
        if self._environment is None:
            raise RuntimeError("RoboCasa environment has not been created; call create() first")
        return self._environment

    def _require_ready(self) -> Any:
        environment = self._require_environment()
        if not self._ready:
            raise RuntimeError("RoboCasa observation is unavailable; reset the environment")
        return environment

    def _observation_vector(self, key: str, length: int) -> np.ndarray:
        self._require_ready()
        if key not in self._latest_obs:
            raise KeyError(f"RoboCasa observation is missing {key!r}")
        value = np.asarray(self._latest_obs[key], dtype=np.float64)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"RoboCasa observation {key!r} must contain {length} finite values")
        return value.copy()

    @staticmethod
    def _validate_seed(seed: int | None) -> None:
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
