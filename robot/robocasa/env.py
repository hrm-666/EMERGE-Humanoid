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
from typing import Any, Callable

import numpy as np

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
    """

    def __init__(
        self,
        *,
        env_name: str,
        split: str | None = "target",
        seed: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
        mujoco_gl: str = "egl",
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
    def sim(self) -> Any:
        """Underlying MuJoCo simulation for later calibration/debugging tools."""
        return self._require_environment().unwrapped.sim

    def create(self) -> None:
        """Replace any owned environment and initialize its first episode."""
        self.close()
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
        if seed is None:
            obs, info = environment.reset()
        else:
            obs, info = environment.reset(seed=int(seed))
        self._store_result(obs, 0.0, False, False, info)
        return self._latest_obs, dict(self._latest_info)

    def step(self, action12: Any) -> StepResult:
        """Execute one policy-format action without clipping or re-encoding it.

        Ordering is xyz(3), rotation(3), gripper(1), base(3), torso(1), mode(1).
        RoboCasa's converter and Gym wrapper retain responsibility for gripper
        and control-mode thresholds. This method returns Gym's five values.
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
        action_dict = self._convert_action(action.copy())
        try:
            obs, reward, terminated, truncated, info = environment.step(action_dict)
        except BaseException:
            # A failed simulator step may already have changed physics state.
            # Require reset rather than exposing the preceding observation as fresh.
            self._clear_observation()
            raise
        self._action_steps += 1
        self._store_result(obs, reward, terminated, truncated, info)
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

    def get_gripper_qpos(self) -> np.ndarray:
        return self._observation_vector("state.gripper_qpos", 2)

    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Return native HWC RGB, already flipped by the RoboCasa Gym wrapper."""
        self._require_ready()
        key = camera_name if camera_name.startswith("video.") else f"video.{camera_name}"
        if key not in self._latest_obs:
            raise KeyError(f"RoboCasa observation is missing camera {key!r}")
        return np.asarray(self._latest_obs[key])

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
            "base_pose": {
                "frame": "world",
                "position_m": base_position.tolist(),
                "orientation_quat_xyzw": base_quaternion.tolist(),
            },
            "gripper_qpos": self.get_gripper_qpos().tolist(),
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
        """Release owned resources; repeated calls are safe."""
        environment, self._environment = self._environment, None
        self._convert_action = None
        self._action_steps = 0
        self._clear_observation()
        if environment is not None:
            environment.close()

    def _clear_observation(self) -> None:
        self._ready = False
        self._latest_obs: dict[str, Any] = {}
        self._latest_reward = 0.0
        self._latest_terminated = False
        self._latest_truncated = False
        self._latest_info: dict[str, Any] = {}

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
