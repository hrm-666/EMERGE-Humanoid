"""Run the unchanged apple-table owner with LightNav-local runtime telemetry.

The upstream MuJoCo scene and runner remain the sole physics owner.  This
wrapper only adds a runtime root offset, a pose PUB stream, and bounded viewer
refresh; it never writes the upstream XML or launcher.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from importlib import import_module

import msgpack
import mujoco
import numpy as np
import zmq

from .tabletop_reset import DEFAULT_RESET_ENDPOINT
from .tabletop_reset import DEFAULT_START_DISTANCE_M
from .tabletop_start import RandomStartSampler
from .tabletop_start import apply_start
from .tabletop_start import table_footprint


class SimulationResetServer:
    """Reset G1 on the physics thread, keeping scene objects and time intact."""

    def __init__(self,
                 runner,
                 endpoint: str,
                 start_sampler=None,
                 radius_m=None) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("simulation reset requires a loopback endpoint")
        self.runner = runner
        model = runner.model
        bodies = {model.body("pelvis").id}
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in bodies:
                bodies.add(body_id)
        self.qpos_indices, self.qvel_indices = [], []
        sizes = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
        for joint_id in range(model.njnt):
            if int(model.jnt_bodyid[joint_id]) in bodies:
                nq, nv = sizes[int(model.jnt_type[joint_id])]
                qa, va = int(model.jnt_qposadr[joint_id]), int(
                    model.jnt_dofadr[joint_id])
                self.qpos_indices.extend(range(qa, qa + nq))
                self.qvel_indices.extend(range(va, va + nv))
        self.initial_qpos = runner.data.qpos[self.qpos_indices].copy()
        self.initial_qvel = runner.data.qvel[self.qvel_indices].copy()
        self.start_sampler = start_sampler
        self.radius_m = radius_m
        self.next_start = None
        if start_sampler is not None:
            qa = _free_pelvis_qpos_adr(model)
            self.fixed_config = {
                "mode": "fixed",
                "position_xy_m": runner.data.qpos[qa:qa + 2].tolist()
            }
            runner.start_config = self.fixed_config.copy()
            if radius_m is not None:
                runner.start_config = start_sampler.sample(radius_m)
                apply_start(runner.data.qpos, _free_pelvis_qpos_adr(model),
                            runner.start_config)
                mujoco.mj_forward(model, runner.data)
                if runner.elastic_band is not None:
                    runner.elastic_band.point = runner.data.xpos[
                        runner.band_body_id].copy()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)
        self.phase = "idle"
        self.request_id = None
        self.deadline = 0.0
        self.support_enabled = False
        print(f"reset_server={endpoint}", flush=True)

    def tick(self) -> None:
        now = time.monotonic()
        if self.phase == "idle":
            try:
                request = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            except ValueError:
                self.socket.send_json({
                    "ok": False,
                    "error": "invalid reset request"
                })
                return
            if not isinstance(request,
                              dict) or request.get("action") != "reset":
                self.socket.send_json({
                    "ok": False,
                    "error": "expected reset action"
                })
                return
            expires = request.get("expires_at")
            if not isinstance(
                    expires,
                (int, float)) or not time.time() <= expires <= time.time() + 10:
                self.socket.send_json({
                    "ok": False,
                    "error": "expired reset request"
                })
                return
            self.request_id = request.get("request_id")
            if not isinstance(self.request_id, str):
                self.socket.send_json({
                    "ok": False,
                    "error": "missing reset request_id"
                })
                return
            try:
                mode = request.get("start_mode", "current")
                if mode not in ("current", "random", "fixed"):
                    raise ValueError(
                        "start_mode must be current, random or fixed")
                if self.start_sampler is None and mode != "current":
                    raise ValueError(
                        "restart sim to enable random-start support")
                radius = request.get(
                    "radius_m",
                    self.radius_m) if mode == "random" else self.radius_m
                if mode == "fixed":
                    radius = None
                if self.start_sampler is not None:
                    self.next_start = (self.fixed_config.copy() if radius
                                       is None else self.start_sampler.sample(
                                           float(radius),
                                           request.get("sample_seed")))
                    self.radius_m = radius
            except (ValueError, TypeError, OverflowError) as exc:
                self.socket.send_json({
                    "ok": False,
                    "request_id": self.request_id,
                    "error": str(exc)
                })
                return
            self.phase, self.deadline = "stopping", now + 0.75
            return
        if now < self.deadline:
            return
        runner = self.runner
        if self.phase == "stopping":
            lock = runner.viewer.lock(
            ) if runner.viewer is not None else nullcontext()
            with lock:
                runner.data.qpos[self.qpos_indices] = self.initial_qpos
                if self.next_start is not None:
                    if self.next_start["mode"] == "random":
                        apply_start(runner.data.qpos,
                                    _free_pelvis_qpos_adr(runner.model),
                                    self.next_start)
                    runner.start_config = self.next_start
                runner.data.qvel[self.qvel_indices] = self.initial_qvel
                runner.data.qacc_warmstart[:] = 0
                runner.data.qfrc_applied[self.qvel_indices] = 0
                mujoco.mj_forward(runner.model, runner.data)
                if runner.elastic_band is not None:
                    self.support_enabled = bool(runner.elastic_band.enable)
                    runner.elastic_band.point = runner.data.xpos[
                        runner.band_body_id].copy()
                    runner.elastic_band.enable = True
            self.phase, self.deadline = "settling", now + 1.0
            return
        if runner.elastic_band is not None:
            runner.elastic_band.enable = self.support_enabled
        qa = _free_pelvis_qpos_adr(runner.model)
        self.socket.send_json({
            "ok": True,
            "request_id": self.request_id,
            "wall_time_s": time.time(),
            "position_m": runner.data.qpos[qa:qa + 3].tolist(),
            "support_enabled": self.support_enabled,
            "start_config": getattr(runner, "start_config", None),
        })
        print(f"SIM_RESET complete; support_enabled={self.support_enabled}",
              flush=True)
        self.phase = "idle"

    def close(self) -> None:
        if self.phase == "settling" and self.runner.elastic_band is not None:
            self.runner.elastic_band.enable = self.support_enabled
        self.socket.close()
        self.context.term()


class RateLimitedViewer:
    """Forward the original viewer API, syncing display at a bounded rate."""

    def __init__(self, viewer, refresh_hz: float) -> None:
        self._viewer = viewer
        self._period_s = 1 / refresh_hz
        self._next_sync = 0.0
        self.sync_seconds = 0.0
        self.sync_count = 0

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def sync(self, *args, **kwargs) -> None:
        started = time.monotonic()
        if started < self._next_sync:
            return
        self._viewer.sync(*args, **kwargs)
        finished = time.monotonic()
        self.sync_seconds += finished - started
        self.sync_count += 1
        self._next_sync = finished + self._period_s


class PosePublisher:
    """Publish the current floating-base pose for LightNav-side event logs."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)
        self.count = 0
        print(f"pose_pub={endpoint}", flush=True)

    def publish(self, runner, qpos_adr: int) -> None:
        qpos = np.asarray(runner.data.qpos[qpos_adr:qpos_adr + 7], dtype=float)
        payload = {
            "wall_time_s": time.time(),
            "sim_time_s": float(runner.data.time),
            "position_m": qpos[:3].tolist(),
            "quaternion_wxyz": qpos[3:7].tolist(),
            "qpos": runner.data.qpos.tolist(),
            "start_config": getattr(runner, "start_config", None),
        }
        self.socket.send(msgpack.packb(payload, use_bin_type=True),
                         flags=zmq.NOBLOCK)
        self.count += 1

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class SimulationTiming:
    """Read-only wall/physics clock diagnostics, independent of the controller."""

    def __init__(self, sim_time_s: float, wall_time_s: float) -> None:
        self.sim_time_s = sim_time_s
        self.wall_time_s = wall_time_s
        self.steps = 0
        self.step_seconds = 0.0

    def record(self, sim_time_s: float, step_seconds: float,
               wall_time_s: float) -> str | None:
        self.steps += 1
        self.step_seconds += step_seconds
        elapsed = wall_time_s - self.wall_time_s
        if elapsed < 5.0:
            return None
        ratio = (sim_time_s - self.sim_time_s) / elapsed
        message = (
            f"SIM_TIMING realtime_ratio={ratio:.3f} physics_hz={self.steps / elapsed:.1f} "
            f"mean_step_ms={1000 * self.step_seconds / self.steps:.2f}")
        if ratio < 0.8:
            message += " timing=BELOW_REALTIME; SONIC runs on the wall clock"
        self.sim_time_s = sim_time_s
        self.wall_time_s = wall_time_s
        self.steps = 0
        self.step_seconds = 0.0
        return message


def _free_pelvis_qpos_adr(model) -> int:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise KeyError("tabletop wrapper requires a pelvis body")
    for joint_id in range(model.njnt):
        if (int(model.jnt_bodyid[joint_id]) == pelvis_id and int(
                model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)):
            return int(model.jnt_qposadr[joint_id])
    raise KeyError("tabletop wrapper requires a free pelvis joint")


def _table_front_x(model, data) -> tuple[float, float]:
    """Return the nearest x edge of the tabletop and its centre y.

    The tabletop box is read from the loaded model at runtime.  This keeps
    the runtime offset independent of XML constants while preserving the
    original table pose and geometry.
    """

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 "tabletop_lab_table")
    if table_id < 0:
        raise KeyError("tabletop wrapper requires tabletop_lab_table")
    candidates = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != table_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id],
                              dtype=float).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        half_x = abs(rotation[0, 0]) * size[0] + abs(rotation[0, 1]) * size[1]
        candidates.append(
            (float(center[2]), float(center[0] - half_x), float(center[1])))
    if not candidates:
        raise KeyError("tabletop wrapper could not find a box tabletop")
    _, front_x, centre_y = max(candidates)
    return front_x, centre_y


def _set_start_pose(runner, distance_m: float) -> tuple[float, float]:
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("--start-distance-m must be finite and positive")
    qpos_adr = _free_pelvis_qpos_adr(runner.model)
    front_x, centre_y = _table_front_x(runner.model, runner.data)
    runner.data.qpos[qpos_adr] = front_x - distance_m
    runner.data.qpos[qpos_adr + 1] = centre_y
    mujoco.mj_forward(runner.model, runner.data)
    if runner.elastic_band is not None:
        runner.elastic_band.point = np.asarray(
            runner.data.xpos[runner.band_body_id], dtype=np.float64).copy()
    return float(runner.data.qpos[qpos_adr]), float(runner.data.qpos[qpos_adr +
                                                                     1])


def main() -> int:
    """Run the existing simulation owner with task-local pose and reset hooks."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    parser.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--start-distance-m",
                        type=float,
                        default=DEFAULT_START_DISTANCE_M)
    parser.add_argument("--reset-endpoint", default=DEFAULT_RESET_ENDPOINT)
    parser.add_argument("--random-start-radius-m", type=float)
    parser.add_argument("--random-start-seed", type=int)
    args, original_args = parser.parse_known_args()
    if not math.isfinite(args.viewer_hz) or not 1 <= args.viewer_hz <= 120:
        parser.error("--viewer-hz must be between 1 and 120")
    if not math.isfinite(args.start_distance_m) or args.start_distance_m <= 0:
        parser.error("--start-distance-m must be positive")

    original = import_module("p5_sim_observation_server")
    import mujoco.viewer

    launch_passive = mujoco.viewer.launch_passive
    original_runner = original.P5MuJoCoRunner

    def launch_viewer(*args_, **kwargs):
        return RateLimitedViewer(launch_passive(*args_, **kwargs),
                                 args.viewer_hz)

    class TimedRunner(original_runner):

        def __init__(self, *args_, **kwargs):
            super().__init__(*args_, **kwargs)
            self._timing = SimulationTiming(float(self.data.time),
                                            time.monotonic())
            self._pose_qpos_adr = _free_pelvis_qpos_adr(self.model)
            self._start_x, self._start_y = _set_start_pose(
                self, args.start_distance_m)
            self.pose_publisher = PosePublisher(args.pose_endpoint)
            sampler = RandomStartSampler(table_footprint(self.model, self.data),
                                         args.random_start_seed)
            self.reset_server = SimulationResetServer(
                self, args.reset_endpoint, sampler, args.random_start_radius_m)
            self._start_x, self._start_y = self.data.qpos[
                self._pose_qpos_adr:self._pose_qpos_adr + 2]
            print(f"START_CONFIG={self.start_config}", flush=True)
            print(
                f"START_DISTANCE_M={args.start_distance_m:g} "
                f"START_POSITION_M=({self._start_x:.3f}, {self._start_y:.3f})",
                flush=True,
            )

        def step(self):
            started = time.monotonic()
            self.reset_server.tick()
            super().step()
            self.pose_publisher.publish(self, self._pose_qpos_adr)
            finished = time.monotonic()
            message = self._timing.record(float(self.data.time),
                                          finished - started, finished)
            if message is not None:
                print(message, flush=True)

        def close(self):
            try:
                try:
                    self.reset_server.close()
                finally:
                    self.pose_publisher.close()
            finally:
                super().close()

    mujoco.viewer.launch_passive = launch_viewer
    original.P5MuJoCoRunner = TimedRunner
    original_argv = sys.argv
    sys.argv = [original.__file__, *original_args]
    print(
        f"SIM_VIEWER refresh_hz={args.viewer_hz:g}; original physics and scene preserved",
        flush=True,
    )
    try:
        return original.main()
    finally:
        sys.argv = original_argv
        original.P5MuJoCoRunner = original_runner
        mujoco.viewer.launch_passive = launch_passive


if __name__ == "__main__":
    raise SystemExit(main())
