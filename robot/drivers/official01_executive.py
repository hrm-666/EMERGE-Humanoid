"""Restricted subprocess client for the existing Official01 text executive.

The client deliberately exposes named operations instead of a generic shell or
stdin method.  The executive remains the sole owner of SONIC and Inspire
publishers, and its native state machine validates every command.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable
from typing import Iterable


class ExecutiveClientError(RuntimeError):
    """The managed executive rejected a command or became unavailable."""


_STATE_RE = re.compile(r"STATE\s*=\s*(READY|ADJUSTING|INFERENCING|MOVING)")
_RESOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REAL_CONFIRMATION = "I_ACKNOWLEDGE_REAL_G1_CONTROL"


class ExecutiveProcessClient:
    """Own one legacy executive process and observe its line protocol."""

    def __init__(
        self,
        control_root: str | Path,
        *,
        runtime: str = "sim",
        startup_timeout_s: float = 90.0,
        command_watchdog_s: float = 0.40,
        linear_speed_mps: float = 0.12,
        yaw_rate_rad_s: float = 0.25,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.control_root = Path(control_root).expanduser().resolve()
        if runtime not in {"sim", "real", "dry-run"}:
            raise ValueError("runtime must be sim, real, or dry-run")
        self.runtime = runtime
        self.startup_timeout_s = self._positive(startup_timeout_s,
                                                "startup_timeout_s")
        self.command_watchdog_s = self._positive(command_watchdog_s,
                                                 "command_watchdog_s")
        self.linear_speed_mps = self._positive(linear_speed_mps,
                                               "linear_speed_mps")
        self.yaw_rate_rad_s = self._positive(yaw_rate_rad_s, "yaw_rate_rad_s")
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._condition = threading.Condition()
        self._state = "STARTING"
        self._state_version = 0
        self._line_version = 0
        self._lines: deque[tuple[int, str]] = deque(maxlen=300)

    @staticmethod
    def _positive(value: float, name: str) -> float:
        result = float(value)
        if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        argv = self._build_argv()
        self._process = self._popen_factory(
            argv,
            cwd=str(self.control_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._runtime_env(),
            start_new_session=False,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name="official01-executive-output",
            daemon=True,
        )
        self._reader.start()
        self.wait_for_state({"READY"}, timeout_s=self.startup_timeout_s)

    def _build_argv(self) -> list[str]:
        common = [
            "--keyboard-slow-duration-s",
            str(self.command_watchdog_s),
            "--keyboard-normal-duration-s",
            str(self.command_watchdog_s),
            "--keyboard-yaw-duration-s",
            str(self.command_watchdog_s),
            "--keyboard-slow-linear-speed",
            str(self.linear_speed_mps),
            "--keyboard-normal-linear-speed",
            str(self.linear_speed_mps),
            "--keyboard-yaw-rate",
            str(self.yaw_rate_rad_s),
        ]
        if self.runtime == "sim":
            core = Path(
                os.environ.get("GR00T_ROOT",
                               Path.home() /
                               "GR00T-WholeBodyControl")).expanduser()
            script = self.control_root / "integrations/official01_real_deployment/official01_executive.py"
            resources = self.control_root / "integrations/official01_real_deployment"
            argv = [
                str(core / ".venv_sim/bin/python"),
                str(Path(__file__).with_name("official01_runtime.py")),
                "--control-root",
                str(self.control_root),
                "--run",
                "--runtime",
                "sim",
                "--confirm-sim",
                "--network-interface",
                "lo",
                "--policy-host",
                "127.0.0.1",
                "--policy-port",
                "5550",
                "--state-host",
                "127.0.0.1",
                "--state-port",
                "5557",
                "--action-host",
                "127.0.0.1",
                "--action-port",
                "5556",
                "--observation-host",
                "127.0.0.1",
                "--observation-port",
                "5560",
                "--stale-timeout",
                "1.0",
                "--poses-dir",
                str(resources / "executive_poses/poses"),
                "--trajectories-dir",
                str(resources / "executive_poses/trajectories"),
                "--right2-body-prepare-config",
                str(resources / "body_prepare/right2_to_sonic_ready_v1.yaml"),
                *common,
            ]
        else:
            script = self.control_root / "integrations/official01_real_deployment/run_vla_executive_official01.sh"
            if self.runtime == "real":
                if os.environ.get(
                        "EMERGE_G1_CONFIRM_REAL") != _REAL_CONFIRMATION:
                    raise ExecutiveClientError(
                        "real robot control requires the operator to export "
                        f"EMERGE_G1_CONFIRM_REAL={_REAL_CONFIRMATION} before starting the Controller"
                    )
                argv = [
                    str(script), "--run", "--runtime", "real",
                    "--confirm-live-robot", *common
                ]
            else:
                argv = [str(script), "--dry-run", "--runtime", "sim", *common]
        if not script.is_file():
            raise ExecutiveClientError(
                f"executive launcher not found: {script}")
        return argv

    def _runtime_env(self) -> dict[str, str]:
        """Resolve existing policy dependencies without changing shared installs."""
        env = os.environ.copy()
        home = Path.home()
        core = Path(env.get("GR00T_ROOT",
                            home / "GR00T-WholeBodyControl")).expanduser()
        isaac = Path(env.get("ISAAC_ROOT",
                             home / "Isaac-GR00T-N1.7")).expanduser()
        paths = [
            isaac, core, core / "external_dependencies/unitree_sdk2_python",
            self.control_root, self.control_root / "src",
            self.control_root / "integrations/inspire_mujoco",
            self.control_root / "integrations/official01_sim_executive"
        ]
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                with self._condition:
                    self._line_version += 1
                    self._lines.append((self._line_version, line))
                    match = _STATE_RE.search(line)
                    if match:
                        self._state = match.group(1)
                        self._state_version += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                if self._state != "CLOSED":
                    self._state = "FAILED"
                    self._state_version += 1
                self._condition.notify_all()

    def _write(self, command: str) -> tuple[int, int]:
        process = self._process
        if process is None or process.poll(
        ) is not None or process.stdin is None:
            raise ExecutiveClientError("Official01 executive is not running")
        if any(ord(character) < 32 for character in command):
            raise ExecutiveClientError(
                "control characters in executive commands are forbidden")
        with self._condition:
            state_version = self._state_version
            line_version = self._line_version
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ExecutiveClientError(
                f"cannot send command to executive: {exc}") from exc
        return state_version, line_version

    def wait_for_state(
        self,
        states: Iterable[str],
        *,
        timeout_s: float,
        after_state_version: int | None = None,
        after_line_version: int | None = None,
    ) -> str:
        targets = set(states)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                rejection = self._rejection_after(after_line_version)
                if rejection:
                    raise ExecutiveClientError(rejection)
                changed = after_state_version is None or self._state_version > after_state_version
                if changed and self._state in targets:
                    return self._state
                process = self._process
                if process is not None and process.poll() is not None:
                    tail = " | ".join(
                        line for _, line in list(self._lines)[-8:])
                    raise ExecutiveClientError(
                        f"Official01 executive exited rc={process.returncode}; {tail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tail = " | ".join(
                        line for _, line in list(self._lines)[-8:])
                    raise ExecutiveClientError(
                        f"timed out waiting for state {sorted(targets)}; state={self._state}; {tail}"
                    )
                self._condition.wait(min(0.1, remaining))

    def _rejection_after(self, version: int | None) -> str | None:
        if version is None:
            return None
        for line_version, line in self._lines:
            if line_version <= version:
                continue
            if "command rejected:" in line or "policy worker stopped:" in line:
                return line.strip()
        return None

    def begin_inference(self, instruction: str, *, timeout_s: float) -> None:
        if not instruction.strip():
            raise ExecutiveClientError("GR00T instruction must be non-empty")
        try:
            shlex.split(f"infer {instruction.strip()}")
        except ValueError as exc:
            raise ExecutiveClientError(
                f"GR00T instruction has invalid quoting: {exc}") from exc
        state_version, line_version = self._write(
            f"infer {instruction.strip()}")
        self.wait_for_state(
            {"INFERENCING"},
            timeout_s=timeout_s,
            after_state_version=state_version,
            after_line_version=line_version,
        )

    def pause_inference(self, *, timeout_s: float) -> None:
        if self.state == "READY":
            return
        state_version, line_version = self._write("pause")
        self.wait_for_state(
            {"READY"},
            timeout_s=timeout_s,
            after_state_version=state_version,
            after_line_version=line_version,
        )

    def run_fixed(self, kind: str, name: str, *, timeout_s: float) -> None:
        if kind not in {"prepare", "pose", "trajectory"}:
            raise ExecutiveClientError(
                "fixed action kind must be prepare, pose, or trajectory")
        if not _RESOURCE_RE.fullmatch(name):
            raise ExecutiveClientError(
                "fixed action name contains forbidden characters")
        if kind == "prepare" and name != "right2":
            raise ExecutiveClientError(
                "the only validated preparation is right2")
        verb = {
            "prepare": "prepare",
            "pose": "goto",
            "trajectory": "traj"
        }[kind]
        state_version, line_version = self._write(f"pose {verb} {name}")
        self.wait_for_state(
            {"READY"},
            timeout_s=timeout_s,
            after_state_version=state_version,
            after_line_version=line_version,
        )

    def begin_moving(self, *, timeout_s: float) -> None:
        state_version, line_version = self._write("move keyboard")
        self.wait_for_state(
            {"MOVING"},
            timeout_s=timeout_s,
            after_state_version=state_version,
            after_line_version=line_version,
        )

    def send_nav_key(self, key: str) -> None:
        if key not in {"w", "a", "s", "d", "q", "e"}:
            raise ExecutiveClientError(
                f"unsupported bounded navigation key: {key!r}")
        if self.state != "MOVING":
            raise ExecutiveClientError(
                f"navigation key requires MOVING; state={self.state}")
        self._write(key)

    def stop_moving(self, *, timeout_s: float) -> None:
        if self.state == "READY":
            return
        if self.state != "MOVING":
            raise ExecutiveClientError(
                f"cannot stop locomotion from state {self.state}")
        state_version, line_version = self._write("stop")
        self.wait_for_state(
            {"READY"},
            timeout_s=timeout_s,
            after_state_version=state_version,
            after_line_version=line_version,
        )

    def recent_output(self, limit: int = 20) -> list[str]:
        with self._condition:
            return [line for _, line in list(self._lines)[-max(1, limit):]]

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.wait(timeout=8.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
        with self._condition:
            self._state = "CLOSED"
            self._state_version += 1
            self._condition.notify_all()
        self._process = None
