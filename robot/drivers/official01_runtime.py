"""Run the external Executive with stable arm targets across navigation phases.

The external implementation is loaded into this private subprocess. No files in
its checkout are modified. Its parser and control loop remain in charge.
Navigation retains its arm reference and verifies resting arm tracking;
pose preparation retains the original reference-pose gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class NavigationHoldMixin:
    """Keep navigation arm references and distinguish rest from pose attainment.

    SONIC imitates a reference through a learned decoder; its motor targets can
    differ from that reference. Navigation rest checks attained reference or
    arm motor tracking, together with low velocity. ADJUSTING still uses the
    external reference-pose gate.
    """

    def _handle_command(self, command: Any) -> None:
        preserve = command.name == "move_keyboard" and self.state == "READY"
        target = self.arm_hold.copy() if preserve else None
        try:
            super()._handle_command(command)
        finally:
            if target is not None:
                self.arm_hold = target
                if self.state == "MOVING":
                    self._stable_since = None

    def _stable(self, debug: dict[str, np.ndarray]) -> bool:
        if self.state != "MOVING":
            return super()._stable(debug)
        self._stable_since = getattr(self, "_stable_since", None)
        if not self.stop_requested or np.any(self.nav):
            self._stable_since = None
            return False
        feedback = self.body.read()
        motor_target = feedback.get("last_action") if isinstance(
            feedback, dict) else None
        try:
            motor_target = np.asarray(motor_target, dtype=float)
        except (TypeError, ValueError):
            motor_target = np.empty(0)
        if motor_target.shape != (29,) or not np.all(np.isfinite(motor_target)):
            self._stable_since = None
            return False

        roll, pitch = self._roll_pitch(debug["base_quat"])
        arm_motor_error = float(
            np.max(np.abs(debug["measured"][15:29] - motor_target[15:29])))
        lower_error = float(
            np.max(np.abs(debug["measured"][:12] - debug["target"][:12])))
        arm_speed = float(np.max(np.abs(debug["dq"][15:29])))
        reference_error = float(
            np.max(np.abs(debug["measured"][15:29] - self.arm_hold)))
        # A learned imitation reference and a torque-producing motor target
        # are different contracts. Accept the original reference gate when the
        # pose is reached; otherwise require motor tracking. Both paths must
        # be at rest and pass the unchanged body guards.
        arm_tracking = min(reference_error,
                           arm_motor_error) <= self.args.ready_error_rad
        stable = (arm_tracking and lower_error <= self.args.ready_error_rad and
                  arm_speed <= 0.15 and
                  max(abs(roll), abs(pitch)) <= self.args.ready_tilt_rad and
                  float(np.linalg.norm(
                      debug["base_ang_vel"])) <= self.args.ready_gyro_rad_s)
        now = time.monotonic()
        if not stable:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = now
        settled = now - self._stable_since >= self.args.stable_hold_s
        if settled:
            print(
                "[integration] navigation rest verified: "
                f"arm_motor_error={arm_motor_error:.3f}rad "
                f"arm_speed={arm_speed:.3f}rad/s "
                f"arm_reference_error={reference_error:.3f}rad; "
                "grasp preparation remains a separate check",
                flush=True,
            )
        return settled


def main() -> int:
    """Load the configured runtime and retain its CLI validation and shutdown."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-root", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    path = args.control_root.expanduser().resolve() / (
        "integrations/official01_real_deployment/official01_executive.py")
    spec = importlib.util.spec_from_file_location("_emerge_official01_runtime",
                                                  path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Executive: {path}")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)

    class IntegrationExecutive(NavigationHoldMixin,
                               runtime.Official01Executive):
        """Use the native FSM with a persistent navigation arm hold."""

    runtime.Official01Executive = IntegrationExecutive
    sys.argv = [str(path), *remaining]
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
