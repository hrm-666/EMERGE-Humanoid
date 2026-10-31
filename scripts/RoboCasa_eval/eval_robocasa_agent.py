#!/usr/bin/env python3
"""Run one native RoboCasa episode with EMERGE as the decision maker.

The controller owns physics in its Python environment. The existing headless
runtime owns planning and specialists in another. Only the controller's private
evaluation status establishes benchmark success, never an agent's reply.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Emerge.runtime.protocol import RunRequest
from Emerge.runtime.storage import atomic_json, atomic_text
from Emerge.utils.helpers import sync_workspace_templates
from robot.mujoco_simulation.scene_io import (
    default_robot_state_doc, load_robot_state_doc, save_robot_state_doc,
)

from scripts.Libero_eval.eval_libero_agent import (
    _action_metrics as action_metrics,
    _load_json as read_json,
    _read_agent_result as read_agent_result,
    _runtime_environment as libero_runtime_environment,
    _should_continue_after_agent_exit,
    _terminate_process as stop_process,
)

read_status = read_json

EVALUATION_RULES = """## RoboCasa Evaluation Rules

Complete the supplied task autonomously through execute_robot_action.
Use the installed EMBODIED.md profile, planning tools and registered specialists.
Inspect fresh ROBOT_STATE.md after each action; action completion is not task success.
The official success signal is robots.robocasa.success. Visual judgments inform
planning but do not replace that signal. Stop on official success, done=true,
ready=false, or when no safe useful action remains. Never change state files
to manufacture success. Do not inspect files or processes outside this workspace.
"""


class EpisodeError(Exception):
    def __init__(self, reason: str, message: str, stage: str) -> None:
        super().__init__(message)
        self.reason, self.stage = reason, stage


def terminal_outcome(status: dict[str, Any]) -> tuple[str, str] | None:
    if status.get("success"):
        return "success", "success"
    if status.get("done"):
        reason = status.get("termination_reason") or "environment_done"
        if reason in {"error", "closed"}:
            return "infrastructure_error", str(reason)
        return "task_failure", str(reason)
    return None


def prepare_episode(args: argparse.Namespace, directory: Path) -> Path:
    """Allocate fresh files; API credentials stay exclusively in their existing config."""
    base = read_json(args.driver_config)
    config = deepcopy(base)
    workspace = directory / "workspace"
    # The CLI workspace must be the sole workspace used by controller/agent.
    config.pop("workspace", None)
    config["robocasa"].update(env_name=args.env_name, split=args.split, seed=args.seed)
    config.setdefault("vla", {})["stop_on_success"] = True
    if args.policy_server_url:
        config["vla"]["server_url"] = args.policy_server_url
    evaluation = config.setdefault("evaluation", {})
    evaluation.update(enabled=True, status_path=str(directory / "status.json"))
    if args.max_steps is not None:
        evaluation["max_action_steps"] = args.max_steps
    evaluation.pop("metadata", None)
    evaluation["video_path"] = str(directory / "rollout.mp4") if args.record_video else None
    workspace.mkdir()
    sync_workspace_templates(workspace, silent=True)
    agents = workspace / "AGENTS.md"
    atomic_text(agents, agents.read_text(encoding="utf-8").rstrip()+"\n\n"+EVALUATION_RULES)
    state = default_robot_state_doc()
    state["task"] = dict(benchmark="RoboCasa", env_name=args.env_name, split=args.split, seed=args.seed)
    save_robot_state_doc(workspace / "ROBOT_STATE.md", state)
    atomic_text(workspace / "ACTION.md", "")
    atomic_json(directory / "driver_config.json", config)
    return workspace


def runtime_environment() -> dict[str, str]:
    env = libero_runtime_environment()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


def start_process(command: list[str], workspace: Path, log: Any) -> subprocess.Popen:
    # Native simulator logs and accidental relative artifacts stay in this episode.
    return subprocess.Popen(command, cwd=workspace, env=runtime_environment(),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def start_agent(args: argparse.Namespace, directory: Path, workspace: Path,
                session_id: str, instruction: str, status: dict, deadline: float,
                feedback_round: int, log: Any) -> subprocess.Popen:
    message = (
        f"Complete this RoboCasa simulated robot task:\n\n{instruction}\n\n"
        "Act autonomously through execute_robot_action using the installed embodiment profile. "
        "Inspect the current state and use the registered specialists when needed. "
        "Official completion is robots.robocasa.success in ROBOT_STATE.md; "
        "stop when it is true or the episode is done."
    )
    if feedback_round:
        message += (
            "\n\nThe evaluator reports success=false after your previous response. "
            "Continue the same episode; inspect fresh ROBOT_STATE.md, ACTION.md and PLAN.md. "
            f"Action budget: {status['action_steps']}/{status['max_action_steps']} used. "
            "Reassess the execution rather than repeating a completion summary."
        )
    request = RunRequest(
        message=message, workspace=str(workspace), config=str(args.agent_config),
        session_id=session_id, restrict_to_workspace=True, stream=False,
        timeout_s=max(.001, deadline-time.monotonic()),
        max_iterations=args.max_agent_iterations,
    )
    request_path = directory / ".agent_request.json"
    output = directory / "agent_runs" / request.run_id
    atomic_json(request_path, request.model_dump(mode="json"))
    process = start_process(
        [args.agent_python, "-B", "-m", "Emerge.cli.headless", "--request", str(request_path),
         "--output-dir", str(output)], workspace, log,
    )
    process.emerge_run_id = request.run_id
    process.emerge_session_id = session_id
    process.emerge_result_path = output / "result.json"
    return process


def agent_exit_outcome(result: dict[str, Any]) -> tuple[str, str]:
    reason = result["finish_reason"]
    if result["run_status"] == "timed_out":
        return "task_failure", "episode_timeout"
    if reason in {"runtime_error", "invalid_request", "cleanup_error", "cancellation_unconfirmed", "error"}:
        return "infrastructure_error", f"agent_{reason}"
    return "task_failure", f"agent_{reason}"


class EpisodeSupervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.signal_number: int | None = None

    def check_interrupted(self) -> None:
        if self.signal_number is not None:
            raise EpisodeError("interrupted", f"Received signal {self.signal_number}", "supervisor")

    def wait_ready(self, controller: subprocess.Popen, directory: Path, workspace: Path) -> dict:
        deadline = time.monotonic()+self.args.ready_timeout_s
        while True:
            self.check_interrupted()
            status = read_status(directory / "status.json")
            terminal = terminal_outcome(status)
            if terminal and terminal[0] == "infrastructure_error":
                raise EpisodeError("initialization_error", status.get("metadata", {}).get("error_message") or terminal[1],
                                   "environment_initialization")
            robot = load_robot_state_doc(workspace / "ROBOT_STATE.md").get("robots", {}).get("robocasa", {})
            if status.get("ready") and robot.get("ready"):
                return status
            if controller.poll() is not None:
                raise EpisodeError("controller_exited", f"Controller exited during initialization ({controller.returncode}); see controller.log",
                                   "environment_initialization")
            if time.monotonic() >= deadline:
                raise EpisodeError("initialization_timeout", "Controller did not become ready within its timeout", "environment_initialization")
            time.sleep(min(self.args.poll_interval_s, max(0., deadline-time.monotonic())))

    def run(self, directory: Path) -> dict[str, Any]:
        args = self.args
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        controller = agent = None
        workspace = directory / "workspace"
        status: dict[str, Any] = {}
        outcome, reason = "infrastructure_error", "preparation_error"
        feedback_rounds = 0
        instruction = ""
        errors: list[dict[str, Any]] = []
        phase = "preparation"
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda number, _frame: setattr(self, "signal_number", number))
        try:
            workspace = prepare_episode(args, directory)
            with (directory / "controller.log").open("w", encoding="utf-8") as controller_log, (directory / "agent.log").open("w", encoding="utf-8") as agent_log:
                try:
                    phase = "environment_initialization"
                    controller = start_process(
                        [args.controller_python, "-B", "-m", "robot.controller", "--driver", "robocasa",
                         "--workspace", str(workspace), "--driver-config", str(directory / "driver_config.json"),
                         "--interval", str(args.poll_interval_s)], workspace, controller_log,
                    )
                    status = self.wait_ready(controller, directory, workspace)
                    instruction = status["metadata"]["instruction"]
                    print(f"READY {args.env_name} seed={args.seed} budget={status['max_action_steps']}", flush=True)
                    phase = "episode_execution"
                    deadline = time.monotonic()+args.episode_timeout_s
                    session_id = "cli:robocasa-eval-"+uuid4().hex
                    while True:
                        self.check_interrupted()
                        status = read_status(directory / "status.json")
                        terminal = terminal_outcome(status)
                        if terminal:
                            outcome, reason = terminal
                            if status.get("metadata", {}).get("error_message"):
                                errors.append({"stage": "environment_execution", "message": status["metadata"]["error_message"]})
                            break
                        if controller.poll() is not None:
                            raise EpisodeError("controller_exited", f"Controller exited unexpectedly ({controller.returncode}); see controller.log", "controller")
                        if time.monotonic() >= deadline:
                            outcome, reason = "task_failure", "episode_timeout"
                            break
                        if agent is None:
                            agent = start_agent(args, directory, workspace, session_id,
                                                instruction, status, deadline, feedback_rounds, agent_log)
                        elif agent.poll() is not None:
                            try:
                                result = read_agent_result(agent)
                            except (OSError, ValueError) as exc:
                                raise EpisodeError("invalid_agent_result", str(exc), "agent_execution") from exc
                            # Read again after exit so a final successful action wins.
                            status = read_status(directory / "status.json")
                            terminal = terminal_outcome(status)
                            if terminal:
                                outcome, reason = terminal
                                break
                            agent_returncode = agent.returncode
                            stop_process(agent, grace_s=args.shutdown_grace_s)
                            agent = None
                            if _should_continue_after_agent_exit(
                                returncode=agent_returncode, status=status,
                                feedback_rounds=feedback_rounds,
                                max_feedback_rounds=args.max_agent_feedback_rounds,
                            ):
                                feedback_rounds += 1
                                print(f"CONTINUE round={feedback_rounds} success=false", flush=True)
                                continue
                            outcome, reason = agent_exit_outcome(result)
                            if result["run_status"] != "completed":
                                errors.append({"stage": "agent_execution",
                                               "message": (result.get("error") or {}).get("message", result["finish_reason"])})
                            break
                        time.sleep(min(args.poll_interval_s, max(0., deadline-time.monotonic())))
                finally:
                    # Keep the controller alive while headless acknowledges cancellation.
                    for label, process in (("agent", agent), ("controller", controller)):
                        try:
                            stop_process(process, grace_s=args.shutdown_grace_s)
                        except Exception as exc:
                            errors.append({"stage": "cleanup", "process": label, "message": f"{type(exc).__name__}: {exc}"})
        except EpisodeError as exc:
            outcome = "cancelled" if exc.reason == "interrupted" else "infrastructure_error"
            reason = exc.reason
            errors.append({"stage": exc.stage, "message": str(exc)})
        except Exception as exc:
            outcome, reason = "infrastructure_error", f"{phase}_error"
            errors.append({"stage": phase, "message": f"{type(exc).__name__}: {exc}"})
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

        # Cleanup may close an unfinished environment, but must not replace the
        # actual stop reason with 'closed' or discard the final applied action.
        try:
            final = read_status(directory / "status.json")
            if final:
                status = final
        except (OSError, ValueError) as exc:
            errors.append({"stage": "final_status", "message": str(exc)})
        if status.get("success"):
            outcome, reason = "success", "success"
        (directory / ".agent_request.json").unlink(missing_ok=True)
        with (directory / "agent.log").open("a", encoding="utf-8") as log:
            for record in errors:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
        if outcome == "infrastructure_error":
            reason = f"infrastructure_error:{reason}"
        result = {
            "schema_version": "Emerge.robocasa_episode_result.v1",
            "episode_key": f"{args.split}/{args.env_name}/seed_{args.seed}",
            "env_name": args.env_name, "split": args.split, "seed": args.seed,
            "instruction": instruction,
            "success": bool(status.get("success", False)),
            "termination_reason": reason,
            "action_steps": int(status.get("action_steps", 0)),
            "environment_steps": int(status.get("environment_steps", 0)),
            "max_action_steps": status.get("max_action_steps", args.max_steps),
            "agent_feedback_rounds": feedback_rounds,
            "agent_runs_dir": str(directory / "agent_runs"),
            "duration_s": round(time.monotonic()-started, 3),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "attempt_dir": str(directory),
            **action_metrics(workspace / "ACTION.md"),
        }
        atomic_json(directory / "result.json", result)
        return result



def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True, help="Native RoboCasa task name")
    parser.add_argument("--split", choices=("target", "pretrain", "all"), default="target")
    parser.add_argument("--seed", type=nonnegative_int, default=7)
    parser.add_argument("--driver-config", type=Path, default=REPO_ROOT / "dev/robocasa_driver.json")
    parser.add_argument("--agent-config", type=Path, default=REPO_ROOT / ".emerge/config.json")
    parser.add_argument("--controller-python", "--watchdog-python", default=os.environ.get("ROBOCASA_PYTHON", sys.executable))
    parser.add_argument("--agent-python", default=sys.executable)
    parser.add_argument("--policy-server-url", help="Override the driver's OpenPI service URL")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True,
                        help="Save rollout.mp4 from the native left camera (enabled by default)")
    parser.add_argument("--max-steps", type=positive_int, help="Override native registry horizon; all control actions share it")
    parser.add_argument("--max-agent-iterations", type=positive_int, help="Override iterations per headless run")
    parser.add_argument("--max-agent-feedback-rounds", type=nonnegative_int, default=2)
    parser.add_argument("--episode-timeout-s", type=positive_float, default=1800., help="Execution timeout after readiness, shared by continuations")
    parser.add_argument("--ready-timeout-s", type=positive_float, default=300.)
    parser.add_argument("--poll-interval-s", type=positive_float, default=.25)
    parser.add_argument("--shutdown-grace-s", type=positive_float, default=90.)
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; existing directories are never overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.driver_config = args.driver_config.expanduser().resolve()
    args.agent_config = args.agent_config.expanduser().resolve()
    # Validate the existing API config without copying or changing it.
    from Emerge.config.loader import load_config
    try:
        load_config(args.agent_config)
    except ValueError as exc:
        parser.error(str(exc))
    directory = (args.output_dir.expanduser().resolve() if args.output_dir else
                 REPO_ROOT / "artifacts/robocasa_agent_eval" / args.env_name /
                 (datetime.now().strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8]))
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists; choose a fresh directory: {directory}")
    print(f"EPISODE {directory}", flush=True)
    result = EpisodeSupervisor(args).run(directory)
    print(json.dumps({key: result[key] for key in ("success", "termination_reason", "action_steps")}, ensure_ascii=False), flush=True)
    print(f"RESULT {directory / 'result.json'}", flush=True)
    if result["termination_reason"] == "interrupted":
        return 130
    return 1 if result["termination_reason"].startswith("infrastructure_error:") else 0


if __name__ == "__main__":
    raise SystemExit(main())
