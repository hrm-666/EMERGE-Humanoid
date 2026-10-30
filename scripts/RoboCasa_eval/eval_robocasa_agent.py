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
import re
import signal
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4
from urllib import request as urllib_request
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Emerge.runtime.protocol import RunRequest, RunResult
from Emerge.runtime.storage import atomic_json, atomic_text
from Emerge.utils.action_queue import normalize_action_document, parse_action_markdown
from Emerge.utils.helpers import sync_workspace_templates
from robot.mujoco_simulation.scene_io import (
    default_robot_state_doc, load_robot_state_doc, save_robot_state_doc,
)

STATUS_SCHEMA = "Emerge.robocasa_evaluation_status.v1"
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_status(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """A missing publication is transient; an incompatible publication is an error."""
    try:
        status = read_json(path)
    except FileNotFoundError:
        return {}
    if status.get("schema_version") != STATUS_SCHEMA:
        raise ValueError("Unexpected RoboCasa evaluation status schema")
    for key, expected in (("env_name", args.env_name), ("split", args.split), ("seed", args.seed)):
        if status.get(key) != expected:
            raise ValueError(f"Evaluation status does not match episode {key}")
    for key in ("ready", "done", "success"):
        if type(status.get(key)) is not bool:
            raise ValueError(f"Evaluation status {key} must be boolean")
    for key in ("action_steps", "max_action_steps"):
        value = status.get(key)
        if key == "max_action_steps" and value is None:
            continue
        if type(value) is not int or value < (1 if key == "max_action_steps" else 0):
            raise ValueError(f"Evaluation status {key} is invalid")
    return status


def terminal_outcome(status: dict[str, Any]) -> tuple[str, str] | None:
    if status.get("success"):
        return "success", "success"
    if status.get("done"):
        reason = status.get("termination_reason") or "environment_done"
        if reason in {"error", "closed"} or status.get("error_message"):
            return "infrastructure_error", str(reason)
        return "task_failure", str(reason)
    return None


def prepare_episode(args: argparse.Namespace, directory: Path) -> Path:
    """Allocate fresh files; API credentials stay exclusively in their existing config."""
    base = read_json(args.driver_config)
    if not isinstance(base.get("robocasa"), dict):
        raise ValueError("driver config must contain a robocasa object")
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
    evaluation["metadata"] = {
        **(evaluation.get("metadata") or {}), "episode_id": directory.name,
        "env_name": args.env_name, "split": args.split, "seed": args.seed,
    }
    # Validate the assembled driver without importing RoboCasa or allocating physics.
    from robot.drivers import load_driver
    driver = load_driver("robocasa", workspace=workspace, **config)
    try:
        if not driver.get_profile_path().is_file():
            raise ValueError(f"Embodiment profile does not exist: {driver.get_profile_path()}")
        if not config.get("observation", {}).get("enabled"):
            raise ValueError("agent evaluation requires observation.enabled=true")
    finally:
        driver.close()
    workspace.mkdir()
    sync_workspace_templates(workspace, silent=True)
    agents = workspace / "AGENTS.md"
    atomic_text(agents, agents.read_text(encoding="utf-8").rstrip()+"\n\n"+EVALUATION_RULES)
    state = default_robot_state_doc()
    state["task"] = dict(benchmark="RoboCasa", env_name=args.env_name, split=args.split, seed=args.seed)
    save_robot_state_doc(workspace / "ROBOT_STATE.md", state)
    atomic_text(workspace / "ACTION.md", "")
    atomic_json(directory / "driver_config.json", config)
    atomic_json(directory / "episode.json", {
        "schema_version": "Emerge.robocasa_episode_spec.v1", **state["task"],
        "agent_config": str(args.agent_config), "controller_python": args.controller_python,
        "agent_python": args.agent_python, "episode_timeout_s": args.episode_timeout_s,
        "max_agent_feedback_rounds": args.max_agent_feedback_rounds,
    })
    return workspace


def runtime_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
    env.update(NO_COLOR="1", TERM="dumb", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


def check_services(args: argparse.Namespace, directory: Path, checks: list[dict]) -> None:
    """Probe model servers directly, independently of the remote LLM provider."""
    from Emerge.config.loader import load_config
    config = load_config(args.agent_config)
    driver = read_json(directory / "driver_config.json")
    endpoints = {
        "OpenPI": driver.get("vla", {}).get("server_url", "ws://localhost:8000"),
        "VGGT": config.subagents.object_location.vggt_url,
        "SAM3": config.subagents.object_location.sam3_url,
    }
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    for name, endpoint in endpoints.items():
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"ws", "wss", "http", "https"} or not parsed.hostname or parsed.username:
            raise EpisodeError("invalid_service_url", f"Invalid {name} service URL", "service_preflight")
        scheme = "https" if parsed.scheme in {"wss", "https"} else "http"
        url = urlunparse((scheme, parsed.netloc, "/healthz", "", "", ""))
        check = {"service": name, "health_url": url, "healthy": False}
        checks.append(check)
        try:
            with opener.open(url, timeout=5) as response:
                if response.status != 200:
                    raise ValueError("non-200 health response")
            check["healthy"] = True
        except (OSError, ValueError) as exc:
            raise EpisodeError("service_unavailable", f"{name} health check failed at {url} ({type(exc).__name__})",
                               "service_preflight") from exc


def start_process(command: list[str], workspace: Path, log: Any) -> subprocess.Popen:
    # Native simulator logs and accidental relative artifacts stay in this episode.
    return subprocess.Popen(command, cwd=workspace, env=runtime_environment(),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_process(process: subprocess.Popen | None, grace_s: float) -> dict[str, Any] | None:
    """Stop only an owned process group; allow headless action cancellation first."""
    if process is None:
        return None
    forced = False

    def send(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        send(signal.SIGINT)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            forced = True
            send(signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                send(signal.SIGKILL)
                process.wait(timeout=3)
    # A tool subprocess can outlive an already exited headless parent.
    if os.name == "posix":
        send(signal.SIGKILL)
    return {"pid": process.pid, "returncode": process.returncode, "forced": forced}


def start_agent(args: argparse.Namespace, directory: Path, workspace: Path,
                session_id: str, instruction: str, status: dict, deadline: float,
                feedback_round: int, log: Any) -> tuple[subprocess.Popen, RunRequest, Path]:
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
        cancel_timeout_s=min(10., max(.1, args.shutdown_grace_s-1)),
        max_iterations=args.max_agent_iterations,
    )
    request_path = directory / "agent_requests" / f"{request.run_id}.json"
    output = directory / "agent_runs" / request.run_id
    atomic_json(request_path, request.model_dump(mode="json"))
    process = start_process(
        [args.agent_python, "-B", "-m", "Emerge.cli.headless", "--request", str(request_path),
         "--output-dir", str(output)], workspace, log,
    )
    return process, request, output / "result.json"


def read_agent_result(path: Path, request: RunRequest, returncode: int) -> RunResult:
    result = RunResult.model_validate(read_json(path))
    if (result.run_id != request.run_id or result.session_id != request.session_id
            or result.exit_code != returncode):
        raise ValueError("Agent result does not match its process, run or session")
    return result


def agent_exit_outcome(result: RunResult) -> tuple[str, str]:
    reason = result.finish_reason
    if result.run_status == "cancelled":
        return "infrastructure_error", "agent_cancelled"
    if result.run_status == "timed_out":
        return "task_failure", "episode_timeout"
    if reason in {"runtime_error", "invalid_request", "cleanup_error", "cancellation_unconfirmed", "error"}:
        return "infrastructure_error", f"agent_{reason}"
    return "task_failure", f"agent_{reason}"


def action_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"action_count": 0, "actions_by_type": {}, "actions_by_status": {}}
    raw = parse_action_markdown(path.read_text(encoding="utf-8"))
    document = normalize_action_document(raw) if raw else None
    counts: dict[str, dict[str, int]] = {"actions_by_type": {}, "actions_by_status": {}}
    actions = document["actions"] if document else []
    for action in actions:
        for field, column in (("action_type", "actions_by_type"), ("status", "actions_by_status")):
            value = str(action.get(field, "unknown"))
            counts[column][value] = counts[column].get(value, 0)+1
    return {"action_count": len(actions), **counts}


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
            status = read_status(directory / "status.json", self.args)
            terminal = terminal_outcome(status)
            if terminal and terminal[0] == "infrastructure_error":
                raise EpisodeError("initialization_error", status.get("error_message") or terminal[1],
                                   "environment_initialization")
            robot = load_robot_state_doc(workspace / "ROBOT_STATE.md").get("robots", {}).get("robocasa", {})
            if status.get("ready") and robot.get("ready"):
                manifest = read_json(workspace / "artifacts/observations/observation.json")
                views = manifest.get("views", [])
                if (not (workspace / "EMBODIED.md").is_file() or not views
                        or not all((workspace / v["image_path"]).is_file() for v in views)):
                    raise EpisodeError("observation_unavailable", "Initial profile/images are incomplete", "environment_initialization")
                if not status.get("instruction") or status.get("max_action_steps") is None:
                    raise EpisodeError("initialization_error", "Task instruction or horizon is missing", "environment_initialization")
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
        agent_runs: list[dict[str, Any]] = []
        process_cleanup: list[dict[str, Any]] = []
        service_checks: list[dict[str, Any]] = []
        phase = "preparation"
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda number, _frame: setattr(self, "signal_number", number))
        try:
            workspace = prepare_episode(args, directory)
            if not args.skip_service_checks:
                phase = "service_preflight"
                check_services(args, directory, service_checks)
            with (directory / "controller.log").open("w", encoding="utf-8") as controller_log, (directory / "agent.log").open("w", encoding="utf-8") as agent_log:
                try:
                    phase = "environment_initialization"
                    controller = start_process(
                        [args.controller_python, "-B", "-m", "robot.controller", "--driver", "robocasa",
                         "--workspace", str(workspace), "--driver-config", str(directory / "driver_config.json"),
                         "--interval", str(args.poll_interval_s)], workspace, controller_log,
                    )
                    status = self.wait_ready(controller, directory, workspace)
                    instruction = status["instruction"]
                    print(f"READY {args.env_name} seed={args.seed} budget={status['max_action_steps']}", flush=True)
                    phase = "episode_execution"
                    deadline = time.monotonic()+args.episode_timeout_s
                    session_id = "cli:robocasa-eval-"+uuid4().hex
                    request = result_path = None
                    while True:
                        self.check_interrupted()
                        status = read_status(directory / "status.json", args)
                        if not status:
                            raise EpisodeError("status_unavailable", "Environment status disappeared after readiness", "environment_execution")
                        terminal = terminal_outcome(status)
                        if terminal:
                            outcome, reason = terminal
                            if status.get("error_message"):
                                errors.append({"stage": "environment_execution", "message": status["error_message"]})
                            break
                        if controller.poll() is not None:
                            raise EpisodeError("controller_exited", f"Controller exited unexpectedly ({controller.returncode}); see controller.log", "controller")
                        if time.monotonic() >= deadline:
                            outcome, reason = "task_failure", "episode_timeout"
                            break
                        if agent is None:
                            agent, request, result_path = start_agent(args, directory, workspace, session_id,
                                                                     instruction, status, deadline, feedback_rounds, agent_log)
                            agent_runs.append({"run_id": request.run_id, "session_id": session_id,
                                               "result_path": str(result_path)})
                        elif agent.poll() is not None:
                            try:
                                result = read_agent_result(result_path, request, agent.returncode)
                            except (OSError, ValueError) as exc:
                                raise EpisodeError("invalid_agent_result", str(exc), "agent_execution") from exc
                            agent_runs[-1].update(returncode=agent.returncode, run_status=result.run_status,
                                                  finish_reason=result.finish_reason, usage=result.usage)
                            # Read again after exit so a final successful action wins.
                            status = read_status(directory / "status.json", args)
                            terminal = terminal_outcome(status)
                            if terminal:
                                outcome, reason = terminal
                                break
                            stop_process(agent, args.shutdown_grace_s)
                            agent = None
                            if result.run_status == "completed" and feedback_rounds < args.max_agent_feedback_rounds:
                                feedback_rounds += 1
                                print(f"CONTINUE round={feedback_rounds} success=false", flush=True)
                                continue
                            outcome, reason = agent_exit_outcome(result)
                            if result.run_status != "completed":
                                errors.append({"stage": "agent_execution",
                                               "message": result.error.message if result.error else result.finish_reason})
                            break
                        time.sleep(min(args.poll_interval_s, max(0., deadline-time.monotonic())))
                finally:
                    # Keep the controller alive while headless acknowledges cancellation.
                    for label, process in (("agent", agent), ("controller", controller)):
                        try:
                            cleanup = stop_process(process, args.shutdown_grace_s)
                            if cleanup is not None:
                                process_cleanup.append({"process": label, **cleanup})
                                if cleanup["forced"]:
                                    errors.append({"stage": "cleanup", "process": label,
                                                   "message": "Process exceeded its graceful shutdown budget and required termination"})
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
            final = read_status(directory / "status.json", args)
            if final:
                status = final
        except (OSError, ValueError) as exc:
            errors.append({"stage": "final_status", "message": str(exc)})
        if status.get("success"):
            outcome, reason = "success", "success"
        elif status.get("termination_reason") == "error" and outcome != "cancelled":
            outcome = "infrastructure_error"
            if not any(e.get("message") == status.get("error_message") for e in errors):
                errors.append({"stage": "environment_execution", "message": status.get("error_message") or "Environment error"})
        elif errors and any(e["stage"] == "cleanup" for e in errors) and outcome != "cancelled":
            outcome = "infrastructure_error"
        for run in agent_runs:
            try:
                result = RunResult.model_validate(read_json(Path(run["result_path"])))
                if result.run_id == run["run_id"] and result.session_id == run["session_id"]:
                    run.update(run_status=result.run_status, finish_reason=result.finish_reason, usage=result.usage)
            except (OSError, ValueError):
                pass  # A force-stopped agent need not have produced a protocol result.
        try:
            metrics = action_metrics(workspace / "ACTION.md")
            robot = load_robot_state_doc(workspace / "ROBOT_STATE.md").get("robots", {}).get("robocasa", {})
            execution = robot.get("last_execution") or {}
            if (not status.get("success") and outcome != "cancelled"
                    and (execution.get("reason") == "server_unavailable"
                         or execution.get("error_stage") in {"inference", "observation_publish", "environment_step"})):
                outcome = "infrastructure_error"
                reason = ("policy_service_unavailable" if execution.get("reason") == "server_unavailable"
                          else f"driver_{execution['error_stage']}_error")
                errors.append({"stage": "driver_execution", "message": execution.get("error_message") or execution.get("reason")})
        except (OSError, ValueError) as exc:
            metrics = {"action_count": None, "actions_by_type": {}, "actions_by_status": {}}
            errors.append({"stage": "artifact_summary", "message": str(exc)})
            if not status.get("success") and outcome != "cancelled":
                outcome = "infrastructure_error"
        result = {
            "schema_version": "Emerge.robocasa_episode_result.v1", "benchmark": "RoboCasa",
            "env_name": args.env_name, "split": args.split, "seed": args.seed,
            "instruction": instruction or status.get("instruction", ""),
            "success": bool(status.get("success", False)), "outcome": outcome,
            "termination_reason": reason, "action_steps": status.get("action_steps", 0),
            "max_action_steps": status.get("max_action_steps"), "environment_status": status,
            "agent_feedback_rounds": feedback_rounds, "agent_runs": agent_runs, "errors": errors,
            "process_cleanup": process_cleanup,
            "service_checks": service_checks, "service_checks_skipped": args.skip_service_checks,
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic()-started, 3), "workspace": str(workspace),
            **metrics,
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
    parser.add_argument("--skip-service-checks", action="store_true",
                        help="Skip local model health probes for controlled diagnostics; recorded in result.json")
    parser.add_argument("--max-steps", type=positive_int, help="Override native registry horizon; all control actions share it")
    parser.add_argument("--max-agent-iterations", type=positive_int, help="Override iterations per headless run")
    parser.add_argument("--max-agent-feedback-rounds", type=nonnegative_int, default=2)
    parser.add_argument("--episode-timeout-s", type=positive_float, default=1800., help="Execution timeout after readiness, shared by continuations")
    parser.add_argument("--ready-timeout-s", type=positive_float, default=300.)
    parser.add_argument("--poll-interval-s", type=positive_float, default=.25)
    parser.add_argument("--shutdown-grace-s", type=positive_float, default=15.)
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; existing directories are never overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.env_name):
        parser.error("env-name must be a native task class name")
    args.driver_config = args.driver_config.expanduser().resolve()
    args.agent_config = args.agent_config.expanduser().resolve()
    for path in (args.driver_config, args.agent_config):
        if not path.is_file():
            parser.error(f"Configuration does not exist: {path}")
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
    print(json.dumps({key: result[key] for key in ("success", "outcome", "termination_reason", "action_steps")}, ensure_ascii=False), flush=True)
    print(f"RESULT {directory / 'result.json'}", flush=True)
    if result["outcome"] == "cancelled":
        return 130
    return 1 if result["outcome"] == "infrastructure_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
