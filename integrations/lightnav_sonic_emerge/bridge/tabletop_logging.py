"""Per-navigation logs alongside the complete tabletop control session."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class NavigationTaskLog:
    task_id: str
    generation: int
    directory: Path
    summary: dict


class TabletopRunLog:
    """Keep cancelled/late replies attached to their original navigation task.

    Only the control thread writes logs. Ended tasks remain indexed by generation
    so an in-flight reply cannot leak into a replacement task's records.
    """

    def __init__(self, output: Path) -> None:
        self.output = output
        self.session = (output / "events.jsonl").open("w",
                                                      encoding="utf-8",
                                                      buffering=1)
        self.tasks: dict[int, NavigationTaskLog] = {}
        self.active: NavigationTaskLog | None = None

    @staticmethod
    def _timestamp(wall_time_s: float) -> str:
        return datetime.fromtimestamp(wall_time_s).astimezone().isoformat(
            timespec="milliseconds")

    @staticmethod
    def _write_summary(task: NavigationTaskLog) -> None:
        path = task.directory / "summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(task.summary, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    def start(
        self,
        generation: int,
        instruction: str,
        robot_pose: dict | None,
        *,
        control_config: dict | None = None,
    ) -> NavigationTaskLog:
        if self.active is not None:
            raise RuntimeError(
                "finish the active navigation log before starting another")
        task_id = f"task-{len(self.tasks) + 1:04d}"
        directory = self.output / "tasks" / task_id
        directory.mkdir(parents=True, exist_ok=False)
        now = time.time()
        task = NavigationTaskLog(
            task_id, generation, directory, {
                "task_id": task_id,
                "generation": generation,
                "instruction": instruction,
                "status": "running",
                "started_at": self._timestamp(now),
                "start_wall_time_s": now,
                "start_robot_pose": robot_pose,
                "control_config": control_config,
                "ended_at": None,
                "end_reason": None,
                "prediction_count": 0,
                "discarded_prediction_count": 0,
                "late_prediction_count": 0,
            })
        self.tasks[generation] = task
        self.active = task
        self._write_summary(task)
        (directory / "inference.txt").write_text(
            f"TASK START {task_id} | {self._timestamp(now)}\nInstruction: {instruction}\n\n"
        )
        (self.output / "latest_task.txt").write_text(f"tasks/{task_id}\n")
        self.record("navigation_start",
                    generation=generation,
                    instruction=instruction,
                    robot_pose=robot_pose)
        return task

    def finish(self, reason: str,
               robot_pose: dict | None) -> NavigationTaskLog | None:
        task = self.active
        if task is None:
            return None
        now = time.time()
        task.summary.update(
            status="ended",
            ended_at=self._timestamp(now),
            end_wall_time_s=now,
            duration_s=now - task.summary["start_wall_time_s"],
            end_reason=reason,
            end_robot_pose=robot_pose,
            model_stop_at_end=task.summary.get("last_model_stop"),
        )
        self.record("navigation_end",
                    generation=task.generation,
                    reason=reason,
                    robot_pose=robot_pose)
        with (task.directory / "inference.txt").open(
                "a", encoding="utf-8") as stream:
            stream.write(
                f"\nTASK END {task.task_id} | {self._timestamp(now)} | {reason}\n"
            )
        self._write_summary(task)
        self.active = None
        return task

    def task_for(self,
                 generation: int | None = None) -> NavigationTaskLog | None:
        return self.active if generation is None else self.tasks.get(generation)

    def save_frame(self,
                   name: str,
                   jpeg: bytes,
                   generation: int | None = None) -> str:
        task = self.task_for(generation)
        directory = self.output if task is None else task.directory
        path = directory / name
        path.write_bytes(jpeg)
        return path.relative_to(self.output).as_posix()

    def record(self, event: str, **data: object) -> None:
        task = self.task_for(data.get("generation"))
        entry = {
            "event": event,
            "wall_time_s": time.time(),
            **data, "task_id": None if task is None else task.task_id
        }
        if task is not None and task.summary[
                "status"] == "ended" and event != "navigation_end":
            entry["after_task_end"] = True
        self.session.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if task is None:
            return
        task_entry = dict(entry)
        if isinstance(entry.get("frame"), str):
            prefix = f"tasks/{task.task_id}/"
            task_entry["frame"] = entry["frame"].removeprefix(prefix)
        with (task.directory / "events.jsonl").open("a",
                                                    encoding="utf-8") as stream:
            stream.write(json.dumps(task_entry, ensure_ascii=False) + "\n")
        if event == "prediction":
            task.summary["prediction_count"] += 1
            discarded = str(data.get("reason", "")).startswith("discarded:")
            task.summary["discarded_prediction_count"] += int(discarded)
            task.summary["late_prediction_count"] += int(
                entry.get("after_task_end", False))
            task.summary["last_prediction_reason"] = data.get("reason")
            task.summary["last_model_stop"] = data.get("stop")
            reply = data.get("reply", {})
            with (task.directory / "inference.txt").open(
                    "a", encoding="utf-8") as stream:
                stream.write(
                    f"[{task.summary['prediction_count']:03d}] {self._timestamp(entry['wall_time_s'])} "
                    f"seq={reply.get('seq')} stop={data.get('stop')} "
                    f"late={entry.get('after_task_end', False)}\n"
                    f"  raw_text: {data.get('raw_text')}\n"
                    f"  pointing: {json.dumps(data.get('pointing'), ensure_ascii=False)}\n"
                    f"  first_waypoint: {data.get('first_waypoint')}\n"
                    f"  robot_pose: {json.dumps(data.get('robot_pose'), ensure_ascii=False)}\n"
                    f"  reason: {data.get('reason')}\n\n")
            self._write_summary(task)

    def close(self) -> None:
        self.session.close()
