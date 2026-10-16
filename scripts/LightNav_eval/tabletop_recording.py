"""Task-local RGB/scene capture and asynchronous offline artifact generation."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor


def write_json(path, payload) -> None:
    """Atomically replace a JSON artifact in an existing task directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class TaskRecording:
    """Save fresh camera frames and full scene poses at a bounded wall-clock rate."""

    def __init__(self, task, fps: float, enabled: bool = True) -> None:
        self.directory = task.directory
        self.fps = fps
        self.enabled = enabled
        self.start = task.summary["start_wall_time_s"]
        self.last_frame = self.last_pose = float("-inf")
        self.frames = self.poses = 0
        self.error = None
        self.jpeg_dir = self.directory / "video_frames"
        if enabled:
            self.jpeg_dir.mkdir()
        self.frame_index = (self.directory / "ego_frames.jsonl").open("w")
        self.scene_index = (self.directory / "scene_poses.jsonl").open("w")

    def capture(self, frame, pose) -> None:
        if not self.enabled or self.error is not None:
            return
        try:
            if frame is not None and frame.wall_time_s >= self.last_frame + 1 / self.fps - 0.002:
                name = f"{self.frames:06d}.jpg"
                (self.jpeg_dir / name).write_bytes(frame.jpeg)
                self.frame_index.write(
                    json.dumps({
                        "file": f"video_frames/{name}",
                        "wall_time_s": frame.wall_time_s,
                        "sim_time_s": frame.sim_time_s
                    }) + "\n")
                self.frames += 1
                self.last_frame = frame.wall_time_s
            if (pose is not None and pose.qpos is not None and
                    pose.wall_time_s >= self.last_pose + 1 / self.fps - 0.002):
                self.scene_index.write(
                    json.dumps({
                        "wall_time_s": pose.wall_time_s,
                        "sim_time_s": pose.sim_time_s,
                        "qpos": list(pose.qpos)
                    }) + "\n")
                self.poses += 1
                self.last_pose = pose.wall_time_s
        except OSError as exc:
            # Artifact storage errors must not interrupt sending control/IDLE.
            self.error = str(exc)

    def finish(self, task) -> None:
        self.frame_index.close()
        self.scene_index.close()
        write_json(
            self.directory / "recording.json", {
                "enabled":
                    self.enabled,
                "fps":
                    self.fps,
                "frame_count":
                    self.frames,
                "pose_count":
                    self.poses,
                "start_wall_time_s":
                    self.start,
                "end_wall_time_s":
                    task.summary["end_wall_time_s"],
                "capture_error":
                    self.error,
                "time_base":
                    "wall clock",
                "ego_source":
                    "observation RGB JPEG; same source supplied to LightNav",
                "third_person_source":
                    "offline qpos replay, mj_forward only; no physics stepping",
            })


class TaskArtifactWorker:
    """One encoding worker; trials continue while finished artifacts are rendered."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="trial-artifacts")
        self.jobs = {}

    def submit(self, task) -> None:
        write_json(task.directory / "artifacts.json", {"status": "pending"})
        self.jobs[self.executor.submit(self._process,
                                       task.directory)] = task.directory

    @staticmethod
    def _process(directory) -> None:
        from .local_tabletop import build_command

        try:
            command, env = build_command(
                "postprocess",
                ["--task-dir", str(directory.resolve())])
            with (directory / "artifacts.log").open("w") as stream:
                result = subprocess.run(command,
                                        env=env,
                                        stdout=stream,
                                        stderr=subprocess.STDOUT,
                                        check=False)
            if result.returncode:
                raise RuntimeError(
                    f"artifact process exited {result.returncode}; see artifacts.log"
                )
        except Exception as exc:
            write_json(directory / "artifacts.json", {
                "status": "error",
                "error": str(exc)
            })

    def poll(self) -> None:
        for future, directory in list(self.jobs.items()):
            if future.done():
                future.result()
                status = json.loads((directory / "artifacts.json").read_text())
                print(
                    f"TASK ARTIFACTS {directory.name}: {status['status']} | {directory}",
                    flush=True)
                del self.jobs[future]

    def close(self) -> None:
        if self.jobs:
            print("Finishing task videos/evaluations; motion has stopped.",
                  flush=True)
        self.executor.shutdown(wait=True)
        self.poll()


def timeline_samples(rows: list[dict], start: float, end: float, fps: float):
    """Hold the most recent source sample; keep gaps and duration in the MP4."""
    if not rows:
        raise ValueError(
            "no samples recorded (restart sim for full qpos telemetry)")
    rows = sorted(rows, key=lambda row: row["wall_time_s"])
    index = 0
    for frame_id in range(max(1, math_ceil_frames(end - start, fps))):
        at = start + frame_id / fps
        while index + 1 < len(rows) and rows[index + 1]["wall_time_s"] <= at:
            index += 1
        yield rows[index]


def math_ceil_frames(duration: float, fps: float) -> int:
    """Return the nonnegative frame count covering a duration at the given FPS."""
    import math

    return math.ceil(max(0.0, duration) * fps)
