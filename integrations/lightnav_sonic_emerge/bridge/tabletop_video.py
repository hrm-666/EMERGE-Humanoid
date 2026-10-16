"""Generate trial MP4s and evaluation without connecting to any control transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .tabletop_evaluate import evaluate_task
from .tabletop_recording import timeline_samples
from .tabletop_recording import write_json
from .tabletop_start import table_footprint


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _encode(path: Path, images, fps: float) -> dict:
    import cv2

    writer = None
    count = 0
    try:
        for rgb in images:
            height, width = rgb.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open the MP4 encoder")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        if writer is not None:
            writer.release()
    if not count:
        raise ValueError("no video frames")
    return {
        "file": path.name,
        "frames": count,
        "fps": fps,
        "duration_s": count / fps
    }


def render_ego(directory: Path, metadata: dict) -> dict:
    """Encode recorded ego JPEGs on the trial's wall-clock timeline.

    Args:
        directory: Task directory containing ego_frames.jsonl and JPEG frames.
        metadata: Recording timestamps and frame rate from recording.json.

    Returns:
        Filename, frame count, frame rate, and duration for ego.mp4.

    Raises:
        ValueError: A source image is unreadable or there are no samples.
        RuntimeError: OpenCV cannot open the video encoder.
    """
    import cv2

    samples = timeline_samples(_read_rows(directory / "ego_frames.jsonl"),
                               metadata["start_wall_time_s"],
                               metadata["end_wall_time_s"], metadata["fps"])

    def frames():
        last_file, rgb = None, None
        for sample in samples:
            if sample["file"] != last_file:
                bgr = cv2.imread(str(directory / sample["file"]))
                if bgr is None:
                    raise ValueError(
                        f"unreadable recorded JPEG: {sample['file']}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last_file = sample["file"]
            yield rgb

    return _encode(directory / "ego.mp4", frames(), metadata["fps"])


def render_third_person(directory: Path, metadata: dict,
                        model_path: Path) -> dict:
    """Render recorded qpos without advancing physics or issuing commands.

    Args:
        directory: Task directory containing scene_poses.jsonl.
        metadata: Recording timestamps and frame rate from recording.json.
        model_path: Original MuJoCo scene for replay.

    Returns:
        Video metadata, camera description, and source model hash.

    Raises:
        ValueError: Recorded poses are empty or incompatible with the model.
        RuntimeError: The renderer or encoder cannot initialize.
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table = table_footprint(model, data)
    rows = _read_rows(directory / "scene_poses.jsonl")
    if not rows:
        raise ValueError(
            "no scene poses; restart the updated simulation terminal")
    pelvis = model.body("pelvis").id
    joint = int(model.body_jntadr[pelvis])
    qa = int(model.jnt_qposadr[joint])
    for row in rows:
        if len(row["qpos"]) != model.nq or not np.isfinite(row["qpos"]).all():
            raise ValueError("recorded scene qpos does not match model")
    xy = np.array([row["qpos"][qa:qa + 2] for row in rows] +
                  [[table.centre_x, table.centre_y]])
    low, high = xy.min(axis=0) - 0.9, xy.max(axis=0) + 0.9
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*(low + high) / 2, 0.6]
    camera.azimuth, camera.elevation = 135.0, -40.0
    span = float(np.linalg.norm(high - low))
    camera.distance = max(
        4.0, span /
        (2 * math.tan(math.radians(float(model.vis.global_.fovy)) / 2)) * 1.4)
    renderer = mujoco.Renderer(model, height=480, width=640)
    samples = timeline_samples(rows, metadata["start_wall_time_s"],
                               metadata["end_wall_time_s"], metadata["fps"])

    def frames():
        for sample in samples:
            data.qpos[:] = sample["qpos"]
            data.time = sample["sim_time_s"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            yield renderer.render()

    try:
        result = _encode(directory / "third_person.mp4", frames(),
                         metadata["fps"])
    finally:
        renderer.close()
    result.update(source="offline scene qpos replay",
                  camera="fixed free overview",
                  model_sha256=hashlib.sha256(
                      model_path.read_bytes()).hexdigest())
    return result


def _cleanup_intermediate_images(directory: Path) -> list[str]:
    """Remove per-task JPEG intermediates after both MP4s are complete."""
    removed = []
    frame_dir = directory / "video_frames"
    if frame_dir.is_dir():
        for path in frame_dir.glob("*.jpg"):
            path.unlink(missing_ok=True)
            removed.append(path.name)
        frame_dir.rmdir()
    # These are all task-local LightNav capture images.  The readable inference
    # log keeps the decoded model fields, so retaining duplicate JPEGs is not
    # necessary after the videos have been encoded.
    for pattern in ("frame-*.jpg", "state-*.jpg", "end_ego.jpg"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def postprocess(directory: Path, model_path: Path) -> dict:
    """Evaluate and encode an ended trial, recording each artifact failure.

    Args:
        directory: Task directory containing recording and summary metadata.
        model_path: Original MuJoCo scene for evaluation and qpos replay.

    Returns:
        Status and per-artifact errors also written to artifacts.json. Source
        images are removed only after all requested outputs finish successfully.
    """
    status = {"status": "processing", "errors": {}, "videos": {}}
    write_json(directory / "artifacts.json", status)
    try:
        evaluate_task(directory, model_path)
        status["evaluation"] = "evaluation.json"
    except Exception as exc:
        status["errors"]["evaluation"] = str(exc)
    try:
        metadata = json.loads((directory / "recording.json").read_text())
        if metadata.get("capture_error"):
            status["errors"]["capture"] = metadata["capture_error"]
        if metadata["enabled"]:
            for name, render in (
                ("ego", lambda: render_ego(directory, metadata)),
                ("third_person",
                 lambda: render_third_person(directory, metadata, model_path))):
                try:
                    status["videos"][name] = render()
                except Exception as exc:
                    status["errors"][name] = str(exc)
    except Exception as exc:
        status["errors"]["recording"] = str(exc)
    status["status"] = "error" if status["errors"] else "complete"
    if status["status"] == "complete" and status["videos"]:
        status["removed_intermediate_images"] = _cleanup_intermediate_images(
            directory)
        status["image_retention"] = "mp4_only"
    write_json(directory / "artifacts.json", status)
    return status


def main() -> None:
    """Build offline video and evaluation artifacts for a completed trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(postprocess(args.task_dir, args.model), indent=2))


if __name__ == "__main__":
    main()
