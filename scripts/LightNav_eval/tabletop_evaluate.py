"""Read-only stop/arrival evaluation for the fixed apple-table simulation.

Scene geometry is used only after a trial; it is never supplied to LightNav or
used to stop the robot. Distances refer to the logged pelvis, not foot clearance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate_pose(
    pose: dict | None,
    front_x: float,
    centre_y: float,
    *,
    min_distance_m: float = 0.8,
    max_distance_m: float = 1.3,
    max_lateral_m: float = 0.35,
    max_yaw_deg: float = 20.0,
) -> dict:
    """Evaluate a declared standing region on the table's negative-X side."""
    if not isinstance(pose, dict):
        return {"geometry_arrival": None, "reason": "missing end robot pose"}
    try:
        x, y, _ = (float(value) for value in pose["position_m"])
        yaw = float(pose["yaw_rad"])
        w, qx, qy, qz = (float(value) for value in pose["quaternion_wxyz"])
        norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
        if not all(math.isfinite(v) for v in (x, y, yaw, norm)) or norm < 1e-8:
            raise ValueError("invalid pose")
        # World vertical component of the pelvis's local up axis.
        upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy) / (norm * norm)
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, upright_cos))))
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"geometry_arrival": None, "reason": "invalid end robot pose"}
    distance = front_x - x
    lateral = y - centre_y
    yaw_deg = math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
    checks = {
        "distance_in_band": min_distance_m <= distance <= max_distance_m,
        "lateral_in_band": abs(lateral) <= max_lateral_m,
        "facing_table": abs(yaw_deg) <= max_yaw_deg,
        "pelvis_upright": tilt_deg <= 15.0,
    }
    return {
        "geometry_arrival": all(checks.values()),
        "pelvis_to_front_edge_m": distance,
        "lateral_offset_m": lateral,
        "yaw_error_deg": yaw_deg,
        "pelvis_tilt_deg": tilt_deg,
        "checks": checks,
    }


def evaluate_summary(summary: dict, front_x: float, centre_y: float,
                     **limits) -> dict:
    """Evaluate model termination separately from geometric arrival.

    Args:
        summary: Completed task summary with timestamped end pose.
        front_x: World X coordinate of the table's negative-X front edge.
        centre_y: World Y coordinate of the table centre.
        **limits: Distance, lateral, and heading limits for evaluate_pose.

    Returns:
        Evidence for geometric arrival and model stop. Arrival is None when
        the end pose is missing, invalid, or stale.
    """
    geometry = evaluate_pose(summary.get("end_robot_pose"), front_x, centre_y,
                             **limits)
    pose = summary.get("end_robot_pose")
    end_time = summary.get("end_wall_time_s")
    pose_time = pose.get("wall_time_s") if isinstance(pose, dict) else None
    if isinstance(end_time,
                  (int, float)) and isinstance(pose_time, (int, float)):
        age = end_time - pose_time
        if not math.isfinite(age) or not -0.25 <= age <= 1.0:
            geometry.update(geometry_arrival=None,
                            reason="stale end robot pose")
    else:
        geometry.update(geometry_arrival=None,
                        reason="missing end pose timestamps")
    model_stop = (summary.get("end_reason") == "LightNav stop=true" and
                  summary.get("model_stop_at_end",
                              summary.get("last_model_stop")) is True)
    arrived = geometry["geometry_arrival"]
    return {
        "task_id":
            summary.get("task_id"),
        "instruction":
            summary.get("instruction"),
        "end_reason":
            summary.get("end_reason"),
        "model_stop":
            model_stop,
        **geometry,
        "model_stop_at_goal":
            None if arrived is None else bool(model_stop and arrived),
        "evaluation_scope":
            "stop-time pelvis pose; contact and post-stop drift are not verified",
    }


def table_geometry(model_path: Path) -> tuple[float, float]:
    """Read the original table's front edge without stepping simulation.

    Args:
        model_path: Path to the original apple-table MuJoCo scene.

    Returns:
        World X of the negative-X front edge and world Y of the table centre.

    Raises:
        ValueError: The expected collision box or orientation is absent.
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,
                      data)  # No viewer, stepping, DDS or motor publisher.
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                "tabletop_lab_table_collision_top")
    if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("scene has no expected tabletop collision box")
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if not np.isclose(abs(rotation[0, 1]), 1.0, atol=1e-5):
        raise ValueError(
            "evaluation requires the original tabletop orientation")
    half_x = float(np.abs(rotation[0]) @ model.geom_size[geom_id])
    return float(data.geom_xpos[geom_id, 0] - half_x), float(
        data.geom_xpos[geom_id, 1])


def evaluate_any_side(summary: dict, table, **limits) -> dict:
    """Any-side tabletop arrival, separately reported from the historical front region."""
    # Reuse the timestamp/upright validation without changing the old criterion.
    validated = evaluate_summary(summary, table.centre_x, table.centre_y,
                                 **limits)
    if validated["geometry_arrival"] is None:
        return {
            "geometry_arrival": None,
            "model_stop_at_goal": None,
            "reason": validated.get("reason")
        }
    pose = summary["end_robot_pose"]
    x, y, _ = pose["position_m"]
    lx, ly = table.local_xy(x, y)
    dx = max(abs(lx) - table.half_x, 0.0)
    dy = max(abs(ly) - table.half_y, 0.0)
    distance = math.hypot(dx, dy)
    bearing = math.atan2(table.centre_y - y, table.centre_x - x)
    error = math.degrees(
        math.atan2(math.sin(pose["yaw_rad"] - bearing),
                   math.cos(pose["yaw_rad"] - bearing)))
    checks = {
        "outside_table":
            not table.contains(x, y),
        "distance_in_band":
            limits.get("min_distance_m", 0.8) <= distance <= limits.get(
                "max_distance_m", 1.3),
        "facing_table_centre":
            abs(error) <= limits.get("max_yaw_deg", 20.0),
        "pelvis_upright":
            validated["checks"]["pelvis_upright"],
    }
    arrived = all(checks.values())
    return {
        "geometry_arrival":
            arrived,
        "model_stop_at_goal":
            bool(validated["model_stop"] and arrived),
        "pelvis_to_table_edge_m":
            distance,
        "yaw_error_deg":
            error,
        "checks":
            checks,
        "criterion":
            "any side, nearest rectangular edge and facing table centre"
    }


def evaluate_task(task_dir: Path, model_path: Path, **limits) -> dict:
    """Write offline arrival evidence for one completed simulation trial.

    Args:
        task_dir: Task directory containing summary.json.
        model_path: Original MuJoCo scene used by the trial.
        **limits: Overrides for distance, lateral, and heading criteria.

    Returns:
        Evaluation fields also written to evaluation.json and evaluation.txt.

    Raises:
        ValueError: The trial is not ended or the scene geometry is unsupported.
    """
    import mujoco

    from .tabletop_start import table_footprint

    summary = json.loads((task_dir / "summary.json").read_text())
    if summary.get("status") != "ended":
        raise ValueError("evaluate an ended trial")
    defaults = dict(min_distance_m=0.8,
                    max_distance_m=1.3,
                    max_lateral_m=0.35,
                    max_yaw_deg=20.0)
    defaults.update(limits)
    front_x, centre_y = table_geometry(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table = table_footprint(model, data)
    result = evaluate_summary(summary, front_x, centre_y, **defaults)
    result.update(criteria=defaults,
                  table_front_x_m=front_x,
                  table_centre_y_m=centre_y,
                  control_config=summary.get("control_config"),
                  start_config=(summary.get("start_robot_pose") or
                                {}).get("start_config"),
                  any_side=evaluate_any_side(summary, table, **defaults))
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    (task_dir / "evaluation.json").write_text(text)
    (task_dir / "evaluation.txt").write_text(
        f"Task: {result['task_id']}\nInstruction: {result['instruction']}\n"
        f"End reason: {result['end_reason']}\nModel stop: {result['model_stop']}\n"
        f"Original front arrival: {result['geometry_arrival']}\n"
        f"Original front model success: {result['model_stop_at_goal']}\n"
        f"Any-side arrival: {result['any_side']['geometry_arrival']}\n"
        f"Any-side model success: {result['any_side']['model_stop_at_goal']}\n\n"
        + text)
    return result


def main() -> None:
    """Evaluate a completed trial using explicit scene and tolerance arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--min-distance-m", type=float, default=0.8)
    parser.add_argument("--max-distance-m", type=float, default=1.3)
    parser.add_argument("--max-lateral-m", type=float, default=0.35)
    parser.add_argument("--max-yaw-deg", type=float, default=20.0)
    args = parser.parse_args()
    limits = {
        key: getattr(args, key) for key in ("min_distance_m", "max_distance_m",
                                           "max_lateral_m", "max_yaw_deg")
    }
    if not all(math.isfinite(value) and value > 0 for value in limits.values()):
        parser.error("evaluation limits must be finite and positive")
    if args.min_distance_m >= args.max_distance_m:
        parser.error("min distance must be smaller than max distance")
    result = evaluate_task(args.task_dir, args.model, **limits)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Evaluation: {args.task_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
