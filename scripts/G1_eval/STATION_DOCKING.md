# Simulation station docking

This is privileged simulation-pose-assisted docking, not improved LightNav
accuracy or real-robot localization. LightNav gets to the station entry region;
`dock_to_station` then walks through the existing Executive → SONIC chain.
No new motion publisher, object-pose policy input, or qpos teleport is used.

## Configuration and commissioning

Use the isolated `~/EMERGE-Humanoid-integration` worktree. The optional Driver
config is `dev/official01_g1_sim_docking.json`. It loads
`dev/official01_sim_stations.json`. The current `apple_table_front` entry is an
enabled `trial` calibration for the unchanged room scene. It is not a validated
real-robot pose, and synthetic coordinates in tests must never replace it.

The station registry requires `runtime: "sim"` and an explicit boolean
`docking_only`. With `docking_only: true`, the Driver rejects preparation,
poses/trajectories, micro-motion and GR00T before and after docking. With
`docking_only: false`, successful measured docking enables the configured
preparation and grasp path; every manipulation action rechecks a fresh pose.

After verifying the actual scene, locomotion arm pose and control ownership,
calibrate one station in the GUI. Record these operator-owned fields:

| Field | Meaning |
| --- | --- |
| `entry` | Root-center rectangle `{xmin,xmax,ymin,ymax}` where LightNav hands off. |
| `corridor` | Obstacle-free root-center rectangle, contracted for robot/hand envelope and stopping clearance; it contains the entry and target tolerance region. |
| `target_x`, `target_y`, `target_yaw` | Measured desired robot-root pose in the unchanged scene; yaw in radians. |
| `position_tolerance_m`, `yaw_tolerance_rad` | Independent arrival tolerances; not a grasp-workspace model. |
| `stable_duration_s` | Fresh, unique post-READY samples must remain in tolerance and below measured speed limits for this dwell. |
| `max_speed_mps`, `max_yaw_rate_rad_s` | Command bounds, no greater than the existing Executive limits. |
| `max_runtime_s`, `max_path_m` | Time (at most 120 s) and observed cumulative travel budgets. |
| `max_pose_age_s`, `max_step_m`, `max_step_yaw_rad` | Freshness, discontinuity and observation-gap checks. |
| `min_base_height_m`, `max_base_height_m`, `max_tilt_rad` | Reject falls or unsupported root poses. |
| `calibration_status` | `trial` for a controlled calibration run, `validated` only after recorded acceptance. |

Defaults in `Station` are conservative **untuned controller settings**.
The geometry and stopping reserve must be assessed against the actual G1 gait,
feet, body and held arm pose, not just pelvis-to-table distance. A rectangle
does not perform dynamic obstacle avoidance. Set `enabled: true` only after
recording a reviewable trial geometry. Do not invent a 0.30 m safe/grasp bound.

## Visible lab trial

1. Confirm `hostname` is `embodied-MS-7E06`, branch is `humanoid-integration`,
   and review every changed file against current uncommitted work. Do not
   overwrite the colleague's `~/EMERGE-Humanoid` or replace existing Skills.
2. Keep the actual MuJoCo GUI and native EMERGE terminal visible on the lab
   desktop. An ego camera alone is insufficient. Preserve old run artifacts.
3. Re-read current PIDs and observation diagnostics. Restore simulator support
   before stopping the existing Controller/SONIC. Controller shutdown also
   exits SONIC: restart it, enter standing mode, then start one Controller.
4. Initialize a new workspace while it is inactive. From the repository root,
   start that Controller with `--driver official01_g1 --driver-config
   dev/official01_g1_sim_docking.json --workspace <new-workspace>`. Do not run
   two control owners. Keep model servers that belong to the current stack.
5. Require fresh RGB/robot pose, Executive READY and a locomotion-safe arm
   configuration; then release support and verify the observed flag is false.
   A robot left in the old grasp-prepared pose needs operator review/reset
   before locomotion; this milestone cannot dispatch preparation as a shortcut.
6. Submit the symbolic station action through the native action queue. The
   Skill example is in `sim-station-docking/SKILL.md`. Deterministic queue tests
   do not need company API access; a later full LLM UI trial needs its own
   authorized company-side configuration. Never copy the company API key.
7. Observe actual walking and stop in MuJoCo. Retain ego/third-person images,
   observed root trajectory, table clearance and stopping drift in evaluation
   artifacts. Keep world coordinates out of the Agent's observation contract.
8. Test a centered staging start, a small lateral/yaw-offset start, and failure
   cases (out-of-entry, stale observations, cancellation). Reach starts through
   the existing motion owner; never teleport to manufacture a successful run.
9. Success requires in-tolerance pose **after** native stop/READY and subsequent
   measured rest. Check clearance in the GUI independently. No preparation,
   micro-motion or grasp is authorized by a `docked` return in this milestone.

The tool ignores image-only completion interrupts for this metric action;
operator cancellation and controller fault handling remain active. A failed
stop is never converted to success. A historical docking result is not live
proof of being docked. Navigation and restart invalidate it; later alignment
will need explicit fresh revalidation and a shared two-correction budget.

## Offline checks

```bash
python -m unittest \
  scripts.G1_eval.checks.test_station_docking \
  scripts.G1_eval.checks.test_docking_driver \
  scripts.G1_eval.checks.test_docking_native -v
```

These exercise message decoding, real local IPC, deterministic kinematics,
Driver guards and the native Skill/tool/queue/Controller handoff. They do not
exercise SONIC dynamics, obstacle clearance, an LLM, a real lab UI or grasping.
Run the existing navigation, native handoff, Executive and camera regressions
as well. Format changed Python with Google YAPF and the repository-scoped Ruff
configuration. Do not modify global environments to run the tests.
