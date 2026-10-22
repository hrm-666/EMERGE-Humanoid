---
name: sim-station-docking
description: Finish the simulation apple-table approach using a named docking station and measured robot pose; use after LightNav staging for 近桌、贴近桌前沿、站点对接. Use measured station arrival before manipulation in the combined simulation task.
---

# Simulation Station Docking

Read `EMBODIED.md`, `ROBOT_STATE.md`, and `PLAN.md`. Use only the unified
`official01_g1` embodiment with `station_docking.mode` equal to
`simulation_docking_only` or `simulation_docking_and_grasp`. If the capability is absent, do not invent it.
Read `lightnav-sonic-navigation` for the preceding visual-navigation phase.

## Goal and evidence

LightNav finds the apple table and reaches its approach region. Its STOP, a
visible/centered apple, and Executive READY do not establish close approach.
Use a separate plan phase for station docking. Its Done Criterion is the
current `dock_to_station` action returning `result="docked"`,
`reason="pose_and_rest_verified"`, and Executive READY, plus observations
consistent with the intended table. Visual observations can invalidate a wrong
target or unsafe scene; they cannot replace the measured docking result.

The deterministic executor reads privileged simulation robot pose and an
operator-owned station registry. This is a simulation-assisted capability,
not proof of precise LightNav or pure visual navigation. Only the execution
layer uses world coordinates. Do not ask for those coordinates, the station
file, object metadata, arbitrary displacements, or target-pose overrides.

## Dispatch

Use only the native action queue. Do not launch another publisher or move the
robot by changing MuJoCo qpos.

```python
execute_robot_action(
    action_type="dock_to_station",
    parameters={"station": "apple_table_front"},
    reasoning="Finish the current table-approach phase and verify physical arrival before any manipulation.",
)
```

The Driver accepts only a station name. Station bounds, target pose, speed,
freshness, stability, and time/path budgets belong to operator configuration.
The operator configures the enabled station and its accepted resting tolerance.
Do not edit configuration to make a task pass.

The existing Executive remains the sole SONIC owner. Arrival requires both
position and heading tolerance, acknowledged stop/READY, and subsequent fresh
measurements showing low base translation and rotation for a stable dwell.
Support-enabled, missing/stale/reset pose, excess tilt, unexpected base height,
entry/corridor violation, exhausted budget, or failed stop cannot produce
`docked`. Metric error magnitudes are execution evidence, not apple reach.

## Recovery and this milestone's boundary

- `outside_entry_region` / `outside_entry_heading`: remain in the approach
  phase; obtain a fresh image and use a bounded, grounded LightNav recovery.
  Count it within the navigation Skill's existing attempt allowance.
- `drifted_after_stop`: inspect current pose/scene; do not declare success or
  automatically repeat forward motion. Preserve evidence for calibration.
- Stale/reset data, disabled station, control failure, bounds violations, or
  exhausted budget: stop and report the specific unresolved cause. A failed
  stop requires operator/runtime diagnosis before further robot actions.
- Cancellation: stop the action and do not automatically restart.

After successful docking, inspect `station_docking.mode`. In
`simulation_docking_only`, report the result and end the trial. In
`simulation_docking_and_grasp`, continue with `gr00t-manipulation`: observe,
run `prepare/right2`, then `pose/pick_ready_r`, re-observe, and attempt the grasp.
`grasp_ready=false` means docking alone does not establish arm preparation or
apple reach; `manipulation_allowed=true` permits the preparation sequence.
Do not repeat navigation or cosmetic alignment when the measured docking has
succeeded. Allow at most two total observation-driven `move_micro` corrections,
including failed-grasp recovery; the Driver also enforces this run's budget.
Navigation invalidates docking. Each manipulation action checks a fresh robot
pose; if it reports changed station pose, observe before deciding to redock.
Never treat READY or a docking result as proof that the apple was lifted.
