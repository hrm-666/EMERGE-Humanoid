---
name: official01-g1-control
description: Safely invoke LightNav navigation, bounded GR00T inference, fixed actions, and micro motion on the Official01 G1 through one serialized executive.
---

# Official01 G1 Control

Use this Skill only when `EMBODIED.md` selects the `official01_g1` driver. Read
`ROBOT_STATE.md` before and after every action. Call only
`execute_robot_action`; never invoke the G1 launchers, executive text protocol,
or SONIC publisher with `exec`.

## Actions

### `navigate`

```python
execute_robot_action(
    action_type="navigate",
    parameters={
        "instruction": "move toward the table with the apple and stop",
        "max_steps": 80,
        "control_hz": 4.0,
        "max_runtime_s": 60.0,
    },
    reasoning="Approach a preset safe observation stand-off before manipulation.",
)
```

`instruction` is required. Budgets are positive and capped by the Driver.
`navigate_vln` remains an alias for existing plans. A returned LightNav stop or
budget boundary means only that the navigation phase ended. Verify visual
arrival and never claim accurate grasp alignment without localization evidence.

### `infer_gr00t`

```python
execute_robot_action(
    action_type="infer_gr00t",
    parameters={
        "instruction": "pick up the apple",
        "duration_s": 10.0,
        "timeout_s": 30.0,
    },
    reasoning="Run the current manipulation phase for a bounded window.",
)
```

The Driver accepts inference only from READY, pauses it at the end of the time
window, and waits for READY. Completion is not semantic success; verify the
effect from current observations.

### `run_fixed_action`

```python
execute_robot_action(
    action_type="run_fixed_action",
    parameters={"kind": "pose", "name": "pick_ready_r", "timeout_s": 30.0},
    reasoning="Move to an existing validated preparation pose.",
)
```

`kind` is `prepare`, `pose`, or `trajectory`. Resource names must exist in the
operator-configured Official01 inventory; arbitrary paths and shell strings are
rejected. The validated preparation is `kind="prepare", name="right2"`.

### `move_micro`

```python
execute_robot_action(
    action_type="move_micro",
    parameters={"direction": "forward", "duration_s": 0.5, "timeout_s": 15.0},
    reasoning="Make one conservative body-relative correction in clear space.",
)
```

Directions are `forward`, `backward`, `left`, `right`, `turn_left`, and
`turn_right`. Duration is positive and capped by the Driver. The returned
distance or angle is a command-derived maximum, not measured odometry. Do not
chain another action until the result and `ROBOT_STATE.md` both show READY.

## Result contract and recovery

Every action returns text containing one JSON object with `status`, `result`,
and `reason`, plus action-specific evidence. `failed` means diagnose before a
retry. `cancelled` means the Driver attempted its normal stop/pause path.
`completed` means the executive returned READY; it does not by itself verify a
navigation or manipulation goal.

After navigation or inference, use current visual evidence and the active plan's
observable criterion. If evidence is missing or ambiguous, report uncertainty.
Do not compensate by increasing motion budgets indefinitely.
