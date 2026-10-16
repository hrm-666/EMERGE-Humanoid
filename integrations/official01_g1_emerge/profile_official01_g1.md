# Official01 Unitree G1 + Inspire Embodiment

## Identity and control ownership

- **Driver:** `official01_g1`
- **Runtime:** selected by the operator (`sim`, `real`, or `dry-run`)
- **Control owner:** one persistent Official01 executive managed by the Driver
- **State machine:** `READY`, `ADJUSTING`, `INFERENCING`, `MOVING`

Read the built-in `official01-g1-control` Skill before using this embodiment.
All physical commands must use `execute_robot_action`; never start a shell,
keyboard teleoperation process, or second SONIC publisher from the Main Agent.

## Supported actions

- `navigate`: bounded LightNav navigation. Completion does not prove semantic arrival.
- `infer_gr00t`: bounded natural-language policy inference, followed by `pause` and READY.
- `run_fixed_action`: allow-listed `prepare`, `pose`, or `trajectory` resource.
- `move_micro`: one bounded body-relative translation or turn, followed by stop and stability wait.
- `navigate_vln`: backward-compatible alias for `navigate`.

Only one queue action runs at a time. The executive additionally rejects state-
invalid commands. Every locomotion exit sends `stop` and waits until the native
stability gate reports READY. Treat missing observations, timeouts, command
rejection, or inability to return READY as failures.

## Known limitations

- Navigation stop or budget exhaustion is not evidence of an accurate grasp pose.
- Micro-motion distance/angle is an upper bound derived from command rate and
  duration; it is not odometry-confirmed displacement.
- GR00T inference is time-bounded because the current policy protocol has no
  semantic task-completion result. Verify the task separately from observations.
- Real mode requires a human-set process environment confirmation before the
  Controller starts. Simulation is the default validation target.
