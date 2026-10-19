# Official01 Unitree G1 + Inspire Embodiment

## Identity and control ownership

- **Driver:** `official01_g1`
- **Runtime:** selected by the operator (`sim`, `real`, or `dry-run`)
- **Control owner:** one persistent Official01 executive managed by the Driver
- **State machine:** `READY`, `ADJUSTING`, `INFERENCING`, `MOVING`

Read `lightnav-sonic-navigation` for approach/standing-position corrections and
`gr00t-manipulation` for preparation/grasp. For a combined request, read both.
The Agent plans the task; the Executive only manages execution modes.
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
- Micro-motion `commanded_distance_m`/`commanded_angle_rad` integrate the
  configured command over its duration. They are not measured motion or hard
  physical bounds; gait startup, lag and stopping dynamics can change displacement.
- The selected GR00T policy controls whole-body motion and both hands. Its
  inference may translate the base; it does not guarantee stationary grasping.
- GR00T inference is time-bounded because the current policy protocol has no
  semantic task-completion result. Verify the task separately from observations.
- Real mode requires a human-set process environment confirmation before the
  Controller starts. Simulation is the default validation target.

## Policy and preparation contract

The current simulation uses the separately launched GR00T checkpoint selected by
the operator. The existing policy runbook
requires `prepare/right2` followed by `pose/pick_ready_r` before the first grasp.
The former performs body preparation; the latter sets the recorded grasp arm/hand
pose. Check successful completion and current observations after each action.
Their existence is checked by the Driver, but suitability after arbitrary
navigation is an integration acceptance question, not a guaranteed capability.

`READY` after navigation means zero commanded locomotion, stable body feedback,
low arm speed, and either attained arm reference or tracking of SONIC motor
targets. It does not certify reaching
the upper-body reference or the grasp preparation pose; the latter retains its
separate reference-pose gate. Arm/hand references are retained. Navigation and
GR00T share this same owner. No calibrated object localization or measured base
X/Y is exposed to this Agent. Simulation ground truth is reserved for evaluation.
Do not use the LIBERO manipulation Skill or its coordinate primitives here.
