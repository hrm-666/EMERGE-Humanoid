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

When `station_docking.mode` is `simulation_docking_only` or
`simulation_docking_and_grasp`, read `sim-station-docking`. The additional action
is `dock_to_station` with only `{"station": "apple_table_front"}`. LightNav reaches
the entry region; docking uses simulation robot pose through the same Executive.
Only its measured `docked` result establishes arrival. In docking-only mode,
report and stop. In docking-and-grasp mode, successful docking permits preparation
and grasp, with fresh pose checks before each action. Navigation invalidates
that handoff. The Driver enforces at most two total micro corrections per run.
Robot world coordinates stay inside the execution layer; object ground truth
remains evaluation-only. Docking does not itself prove reach or grasp success.

- Navigation stop or budget exhaustion does not certify close approach or a
  grasp pose. Require positive current/recent visual evidence of being at the
  identified table, not just target visibility, apple image centering or absence
  of obstacles. Complete a remaining approach gap through navigation before
  GR00T. Do not use GR00T's body motion to finish an incomplete approach.
  Once close, approximate alignment permits preparation and a short grasp trial;
  do not demand simultaneous apple/edge/floor visibility or calibrated reach.
  Do not retreat solely because a close-up hides the floor or apple.
- Allow at most two near-table micro corrections for the same acquisition,
  including failed-grasp recovery; use 0.25–0.5 s pulses with observation after
  each. Prefer reserving one for an observed miss. Do not reset this allowance
  through replanning. Diagnose the first failed grasp; correct a supported
  geometric error before retrying instead of changing only the text instruction.
  In docking-and-grasp mode the Driver additionally enforces the two-correction budget. Verify actual
  held-and-lifted evidence for grasp success; READY is only a control state.
- Simulation navigation preserves simultaneous bounded body translation and yaw
  through the single Executive and its command watchdog. Continuous navigation
  has not been enabled for real hardware.
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
