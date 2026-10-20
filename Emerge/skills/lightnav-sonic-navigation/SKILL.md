---
name: lightnav-sonic-navigation
description: Find and approach a visually identifiable destination with LightNav-0 and SONIC; adjust G1 standing position and heading before manipulation. Use for 寻找、走到、导航、站位调整 and find, approach, walk, align requests.
---

# lightnav-sonic-navigation — Visual Navigation

This Operational Skill uses LightNav-0 for local visual-language waypoints and
SONIC for locomotion. Invoke it through the native `execute_robot_action` tool;
the active Controller owns dispatch, cancellation, and the planner publisher.

## Preconditions and Grounding

Read `EMBODIED.md`, `ROBOT_STATE.md`, and the current `PLAN.md`. Use this Skill
only when the active embodiment is `lightnav_sonic_g1` or `official01_g1`
and lists `navigate_vln` (the unified driver also accepts `navigate`).
The LightNav service, simulation RGB stream, and SONIC yaw feedback must be
available. The driver checks live inputs during execution; a configured/ready
connection state alone does not prove that every service is healthy.

Bind the action to the current Main Line pointer or top recovery branch. Use
`update_plan` for all plan changes and provide a complete snapshot. Before the
first action, establish an observable Done Criterion for the destination. If
needed, use `delegate_subagent` with `agent_name="task_verification"` to assess
whether the intended target is identifiable in the current ego view. Missing or
expired observations and ambiguous targets are uncertainty, never success.

This embodiment exposes a single RGB view without calibrated world X/Y. Do not
request metric target localization from the LIBERO `object_location` specialist,
invent world coordinates, or claim a measured distance or physical stability
from an image. If the user's criterion needs unavailable measurements, report
that limitation instead of silently weakening it.

## Phase-Local Parameters

Use one short imperative sentence naming only the active destination and local
navigation intent. For a find/approach/grasp mission, preserve the close-approach
intent, for example `Approach the apple table until it is within arm reach.
Face the apple.` A vague `move toward the table and stop` can cause an early
policy stop. This wording is a policy request, not a reachability measurement.
For a navigation-only mission, retain the user's requested stopping criterion.
Do not copy the full Mission or include a future grasp/place operation.

| Parameter | Contract |
|---|---|
| `instruction` | Required non-empty string; one navigation phase. |
| `max_steps` | Positive integer; default 120, at most the driver ceiling (120 by default). Includes stale-input ticks. |
| `control_hz` | Finite positive number; use the configured default, normally 4 Hz. |
| `max_runtime_s` | Finite positive number; default 120 s, at most the configured runtime ceiling. |

Steps and runtime are budgets, not distance or guaranteed completion. At 4 Hz,
120 ticks take at least about 30 seconds when all ticks run; inference can slow
the loop. Cancellation/runtime expiry is checked between operations and after
inference; an outstanding request is bounded by `inference_timeout_s` (10 s by
default), rather than being preempted immediately.

```python
execute_robot_action(
    action_type="navigate_vln",
    parameters={
        "instruction": "Approach the apple table until it is within arm reach. Face the apple.",
        "max_steps": 120,
        "control_hz": 4.0,
        "max_runtime_s": 120.0,
    },
    reasoning="The active plan phase requires a close approach to the apple table before grasp readiness is assessed.",
)
```

Do not launch standalone bridge runners or keyboard teleoperation from shell
commands: only the Controller may publish SONIC planner commands in this mode.

## Execution Postconditions

The tool returns an action-queue status and a textual result containing JSON
execution evidence (`status`, `reason`, `ticks`, `command_steps`, `elapsed_s`,
`lightnav_stop`, last waypoint/yaw, and any error). `ROBOT_STATE.md` records the
navigation state. The standalone driver attempts locomotion IDLE on exit.
The unified `official01_g1` driver stops locomotion and waits for Executive
`READY`, retaining arm/hand hold. In the integration simulation, first-waypoint commands
preserve simultaneous translation and yaw, bounded by the Executive's configured
speed limits. Commands expire through its existing watchdog; the Agent cannot
bypass it. A small non-stop waypoint is not treated as a completed phase.
Neither is proof of semantic arrival; never
start a second publisher or switch drivers between navigation and manipulation.

| Backend result | Meaning and next step |
|---|---|
| `completed` / `lightnav_stop` | Policy requested stop; verify the Done Criterion. |
| `budget_exhausted` / `max_steps` or `max_runtime_s` | Budget ended; inspect progress and verify before deciding whether to continue. |
| `cancelled` | Action interrupted; inspect the reason and any current visual-monitor evidence. |
| `failed` | Execution failed, including stale inputs or inference errors; diagnose before retrying. |

The queue may label a normally returned budget as `completed`. Neither that
label, LightNav stop, nor a last waypoint is proof that the destination was
reached. Invalid requests can return `Failed:` without backend JSON.

## Verification and Recovery

Allow at most **five navigation attempts per main-line subgoal**, including the
initial action and all recovery actions. Count dispatched navigation actions
from ACTION.md after the current mission began, including actions on recovery
branches; keep this count across turns and replans of the same semantic goal.
Verification calls do not consume this budget. Do not confuse this total with
PLAN.md Retries, which counts failures or lack of progress.

After each attempt, refresh state and verify before deciding whether to act
again. Stop early when the goal is achieved, the user cancels, live inputs are
unavailable, or no useful supported recovery remains. After the fifth attempt,
report any unresolved criterion and do not dispatch a sixth action. This is an
Agent instruction budget, not a controller-enforced action counter.

After every action, re-read `ROBOT_STATE.md` and obtain current visual evidence
through the native Evaluation Skill:

```python
delegate_subagent(
    agent_name="task_verification",
    task=(
        "Read the active PLAN.md Done Criterion and verify every predicate "
        "from fresh observations. For a close approach before grasping, report "
        "target identification, heading and evidence for a suitable standing "
        "position separately. Seeing the apple or table edge does not establish "
        "reachability. Report uncertain for conditions the available view "
        "cannot establish; do not infer metric distance from pixels."
    ),
)
```

The example criterion is only suitable when it matches the user's actual goal.
The specialist uses `observe_scene` and `submit_task_verification`, returning
`achieved`, `not_achieved`, or `uncertain` with predicate evidence. If the native
visual monitor already supplies a current verification of the same full
criterion, reuse that evidence instead of duplicating the call.

1. On `achieved`, use `update_plan(update_kind="progress", ...)` to complete
   the current phase/branch and advance the pointer only as appropriate.
2. On `not_achieved` or `uncertain`, keep the main subgoal active. Continue a
   bounded phase only if current evidence shows useful progress and the target
   is unambiguous. A budget boundary alone is not a failure or a recovery plan.
3. For a blocked/no-progress phase, increment its retries and push a localized
   Branch Stack entry using `update_plan`. Each branch needs its own observable
   criterion. Use `update_kind="replan"` when changing the Main Line itself.
4. After two consecutive no-progress calls, change strategy before continuing
   within the five-attempt budget. The planning tool's retry warning requires
   reassessment/replanning, not an automatic termination after three attempts.
   Use a supported, evidence-grounded recovery with its own observable criterion.
   Do not invent a scan/turn primitive, use an unsupported manipulation action,
   reset retries, or increase budgets as the sole response to wrong-target
   behavior or absent observations.

This Skill is simulation-only. Hardware deployment belongs to future `real-dev`
work and is not supplied by this Skill.

## Handoff to Manipulation on `official01_g1`

For a request such as "寻找放有苹果的桌子，走到桌子面前抓起苹果" or
"Find the table with an apple, walk up to it, and pick up the apple", read
`gr00t-manipulation` as well. The user need not name Skills or supply budgets.
Use the native plan to separate approach, manipulation readiness and grasp
verification; adapt these phases to observations rather than replaying a fixed
sequence. Send short English phase instructions to the learned policies even
when the user speaks Chinese; respond to the user in their language.

Separate target discovery, close approach, and grasp readiness in the plan.
Finding the table does not complete a close-approach subgoal. A visible near edge,
a centered apple, a LightNav stop, and Executive READY do not establish that the
robot is close enough to grasp. Pass the full active criterion to verification;
do not replace it with the weaker question of whether the table is visible.
READY can support the stopped/controller predicate only. It cannot override an
uncertain distance, alignment or reachability verdict.

If the policy stops while the fresh view still shows an incomplete approach,
keep that subgoal active and consider a more explicit close-approach instruction
within the existing attempt budget. For example, if the table is still ahead
and fresh observations support a clear approach, use `Move closer to the front
edge of the table with the apple.` Re-verify the original criterion afterward;
a changed instruction is not evidence of success. Do not override the model's stop with a
blind forward command. Obtain fresh images before refining standing position.
Do not repeatedly drive forward just because the apple is off-center or hidden.
Check the known scene, table edge, robot heading and visible clearance first.

Only the unified driver supports `move_micro`. Use `execute_robot_action` with
`action_type="move_micro"` and `parameters={"direction": "turn_left",
"duration_s": 0.5, "timeout_s": 15}` for one grounded correction. Supported
`direction` values are `forward`, `backward`, `left`, `right`, `turn_left`,
`turn_right`. Default to 0.25–0.5 s pulses; never exceed the configured 2 s ceiling.
Refresh observation after every pulse. Returned commanded distance and angle
are command integrals, not measured displacement or guaranteed physical bounds. Do not infer clearance in occluded
space or use uncalibrated image pixels as metric distances.

Count these dispatched corrections with navigation actions toward the existing
five-attempt limit for the same standing-position subgoal. Do not evade this
limit by renaming or replanning the goal. Use early stopping and the existing
no-progress recovery rules. If the initial approach is achieved, a separately
planned grasp-alignment subgoal may have its own five-attempt budget; this must
not disguise an unsuccessful approach.

Let the manipulation Skill define required reachability and preparation.
If it finds the base pose unsuitable, keep grasping pending and correct the
standing position through this Skill. After any base motion, invalidate prior
readiness conclusions and re-observe; after an arm preparation changes the view,
recheck the target. Never use an arm reset/preparation while carrying an object.


## Control Failure Is Not a Motion-Recovery Request

If an action cannot return the Executive to READY, stop dispatching robot actions
and report the unresolved controller state. A forward micro-motion, a new
navigation instruction, a zero/tiny duration, or another inference is not a stop
primitive and cannot recover a non-READY Executive. Only the operator/runtime
may diagnose and reset that control failure. Task-level visual recovery is
permitted only while the controller is healthy and READY.
