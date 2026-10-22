---
name: lightnav-sonic-navigation
description: Find and approach a visually identifiable destination with LightNav-0 and SONIC; adjust G1 standing position and heading before manipulation. Use for 寻找、走到、导航、站位调整 and find, approach, walk, align requests.
---

# lightnav-sonic-navigation — Visual Navigation

This Operational Skill uses LightNav-0 for local visual-language waypoints and
SONIC for locomotion. Invoke it through the native `execute_robot_action` tool;
the active Controller owns dispatch, cancellation, and the planner publisher.

## Preconditions and Grounding

For `official01_g1` with `station_docking.mode` equal to
`simulation_docking_only` or `simulation_docking_and_grasp`, read
`sim-station-docking`. LightNav finds the apple table and reaches the approach
region; then call `dock_to_station` to finish the approach. STOP and visual
verification cannot replace measured docking. In docking-only mode, end after
docking. In docking-and-grasp mode, continue with preparation and grasp after
`docked`; do not continue navigating closer after that measured success.
After LightNav ends, call dock_to_station as the next robot action. Do not require a second visual close-approach verdict before the measured docking attempt. The Agent receives no world coordinates. A disabled station is not permission
to fall back to a qualitative close-table judgment.

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
intent from the first call, for example `Walk up to the front edge of the
table with the apple and stop close enough to reach the apple.` A vague `move toward the table and stop` can cause an early
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
        "instruction": "Walk up to the front edge of the table with the apple and stop close enough to reach the apple.",
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
        "Assess close approach to the identified apple table, separately from "
        "target recognition. Cite positive visual evidence that the robot is "
        "at the table, rather than merely facing it across a remaining gap. "
        "Use the current view: report a visible approach gap as "
        "not_achieved; insufficient proximity evidence as uncertain. STOP, "
        "READY, apple image centering and absence of obstacles are not proof "
        "of proximity. Do not demand simultaneous apple/edge/floor visibility "
        "or invent metric distances. Describe visible apple/right-hand "
        "alignment separately; grasp success is a later criterion."
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

### Close approach before preparation

Separate **target found**, **close approach**, and **grasp alignment**. Finding
the apple, centering it in the image, a policy STOP, and READY do not establish
close approach. Before advancing the approach subgoal, cite positive evidence
from current/recent ego observations that the base is at the intended table:
for example, the previously visible approach gap has closed and the near table
boundary is immediately ahead in the image sequence. Explain the actual cues;
do not infer proximity just because no obstacle or contradiction was reported.
These are qualitative trial-readiness cues, not calibrated reach measurements.
The task verifier sees current camera images only. The Main Agent may compare
its previous grounded reports with the new report; do not ask the verifier to
see an unavailable image history or to choose a robot action.
Do not require the apple, table edge and floor to appear simultaneously. A
close-up crop alone neither establishes nor disproves proximity; keep recent
target context when the apple or floor leaves the image.

Make the first LightNav request explicitly seek the front edge and a close
stop. Honor its STOP, then verify. If a substantial approach gap remains, keep
the same approach subgoal active and request a bounded local close approach,
such as `Move closer to the front edge of the table with the apple.` Verify
again before preparation. This is a new, evidence-grounded action, not ignoring
STOP or blindly appending forward motion. Prefer one such local completion;
if it also stops far away without useful progress, reassess and report the
unresolved approach rather than letting GR00T walk the missing distance. Do not
claim one model call is guaranteed to produce a grasp-ready stop.

On insufficient proximity evidence, request a fresh observation and explain the
uncertainty. Do not silently advance to grasp, invent a metric distance, or
retreat merely to recover a wide floor view. Once close approach is supported,
do not keep calling long LightNav actions to optimize a cosmetic camera view.
Proceed to preparation, with a small correction only for a concrete alignment
error. Do not require a perfect pose before the first bounded grasp trial.

### At most two small near-table corrections

For the same apple-table acquisition, allow **at most two `move_micro` actions
in total**, including pre-grasp alignment and corrections after failed grasps.
Prefer zero or one before the first grasp, retaining one for an observed miss.
Count from ACTION.md across turns, branches and replans; a renamed alignment
subgoal does not reset the allowance. These actions also count toward the
existing five navigation-action budget for this acquisition. This is an Agent
instruction budget, not a Driver-enforced displacement/count limit.

Only the unified driver supports `move_micro`. Use `execute_robot_action` with
`action_type="move_micro"` and `parameters={"direction": "turn_left",
"duration_s": 0.5, "timeout_s": 15}` for one grounded correction. Supported
directions are `forward`, `backward`, `left`, `right`, `turn_left`, `turn_right`.
Use a 0.25–0.5 s pulse for near-table alignment; do not use the general 2 s
Driver ceiling here, chain pulses without observation, or replace a needed
close approach with a long blind micro-motion. At the current 0.25 m/s and
0.25 rad/s settings, these pulses command approximately 6–12.5 cm or 3.6–7.2
degrees. These are command integrals, **not measured displacement or physical
bounds**: gait startup and stopping dynamics can produce different motion.

Choose one error to correct, cite the evidence and predict its visual effect.
Use a small turn for a supported heading mismatch, lateral motion for a
supported sideways offset, and forward motion only for a small remaining reach
gap with visible clearance. Image centering alone is not right-hand alignment;
consider the apple relative to the prepared right-hand approach. If the direction
cannot be grounded, do not guess from an occluded apple. Re-observe after each
pulse and compare the predicted effect before using the remaining allowance.

Require READY and confirmed empty hands before a post-grasp base correction.
Do not move the base when the hand may be holding the apple or is engaged with
the table. Let `gr00t-manipulation` choose the supported empty-hand pose reset
when needed. After any base motion, invalidate prior readiness conclusions and
re-observe. After preparation changes the view, recheck target context without
reinstating the simultaneous-visibility gate. Never reset arms while carrying.
When the correction allowance is exhausted, do not disguise another correction
as a new LightNav/GR00T instruction; report the remaining alignment limitation.


## Control Failure Is Not a Motion-Recovery Request

If an action cannot return the Executive to READY, stop dispatching robot actions
and report the unresolved controller state. A forward micro-motion, a new
navigation instruction, a zero/tiny duration, or another inference is not a stop
primitive and cannot recover a non-READY Executive. Only the operator/runtime
may diagnose and reset that control failure. Task-level visual recovery is
permitted only while the controller is healthy and READY.
