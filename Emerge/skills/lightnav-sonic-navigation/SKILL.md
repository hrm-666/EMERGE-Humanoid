---
name: lightnav-sonic-navigation
description: Use LightNav-0 and SONIC to navigate a simulated Unitree G1 toward a visually identifiable destination through bounded navigate_vln actions, then verify the active plan criterion.
---

# lightnav-sonic-navigation — Visual Navigation

This Operational Skill uses LightNav-0 for local visual-language waypoints and
SONIC for locomotion. Invoke it through the native `execute_robot_action` tool;
the active Controller owns dispatch, cancellation, and the planner publisher.

## Preconditions and Grounding

Read `EMBODIED.md`, `ROBOT_STATE.md`, and the current `PLAN.md`. Use this Skill
only when the active embodiment is `lightnav_sonic_g1` and lists `navigate_vln`.
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
navigation intent, for example `move toward the table with the apple and stop`.
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
        "instruction": "move toward the table with the apple and stop",
        "max_steps": 120,
        "control_hz": 4.0,
        "max_runtime_s": 120.0,
    },
    reasoning="The active plan phase is approaching the visible apple table.",
)
```

Do not launch standalone bridge runners or keyboard teleoperation from shell
commands: only the Controller may publish SONIC planner commands in this mode.

## Execution Postconditions

The tool returns an action-queue status and a textual result containing JSON
execution evidence (`status`, `reason`, `ticks`, `command_steps`, `elapsed_s`,
`lightnav_stop`, last waypoint/yaw, and any error). `ROBOT_STATE.md` records the
navigation state. The driver attempts locomotion IDLE and closes action resources
on exit; this is not sensor-confirmed standing stability or a SONIC shutdown.

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
        "Verify the active navigation Done Criterion from the current ego view. "
        "Check whether the intended apple table is identifiable ahead and its "
        "near edge is visible in the foreground. Report uncertain if the view "
        "cannot establish a condition; do not estimate metric distance."
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
