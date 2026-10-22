---
name: gr00t-manipulation
description: Prepare and execute bounded GR00T manipulation on the Official01 G1, coordinating standing-position correction with navigation and verifying physical effects. Use for 抓取、拿起苹果、操作 and grasp, pick up, manipulate requests.
---

# GR00T Manipulation

This Operational Skill uses the active `official01_g1` Controller and its single
Executive. Read `EMBODIED.md`, `ROBOT_STATE.md`, and `PLAN.md`. Use only supported
`execute_robot_action` calls. Do not start policies, shells, keyboard control,
second publishers, or a different Driver. This integration is simulation-only. End-to-end grasping in the room is not
yet validated; dry-run checks are not physical task validation.

## Readiness and Navigation Handoff

First inspect `station_docking.mode`. When it is
`simulation_docking_only`, use `sim-station-docking` and stop after the docking
trial. Do not run preparation, pose/trajectory, micro-motion, or GR00T actions;
the Driver rejects them in this milestone even after measured docking succeeds.
Neither a centered apple nor a visual `achieved` report overrides this mode.
In `simulation_docking_and_grasp`, first require a successful measured
`dock_to_station` result, then follow the preparation and grasp sequence below.
The Driver checks a fresh station pose before each manipulation action and
allows at most two total micro corrections in this acquisition workspace.
The remaining guidance applies only when docking-only mode is not active.

For a combined find/approach/grasp request, also read `lightnav-sonic-navigation`.
Use native planning and visual verification; the user should give only the task,
in Chinese or English. Translate the current policy subgoal into concise English
without adding unrequested tasks such as placing the apple in a drawer.

Read the navigation Skill's close-approach contract. Require fresh inputs,
Executive READY, identified apple/table context, and **positive evidence of
close approach** before preparation. Neither "the table is visible" nor "no
obstruction is reported" establishes proximity. If an approach gap remains,
return to navigation before starting GR00T; its whole-body output must not be
used to compensate for an incomplete approach.

Once close approach is supported, approximate alignment is sufficient for a
bounded simulation grasp trial. Do not demand certified metric reachability,
simultaneous apple/table-edge/floor visibility, or repeated cosmetic alignment.
An apple temporarily outside a close-up crop does not alone require retreat;
retain recent grounded target context and re-observe after preparation. Lack of
positive proximity evidence remains uncertainty, not trial readiness. Use the
native task verifier for these separate predicates. Do not invoke LIBERO
geometry or invent world coordinates from RGB.

Confirm empty hands before preparation. Known empty hands in the fresh
simulation or recent hand-state evidence may establish this; a camera view of
both hands is not mandatory. Stale input, wrong target, unresolved control
failure or clear obstruction requires stopping or correction.

Body translation/heading corrections belong to navigation (`move_micro`);
hand/arm preparation belongs here (`run_fixed_action`). Share navigation's
**two-pulse total near-table correction allowance across the entire acquisition,
including failed-grasp recovery**. Prefer preparation and one short grasp once
close, rather than spending both corrections to seek a perfect initial view.
After preparation or base motion, re-observe the apple/right-hand approach and
table clearance. Do not assume the pre-action geometry is unchanged.
Ask the verifier to describe whether the apple lies left/right of the prepared
right hand and whether the approach is obstructed, as well as proximity. Retain
that report in the plan context for comparison. The verifier sees only current
images; the Main Agent compares reports and chooses actions. A prior report
does not prove the target stayed put after body motion or contact.

## Preparation

Use only resources listed in the active profile and present in the configured
inventory. The current Official01 G1 policy follows the existing simulation
runbook: complete `prepare/right2`, then `pose/pick_ready_r`, before starting the
first grasp inference. The first action performs body preparation and its
terminal gate; the second establishes the recorded grasp arm pose and open-hand
targets. Neither action substitutes for the other. This sequence is specific to
the configured policy, not a universal preparation for all manipulation models.

Execute one action at a time. Require successful completion, Executive `READY`,
and fresh observations after each action; do not continue after a timeout or
tracking failure. Before opening/resetting the hands, verify they are empty.

```python
execute_robot_action(
    action_type="run_fixed_action",
    parameters={"kind": "prepare", "name": "right2", "timeout_s": 90.0},
    reasoning="The empty-handed robot is stopped and needs the configured body preparation.",
)
# Inspect the result and refresh state before dispatching the second action.
execute_robot_action(
    action_type="run_fixed_action",
    parameters={"kind": "pose", "name": "pick_ready_r", "timeout_s": 30.0},
    reasoning="Body preparation completed; establish the configured grasp arm and hand pose.",
)
```

Read the result and current state. Preparation completion does not prove target
reachability or grasp success. If preparation changes the body position or view,
recheck alignment. Do not repeat preparation during a grasp without evidence
that resetting the hand is appropriate. Legacy preparation is not immediately
preemptible; cancellation waits for it to finish or reports an unresolved stop.

## Bounded Policy Execution

```python
execute_robot_action(
    action_type="infer_gr00t",
    parameters={"instruction": "pick up the apple", "duration_s": 5.0, "timeout_s": 30.0},
    reasoning="Fresh observations support the current grasp phase after preparation.",
)
```

Use one current manipulation phase, not the whole navigation-and-grasp mission.
The selected checkpoint outputs a whole-body motion token and hand targets;
the body can translate during grasp inference. It is not an arm-only controller
and does not guarantee a fixed base position. Require visible clearance for
whole-body motion, and reassess body position as well as the hand/object after
each window. Do not extend duration merely because the hand has not closed.

Start with a 5 s observation window (Driver ceiling 60 s), then inspect the
apple/hand relationship before continuing. This limits unobserved whole-body
motion compared with the previous 10 s first window; it is not a displacement
bound or a claim that a grasp should finish in 5 s. An unfinished but progressing
reach can continue for another 5–10 s within the total window budget. The
Executive pauses inference at
window end and holds measured arm posture and the last hand targets while SONIC
continues controlling the body. `READY` and `semantic_task_verified=false` mean
execution ended, not that the apple was picked up.

## Verification and Recovery

After every inference, refresh state and verify that the intended apple is
visibly held and lifted clear of its support. Occlusion, a closed hand, elapsed
time, or an apple no longer visible on the table alone is insufficient evidence.
Use `delegate_subagent(agent_name="task_verification", task=...)` with these
observable predicates. Keep `uncertain` distinct from `achieved`.

If grasp is confirmed, end this user's task while retaining the hold. Do not
reset arms, open the hand, walk away, or place the apple unless requested.

After the **first failed grasp window**, diagnose before dispatching another:

1. Ask the verifier for current apple/hand contact, lift and relative alignment;
   compare its current and previous reports for body translation
   and the right-hand approach relative to the apple. Distinguish ongoing
   contact/progress, an empty-hand geometric miss, and insufficient evidence.
   If the apple is occluded and holding cannot be excluded, re-observe; do not
   reset/open hands or move the base on an assumption of failure.
2. For an empty-hand miss caused by a supported reach/heading/lateral error,
   add a localized recovery branch stating that error and the expected visual
   improvement. Request **one** navigation `move_micro` pulse within the shared
   two-pulse allowance. Require READY, empty hand and clearance; if the arm is
   engaged with the table, first use a supported empty-hand preparation/pose
   only when its path is clear. Do not guess a turn direction or use long
   LightNav movements near the table.
3. Re-observe after the correction. If the grasp displaced the arm/closed the
   empty hand, restore `pose/pick_ready_r` as needed and recheck the scene. Repeat
   `prepare/right2` only if its body preparation is actually invalidated, not
   automatically after every miss. Require each preparation to finish READY.
4. If alignment improved and inputs remain valid, retry one short window with
   the same concise task (`pick up the apple`). Changing only instruction wording
   is not a recovery for unchanged bad geometry. A no-correction continuation
   is appropriate only when observations support ongoing grasp progress or a
   specific non-geometric recovery, not mere hope that another window will help.

Allow at most five inference windows for the same grasp subgoal, counted from
ACTION.md across turns and recovery branches. This ceiling is not a requirement
to use all five. After two no-progress windows, reassess the failure hypothesis;
do not simply repeat, extend duration, restart budgets or manufacture a different
instruction. If the body moved substantially during GR00T, reassess proximity
and clearance; do not treat a large displacement as a small alignment error.
Stop on stale input, unresolved control errors, cancellation, exhausted budgets,
or no grounded recovery. Report uncertainty and remaining start-pose/checkpoint
limitations. These are Agent instructions, not hard Driver counters.


## Control Failure Is Not a Motion-Recovery Request

If an action cannot return the Executive to READY, stop dispatching robot actions
and report the unresolved controller state. A forward micro-motion, a new
navigation instruction, a zero/tiny duration, or another inference is not a stop
primitive and cannot recover a non-READY Executive. Only the operator/runtime
may diagnose and reset that control failure. Task-level visual recovery is
permitted only while the controller is healthy and READY.
