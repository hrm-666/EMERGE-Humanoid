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

For a combined find/approach/grasp request, also read `lightnav-sonic-navigation`.
Use native planning and visual verification; the user should give only the task,
in Chinese or English. Translate the current policy subgoal into concise English
without adding unrequested tasks such as placing the apple in a drawer.

Before inference, require current observation and Executive `READY`. Check:

- The intended apple and relevant table surface are identifiable.
- The robot is stopped; the standing position and heading are compatible with
  the active policy's documented training/start conditions and visible clearance.
- The configured preparation has completed and the target remains visible after
  the camera view changes. A centered apple alone does not establish reachability.
- The hand is not already holding an object that preparation would release.

Use the native `task_verification` specialist for visual criteria. Distinguish
visible evidence from calibrated reachability: RGB alone cannot certify metric
reach, contact force or grasp stability. Do not invoke LIBERO-specific geometry
or invent a world pose. If the initial conditions cannot be established, report
uncertainty or request a supported standing-position correction rather than
claiming readiness from a successful action status.

Carry the full readiness criterion into the verification request. A verifier's
uncertain reach/alignment verdict stays uncertain even when the Controller is
READY. READY establishes the control state, not the object's reachability.
If navigation merely found the table, keep the incomplete approach active;
do not use a grasp window to compensate for missing approach evidence.

Body translation/heading corrections belong to navigation (`move_micro`).
Hand/arm preparation belongs here (`run_fixed_action`). After a correction,
re-observe and re-evaluate the same readiness criterion before inference.

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
    parameters={"instruction": "pick up the apple", "duration_s": 10.0, "timeout_s": 30.0},
    reasoning="Fresh observations support the current grasp phase after preparation.",
)
```

Use one current manipulation phase, not the whole navigation-and-grasp mission.
The selected checkpoint outputs a whole-body motion token and hand targets;
the body can translate during grasp inference. It is not an arm-only controller
and does not guarantee a fixed base position. Require visible clearance for
whole-body motion, and reassess body position as well as the hand/object after
each window. Do not extend duration merely because the hand has not closed.

Start with a 10 s window (Driver ceiling 60 s). The Executive pauses inference at
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
If a partial grasp is progressing, another bounded inference window may be
appropriate without resetting the arms. If the target is out of reach, stop
inference, verify empty-hand state and obtain navigation correction first.

Allow at most five inference windows for the same grasp subgoal, counted from
ACTION.md across turns and recovery branches. After two no-progress windows,
change an evidence-supported strategy instead of only increasing duration.
Stop on stale inputs, unresolved control errors, cancellation, exhausted budget,
or no grounded recovery. Preserve uncertainty in the final report. These are
Agent instructions, not a hard Driver counter.


## Control Failure Is Not a Motion-Recovery Request

If an action cannot return the Executive to READY, stop dispatching robot actions
and report the unresolved controller state. A forward micro-motion, a new
navigation instruction, a zero/tiny duration, or another inference is not a stop
primitive and cannot recover a non-READY Executive. Only the operator/runtime
may diagnose and reset that control failure. Task-level visual recovery is
permitted only while the controller is healthy and READY.
