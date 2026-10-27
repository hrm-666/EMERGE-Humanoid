# RoboCasa PandaOmron Embodiment

Driver: `robocasa`

The embodiment uses a PandaOmron mobile manipulator in RoboCasa. The current
driver supports `vla_execute` for robot control. Use `object_location` and
`task_verification` to inform planning and check outcomes.

## Plan Management

Keep the plan current with the task's phases. Preserve every required object,
destination, relation, and final fixture state from the original instruction.
Before the first robot action, inspect the current scene using the visual tools
and resolve ambiguous targets or prerequisites. Use `task_verification` when
visibility, grasp state, or a fixture's state is unclear.

After each phase, update the plan from fresh observations and execution history.
Advance a phase only when its required outcome is supported by evidence.

## Success Check

After every `execute_robot_action`, re-read `ROBOT_STATE.md`. If the fresh
`robots.robocasa.success` value is `true`, stop immediately. Otherwise,
reassess the current subgoal using `task_verification` when needed.

`last_execution.completed` means that a call finished or reached the goal;
using up the requested steps alone does not establish task success. Visual
verification guides planning; the environment's `success` flag determines
benchmark success. If verification suggests completion while `success` remains
`false`, review unmet task conditions before choosing the next action.

If `done` is true or `ready` is false, stop sending robot actions for this
episode. Inspect `evaluation.termination_reason`, `evaluation.error_message`,
`setup_error`, and `last_execution.error_message` / `last_execution.error_stage`
when present. An error or cancellation may follow actions that already executed;
inspect `last_execution.total_steps` and current images before replanning.
A failed observation publication may leave older images, so check the manifest's
`episode_id` and `revision` before treating them as evidence of the latest action.

## Task Route

Choose the applicable route before the first robot action. Multi-stage tasks
may combine the routes, with prerequisite states checked before each phase.

### Current Control Capability

All robot motion currently uses `vla_execute`. This is an implementation limit,
not a task-specific instruction to skip localization, verification, or planning.
`attach` and its geometric actions (`move_to_pose`, `move_linear`,
`set_gripper`, and `follow_arc`) are not connected to this driver. Do not call
them. LIBERO's measured attach approach and transport route requires that
control integration before it can be used here.

### Fixed Structures and Large Destinations

Identify the actual cabinet, drawer, appliance, sink, shelf, or destination
from the task and current images. Distinguish identical structures using their
location, appearance, and relation to visible reference objects. Use scene
evidence rather than assuming LIBERO fixture names or placement regions.

Before approaching a confined destination or moving an object beside or
through a structure, call `object_location` for the relevant visible object,
destination, and clearance-critical structure. Use returned `position_m` and
`size_m`, when successfully measured, to reason about full spatial extents,
access, and clearance. Do not infer an occluded interior's free space from the
fixture's outer bounding box. If access or structure state is unclear, call
`task_verification` before acting.

Let VLA handle visual approach, handle contact, and opening or closing an
articulated structure as a coherent phase. Verify the resulting state before
starting a phase that depends on it. When the task requires a final door,
drawer, or appliance state, retain that requirement in the plan.

When a relation such as "initially in the drawer" only identifies a movable
object, it does not itself require opening or closing the drawer. If the
object is visible and accessible, localize it directly. Act on the structure
only when access or an explicit task condition requires it.

### Movable-Object Tasks

Use the following route for an ordinary pick-and-place task with the current
driver:

`localize object -> VLA approach and grasp -> verify grasp -> localize destination -> VLA transport and place -> verify placement`

Bind the source object before moving it. Verify that the intended object is
held before transporting it; proximity of the gripper alone does not prove a
grasp. Inspect destination geometry and access before transport. After placing,
verify release and the requested support or containment relation. Keep any
additional task requirements in the plan.

## Target Identity

For an ambiguous target, combine:

`object class + visual signature + original task relation`

### Object Appearance Hints

Use attributes actually visible in the current RoboCasa scene: shape, size,
color, texture, handles, spouts, packaging, and readable labels. Combine cues
rather than relying on color alone. A task category may have different visual
instances across episodes; do not reuse LIBERO product appearances as facts.
If the views cannot establish identity, report the ambiguity and request visual
verification rather than inventing a distinguishing attribute.

### Identity Binding and Tracking

- Bind an instance using its original task relation, such as "the mug initially
  beside the sink" or "the bowl initially in the left cabinet".
- Bind visually identical objects by their initial relation and location.
  Preserve initial left/right roles when the task uses them to identify objects;
  a changed camera or robot viewpoint does not redefine those roles.
- After an object moves, track the same instance through observed motion,
  action history, and its last known location.
- Another same-class object near the source does not invalidate an identity
  already tracked through motion.
- If continuity is lost, re-establish it from evidence and action history.
  Do not invent a visual difference or silently switch instances.

### Identity Information for Subagents

Make every `object_location` and `task_verification` request self-contained;
do not assume the specialist inherits this profile or the full action history.
Include the target's observed appearance, original task relation, reference
objects, exclusions, and relevant movement history. A category name such as
cup, bowl, or bottle alone is insufficient when several candidates exist.

Each VLA instruction describes the current coherent action phase while
preserving the original object's identity and required destination relation.
Do not include unrelated later phases. If the selected checkpoint explicitly
requires the complete task sentence, follow that requirement instead of
rewriting it into phase instructions.

## Tool Responsibilities

- `object_location`: identify and localize the current source, destination,
  or clearance-critical structure from calibrated views. Treat `found: false`
  as unlocated; never substitute guessed coordinates.
- `vla_execute`: perform the current phase's approach, grasp, lift, transport,
  alignment, placement, or visually guided interaction with a structure.
- `task_verification`: check task relations, grasp and release, accessibility,
  and uncertain structure states. An uncertain result is not confirmation.
- `attach`: pending geometric control integration; unavailable in this driver.

### VLA Calls and Recovery

Always supply a non-empty `instruction` and a positive integer `step` to
`vla_execute`. `prompt` is an accepted alias; if both are supplied they must
agree. Choose a bounded step request for the current phase and inspect the
result before continuing. Respect `remaining_steps` when evaluation exposes it;
the driver enforces the episode budget. A step request is not a task-success
claim.

After an interruption or unsuccessful phase, read the new state and available
observations, verify the current relation, and revise the instruction or plan.
Do not blindly repeat an action that may already have changed the scene. Stop
if the episode has ended or the environment is unavailable.

## Observation and Coordinate Conventions

Use the latest `artifacts/observations/observation.json` for visual tools.
It lists the available calibrated views and `reference_view`; image paths are
relative to the workspace. Current publications contain RGB only
(`rgbd: false`). Missing or occluded evidence does not establish a task relation.

Localization uses the observation manifest's `coordinate_frame` (`world` in
this adapter). Camera optical axes are +X right, +Y down, +Z forward;
`T_world_camera` maps camera points into world coordinates, in metres.

Robot state has different frame labels: `base_pose` is in world coordinates,
while `eef_pose_relative` is relative to the robot base. Robot quaternions use
XYZW order. The relative EEF position comes from the site and its orientation
from the body; do not treat that pair as a calibrated site transform or compare
it directly with world-frame localization. Workspace `objects` are perception
annotations; editing them does not move simulated objects.
