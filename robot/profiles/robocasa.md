# RoboCasa PandaOmron Embodiment

Driver: `robocasa`

This embodiment uses RoboCasa's PandaOmron mobile manipulator. The driver supports
`vla_execute`, `move_to_pose`, `move_linear`, `set_gripper`, and `move_base`.
Use `object_location` and `task_verification` to inform planning and check outcomes.

**Remember to keep the plan up to date.**

Update and review the plan as execution progresses through its phases.

## Success Check

After every `execute_robot_action`, re-read `ROBOT_STATE.md`.
If the fresh `robots.robocasa.success` value is `true`, stop immediately.
Otherwise reassess the current subgoal, using `task_verification` when needed.
Action completion does not establish task success.

If `robots.robocasa.done` is `true` or `robots.robocasa.ready` is `false`,
stop sending robot actions.

## Task Routes

Choose the applicable route for the current subgoal. Organize composite tasks
into phases according to the original instruction. Continue from the state
already reached, retaining all unfinished object, spatial-relation, and
appliance-state requirements.

### Fixed Structures and Large Destinations

Identify the target structure and its actual interaction parts from the task
and current scene:

- Cabinets, drawers, and refrigerators: distinguish doors, drawers, interior
  spaces, and specified shelves.
- Microwaves and toaster ovens: distinguish doors, interior placement regions,
  start buttons, and timer knobs.
- Dishwashers: distinguish doors, sliding racks, and dish-placement regions.
- Stoves: distinguish the target burner and its controlling knob.
- Sinks: distinguish the basin, water outlet, and faucet controls.
- Coffee machines: distinguish the cup-placement region and start button.
- Other appliances: distinguish the blender lid and jar opening, stand-mixer
  head, electric-kettle switch, and toaster slots and lever as required.
- Counters, dining tables, and containers: identify the final supporting surface
  or the interior of a basket, tray, bowl, cup, pot, or other container.

Before carrying an object beside, above, or through these structures,
call `object_location` to measure the relevant objects and structures'
`position_m` and `size_m`. Plan around their full spatial extent,
leaving clearance for the gripper and held object. A structure's overall bounding
box does not establish its usable interior space.

Let VLA handle visual approach, handle contact, opening or closing doors,
pulling drawers or racks, pressing buttons, and turning knobs as coherent
phases. Use attach for measured motion through clear space.

When a relation such as "initially inside the drawer or cabinet" only identifies
an object that is already visible and accessible, locate it directly. Operate
the structure only when access requires it or the task explicitly requests it.

### Movable-Object Tasks

For ordinary pick-and-place tasks, use:

`locate object -> attach approach -> VLA grasp -> locate destination -> attach lift and transport -> VLA placement`

Maintain the same target instance. When transporting bowls, baskets, trays,
pots, or other containers holding food or objects, preserve the required
orientation and contents. Leave final alignment, contact, and release to VLA.

Call `task_verification` when target identity, grasp state, access, or a phase's
outcome is unclear. When the outcome is clear, continue to the next phase without
inserting a verification call at every transition.

### Continuous-Contact and Composite Tasks

Let VLA perform washing, scrubbing, stirring, pouring, and similar operations as
coherent phases with complete action semantics. Merely moving a tool near the
target does not complete these operations.

Connect the routes above according to the original composite-task instruction.
For example, load food before carrying its container, place the cup before
starting the coffee machine, and load dishes before closing the dishwasher.
Preserve required object counts, categories, relative positions, container
contents, and final door or switch states. After a failure, recover the
unfinished phases from the current state.

### CloseBlenderLid (task: "Close the lid blender by securely placing the lid on top.")

- Use a measured attach approach and VLA to grasp and seat the lid. Once it is
  centered and level on the rim, open in place with `set_gripper`.
- Adjust the release pose first if a fingertip is visibly caught. If the lid is
  seated and the fingertips are clear or obscured, try one 1 cm `move_linear`
  along a clear direction with the gripper open, the current wrist orientation,
  and `position_tolerance_m: 0.002`.
- Save a side view before the probe and compare the same view afterward. Nudge
  or retreat only if the fingertips are visibly clear and the lid stayed put.
  Verify again if unclear. If the lid is off-center on the rim, use `move_linear`
  to nudge it horizontally inward from the overhanging side by 5–10 mm
  (`position_tolerance_m: 0.002`). Recheck and continue only if it improves.
  Once flush, keep the gripper open, retreat diagonally up and toward the robot
  along a clear path, and check success once >0.15 m from the lid.

### CoffeeSetupMug (task: "Pick the mug from the counter and place it under the coffee machine dispenser.")

- Expose the machine; localize the mug and perforated drip tray (not the dispenser
  face or front counter); re-localize after base motion or machine contact.
- Keep the mug upright and gripper closed; prefer one continuous VLA phase without
  small base/arm moves; if stalled, route around an exposed machine end in short segments.
- Place centered on the tray, release when supported, then withdraw the open gripper
  horizontally by >0.25 m before lifting.

### TurnOffStove

- Identify the instructed burner's knob (hub + radial grip), not the oven-handle
  mounts; clear the arm ~0.20 m away and 0.15 m up for a close oblique view.
- Let VLA approach, engage, and rotate as one continuous phase; if a chunk ends with
  the panel visible, continue the same native instruction without manual moves.
- After each chunk, confirm the knob row is in view; if the stove leaves the frame,
  restore the view/stance first. Wrist rotation or gripper closure does not prove
  the burner is off.

### TurnOnSinkFaucet (task: "Turn on the sink faucet.")

- Target the short thin lever, not the arched spout or outlet; localize its raised
  surface; re-localize after any base or wrist change.
- Approach via a nearby hover in <0.30 m segments; straighten a tilted wrist, descend
  from ~0.08 m above to ~0.02 m below the lever's upper surface, close, lift; use short
  geometric steps or a bounded VLA chunk, not one long VLA.
- Check official success and visible water; a grasp, moved spout, or timeout is not
  success. If unchanged and water is off, retract, re-localize, retry; adjust the base
  if reach/visibility is poor, or restage the wrist and try a bounded full-body VLA.

## Target Identity

For ambiguous targets, combine:

`object category + visual features + original task relation`

### Object Appearance Hints

Use these hints to support identification. Combine the current instance's
shape, color, texture, labels, and original task relation. Do not treat one
model's color scheme as a fixed property of its entire category.

- **Mug (`mug`):** An open cup body with a side handle; plain and printed models
  exist. Do not assume it is white or confuse it with a handleless drinking glass.
- **Drinking cup / glass (`cup` / `glass_cup`):** A handleless open vessel, often
  cylindrical or wider at the top. Both transparent and opaque models exist;
  transparency alone does not identify the target.
- **Bowl (`bowl`):** An open container with noticeable depth and curved sidewalls
  forming a cavity. Distinguish it from a shallow plate, tray, or inverted lid.
- **Plate (`plate`):** A shallow, wide, circular dish with a low rim; plain and
  patterned models exist.
- **Colander (`colander`):** A bowl-like container with dense holes or mesh in its
  walls or bottom, sometimes with side handles or feet; not a solid bowl.
- **Food-storage container (`tupperware`):** A relatively deep open box, often
  rectangular, with clear sidewalls and interior space; not a low-sided tray.
- **Tray (`tray`):** A broad, shallow, flat carrying surface with low edges,
  sometimes with handles; not a deep food-storage box or rimless cutting board.
- **Basket (`basket`):** A relatively deep carrying space, with wire-mesh and
  woven models in round or rectangular shapes. Do not identify it solely by
  brown coloring or one particular weave.
- **Pan (`pan`):** A shallow circular body with a long side handle; distinguish
  it from a plate or deep pot.
- **Pot (`pot`):** A relatively deep body with side handles or loop handles;
  distinguish it from pans and bowls using wall depth and handle shape.
- **Non-electric kettle (`kettle_non_electric`):** A body with a spout and a top
  or side handle, in metal or ceramic variants; not a coffee machine, cup, or
  electric kettle with a powered base.

- **Cutting board (`cutting_board`):** A flat board, often rectangular or
  rounded-rectangular with wood grain and a flat main surface; not a rimmed tray
  or digital scale with a display.
- **Digital scale (`digital_scale`):** A base with a weighing platform, display,
  and controls; do not mistake a nearby board or ordinary tray for its platform.
- **Knife (`knife`):** A handle joined to a relatively long, flat blade; not a
  thin straw, utensil handle, or food skewer.
- **Peeler (`peeler`):** A handle joined to an open blade frame, including
  Y-shaped models; not a knife with a continuous long blade.
- **Spatula (`spatula`):** A long handle ending in a broad, flat head; not a
  spoon or ladle with a concave head.
- **Fork / spoon / ladle (`fork` / `spoon` / `ladle`):** Identify them by tines,
  a shallow concave head, or a deep scooping head respectively. Color and handle
  length alone do not determine the category.
- **Tongs / whisk (`tongs` / `whisk`):** Tongs have opposing arms; a whisk ends
  in wire loops or a cage-like structure rather than a solid spatula head.
- **Sponge (`sponge`):** A thick, porous object, with green-yellow layered,
  single-color, and rounded models; not thin food or a sharply edged ice cube.
- **Straw (`straw`):** A very thin tube, sometimes bent, with varying colors;
  distinguish it from a knife, utensil handle, or skewer carrying food.

- **Bread (`bread`):** Includes round loaves and irregular pieces with a baked
  crust, cracks, or scoring; do not assume all bread is square sandwich bread.
- **Sandwich bread (`sandwich_bread`):** A flat, roughly square slice with a
  pale center and darker crust; not a whole loaf or hotdog bun.
- **Croissant (`croissant`):** Golden to brown, with rolled layers and transverse
  surface texture; may be crescent-shaped or short and spindle-shaped, unlike
  a round cupcake.
- **Baguette / hotdog bun (`baguette` / `hotdog_bun`):** Both may be elongated.
  Baguettes often have surface scoring; hotdog buns have a lengthwise split.
  Distinguish them using the actual cuts and task name.
- **Pancake / waffle (`pancake` / `waffle`):** Pancakes are flat round cakes or
  stacked layers, sometimes with toppings; waffles have a distinct grid pattern.
- **Cupcake (`cupcake`):** A cup-shaped cake body or paper liner, sometimes
  topped with raised cream, frosting, or decorations; not stacked pancakes.
- **Sausage / kebab skewer (`sausage` / `kebab_skewer`):** A sausage is one
  continuous elongated meat product; a kebab has a rod through separate food
  pieces. Do not identify them only by their elongated outline.
- **Chicken drumstick (`chicken_drumstick`):** A thick meaty end tapering toward
  a leg bone or bone handle; not a sausage of relatively uniform thickness.
- **Steak (`steak`):** A flat or thick meat cut with visible meat texture, fat
  edges, or grill marks. Raw and cooked variants differ in color; red or brown
  coloring alone is insufficient.

- **Lettuce (`lettuce`):** Wrinkled leaves, layered edges, or visible veins,
  in whole-head or leaf-like variants; not broccoli's dense flower buds.
- **Broccoli (`broccoli`):** A dense, granular crown with a thick stalk, unlike
  lettuce's spreading or curled leaves.
- **Carrot (`carrot`):** A mainly orange, elongated tapered root, sometimes with
  a green top; distinguish it from a cucumber with relatively rounded ends.
- **Cucumber / eggplant (`cucumber` / `eggplant`):** Cucumbers are often elongated
  with rounded ends and may have longitudinal texture. Eggplants may be long
  or round and typically have a distinct calyx. Combine shape, skin, and task name.
- **Strawberry (`strawberry`):** Mainly red with dense surface seeds and a conical
  or heart-like outline; a green leafy crown is not clearly visible on every model.
- **Lemon wedge (`lemon_wedge`):** Fan-shaped or wedge-shaped flesh with peel on
  the curved edge. This asset library includes green models; do not exclude one
  because it is not yellow.
- **Ice cube (`ice_cube`):** A white or translucent angular block, often roughly
  cubic; not a porous sponge or a piece of fruit with peel.

- **Ketchup / condiment bottle (`ketchup` / `condiment_bottle`):** Combine sauce
  color, bottle outline, cap, and readable label; flat bottles, slender bottles,
  and different packaging exist.
- **Shaker (`shaker`):** Often a relatively short container; examine the dispensing
  lid and body rather than relying only on color to distinguish it from a bottle.
- **Jam (`jam`):** A lidded jar or cup of food, including glass jars and plastic
  cups. Combine the label, lid, and contents; do not assume transparency or a
  checkered lid.
- **Bottled water (`bottled_water`):** Often has a screw cap, body ribs, and a
  wraparound label; a transparent bottle alone does not establish bottled water.
- **Alcohol (`alcohol`):** Includes beer, wine, and other instances with varying
  bottle shapes, colors, and labels. Use the neck, cap, label, and task reference
  group to distinguish them from bottled water.
- **Boxed / bagged food (`boxed_food` / `bagged_food`):** Identify rigid boxes
  versus flexible bags; determine the actual food from readable labels and the
  original instruction.

- **Blender lid:** A flat lid matching the blender jar opening, sometimes with
  a central protrusion or knob; not the jar, appliance base, or a nearby plate.

### Identity Binding and Tracking

- Bind an instance using its original task relation, such as "the cup initially
  beside the sink" or "the bowl initially inside the left cabinet".
- Bind visually identical objects using their initial relations and positions.
  When initial left/right relations identify objects, retain those roles;
  changing the camera or robot viewpoint does not redefine them.
- After an object moves, track the same instance through observed motion,
  action history, and its last known position.
- Another object of the same category near the source does not invalidate an
  identity already established through motion tracking.
- If tracking continuity is lost, re-establish identity from evidence and action
  history. Do not invent visual differences or silently switch instances.

### Identity Information for Subagents

The `object_location` specialist does not inherit this profile. Every request
must be self-contained: provide the target's appearance, original task relation,
relevant movement history, and necessary exclusions. Category names such as
cup, bowl, or bottle alone are insufficient. Use the same identity information
in `task_verification` requests.

## RoboCasa Control Conventions

This embodiment uses the official episode instruction. Copy
`robots.robocasa.task.instruction` exactly into every `vla_execute` call; keep
the current local phase in `PLAN.md` and the action reasoning rather than
rewriting the policy prompt. `step` is a positive policy action budget.

Follow the attach and VLA skills for action parameters, step budgets, and recovery.

Arm geometric actions use world coordinates in metres. The target position refers
to the end-effector grip site, not the object center. Read the current end-effector
pose from `eef_pose_world`; quaternions use XYZW order. Do not use
`eef_pose_relative` directly as a world-frame target.

`move_to_pose` holds the world-frame grip-site orientation measured at action
start when orientation is omitted. An explicit `orientation_quat` or
`orientation_euler` overrides it; both position and orientation must meet the
configured pose tolerances for `pose_reached`.

Geometric motion preserves the last executed gripper command by default,
including after VLA. A successful environment reset establishes an `open`
command for the new episode without adding simulation steps; this does not
assert that the fingers have already fully opened. If `last_gripper_close` is `null`, explicitly provide
`gripper_command` (`open` / `close`) based on the current grasp state. Keep the
gripper closed while carrying an object; do not open it merely to initialize it.

`set_gripper` supports only `open` and `close`, not continuous opening targets.

### Local Base Movement

Use `move_base` for small stance changes when visibility, reach, or approach
direction is unsuitable. Choose a visibly clear local path before moving;
this primitive does not plan around obstacles or establish contact with fixtures.

```python
execute_robot_action(
    action_type="move_base",
    parameters={"forward_m": 0.10, "lateral_m": 0.0, "yaw_rad": 0.0, "max_steps": 60},
    reasoning="Adjust the stance along a clear path before continuing VLA.",
)
```

Distances are measured in the base frame at call start: `forward_m` follows
local +X, `lateral_m` follows local +Y, and positive `yaw_rad` turns counterclockwise.
Omitted distances and angle default to zero. Each call allows at most 0.3 m
combined translation, pi/4 radians rotation, and 1–100 `max_steps` (default 60).
The arm follows the moving base, torso motion is not requested, and the last
gripper intent is preserved; `gripper_command` can explicitly select `open` or `close`.

`base_pose_reached` means the base target was reached and braking steps were
applied within the budget. `stalled` means the path is blocked: do not repeat
the same request; choose a different direction or leave the adjustment to VLA.
`not_converged` means the budget ran out before arrival. Neither establishes
arrival. The final two steps of a request are braking, so use at least three
steps for any motion.
A zero-distance, zero-angle request completes without advancing physics.
Read actual `base_delta_m`, `base_yaw_rad` and pose errors in `last_execution`.
All movement and braking steps share the episode budget. After moving, inspect
fresh images and robot state, and re-localize motion targets as needed. The next
VLA call rebuilds its history while keeping the original task instruction.
