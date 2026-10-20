# Unified G1 room simulation

The integration worktree is `~/EMERGE-Humanoid-integration` on
`humanoid-integration`, based on `humanoid-nav`. The colleague's working directory
`~/EMERGE-Humanoid` is not modified. All commands below run as `embodied` and
require **no sudo and no password**. They use DDS interface `lo` and local TCP/IPC.
Do not mix these services with the colleague's network-namespace launch scripts.
The default shared simulation ports permit one active stack at a time.

## Dependencies and provenance

Runtime sources/weights stay in existing local installations:

- `~/visionpro_sonic_bridge`: room scene, camera, Executive, preparation resources.
- `~/GR00T-WholeBodyControl`: SONIC binary/weights and `.venv_sim`.
- `~/Isaac-GR00T-N1.7`: GR00T server and selected checkpoint.
- `~/LightNav-0`: LightNav code and checkpoint.

`GR00T_ROOT`, `ISAAC_ROOT`, `VISIONPRO_SONIC_ROOT`, and `LIGHTNAV_ROOT` may select
alternate existing installations. The Driver's `control_root` must agree with
`VISIONPRO_SONIC_ROOT`. No model is installed into the Agent environment.

The three colleague commits `e9b48aa`, `1272fa0`, `5d034eb` were cherry-picked
with their original authors. Reviewed uncommitted camera, observation and recording
changes were then ported with path mapping. LIBERO GR00T backend experiments were
not needed for this G1 path. A source snapshot was saved outside Git under
`~/.cache/emerge-integration/20261019_213229/`; the external dependency manifest
records source hashes, not model weights. Some external files are untracked in
their original repository, so this EMERGE branch alone is not a complete runtime
archive.

## One-time workspace initialization

Use a separate workspace so task resets do not affect another developer:

```bash
cd ~/EMERGE-Humanoid-integration
conda activate EmergePolicy
python - <<'PYCODE'
from pathlib import Path
from Emerge.runtime.workspace import reset_workspace_context
from Emerge.utils.helpers import sync_workspace_templates
workspace = Path.home() / ".Emerge/workspace-integration"
if not workspace.exists():
    sync_workspace_templates(workspace, silent=True)
    reset_workspace_context(workspace)
print(workspace)
PYCODE
```

This creates blank task state instead of treating bundled example objects as the
real room. Do not reset a workspace while its Agent or Controller is running.

## Service startup

Each service needs a persistent terminal. In each terminal first run:

```bash
cd ~/EMERGE-Humanoid-integration
conda activate EmergePolicy
```

Then launch, in order:

```bash
# Terminal 1: room viewer; retain startup support for now.
python -m scripts.G1_eval.local_sim sim
```

```bash
# Terminal 2: SONIC; after Init Done press ] once to start standing.
python -m scripts.G1_eval.local_sim sonic
```

```bash
# Terminal 3: GR00T checkpoint server on 127.0.0.1:5550.
python -m scripts.G1_eval.local_sim gr00t
```

```bash
# Terminal 4: LightNav on 127.0.0.1:8051.
python -m scripts.G1_eval.local_sim lightnav
```

LightNav uses eager mode, a 1 GiB KV cache and 0.50 GPU-memory setting to share
the 24 GiB GPU with GR00T and SONIC. The explicit KV cache overrides vLLM's
utilization-based cache sizing. This is a starting configuration, not a
universal memory guarantee; wait for each model to load and inspect actual use.

```bash
# Terminal 5: owns the only Executive; no separate Executive launcher.
python -m robot.controller \
  --driver official01_g1 \
  --driver-config dev/official01_g1_sim.json \
  --workspace "$HOME/.Emerge/workspace-integration"
```

Wait for `Watching ACTION.md`. Then focus the MuJoCo viewer and press `9` once;
confirm `ElasticBand enable: False`. Starting navigation while support is enabled
is not a valid movement test. Do not run the old standalone navigation Driver
alongside this Controller.

```bash
# Terminal 6: choose your own provider configuration explicitly.
python -m Emerge \
  --workspace "$HOME/.Emerge/workspace-integration" \
  --config "$HOME/.config/emerge-sun-gene/integration-runtime.json"
```

The shown config is Sun-Gene's personal runtime entry, not a repository file or
shared provider default. It requires the personal relay to be running. Other
users must supply their own config; the repository contains no company key.
A shared Linux account is not a security boundary: stop personal relay/tunnel
access after testing and remove transient lab relay credentials when finished.

Enter only the natural task in Chinese or English. Navigation and manipulation
Skills describe planning, readiness checks and bounded recovery. The current
Agent budgets are five navigation/correction actions per semantic position goal
and five inference windows per grasp goal. The Driver separately enforces
per-action limits. These budgets do not guarantee success.

## Acceptance and diagnostics

Check in increasing scope:

1. Room RGB + yaw arrive fresh; SONIC and Executive become ready.
2. After support release, one bounded micro-motion produces visible displacement
   or turning, followed by a stable hold. Sent command counts are not odometry.
3. LightNav uses current RGB and produces actual progress toward the intended table.
4. Observe readiness after stopping. Follow the configured policy runbook:
   `prepare/right2`, then `pose/pick_ready_r`, checking READY and fresh images
   after each. A centered apple is not by itself a reachable grasp pose.
5. Run GR00T; verify the intended apple is held and lifted from its support.
6. Submit the natural-language mission through the UI and retain its complete
   plan/actions, two videos and final verification.

The room's existing task-metric helper currently reports `available=false`
because the room has no `tabletop_plate`, which that helper requires. Never use that report
as a grasp-success metric. Visual evidence remains required until a room-specific
evaluation metric is implemented and validated.

Offline regressions (synthetic model/transport outputs, not policy-quality tests):

```bash
python -m unittest discover -s scripts/G1_eval/checks -v
python -m unittest discover -s scripts/LightNav_eval/checks -v
```

New/migrated Python uses Google-style YAPF formatting and scoped Ruff checks from
`scripts/LightNav_eval/ruff.toml`.

## Records, stop and cleanup

With `runRecording.enabled=true`, each UI task writes under:
`~/.Emerge/workspace-integration/runs/<run_id>/`:
`first_person.mp4`, `third_person.mp4`, `events.jsonl`, `terminal_output.txt`,
`runtime.log`, `request.json`, `result.json`, `recording.json`.
Recording failure is reported separately from task execution.

In the UI use `/stop`, wait for stopping acknowledgement, then `/quit`. Stop
Controller, LightNav, GR00T, SONIC and simulator in that order. Before stopping
the Controller or SONIC, enable startup support in the simulator and confirm
`ElasticBand enable: True`; do not toggle it if it is already enabled. The
Controller shutdown also stops SONIC. Do not use the
colleague's broad cleanup script against shared services. Fixed preparation is
not immediately interruptible by the legacy Executive; an unresolved stop must
remain a failure rather than being reported as successful cancellation.

Simulator/SONIC/model debug outputs default to `outputs/g1_integration/`, which
Git ignores. Use `--output-dir` with a unique directory to preserve separate
sessions. Inspect sizes with `du -sh ~/.Emerge/workspace-integration/runs/*
outputs/g1_integration` and archive/delete only completed runs you no longer need.
Archives still consume disk. Do not remove active logs, model checkpoints or
runtime sources as part of ordinary run cleanup.

## Integration trial on 2026-10-19

Run `e0d3d151e36f4080bff783a1f0f20506` used the real Astra provider, GPU
LightNav, and SONIC through the native UI. It performed four navigation phases
and one micro-turn, then reported that it could not see the apple. GR00T was
not invoked. The UI run status `completed` means the conversation finished;
it does **not** mean the physical mission succeeded.

The trial exposed repeated measured-arm recapture at each READY-to-MOVING
transition. The integration runtime now preserves the existing commanded arm
hold for navigation. Grasp pause still captures the current arm pose. This
prevents accumulating tracking error, but does not eliminate SONIC tracking
error or validate every preparation pose.

Measured simulator throughput was about 0.17 simulation seconds per wall second.
The integration simulator wrapper limits viewer synchronization to 30 Hz and
publishes base position, startup-support state, and measured realtime factor as
diagnostics. These diagnostics do not enter LightNav or GR00T policy inputs.
After a supported simulator reset, live diagnostics measured approximately 1.00
simulation seconds per wall second and the arms returned to the body's sides.
The complete navigation-and-grasp task still requires live validation.

A controller restart sends a stop to SONIC. With startup support released,
restarting the low-level stack can destabilize the robot; one recovery test fell
and its pose action timed out. Reset the simulator with startup support enabled
before restarting SONIC, press `]` to initialize, start the Controller, and only
then release support with `9`. Do not keep attempting actions from that failed
state or report the preparation as successful.

Subsequent support-off micro-motion tests still failed locomotion acceptance:
0.12 m/s commands reached SONIC and triggered SLOW_WALK replanning, but net
translation was only millimeters. The persistent arm target remained unchanged;
measured wrist-pitch error exceeded the existing readiness threshold and a stop
wait timed out. Do not interpret READY from the earlier trial, command-derived
distance limits, or completed action messages as verified locomotion. The
navigation/manipulation end-to-end task remains unvalidated.


### Controlled motion comparison and updated simulation speed

A support-off, four-second standard planner command at 0.12 m/s moved only about
7 mm, while 0.25 m/s moved about 0.75 m. With upper-body hold, the 0.25 m/s case
moved about 0.48 m. These are diagnostic observations, not guaranteed speed
tracking. A relative-frame backward trial did not establish reliable backward
motion; do not extrapolate the forward result to all directions.

The native Driver then used 0.25 m/s without changing readiness thresholds.
A two-second forward action moved about 0.166 m and returned READY. A subsequent
continuous sequence of two forward actions moved about 0.298 m and 0.227 m;
both returned READY, followed by a turn action that also returned READY. Turning
angle was not independently measured in that check. Startup support was off
during every measured action and enabled before closing the Executive.

`dev/official01_g1_sim.json` now selects `linear_speed_mps=0.25` for this simulation.
Generic/real-runtime defaults remain unchanged. This resolves the observed
forward-motion failure in the bounded checks; it does not validate LightNav's
visual goal selection, manipulation preparation, GR00T grasping, or the full UI
mission. The original failed UI run is retained as evidence.


### Navigation stop diagnostics on 2026-10-20

Real LightNav produced about 0.41 m and 0.52 m forward motion in two supported-
startup, support-off trials, then hit a stop timeout. Before support was restored,
arm speed was about 0.004 rad/s and arm motor tracking error about 0.061 rad,
while right wrist reference error was about 0.344 rad. SONIC's learned decoder
outputs motor targets distinct from its motion reference. The previous single
reference-pose gate conflated reaching a requested arm pose with navigation rest.

The integration simulation wrapper now distinguishes them. In MOVING only, a
requested stop requires zero commanded motion, valid motor feedback, arm motor
tracking or reference-pose error within the existing 0.30 rad threshold,
arm speed at most 0.15 rad/s,
and the original lower-body tracking, tilt, gyro and continuous-hold checks.
ADJUSTING keeps the external reference-pose gate. Arm references are not
recaptured and reference error is logged; READY never certifies grasp readiness.
This is a simulation integration change, not validation of a real-robot stop.

A subsequent real LightNav rerun completed 20 ticks in 6.364 s, moved about
0.581 m forward with support off, requested policy stop and returned READY.
This validates execution and navigation rest for that trial; it does not prove
semantic arrival, reachability, grasp preparation or the complete UI mission.

A post-grasp recovery exposed the complementary case: reference error 0.155 rad
and motor-target offset 0.459 rad while arm speed was 0.0053 rad/s. The rest
gate therefore retains the original reached-reference path as well as the
learned-decoder tracking path. Both require low arm speed and all body guards.
The new post-grasp case is covered by regression. Micro-motion return values
are named commanded_distance_m/commanded_angle_rad, not physical upper bounds.


### Full UI trial and remaining manipulation gap

Run `2544ebdcbcf84ea5b3a6df3252b6d9a1` executed the Chinese natural mission with
real Astra high. It selected navigation, right2, pick_ready_r, a 10-second GR00T
window, visual verification, and a recovery branch. The apple stayed on the
table; the verifier correctly reported failure. Navigation recovery then hit
the post-grasp stop issue described above. A later attempted micro-motion in
MOVING was correctly rejected by the Driver. Skills now forbid inventing a
stop/recovery primitive from a small forward action.

The UI conversation completed with an explicit failure report; this is **not**
physical task success. Both videos finalized with 11,452 frames at 10 fps, and
all events/results remain in the run directory. The mission still needs a
validated standing-position handoff and successful grasp evidence.


### Controlled grasp follow-up

The selected `NEW_EMBODIMENT` processor uses one `ego_view` RGB frame, 44 state
values, and a 40-step action horizon containing a 64D `motion_token` and two
6D hand targets. It is a whole-body policy: base motion during inference is
expected to be possible. This interface check does not establish its training
start distribution or success in the room.

Controlled run `/tmp/emerge-integration-nav-20261020_010111` used LightNav, two
1.5-second forward corrections, right2, pick_ready_r, and 10 seconds of GR00T.
All phases returned READY with startup support off; the apple remained on the
table. The base was about x=1.2 m before and after grasping.

Run `/tmp/emerge-integration-nav-20261020_010611` requested 30 seconds of GR00T.
Its starting pose differed: the base was about x=0.67 m after preparation.
GR00T moved the body forward and brought the hand near the apple, but an
evaluation-only base-position guard cancelled inference after 19.741 seconds
at x=1.372 m. The run is therefore censored, not a completed 30-second grasp
failure. Images do not establish a held, lifted apple. Startup support was
restored before closing the Executive. These are controlled diagnostics, not
autonomous UI acceptance or a fixed recipe for the navigation handoff.

The room table is rotated 90 degrees; its near tabletop edge is approximately
x=1.679 m and its top is z=0.764 m. The original static table is 8 cm lower.
Room geometry and images require their own grasp validation; increasing Agent
iterations or inference duration cannot by itself establish reachability.


Follow-up `/tmp/emerge-integration-nav-20261020_011434` used a fresh simulator
and an evaluation base guard at x=1.45 m after checking the table geometry.
After navigation/correction/preparation, the base was x=1.174 m. GR00T moved
forward and triggered cancellation after 9.640 seconds; the paused base was
x=1.492 m, illustrating stopping displacement beyond the trigger. No completed
30-second trial or successful grasp was established. The support-off run remained
upright and support was restored before shutdown. Do not keep widening an
evaluation boundary as a substitute for validating the whole-body policy's
start conditions and allowed movement. These coordinates are evaluation-only.

Remaining acceptance work: reproduce a known-good checkpoint start condition,
validate it in the room (including the 8 cm table-height difference), establish
an observable navigation-to-grasp handoff, and rerun the unmodified natural task
through the UI. No successful room grasp baseline has yet been supplied or
established by these tests.


### Early navigation stop: interface fix and bounded comparisons (2026-10-20)

The UI run `5d694b68df224974837dab20c9865790` stopped at LightNav tick 32.
The policy emitted its stop token and an all-zero waypoint chunk. At that instant
base x was about 0.678 m and the near table edge x about 1.679 m: approximately
1.00 m of base-to-edge separation along world X. The apple was near the top of
the ego image, not its center. This was a real model stop, not a decoded stop bug.
These coordinates are evaluation evidence, never inputs to the learned policy.

Two integration defects were also present. A 0.12 m waypoint deadband discarded
real 0.108 m forward waypoints; converting the remaining waypoints to one keyboard
key discarded simultaneous translation/turning and the commanded speed magnitude.
The simulation adapter now uses a 0.02 m / 0.02 rad deadband and sends bounded
body-frame `(forward_mps, lateral_mps, yaw_rate_rad_s)` through the existing single
Executive. Planar/yaw limits come from the Executive configuration. Both client
and runtime reject non-finite/excess commands and invalid control states. The
existing 0.4 s command watchdog still zeros motion if commands stop arriving.
A non-stop zero waypoint sends zero velocity without READY/MOVING cycling;
a model stop still ends the action. `move_micro` retains its original protocol.
This does not enable continuous navigation on real hardware.

Native Skill instructions now preserve close-approach intent and the complete
visual criterion. Seeing the apple/table edge, policy stop and READY cannot
substitute for grasp reachability. An incomplete approach remains active;
one evidence-grounded local approach can be replanned within the same five-action
budget. The user still supplies only their natural-language task. These are
Agent instructions, not a newly enforced geometric grasp-readiness gate.

Support-off GPU/SONIC comparisons, with support restored before owner shutdown:

| Trial | Base X start -> stopped (m) | Result |
|---|---|---|
| Old adapter, original instruction | -0.091 -> 0.459 | Policy stop, READY, 15 ticks |
| Fixed adapter, original instruction | -0.048 -> 0.757 | Policy stop, READY, 26 ticks |
| Fixed adapter, `Approach the apple table until it is within arm reach. Face the apple.` | 0.106 -> 1.027 | Policy stop, READY, 21 ticks |
| Fixed adapter, close approach followed by `Move closer to the front edge of the table with the apple.` | -0.098 -> 1.416 | Second action cancelled by evaluation boundary, READY |

These are individual diagnostic trials, not success-rate measurements or perfectly
matched starting poses. The last trial crossed the x=1.35 m evaluation boundary
and stopped at x=1.416 m; it did **not** demonstrate a policy-selected final stop.
The boundary was never passed to LightNav and must not become a task controller.
All trials remained upright at about real time. No grasp policy or company LLM
was called by these comparisons; full UI planning and successful grasping have
not been revalidated by this diagnostic.

Evidence is retained locally under `outputs/g1_integration/` (Git-ignored):
`nav-baseline-20261020_122347`, `nav-continuous-20261020_122602`,
`nav-arm-reach-20261020_123505`, and `nav-local-recovery-20261020_123903`.
Each measured trial has `result.json`, `telemetry.json` and `frames/`.
The new trials also retain their diagnostic script. The separate
`navigation-stop-replay.json` records open-loop replay of the old 32 ego frames:
changing wording can remove stop on those frames but does not establish closed-
loop arrival. Do not present replay or the evaluation cancellation as success.

Upstream `LightNav-0/docs/EVAL_HABITAT.md` reports a 3.0 m VLN-CE success radius;
ObjectNav uses distances to goal viewpoints. Those benchmark definitions do not
certify a G1 grasp pose, and are not a hard-coded stop radius in this integration.
There is no demonstrated reliable close-range stopping/readiness gate yet. Next
acceptance requires an observable, calibrated grasp-start envelope, comparison
against known-good GR00T starts in this room, and autonomous UI verification of
approach/recovery without the evaluation boundary supplying task completion.


### UI feedback: approximate near-table grasp trial (2026-10-20)

Run `2c8b8421cbaa4328b69c45a6fccb9136` used real Astra high and the fixed adapter.
The Agent navigated from base x=-0.200 m to x=1.187 m, then to x=1.612 m; both
navigation actions ended on policy STOP and returned READY. It then demanded
simultaneous apple/table-front/floor visibility and retreated instead of trying
preparation. The user judged the near-table pose suitable for a trial from the
MuJoCo view, rejected that overly strict visual gate, and requested cancellation
and a fresh UI run. Cancellation was acknowledged and both videos finalized.

The navigation/manipulation Skills and embodiment now record the user's intended
simulation contract: approximately close and stopped at the identified target
is sufficient to attempt the configured preparation and a bounded GR00T window.
Recent target context is retained when the apple temporarily leaves a close-up
crop. Metric reachability and simultaneous apple/edge/floor visibility are not
preconditions. Missing those alone must not trigger retreat. This changes trial
eligibility, not success reporting: actual held-and-lifted evidence is still
required. Fresh inputs, empty-hand preparation, successful pose gates, READY,
and checks for concrete obstruction/wrong-target evidence remain in force.
