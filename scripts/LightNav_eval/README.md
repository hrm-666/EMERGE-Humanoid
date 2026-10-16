# LightNav-0 + SONIC navigation

Maintained on the humanoid-nav branch.

The built-in Operational Skill is
[Emerge/skills/lightnav-sonic-navigation/SKILL.md](../../Emerge/skills/lightnav-sonic-navigation/SKILL.md).
A Skill describes when and how the Agent uses tools; it does not contain model
weights or replace the Controller. This implementation follows the upstream
OpenPI separation of Skill, driver, policy client, and third-party source.

## Layout

| Responsibility | Location |
| --- | --- |
| Agent instructions | Emerge/skills/lightnav-sonic-navigation/SKILL.md |
| Built-in BaseDriver | robot/drivers/lightnav_sonic_driver.py |
| Policy client, bounded execution, observations, SONIC adapter | robot/navigation/lightnav_sonic/ |
| Robot capability profile | robot/profiles/lightnav_sonic_g1.md |
| Endpoint configurations | dev/lightnav_sonic_tabletop.json, dev/lightnav_sonic_procthor.json |
| Simulation launchers, evaluation, recording, checks | scripts/LightNav_eval/ |
| Upstream LightNav source (submodule) | third_party/lightnav/ |
| Upstream SONIC source (submodule) | third_party/sonic/ |
| Historical experiment notes | docs/lightnav/notes/ |

There is no top-level integrations directory on this branch. The driver is in
DRIVER_REGISTRY and requires no plugin registration, manifest, sys.path injection,
or workspace Skill copy. A legacy workspace Skill can still override the built-in
Skill: review it before use.

LightNav produces local navigation waypoints; SONIC provides locomotion. They
are distinct from OpenPI's manipulation policy. Navigation execution belongs
under robot/navigation; robot/vla remains the manipulation-policy execution
package. Both use the same Skill, Driver, and third-party source layers. The
model is served by the upstream LightNav WebSocket server directly;
no duplicate inference implementation is required in external_model_server.

## Dependencies

Initialize the pinned upstream sources:

~~~bash
git submodule update --init third_party/lightnav third_party/sonic
~~~

Follow their upstream installation instructions, including SONIC's nested
dependencies, LFS assets, model downloads, and build. The pinned revisions are
LightNav 0e9971784a04da2210bfccc446a68d45256e2894 and
SONIC 542842820d35bfc9a87b5bf11afd6fd9262da682.
Models use their own Python environments. Weights, engines, credentials,
recordings, and generated scenes are not committed.

The launcher defaults to the two third_party directories. Existing laboratory
installations can be reused without moving their virtual environments:

~~~bash
export LIGHTNAV_ROOT="$HOME/LightNav-0"
export GR00T_ROOT="$HOME/GR00T-WholeBodyControl"
export MODEL_PATH="$LIGHTNAV_ROOT/checkpoints/LightNav-0"
~~~

Install robot-side client dependencies in the EMERGE environment if missing:

~~~bash
python -m pip install -r scripts/LightNav_eval/requirements.txt
~~~

## Native execution

~~~text
Skill summary -> Astra reads SKILL.md -> update_plan
 -> execute_robot_action(navigate_vln) -> ACTION.md -> Controller
 -> SonicLightNavDriver -> RGB + instruction -> LightNav
 -> waypoint + yaw -> adapter -> SONIC -> simulated G1
 -> fresh observation -> task_verification -> update_plan
~~~

See [SIMULATION.md](SIMULATION.md) for the apple-table launcher and Controller.
Only the Controller may publish planner commands during native Agent execution.

The Skill allows at most five navigation actions per main-line subgoal,
including recovery actions. Verify after every action and stop early on success
or when no supported useful recovery remains. After two no-progress attempts,
change strategy. The global planning retry warning remains unchanged and calls
for reassessment; it is not a hard action limit. Five attempts are an Agent
instruction budget, not a controller-enforced counter or five guaranteed runs.
A new task prompt must not retain the old explicit two-attempt stop instruction.

## Validation and limits

The recorded Astra apple-table run executed two real LightNav GPU / SONIC
navigation actions, each followed by visual verification. Both were uncertain;
the target was not visible well enough to establish arrival. Recovery was
attempted, and no task success was claimed.

The directory migration and five-attempt instruction have only been tested
offline; the full GPU/SONIC experiment has not been rerun after this change.
The downward camera and target visibility remain known limitations. More
attempts do not establish success by themselves.

From the repository root:

~~~bash
PYTHONPATH=. ~/miniconda3/envs/EmergePolicy/bin/python -m unittest discover \
  -s scripts/LightNav_eval/checks -v
MUJOCO_GL=egl PYTHONPATH=. ~/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m unittest discover -s scripts/LightNav_eval/simulation_checks -v
~~~

These checks use synthetic messages, mock policy replies, and small MuJoCo
fixtures. They test interfaces, cancellation, observation freshness, plan
transitions, launchers, and recording; they do not measure navigation success.
Python formatting uses the Google YAPF style in this directory.

This branch is simulation-only. real-dev remains reserved for later hardware
work; this change adds no hardware deployment.
