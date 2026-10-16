# EMERGE → Official01 G1 skills

This integration adds one external EMERGE Driver that owns the existing
Official01 executive as a child process. It does not replace or edit the G1
control repository. The legacy parser, READY/ADJUSTING/INFERENCING/MOVING state
machine, pose store, GR00T worker, planner hold, and stability checks remain the
authority for robot execution.

## Call path

```text
Main Agent
  -> execute_robot_action(action_type, parameters)
  -> ACTION.md (one pending/running action)
  -> robot.controller
  -> Official01G1Driver (one process lock)
  -> existing Official01 text command whitelist/state machine
  -> SONIC + Inspire + GR00T
  -> READY/result -> ACTION.md -> Main Agent
```

For `navigate`, the Driver reuses the current LightNav image subscriber,
heading subscriber, WebSocket client, waypoint validation, and conservative
adapter. Its output is quantized to the executive's bounded movement commands,
so the executive remains the sole publisher on the SONIC action port.

## Action interface

| Action | Required input | Optional bounds | Completion result |
|---|---|---|---|
| `navigate` | `instruction` | `max_steps`, `control_hz`, `max_runtime_s` | LightNav reason and telemetry; semantic arrival remains unverified |
| `infer_gr00t` | `instruction` | `duration_s`, `timeout_s` | inference window ended, paused, and READY |
| `run_fixed_action` | `kind`, `name` | `timeout_s` | resource finished and native stability gate returned READY |
| `move_micro` | `direction` | `duration_s`, `timeout_s` | stopped and stable; command-derived maximum distance/angle |

All results contain `status`, `result`, and `reason`. Invalid input, state
rejection, stale inputs, subprocess failure, timeout, and stop/stability failure
are explicit failures.

## Registration and dry-run

```bash
cd ~/EMERGE-Humanoid
conda activate EmergePolicy
python -c "from robot.plugins import register_plugin; print(register_plugin('integrations/official01_g1_emerge'))"

python -m robot.controller \
  --driver official01_g1 \
  --driver-config integrations/official01_g1_emerge/driver_dry_run.json
```

The dry-run starts the existing executive with no DDS, ZMQ, camera, or policy
connection. It validates fixed actions, inference lifecycle, micro-motion state
transitions, queue serialization, and result handling. Real LightNav navigation
still requires live image and heading streams.

## Simulation launch

Start the existing simulation services exactly as documented in
`~/visionpro_sonic_bridge/integrations/official01_sim_executive/RUNBOOK.md`:
MuJoCo owner, SONIC deploy, GR00T PolicyServer, and LightNav server. Do **not**
start Terminal 4 (`run_executive.sh`) separately; this Driver starts that same
script and must be its sole owner.

Start the Controller from an interactive terminal. Before the managed executive
starts, the Controller runs `sudo -v` in that same terminal, so sudo can prompt
there for the password. The executive remains attached to the same terminal
session while its command input/output use private pipes:

```bash
cd ~/EMERGE-Humanoid
conda activate EmergePolicy
python -m robot.controller \
  --driver official01_g1 \
  --driver-config integrations/official01_g1_emerge/driver_sim.json
```

If the Controller is launched without a terminal (for example as a background
service), startup stops with an explicit interactive-terminal error. Run it in
a terminal so you can answer the sudo prompt.

Then start the Main Agent in another terminal:

```bash
cd ~/EMERGE-Humanoid
conda activate EmergePolicy
python -m Emerge agent
```

Use a preset safe observation stand-off for initial simulation tests. No action
result should be interpreted as precise grasp-position arrival without separate
visual or geometric verification.

## Real robot gate

Real mode is intentionally not provided as the default config. An operator must
create a reviewed config with `"runtime": "real"` and explicitly set this in
the Controller's terminal before startup:

```bash
export EMERGE_G1_CONFIRM_REAL=I_ACKNOWLEDGE_REAL_G1_CONTROL
```

The existing real launcher performs its own exclusivity check and live-robot
confirmation as well. Never set this variable from an Agent tool call.
