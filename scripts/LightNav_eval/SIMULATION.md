# 实验室仿真入口

在 `embodied@embodied-MS-7E06:~/EMERGE-Humanoid`、`humanoid-nav` 使用本说明。
原始代码和历史运行产物仍在 `~/LightNav-0`；新的仿真代码统一在本仓库维护。

## 验证

EMERGE 接入检查（61 项，无 GPU/MuJoCo）：

```bash
cd ~/EMERGE-Humanoid
PYTHONPATH=. ~/miniconda3/envs/EmergePolicy/bin/python -m unittest discover \
  -s scripts/LightNav_eval/checks -v
```

苹果桌仿真检查（80 项，使用已有 MuJoCo/OpenCV 环境）：

```bash
cd ~/EMERGE-Humanoid
MUJOCO_GL=egl PYTHONPATH=. ~/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m unittest discover -s scripts/LightNav_eval/simulation_checks -v
```

该组检查用合成消息和小型 MuJoCo 场景验证坐标、取消、复位、随机起点及录像评估；
WebSocket/ZMQ 检查绑定临时 loopback 端口。无需模型 checkpoint 或 SONIC deployment。
不要把这两组检查表述为真实模型导航成功率。

## ProcTHOR + EMERGE

ProcTHOR 使用 python -m scripts.LightNav_eval.room_sim --help 查看场景启动参数。
`dev/lightnav_sonic_procthor.json` 对应 5555 JPEG 和 5557 yaw；执行由 Controller 独占 5556。
Agent 模式需要已有 provider 配置；导航 Skill 已随仓库内置，无需复制到 workspace。
此次新增动作等待/取消测试不等于真实 LLM 端到端验收。

## 苹果桌：统一启动器

以下命令需要实验室桌面和已有的原苹果桌场景。每个进程独立一个终端：

```bash
cd ~/EMERGE-Humanoid
# 各终端都设置。推理源码/权重仍在外部 LightNav 仓库。
export LIGHTNAV_ROOT="$HOME/LightNav-0"
export GR00T_ROOT="$HOME/GR00T-WholeBodyControl"
export PYTHONPATH="$PWD"
```

终端 1：原苹果桌仿真（默认仍有支撑，保留原场景和相机）：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m scripts.LightNav_eval.local_tabletop sim
```

终端 2：SONIC，使用原二进制并固定 DDS `lo` / domain 0：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m scripts.LightNav_eval.local_tabletop sonic
```

终端 3：键盘/导航客户端，独占 planner：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m scripts.LightNav_eval.local_tabletop control --enable-motion
```

终端 4：LightNav 模型：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m scripts.LightNav_eval.local_tabletop server
```

终端 5：ego 查看器：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m scripts.LightNav_eval.local_tabletop ego
```

等待控制端 `SIMULATION_ONLY STATE = READY` 和模型 READY 后，按原流程在
MuJoCo 窗口按 9 一次，确认支撑释放、站稳，再在终端 3 输入：

```text
nav Go straight to the table directly ahead with the apple on top and stop.
```

`stop` 停止，`move keyboard` 切换键盘；原控制命令与日志功能保持。
如通过 SSH 使用键盘，连接须加 `-t`；MuJoCo GUI 仍由实验室桌面启动。
各组件的 `--help` 查看参数；`evaluate` 和 `postprocess` 可对已有 task 目录重做离线产物。

## 环境与目录

| 变量 | 默认 |
| --- | --- |
| `LIGHTNAV_ROOT` | `third_party/lightnav`，可通过变量复用 `~/LightNav-0` |
| `MODEL_PATH` | `$LIGHTNAV_ROOT/checkpoints/LightNav-0` |
| `LIGHTNAV_PYTHON` | `~/miniconda3/envs/lightnav-0/bin/python` |
| `SONIC_PYTHON` | `~/miniconda3/envs/sonic/bin/python` |
| `GR00T_ROOT` | `third_party/sonic`，可通过变量复用已有安装 |
| `VISIONPRO_SONIC_ROOT` | `~/visionpro_sonic_bridge`，提供原苹果桌场景 |
| `ISAAC_ROOT` | `~/Isaac-GR00T-N1.7` |

启动器的仿真/SONIC/模型日志在 `EMERGE-Humanoid/outputs/`，控制器的 task 目录
由其 `--output-dir` 决定；这些运行产物已被 Git 忽略。
苹果桌读取 5560 Official01 RGB，ProcTHOR 读取 5555 JPEG，二者不是可互换配置。
苹果桌也已提供 EMERGE `navigate_vln` 相机适配：使用 `dev/lightnav_sonic_tabletop.json`。
下面是原生 Skill 路径（替换上文终端 3 的独立 control 进程）：

```bash
cd ~/EMERGE-Humanoid
PYTHONPATH=. ~/miniconda3/envs/EmergePolicy/bin/python -m robot.controller \
  --driver lightnav_sonic_g1 \
  --workspace "$HOME/.Emerge/workspace" \
  --driver-config dev/lightnav_sonic_tabletop.json \
  --interval 0.25
```

仿真、SONIC、模型和 ego 查看器继续使用上面的独立进程。确认仿真支撑已按原流程
手动释放、机器人站稳且 RGB/yaw 有效后，再从 Main Agent 或 enqueue 工具提交动作。
该配置不会写 ProcTHOR 的支撑释放文件。Main Agent 读取内置导航 Skill，先用
`update_plan` 确立当前目标，再调用 `navigate_vln`，由 `task_verification` 验证。

不能同时运行 `local_tabletop control` 和 Controller，二者争用同一 planner 端口。
新的适配复用 RGB/waypoint/SONIC 通道；独立 tabletop control 的 motion gate、键盘、
复位和录像评估流程没有被迁入 Controller。历史 Astra 实验已执行两次真实导航与视觉验证，
均因目标不可见而无法确认成功。本次目录迁移后只做离线回归，没有重跑真实模型。

历史仿真结果和旧命令见 `~/LightNav-0/docs/LIGHTNAV_SONIC_TABLETOP.md`。
本次仅整理源码和做自动回归，没有重新启动桌面仿真、GPU 模型或 SONIC deployment。

## Agent 任务输入

新任务可输入：使用 lightnav-sonic-navigation，让 G1 到苹果桌前停下，不抓取、
不触碰。最多执行 5 次导航动作（含首次和恢复），每次最多 120 ticks、90 秒；
每次刷新状态并视觉验证；两次无进展后改变策略；第五次仍不能确认则停止报告。
确认成功或没有可用的有意义恢复动作时提前停止。

不要复用旧实验中“最多 3 次动作、连续两次无进展后停止”的用户指令；
用户任务中的显式限制会覆盖 Skill 的默认预算。
