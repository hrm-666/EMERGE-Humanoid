# 实验室仿真入口

在 `embodied@embodied-MS-7E06:~/EMERGE-Humanoid`、`humanoid-dev` 使用本说明。
原始代码和历史运行产物仍在 `~/LightNav-0`；新的仿真代码统一在本仓库维护。

## 验证

EMERGE 接入检查（50 项，无 GPU/MuJoCo）：

```bash
cd ~/EMERGE-Humanoid
PYTHONPATH=. ~/miniconda3/envs/EmergePolicy/bin/python -m unittest discover \
  -s integrations/lightnav_sonic_emerge/checks -v
```

苹果桌仿真检查（80 项，使用已有 MuJoCo/OpenCV 环境）：

```bash
cd ~/EMERGE-Humanoid
MUJOCO_GL=egl PYTHONPATH=. ~/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m unittest discover -s integrations/lightnav_sonic_emerge/simulation_checks -v
```

该组检查用合成消息和小型 MuJoCo 场景验证坐标、取消、复位、随机起点及录像评估；
WebSocket/ZMQ 检查绑定临时 loopback 端口。无需模型 checkpoint 或 SONIC deployment。
不要把这两组检查表述为真实模型导航成功率。

## ProcTHOR + EMERGE

沿用 [README 的 ProcTHOR 六终端流程](README.md#end-to-end-procthor-run)。
`driver_procthor.json` 对应 5555 JPEG 和 5557 yaw；执行由 Controller 独占 5556。
Agent 模式额外需要已有 provider 配置和 workspace navigation skill。
此次新增动作等待/取消测试不等于真实 LLM 端到端验收。

## 苹果桌：统一启动器

以下命令需要实验室桌面和已有的原苹果桌场景。每个进程独立一个终端：

```bash
cd ~/EMERGE-Humanoid
# 各终端都设置。推理源码/权重仍在外部 LightNav 仓库。
export LIGHTNAV_ROOT="$HOME/LightNav-0"
export PYTHONPATH="$PWD"
```

终端 1：原苹果桌仿真（默认仍有支撑，保留原场景和相机）：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m integrations.lightnav_sonic_emerge.bridge.local_tabletop sim
```

终端 2：SONIC，使用原二进制并固定 DDS `lo` / domain 0：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m integrations.lightnav_sonic_emerge.bridge.local_tabletop sonic
```

终端 3：键盘/导航客户端，独占 planner：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m integrations.lightnav_sonic_emerge.bridge.local_tabletop control --enable-motion
```

终端 4：LightNav 模型：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m integrations.lightnav_sonic_emerge.bridge.local_tabletop server
```

终端 5：ego 查看器：

```bash
~/miniconda3/envs/sonic/bin/python \
  -m integrations.lightnav_sonic_emerge.bridge.local_tabletop ego
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
| `LIGHTNAV_ROOT` | `~/LightNav-0`，提供 `src/` 和默认 checkpoint |
| `MODEL_PATH` | `$LIGHTNAV_ROOT/checkpoints/LightNav-0` |
| `LIGHTNAV_PYTHON` | `~/miniconda3/envs/lightnav-0/bin/python` |
| `SONIC_PYTHON` | `~/miniconda3/envs/sonic/bin/python` |
| `GR00T_ROOT` | `~/GR00T-WholeBodyControl` |
| `VISIONPRO_SONIC_ROOT` | `~/visionpro_sonic_bridge`，提供原苹果桌场景 |
| `ISAAC_ROOT` | `~/Isaac-GR00T-N1.7` |

启动器的仿真/SONIC/模型日志在 `EMERGE-Humanoid/outputs/`，控制器的 task 目录
由其 `--output-dir` 决定；这些运行产物已被 Git 忽略。
苹果桌读取 5560 Official01 RGB，ProcTHOR 读取 5555 JPEG，二者不是可互换配置。
苹果桌目前仍是独立控制入口，未接到 EMERGE `navigate_vln` 的相机适配层。

历史仿真结果和旧命令见 `~/LightNav-0/docs/LIGHTNAV_SONIC_TABLETOP.md`。
本次仅整理源码和做自动回归，没有重新启动桌面仿真、GPU 模型或 SONIC deployment。
