<div align="center">

# EMERGE-Policy

### A Robot Mind Emerges Beyond a Single Policy

Zhirui Fang<sup>*</sup>, Qingchi Yu<sup>*†</sup>, Ziyang Chen<sup>*</sup>, Longfei Li<sup>*†</sup>, Haoran Ma<sup>†</sup>, Keru Zhou, Xinrun Xu, Samith Va, Yuxuan Hu<sup>†</sup>, Peixuan Song, Qiang Du, Bin Qian, Yongkang Deng<sup>†</sup>, Xin Li<sup>†</sup>, Yezhen Wang, Zhe Li, Hao Luo, Shuyan Li, Ziwei Wang, Weijian Deng, Xiu Li<sup>✉</sup>

Tsinghua University · Nanjing University of Science and Technology · Xi'an Jiaotong University · Xidian University · Harbin Institute of Technology · Peking University · Nanyang Technological University

<sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Work done during internship at Tsinghua University &nbsp;&nbsp; <sup>✉</sup> Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-6c5ce7?style=for-the-badge)](https://emerge-policy.github.io/EMERGE-Policy/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=for-the-badge)](https://arxiv.org/pdf/2608.29896)
[![arXiv](https://img.shields.io/badge/arXiv-2608.29896-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2608.29896)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/EMERGE-Policy/EMERGE-Policy)

</div>

<p align="center">
  <img src="docs/images/paper/framework.png" alt="EMERGE-Policy framework" width="100%">
</p>

## Updates

**[August 30, 2026]** The first version of EMERGE-Policy is now available! This
initial release introduces an embodied control framework in which a primary agent
coordinates planning while multiple sub-agents collaborate on perception and
verification. It also establishes the first end-to-end closed-loop execution
pipeline, marking the project's first step from a single policy toward
multi-policy collaboration and continual evolution.

## Overview

*🤖 EMERGE-Policy is an agentic embodied-control framework designed to move
beyond the limitations of a single policy model. A primary agent coordinates task
planning and recovery while delegating object localization, scene perception, and
task verification to specialized sub-agents. The framework supports both OpenPI
VLA and Cosmos Policy WAM backends. All actions are safely executed through the
Controller, while live visual observations and simulator success signals close
the loop, allowing the robot to perceive, decide, act, and correct itself through
continuous interaction.*

## Humanoid Navigation Skill (humanoid-nav)

This branch contains the simulation navigation work for Unitree G1.
[lightnav-sonic-navigation](Emerge/skills/lightnav-sonic-navigation/SKILL.md)
is an Operational Skill: the main Agent plans a navigation phase, calls the
native execute_robot_action tool with navigate_vln, and verifies the result.
LightNav-0 supplies visual-language navigation (VLN) waypoints; SONIC executes
locomotion. Navigation runtime code belongs under robot/navigation, separately
from the manipulation-policy execution code in robot/vla.

| Component | Location |
| --- | --- |
| Skill contract | [SKILL.md](Emerge/skills/lightnav-sonic-navigation/SKILL.md) |
| Navigation runtime | [robot/navigation/lightnav_sonic](robot/navigation/lightnav_sonic/) |
| Native Controller driver | [lightnav_sonic_driver.py](robot/drivers/lightnav_sonic_driver.py) |
| Apple-table configuration | [lightnav_sonic_tabletop.json](dev/lightnav_sonic_tabletop.json) |
| Upstream model sources | third_party/lightnav and third_party/sonic (pinned submodules) |
| Simulation and evaluation | [scripts/LightNav_eval](scripts/LightNav_eval/) |

~~~text
User task -> main Agent reads Skill -> update_plan
 -> execute_robot_action(navigate_vln) -> Controller / Driver
 -> RGB + instruction -> LightNav -> waypoint -> SONIC -> G1 motion
 -> fresh RGB -> task_verification -> progress, recovery, or stop
~~~

For the navigation setup, use the
[navigation dependencies and architecture guide](scripts/LightNav_eval/README.md)
and [apple-table startup instructions](scripts/LightNav_eval/SIMULATION.md).
The general installation and LIBERO quick start below describe the original
manipulation workflow; they are not the navigation startup sequence.

After the simulator, SONIC, LightNav service, and navigation Controller are
ready, submit a task such as:

> 在当前苹果桌仿真中，使用 lightnav-sonic-navigation，让 G1 到苹果桌前停下，
> 不抓取、不触碰。最多 5 次导航动作（含首次和恢复），每次最多 120 ticks、
> 90 秒。每次刷新状态并视觉验证；两次无进展后改变策略。
> 第五次仍不能确认则停止报告，成功或无有效恢复方式时提前停止。

The five-attempt limit is a Skill instruction for the Agent, not a hard counter
in the Controller. Each attempt requires verification; LightNav stop and an
exhausted action budget do not prove arrival. Start a new task with this budget
instead of reusing the old task's explicit two-attempt stop instruction.

Validation: 61 core checks and 80 simulation checks pass, together with Ruff,
Google-style YAPF, and an installed-wheel Driver import check. The earlier Astra
run exercised two real GPU navigation actions, verification, and recovery; both
verifications were uncertain, so arrival was not established. The directory
migration and five-attempt workflow have not been rerun with real models.
Hardware deployment remains future work on real-dev.

[Historical experiment notes](docs/lightnav/notes/README.md) are retained for
reference; their old directory and registration commands are superseded by the
current navigation guide.

## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules \
  --branch humanoid-nav \
  https://github.com/hrm-666/EMERGE-Humanoid.git
cd EMERGE-Humanoid

git submodule sync --recursive
git submodule update --init --recursive
```

### 2. Install the Main `EmergePolicy` Environment

```bash
conda create -n EmergePolicy python=3.12 -y
conda activate EmergePolicy
python -m pip install --upgrade pip wheel "setuptools<82"

python -m pip install \
  -e ".[libero]" \
  -e third_party/vggt \
  -e third_party/sam3 \
  -e third_party/openpi/packages/openpi-client

bash third_party/imagemagick_env/install.sh
```

### 3. Install the Standalone OpenPI Environment `pi05_server`

```bash
conda create -n pi05_server python=3.12 -y
conda activate pi05_server
python -m pip install --upgrade pip uv

cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  --python "$CONDA_PREFIX/bin/python" \
  -e .
```

## Quick Start

The Python package is named `Emerge`, and the installed command-line entry point
is `emerge`. The default configuration file is `~/.Emerge/config.json`, and the
default workspace is `~/.Emerge/workspace`.

Start the Controller in one terminal:

```bash
python -m robot.controller \
  --driver libero_mujoco \
  --driver-config dev/libero_mujoco_driver_sample.json
```

Run the agent CLI in another terminal:

```bash
python -m Emerge agent
```

Start the default OpenPI, VGGT, and SAM3 services:

```bash
OPENPI_GPU=0,1,2 \
VGGT_GPU=3 \
SAM3_GPU=4 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

## Model Checkpoints

Download the external model checkpoints to `checkpoints/` in the repository root:

| Model | Source and Contents | Destination |
|---|---|---|
| OpenPI π0.5-LIBERO | Fetch `gs://openpi-assets/checkpoints/pi05_libero` from the [official OpenPI checkpoint repository](https://github.com/Physical-Intelligence/openpi#model-checkpoints) | `checkpoints/pi05_libero/` |
| VGGT-1B | Download `model.pt` from [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) | `checkpoints/vggt/model.pt` |
| SAM3 | Download `sam3.pt` from [`facebook/sam3`](https://www.modelscope.cn/models/facebook/sam3/files) on ModelScope | Rename it to `checkpoints/sam3/model.pt` |
| Cosmos Policy (optional) | Download the policy weights, statistics, and T5 cache from [`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `checkpoints/cosmos-policy/` |
| Cosmos Predict2 (optional) | Download the base model from [`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) | `checkpoints/cosmos-policy/Cosmos-Predict2-2B-Video2World/` |

## External Model Service Ports

The startup script launches the selected persistent services, keeping model
loading independent from individual agent episodes.

| Service | Port | Environment | Purpose |
|---|---:|---|---|
| OpenPI | 8000 | `pi05_server` | VLA action inference |
| VGGT | 8001 | `EmergePolicy` | Multi-view geometry estimation |
| SAM3 | 8002 | `EmergePolicy` | Prompt-guided image segmentation |
| Cosmos Policy | 8003 | `cosmos-policy` | WAM candidate action generation and scoring |

For additional startup options and health-check endpoints, see
[`scripts/model_server/README.md`](scripts/model_server/README.md).

## Evaluation

- [Standard LIBERO Evaluation](scripts/Libero_eval/README.md)
- [LIBERO-Plus Robustness Evaluation](scripts/LiberoPlus_eval/README.md)
- [LIBERO-Pro Evaluation](scripts/LiberoPro_eval/README.md)
