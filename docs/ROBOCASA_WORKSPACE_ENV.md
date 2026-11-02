# RoboCasa_Workspace 环境配置记录

> 记录时间：2026-11-02  
> 用途：让协作同学复现同一套 RoboCasa 仿真、OpenPI 策略服务、EMERGE agent 与感知服务。  
> 注意：本文只记录配置，不包含任何 API key / token。`.emerge/config.json` 里的凭据需要每个人自己配置。

当前工作区根目录：

```text
/private/RoboCasa_Workspace
├── EMERGE-Humanoid/   # agent / controller / profiles / 评测入口
├── robocasa/          # RoboCasa 仿真与任务
├── robosuite/         # 仿真框架
├── openpi/            # OpenPI 策略服务与 client
├── checkpoints/       # 策略 / VGGT / SAM3 / LightNav 权重
└── test/              # 评测输出
```

如果换到别的机器，建议保持相同的相对目录结构；如果绝对路径不同，需要重新安装 editable 包（见第 7 节）。

---

## 1. 主机、系统与 GPU

| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04.4 LTS |
| Kernel | 5.15.0-78-generic |
| CPU | 128 cores |
| GPU | 7 × NVIDIA L20，46 GB each |
| NVIDIA driver | 570.172.08 |
| CUDA（nvidia-smi） | 12.8 |
| PyTorch CUDA runtime | 12.6 |
| MuJoCo 渲染 | `MUJOCO_GL=egl` |
| EGL device（仿真） | `MUJOCO_EGL_DEVICE_ID=0` |

说明：

- `MUJOCO_EGL_DEVICE_ID=0` 是容器里的 EGL 设备编号，**不等于** `nvidia-smi` 的 GPU 编号。
- 运行 MuJoCo 时可能出现 `OpenGL error 0x501 in or before mjr_makeContext`，同时在工作目录生成 `MUJOCO_LOG.TXT`；这是 warning 日志，不是配置错误，可以删除。
- 7 张 GPU 的编号：`0–6`。模型服务默认使用 `0/1/2`，其余 GPU 空闲。

---

## 2. 各仓库的版本

| 目录 | 远端 | 分支 | 当前 commit | 说明 |
|---|---|---|---|---|
| `EMERGE-Humanoid` | `https://github.com/hrm-666/EMERGE-Humanoid.git` | `feat/robocasa` | `77ae389aebe8580bd8b38a502d0d7833aa7b4278` | RoboCasa 适配、profile、评测入口；本文和 `docs/env/` 是新增未跟踪文件 |
| `robocasa` | `https://github.com/robocasa/robocasa.git` | `main` | `2ce1aecdd76e29b52d32fdb160436c5a3e230b07` | 2026-10-21；本地有一个未跟踪的 `robocasa/models/assets/README.md`，不影响运行 |
| `robosuite` | `https://github.com/ARISE-Initiative/robosuite` | `master` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | 2026-07-11 |
| `openpi` | `https://github.com/robocasa-benchmark/openpi.git` | `main` | `5a6beda9ff99da30b4e1b59320f6a32971d7c397` | 2026-05-12；**有一处未提交的本地修改**，见第 6 节 |

EMERGE 中与 RoboCasa 感知相关的子模块：

| 子模块 | commit |
|---|---|
| `third_party/vggt` | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| `third_party/sam3` | `402a0275f6e4648f5a35bbdeeda017467c1099d0` |
| `third_party/lightnav-0` | `3015508b70fb6e30bbd66701e94c9b916dcb8df8` |
| `third_party/openpi` | 未初始化（`-215abfb...`），RoboCasa 使用的是工作区独立的 `openpi/` 目录，不是这个子模块 |

版本检查命令：

```bash
cd /private/RoboCasa_Workspace
for r in EMERGE-Humanoid robocasa robosuite openpi; do
  echo "== $r"
  git -c safe.directory="$PWD/$r" -C "$r" rev-parse HEAD
  git -c safe.directory="$PWD/$r" -C "$r" branch --show-current
done
```

> `robocasa` / `robosuite` / `openpi` 的 Git 目录属主可能和当前用户不同；如果出现 `dubious ownership`，使用上面的 `-c safe.directory=...`，不要直接改全局配置。

---

## 3. Conda 环境

系统里使用**两个独立的 Conda 环境**，职责不同，不要混装：

| 环境 | Python | 用途 |
|---|---|---|
| `robocasa` | 3.11.17 | 仿真、Controller、OpenPI policy server、openpi-client |
| `EmergePolicy` | 3.12.16 | EMERGE agent、object_location / task_verification 子 agent、VGGT/SAM3 server |

### 3.1 robocasa 环境

路径：`/opt/conda/envs/robocasa`

关键包版本：

| 包 | 版本 |
|---|---|
| numpy | 2.2.5 |
| scipy | 1.15.3 |
| torch | 2.7.1（CUDA 12.6，可用） |
| jax / jaxlib | 0.5.3 / 0.5.3（可见 7 张 CUDA device） |
| mujoco | 3.3.1 |
| robosuite | 1.5.2（editable） |
| robocasa | 1.0.1（editable） |
| gymnasium | 0.29.1 |
| openpi | 0.1.0（editable） |
| openpi-client | 0.1.0（editable） |
| pydantic | 2.14.1 |
| websockets | 17.3 |
| imageio / imageio-ffmpeg | 2.38.1 / 0.6.0 |
| transformers | 4.53.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |
| tyro | 1.0.16 |
| einops | 0.8.2 |
| opencv-python | 5.0.0.93 |
| h5py | 3.16.0 |
| numba | 0.61.2 |
| ml-collections | 1.0.0 |
| absl-py | 2.5.1 |

editable 安装位置：

```text
openpi          -> /private/RoboCasa_Workspace/openpi
openpi-client   -> /private/RoboCasa_Workspace/openpi/packages/openpi-client
robosuite       -> /private/RoboCasa_Workspace/robosuite
robocasa        -> /private/RoboCasa_Workspace/robocasa
```

> 注意：该环境**没有安装 `pydantic_settings`**。因此 `Emerge.config` / 子 agent 相关代码不要在这个环境里跑；agent 和子 agent 用 `EmergePolicy`。

### 3.2 EmergePolicy 环境

路径：`/opt/conda/envs/EmergePolicy`

关键包版本：

| 包 | 版本 |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| torch | 2.7.1+cu126（CUDA 12.6，可用） |
| pydantic | 2.14.1 |
| pydantic-settings | 2.16.2 |
| websockets | 16.1.1 |
| pillow | 12.3.0 |
| safetensors | 0.8.0 |
| huggingface-hub | 2.3.0 |
| einops | 0.8.2 |
| opencv-python | 4.11.0.86 |

> 该环境没有 gymnasium / robosuite / robocasa / openpi_client，**不要**在这里跑仿真或直接连 policy client。agent 通过 websocket bridge 调用外部 policy server。

### 3.3 完整 pip 快照

为了精确复现，仓库里附了两份 `pip freeze`：

- [robocasa_pip_freeze.txt](env/robocasa_pip_freeze.txt)
- [EmergePolicy_pip_freeze.txt](env/EmergePolicy_pip_freeze.txt)

重新生成：

```bash
/opt/conda/envs/robocasa/bin/pip freeze > docs/env/robocasa_pip_freeze.txt
/opt/conda/envs/EmergePolicy/bin/pip freeze > docs/env/EmergePolicy_pip_freeze.txt
```

`pip freeze` 里的 `-e /private/RoboCasa_Workspace/...` 是 editable 路径；如果 clone 到别的位置，需要重新 `pip install -e`。

---

## 4. Checkpoints

| 路径 | 大小 | 用途 |
|---|---|---|
| `checkpoints/hf_download/pi05_pretrain_human300/multitask_learning/75000` | **12G** | pi05 RoboCasa 策略，step 75000；`assets/norm_stats.json` + `params/` |
| `checkpoints/model.pt` | **4.7G** | VGGT 几何模型（object_location 使用） |
| `checkpoints/sam3.pt` | **3.3G** | SAM3 分割模型（object_location 使用） |
| `checkpoints/lightnav-0` | 14M | LightNav 导航权重（RoboCasa 不用） |

checkpoint 不在 git 里，需要向工作区维护者拷贝，保持上面的目录结构。策略服务使用：

```text
policy.config = pi05_pretrain_human300
policy.dir    = /private/RoboCasa_Workspace/checkpoints/hf_download/pi05_pretrain_human300/multitask_learning/75000
```

---

## 5. 模型服务

启动命令（在 `EMERGE-Humanoid/` 下）：

```bash
bash scripts/model_server/start_external_model_servers.sh --robocasa
```

`--robocasa` 预设的默认值：

| 服务 | 端口 | GPU | Python / 环境 | 模型 |
|---|---|---|---|---|
| OpenPI policy | 8000 | 0 | `/opt/conda/envs/robocasa/bin/python` | `checkpoints/hf_download/pi05_pretrain_human300/multitask_learning/75000` |
| VGGT | 8001 | 1 | Conda `EmergePolicy` | `checkpoints/model.pt` |
| SAM3 | 8002 | 2 | Conda `EmergePolicy` | `checkpoints/sam3.pt` |

关键参数：

```text
OPENPI_ROOT=/private/RoboCasa_Workspace/openpi
OPENPI_CONFIG=pi05_pretrain_human300
OPENPI_PYTHON=/opt/conda/envs/robocasa/bin/python
OPENPI_GPU=0
VGGT_GPU=1
SAM3_GPU=2
VGGT_MAX_BATCH_SIZE=1
SAM3_MAX_BATCH_SIZE=1
SAM3_BACKEND=official
SAM3_CONFIDENCE_THRESHOLD=0.3
SAM3_MASK_THRESHOLD=0.5
```

实际 policy server 命令：

```bash
/opt/conda/envs/robocasa/bin/python -u scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=pi05_pretrain_human300 \
  --policy.dir=/private/RoboCasa_Workspace/checkpoints/hf_download/pi05_pretrain_human300/multitask_learning/75000
```

健康检查（本机 curl 需要绕过代理）：

```bash
NO_PROXY='*' no_proxy='*' curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/healthz
NO_PROXY='*' no_proxy='*' curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/healthz
NO_PROXY='*' no_proxy='*' curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8002/healthz
```

期望都是 `200`。

> 注意：shell 里配置了 `http_proxy/https_proxy` 时，直接 `curl 127.0.0.1:8000` 可能返回 502；这是代理导致的，不是服务挂了。Python 的 OpenPI client 会为 localhost 清理 NO_PROXY。

---

## 6. openpi 的本地必要修改

`openpi/src/openpi/training/config.py` 里 `pi05_pretrain_human300` 有一处**未提交**的本地修改：

```diff
         data=LeRobotRobocasaDataConfig(
-            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human300"],
+            data_dirs=[],
         ),
```

原因：serving 时不需要 RoboCasa 数据集注册；`data_dirs=[]` 可以避免 serve 阶段依赖数据集路径。复现时请应用这个改动，否则 policy server 可能因为找不到数据集注册/路径而启动失败。

---

## 7. 复现步骤

1. **准备目录**：保持 `/private/RoboCasa_Workspace` 结构，或修改所有 editable 路径。
2. **克隆仓库**：按第 2 节的 remote / branch / commit checkout。
3. **创建 Conda 环境**：
   - Python 3.11 → `robocasa`
   - Python 3.12 → `EmergePolicy`
   - 根据两份 freeze 安装依赖；editable 包用下面的命令重新安装：
     ```bash
     /opt/conda/envs/robocasa/bin/pip install -e /private/RoboCasa_Workspace/openpi
     /opt/conda/envs/robocasa/bin/pip install -e /private/RoboCasa_Workspace/openpi/packages/openpi-client
     /opt/conda/envs/robocasa/bin/pip install -e /private/RoboCasa_Workspace/robosuite
     /opt/conda/envs/robocasa/bin/pip install -e /private/RoboCasa_Workspace/robocasa
     ```
   - `EmergePolicy` 环境需要 `pydantic-settings`；EMERGE 代码本身通常通过 `PYTHONPATH=/private/RoboCasa_Workspace/EMERGE-Humanoid` 使用，不需要 pip 安装。
4. **拷贝 checkpoint**：按第 4 节放置。
5. **应用 openpi 的 `data_dirs=[]` 修改**。
6. **启动模型服务**：第 5 节命令；确认三个 healthz 都是 200。
7. **配置 `.emerge/config.json`**：
   - 复制模板后填入**自己的** API key / provider 配置；不要把 key 提交到 git。
   - 当前模型：`openai/gpt-6.1-sol`；provider 使用 OpenAI-compatible endpoint；`api_mode=responses`。
   - 子 agent 配置：VGGT `ws://localhost:8001`，SAM3 `ws://localhost:8002`；object_location `max_iterations=8`、`point_conf_threshold=0.3`、`min_points=80`、`view_center_tolerance_m=0.08`、`ray_consensus_tolerance_m=0.02`、`min_consistent_views=2`；task_verification `max_iterations=2`。
8. **跑一次 smoke eval**：

   ```bash
   conda activate EmergePolicy
   export MUJOCO_GL=egl
   export MUJOCO_EGL_DEVICE_ID=0
   python -B scripts/RoboCasa_eval/eval_robocasa_agent.py \
     --env-name CoffeeSetupMug \
     --split target \
     --seed 7 \
     --controller-python /opt/conda/envs/robocasa/bin/python \
     --agent-python "$CONDA_PREFIX/bin/python" \
     --agent-config .emerge/config.json \
     --driver-config dev/robocasa_driver.json \
     --policy-server-url ws://127.0.0.1:8000 \
     --output-dir tests/robocasa_validation/runs/agent_episode_mug_01
   ```

   注意：`--controller-python` 必须是 `robocasa` 环境，`--agent-python` 必须是 `EmergePolicy` 环境；不要用同一个 Python 跑两边。

---

## 8. 驱动与评测设置

`dev/robocasa_driver.json` 的当前关键值：

```json
{
  "robocasa": {
    "env_name": "OpenCabinet",
    "split": "target",
    "seed": 7,
    "mujoco_gl": "egl"
  },
  "motion": {
    "position_gain": 1.0,
    "rotation_gain": 1.0,
    "position_tolerance_m": 0.02,
    "rotation_tolerance_rad": 0.15,
    "max_steps": 160,
    "move_linear_steps": 80,
    "settle_steps": 40,
    "gripper_steps": 30,
    "publish_every_n_steps": 5
  },
  "vla": {
    "server_url": "ws://localhost:8000",
    "resize_size": 224,
    "replan_steps": 5,
    "publish_every_n_steps": 5,
    "stop_on_success": true
  },
  "observation": {
    "enabled": true,
    "reference_view": "perception_oblique_1",
    "camera_names": [
      "perception_oblique_1",
      "perception_oblique_2",
      "perception_oblique_3",
      "perception_side_1",
      "perception_side_2"
    ]
  },
  "evaluation": {
    "enabled": true,
    "status_path": "artifacts/evaluation/status.json"
  }
}
```

实际评测时 `--env-name` / `--seed` 会覆盖 `robocasa` 段；`motion` / `vla` / `observation` 使用上面的值。

评测输出目录包含：

```text
<output-dir>/
├── result.json
├── status.json
├── rollout.mp4
├── controller.log
├── agent.log
├── agent_runs/
└── workspace/
```

---

## 9. 已知坑与注意事项

1. **两个 Python 环境不能混用**：仿真/Controller/policy server 用 `robocasa`，agent/子 agent/感知 server 用 `EmergePolicy`。
2. **`robocasa` 环境缺 `pydantic_settings`**：`scripts/RoboCasa_eval/eval_robocasa_agent.py` 的 agent 部分必须在 `EmergePolicy` 里跑。
3. **openpi 未提交的 `data_dirs=[]`**：复现时必须手动应用（第 6 节）。
4. **`third_party/openpi` 子模块未初始化**：RoboCasa 使用工作区独立的 `openpi/`，不要去用 submodule 里那一份。
5. **editable 安装路径**：`pip freeze` 里是绝对路径 `/private/RoboCasa_Workspace/...`；换路径后必须重新 `pip install -e`。
6. **policy server 的随机性**：OpenPI policy 的 JAX RNG 初始化为 `jax.random.key(0)`，每次 `infer` 都会 split；**没有 per-episode seed**。同一个 seed/场景重跑结果可能不同。做成功率对比时要跑多次 trial，或者每次 trial 重启 policy server / 增加 per-episode 噪声。
7. **代理**：shell 里配置了 `http_proxy/https_proxy` 时，本机 curl 健康检查可能 502，需要 `NO_PROXY='*'`；Python 客户端会为 localhost 处理。
8. **`MUJOCO_LOG.TXT`**：MuJoCo 的 warning 日志，可能出现在仓库根目录或 episode workspace；不是配置，可以删除，建议加入 `.gitignore`。
9. **`tests/` 被 `.gitignore` 忽略**：`tests/robocasa_validation/scripts/` 下的手动规则脚本和测试改动不会出现在 git status 中；如果要共享给协作者，需要单独拷贝或用 `git add -f`。
10. **GPU/EGL 编号不同**：`MUJOCO_EGL_DEVICE_ID=0` 是 EGL 设备；模型服务的 `CUDA_VISIBLE_DEVICES=0/1/2` 是 nvidia-smi 编号。

---

## 10. 快速版本核对

```bash
git -C /private/RoboCasa_Workspace/EMERGE-Humanoid rev-parse HEAD
git -c safe.directory=/private/RoboCasa_Workspace/robocasa -C /private/RoboCasa_Workspace/robocasa rev-parse HEAD
git -c safe.directory=/private/RoboCasa_Workspace/robosuite -C /private/RoboCasa_Workspace/robosuite rev-parse HEAD
git -c safe.directory=/private/RoboCasa_Workspace/openpi -C /private/RoboCasa_Workspace/openpi rev-parse HEAD

/opt/conda/envs/robocasa/bin/python - <<'PY'
import sys, importlib.metadata as m
print(sys.version.split()[0])
for p in ["numpy","scipy","torch","jax","jaxlib","mujoco","robosuite","robocasa","gymnasium","openpi","openpi-client"]:
    try: print(p, m.version(p))
    except Exception: pass
PY

/opt/conda/envs/EmergePolicy/bin/python - <<'PY'
import sys, importlib.metadata as m
print(sys.version.split()[0])
for p in ["numpy","scipy","torch","pydantic","pydantic-settings","websockets"]:
    print(p, m.version(p))
PY
```

若以上 commit 与版本和本文不一致，先对齐再跑实验。
