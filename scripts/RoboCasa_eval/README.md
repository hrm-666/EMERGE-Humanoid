# EMERGE RoboCasa 单 episode 评测

入口为 `eval_robocasa_agent.py`。它沿用 LIBERO 的进程组织：Controller 持有仿真环境，现有 `Emerge.cli.headless` 持有主 agent、两个子代理与视觉监控，二者通过本次任务独立的 workspace 交换动作和观测。

## 运行准备

- RoboCasa Python 环境能创建原生 PandaOmron 任务，并能导入本仓库的 Controller。
- EmergePolicy 环境能运行 EMERGE 主程序。
- 已有可供 RoboCasa 使用的 OpenPI 服务，以及 VGGT、SAM3 服务。复用之前已跑通的 **RoboCasa checkpoint**；LIBERO checkpoint 不能作为这里的替代品。默认服务端口为 8000、8001、8002。
- 正式 API 配置仍使用仓库 `.emerge/config.json`。启动器只传递路径，不修改它，也不把凭据复制到任务目录。

从仓库根目录运行：

先在一个 tmux 会话启动三个模型，默认分别使用 GPU 0、1、2：

```bash
bash scripts/model_server/start_external_model_servers.sh --robocasa
```

该预设使用相邻 `openpi/` 的原始策略服务及工作区 `checkpoints/` 的权重，复用已有服务启动和退出清理逻辑。另一个 tmux 会话运行评测即可。服务已启动时无需重复启动。

当前容器 EGL 设备索引为 0；它与 `nvidia-smi` 的 GPU 编号不是同一个枚举。评测会话先执行 `export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`，再运行：

```bash
conda activate EmergePolicy
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

示例把首次联调产物放在 `tests/`。正式评测可通过 `--output-dir` 指定新的结果目录；不指定时默认写入 `artifacts/robocasa_agent_eval/<任务>/<时间戳及唯一标识>/`。已存在的输出目录会被拒绝，避免覆盖旧结果。

`--controller-python` 与 `--agent-python` 可以不同。前者也可通过 `ROBOCASA_PYTHON` 环境变量指定；未指定时两者均使用启动器自身的 Python。无需给两个环境安装相互冲突的仿真依赖。

## 执行流程

1. 创建本次任务的 workspace、初始文档和独立 driver 配置，由 Controller 加载配置。
2. 用 RoboCasa Python 启动 `robot.controller --driver robocasa`，安装 profile 并创建原生环境。模型服务连接由已有执行器和子代理处理。
3. 与 LIBERO 一样等待私有状态和机器人状态就绪，从环境获取原始任务指令及预算。
4. 用 EmergePolicy Python 启动现有 headless runtime。主 agent 自主规划、调用子代理并执行几何/VLA 动作；启动器不预写任务动作序列。
5. 监督环境状态。主 agent 正常提前返回、任务仍未完成时，最多进行两轮同 session 的继续反馈。每轮使用独立 run_id，共享最初的执行截止时间。
6. 成功、预算耗尽、超时、异常或收到中断后，先停止主 agent，让它通过仍在运行的 Controller 取消未完成动作，再清理 Controller，关闭视频编码器，最后保存结果。

RoboCasa 保留 `gym.make(seed=...)` 及既有 reset 流程，不引入 LIBERO 的 BDDL、初始状态数组或额外 settling 动作。没有 `--max-steps` 或 driver 配置覆盖时，动作预算由原生任务 registry 解析。VLA 与几何动作共享这个预算。

## 重要参数

| 参数 | 含义 |
| --- | --- |
| `--env-name` | 必填，原生 RoboCasa 任务类名 |
| `--split` / `--seed` | 默认 `target` / `7` |
| `--record-video` / `--no-record-video` | 默认录制 `rollout.mp4`；诊断时可以显式关闭 |
| `--max-steps` | 覆盖整个 episode 的控制步数预算 |
| `--max-agent-iterations` | 覆盖每轮主 agent 的工具迭代上限；默认使用 API 配置中的值 |
| `--max-agent-feedback-rounds` | 主 agent 正常返回但任务未完成时的继续反馈次数，默认 2 |
| `--ready-timeout-s` | 环境初始化超时，默认 300 秒 |
| `--episode-timeout-s` | 就绪后的执行超时，默认 1800 秒，续跑不会重置 |
| `--shutdown-grace-s` | 子进程收到中断后的退出宽限，默认 90 秒，覆盖动作取消与视觉记忆收尾 |

## 输出与判定

```text
<output-dir>/
  driver_config.json    # 本次仿真配置
  status.json           # Controller 写入的官方环境状态，位于 agent workspace 之外
  result.json           # 沿用 LIBERO 字段结构的最终成绩
  rollout.mp4           # H.264 视频，默认录制
  controller.log
  agent.log
  agent_runs/           # EMERGE 原有的每轮请求快照、事件、结果及日志
  workspace/
    AGENTS.md
    EMBODIED.md
    PLAN.md
    ROBOT_STATE.md
    ACTION.md
    artifacts/          # 相机 PNG、标定观测等
    memory/             # 通用运行时生成的视觉记忆
```

只有 `status.json` 中官方 `success=true` 才能判定 benchmark 成功。主 agent 的“完成”回复、动作执行完成或视觉验证结果都不替代它。

`result.json` 沿用 LIBERO 单 episode 结果的字段结构：schema、episode 标识、任务信息、指令、成功标志、结束原因、步数及预算、agent 继续反馈次数、运行目录、耗时与起止时间，以及 `action_count`、`actions_by_type`、`actions_by_status` 动作统计。任务信息使用 RoboCasa 的 `env_name`、`split`、`seed`，不套用 LIBERO 的 `suite`、`task_id`、`trial`。不再另加 `outcome`、`video_path`、顶层异常信息或嵌套整份运行状态。

`status.json` 是覆盖更新的进程通信文件，顶层字段与 LIBERO 一致：`schema_version`、`ready`、`success`、`done`、`reward`、`environment_steps`、`settling_steps`、`action_steps`、`max_action_steps`、`termination_reason`、`metadata`。RoboCasa 没有额外初始化 settling 动作，因此 `settling_steps=0`、`environment_steps=action_steps`；几何动作中的保持步数仍计入 `action_steps`。任务名、split、seed、原生指令与异常详情放在 `metadata` 中。

与 LIBERO 一样，结果中的 `termination_reason` 以 `infrastructure_error:` 前缀标记环境、服务或进程错误；`success` 表示环境成功，`max_action_steps`、`episode_timeout` 等表示任务未完成，`interrupted` 表示用户中断。初始化 NaN 等错误仍与策略失败分开判定。`controller.log` 和 `agent.log` 完整保留，启动器的异常诊断附加到 `agent.log`。每轮 headless 的 `agent_runs/<run_id>/result.json` 继续使用通用运行时原有格式。

启动请求只在运行期间使用临时 `.agent_request.json`，结束后删除；每轮请求快照由 EMERGE 原样保存在 `agent_runs/<run_id>/request.json`，不再另外保留重复的 `agent_requests/`。`sessions/`、`memory/` 及其图片的原有工作机制不变。

CLI 返回码：0 表示评测正常结束（包括任务失败），1 表示执行基础设施错误，130 表示中断，2 表示命令/配置参数错误。判断任务成功请读取 `result.json.success`。

相机图片沿用 LIBERO 的更新方式：每个相机使用固定 PNG 路径，通过临时文件替换当前图片，再更新 `observation.json`，不会按观测版本累积历史目录。历史动作证据由现有视觉监控独立保存在 `memory/images/`。PNG 路径代表当前画面，查看历史画面应使用视觉记忆中的关键图。

默认视频为任务目录下的 `rollout.mp4`，使用 RoboCasa 原生左前相机 `video.robot0_agentview_left`，H.264 编码，可在浏览器和 VS Code 中播放。初始化成功后记录一帧，每个已完成的控制步再记录一帧，帧率取原生环境的 `control_freq`（通常 20 FPS）。VLA 与几何动作、各次 agent 继续调用都记录到同一个视频中，正常结束、失败和 Ctrl+C 清理时关闭编码器。

视频按仿真控制步回放，不把 LLM 等待时间填成静止画面；采集不增加仿真步、不重新渲染、不新增历史 PNG。环境初始化尚未生成可用观测就失败时无法生成视频。视频使用固定文件名 `rollout.mp4`，与 LIBERO 一样不在结果中另存路径字段。编码依赖 RoboCasa Python 环境中的 `imageio` 和 `imageio-ffmpeg`；与现有 OpenPI 录像使用同一依赖。

入口仍只负责一个 episode，批量调度和断点续跑尚未实现。

评测器直接复用 LIBERO 的进程清理、agent 结果校验、继续反馈判定和动作统计。环境已给出成功或步数耗尽等终态后，收尾耗时不改变该回合判定。90秒退出宽限覆盖现有视觉监控的最终验证与记忆保存，不修改通用 agent、视觉监控或记忆逻辑。
