# EMERGE RoboCasa 单 episode 评测

入口为 `eval_robocasa_agent.py`。它沿用 LIBERO 的进程组织：Controller 持有仿真环境，现有 `Emerge.cli.headless` 持有主 agent、两个子代理与视觉监控，二者通过本次任务独立的 workspace 交换动作和观测。

## 运行准备

- RoboCasa Python 环境能创建原生 PandaOmron 任务，并能导入本仓库的 Controller。
- EmergePolicy 环境能运行 EMERGE 主程序。
- 已有可供 RoboCasa 使用的 OpenPI 服务，以及 VGGT、SAM3 服务。复用之前已跑通的 **RoboCasa checkpoint**；LIBERO checkpoint 不能作为这里的替代品。默认服务端口为 8000、8001、8002。
- 正式 API 配置仍使用仓库 `.emerge/config.json`。启动器只传递路径，不修改它，也不把凭据复制到任务目录。

从仓库根目录运行：

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
2. 直接检查 OpenPI、VGGT、SAM3 的 `/healthz`；依赖缺失时记录基础设施错误并退出。通过后，用 RoboCasa Python 启动 `robot.controller --driver robocasa`，安装 profile 并创建原生环境。
3. 等待私有状态、`robots.robocasa`、profile 和初始相机观测就绪，从环境获取原始任务指令及预算。
4. 用 EmergePolicy Python 启动现有 headless runtime。主 agent 自主规划、调用子代理并执行几何/VLA 动作；启动器不预写任务动作序列。
5. 监督环境状态。主 agent 正常提前返回、任务仍未完成时，最多进行两轮同 session 的继续反馈。每轮使用独立 run_id，共享最初的执行截止时间。
6. 成功、预算耗尽、超时、异常或收到中断后，先停止主 agent，让它通过仍在运行的 Controller 取消未完成动作，再清理 Controller，最后保存结果。

RoboCasa 保留 `gym.make(seed=...)` 及既有 reset 流程，不引入 LIBERO 的 BDDL、初始状态数组或额外 settling 动作。没有 `--max-steps` 或 driver 配置覆盖时，动作预算由原生任务 registry 解析。VLA 与几何动作共享这个预算。

## 重要参数

| 参数 | 含义 |
| --- | --- |
| `--env-name` | 必填，原生 RoboCasa 任务类名 |
| `--split` / `--seed` | 默认 `target` / `7` |
| `--max-steps` | 覆盖整个 episode 的控制步数预算 |
| `--max-agent-iterations` | 覆盖每轮主 agent 的工具迭代上限；默认使用 API 配置中的值 |
| `--max-agent-feedback-rounds` | 主 agent 正常返回但任务未完成时的继续反馈次数，默认 2 |
| `--ready-timeout-s` | 环境初始化超时，默认 300 秒 |
| `--episode-timeout-s` | 就绪后的执行超时，默认 1800 秒，续跑不会重置 |
| `--shutdown-grace-s` | 子进程收到中断后的退出宽限，默认 15 秒 |
| `--skip-service-checks` | 仅在受控诊断中跳过本地模型健康检查；结果会明确记录是否跳过 |

## 输出与判定

```text
<output-dir>/
  driver_config.json    # 本次仿真配置
  status.json           # Controller 写入的官方环境状态，位于 agent workspace 之外
  result.json           # 启动器最终结果
  controller.log
  agent.log
  agent_requests/       # headless 请求，只引用正式 API 配置路径
  agent_runs/           # 每轮请求、事件、运行结果和日志
  workspace/
    AGENTS.md
    EMBODIED.md
    PLAN.md
    ROBOT_STATE.md
    ACTION.md
    artifacts/          # 相机 PNG、标定观测等
    memory/             # 通用运行时生成的视觉记忆
```

只有 `status.json` 中官方 `success=true` 才能判定 benchmark 成功。主 agent 的“完成”回复、动作执行完成或视觉验证结果都不替代它。结果中的 `outcome` 区分：

- `success`：环境确认成功。
- `task_failure`：预算耗尽、执行时间耗尽或主 agent 结束时任务仍未完成。
- `infrastructure_error`：环境初始化/仿真、模型服务、协议或进程等执行问题。
- `cancelled`：用户通过 SIGINT/SIGTERM 中断启动器。

`termination_reason`、`errors` 和 `environment_status` 保留具体原因。初始化 NaN 等错误不会记为策略失败；成功后产生的清理异常会保留在 `errors` 中，官方成功标记仍保留。
`process_cleanup` 记录子进程是否需要强制终止；`service_checks` 记录启动时的模型健康检查。纯几何动作诊断可以显式跳过服务检查，正常 EMERGE 评测应保留检查。

CLI 返回码：0 表示评测正常结束（包括任务失败），1 表示执行基础设施错误，130 表示中断，2 表示命令/配置参数错误。判断任务成功请读取 `result.json.success`。

相机图片沿用 LIBERO 的更新方式：每个相机使用固定 PNG 路径，通过临时文件替换当前图片，再更新 `observation.json`，不会按观测版本累积历史目录。历史动作证据由现有视觉监控独立保存在 `memory/images/`。PNG 路径代表当前画面，查看历史画面应使用视觉记忆中的关键图。

第一版只负责一个 episode。批量调度、断点续跑和连续 MP4 录像尚未实现；当前保存的是已发布的相机快照、动作历史及运行时记忆。
