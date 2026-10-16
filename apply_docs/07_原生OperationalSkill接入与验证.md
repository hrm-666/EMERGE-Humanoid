# LightNav-0 + SONIC 原生 Operational Skill 接入

本次修改位于实验室 `embodied-MS-7E06:~/EMERGE-Humanoid` 的 `humanoid-dev`。
目标是把已经具有仿真执行能力的导航后端纳入 EMERGE 的统一 Skill 调用流程。
`real-dev` 保留此前创建的基线分支，本次不加入真机代码，也不移动该分支。

## 官方依据与版本边界

- 论文：[EMERGE-Policy](https://arxiv.org/abs/2608.29896)，阅读 HTML v2 的
  §2.3.1–2.3.3（功能分类、具体工具 API、标准契约及执行生命周期）。
- 官方仓库：[EMERGE-Policy/EMERGE-Policy](https://github.com/EMERGE-Policy/EMERGE-Policy)。
- 本次对照提交：`e4c7134ad2aeb6dded049106ada1c28985b66e69`。
- 已读取的主要官方源码：`Emerge/agent/skills.py`、`context.py`、
  `tools/embodied.py`，以及 `skills/vla`、`wam`、`planning`、
  `task-verification`，`robot/drivers/base_driver.py`、`robot/policy_execution.py`。
- 阅读快照在实验室 `/tmp/emerge-official-skill-audit/`，只作本次审计缓存。

论文按能力在决策循环中的作用分类。LightNav 输出局部 waypoint，SONIC 将其执行为
G1 全身运动，因此组合为一个 Operational Skill；它不负责证明任务已完成。
Evaluation Skill 用当前观测对照 Done Criterion，Main Agent 根据证据决定继续、
推进计划或恢复。LightNav 和 SONIC 无须各自成为新的 Main Agent 工具。

官方可执行范式是 `SKILL.md` + 原生工具 API + Controller/Driver。没有要求新建
一个继承某个 OperationalSkill 基类的导航类，也没有按论文三分类新增三套注册器。
Skill frontmatter 中的一行描述先进入上下文，选中后通过 `read_file` 读取完整说明。
官方 VLA 常驻加载，WAM 按需加载；本次导航采用按需加载以符合论文的上下文设计。

官方最新驱动生命周期和本地分支不是同一版本。本次保留本地
`BaseDriver.load_scene/execute_action/get_runtime_state/close`、已有插件注册以及
原生队列取消协议，没有为导航而整体替换上游驱动/API。

## 契约对应

| 论文契约 | 本次实现 |
| --- | --- |
| Metadata | 内置 `Emerge/skills/lightnav-sonic-navigation/SKILL.md` 的 name/description；现有 SkillsLoader 自动发现 |
| Parameters | Skill 参数表、原生 `execute_robot_action` schema、Driver 参数校验 |
| Preconditions | 当前 embodiment 支持动作，目标/当前阶段明确，相机、yaw 和模型可用；计划和可观察完成条件已建立 |
| Execution postconditions | JSON 执行状态、原因、ticks、指令数、时间、stop/waypoint/yaw；末尾尝试发送 IDLE，释放执行资源 |
| Outcome postconditions | `task_verification` 对实际 Done Criterion 返回 achieved / not_achieved / uncertain 和证据；Main Agent 负责计划转换 |

注意：前置条件中，语义目标/计划由 Agent 遵循 Skill 指令建立；实时输入和参数由
Driver 检查。没有新增自动语义强制门禁。符合官方范式不等于能保证 LLM 总是遵守。
`update_plan` 负责校验并持久化计划，不会自主判断导航是否成功。

## 原生执行链

```text
ContextBuilder / SkillsLoader
  -> read_file(内置导航 SKILL.md)
  -> update_plan(当前阶段 + Done Criterion)
  -> 必要时 delegate_subagent(task_verification) 做定性目标确认
  -> execute_robot_action(action_type="navigate_vln")
  -> workspace/ACTION.md
  -> robot.controller._poll_once
  -> SonicLightNavDriver -> NavigationBackend
  -> LightNav RGB+instruction -> waypoint -> SONIC planner/WBC
  -> ACTION.md 结果 + ROBOT_STATE.md + 当前 ego 观测
  -> delegate_subagent(task_verification)
     -> observe_scene -> submit_task_verification
  -> update_plan(progress / Branch Stack / replan)
```

`completed/lightnav_stop` 表示模型希望停止，`budget_exhausted` 表示预算耗尽。
队列可以把正常预算返回记为 completed，仍然不表示语义到达。
单目图像不能证明米制距离或物理站稳；不可凭 yaw 虚构世界 X/Y。
若原生 visual monitor 已给出同一完整 criterion 的当前验证，可复用该证据。
未知、错误目标或无进展不能通过无条件加大预算解决。

## 代码变更与原因

| 文件 | 作用 |
| --- | --- |
| `Emerge/skills/lightnav-sonic-navigation/SKILL.md` | 唯一的导航 Skill：前置条件、阶段指令、参数、后置状态、原生验证和恢复 |
| `Emerge/skills/README.md` | 内置能力目录 |
| `robot/controller.py` | 对 lightnav_sonic_g1 使用 Controller 选定 workspace，防止队列与图像落在不同目录 |
| `integrations/lightnav_sonic_emerge/driver.py` | 空闲/动作前后刷新只读 RGB；暴露 observation_state；关闭时清理订阅 |
| `camera.py` | SONIC JPEG 与 Official01 RGB 复用一个只读 JPEG 接口 |
| `navigation_backend.py` | 执行时使用同一相机选择；相机失效时撤回旧观测 |
| `observation_writer.py` | 原子发布图像与 manifest，携带发布时间/失效时间，支持撤回 manifest |
| `Emerge/subagents/task_verification/tools/observation.py` | 读取时拒绝过期导航观测，兼容没有 expires_at 的既有 manifest |
| `Emerge/agent/visual_monitor.py` | 跳过已过期的导航图像，避免旧画面触发成功中断 |
| `driver_tabletop.json` | 苹果桌 5560 Official01 RGB，支撑释放文件禁用，沿用同一导航动作 |
| `checks/test_native_skill.py` | 原生发现、示例 schema、Controller、验证子代理、计划转换和观测生命周期回归 |
| `checks/test_camera.py` | 两种传输适配、默认 factory 选择及错误配置拒绝 |

被动相机只订阅 RGB，不创建运动 publisher。Controller 默认示例轮询 0.25 s；
应小于输入超时 0.75 s。动作中仍由有界导航循环发布图像。TTL 主要保证发布活性：
SONIC JPEG 用本地接收时间，Official01 还检查源时间戳；不是严格同步的采集时间保证。
过期后原 PNG 保留用于诊断，验证不得读取它充当当前证据。

## 安装、迁移与运行

原生 Skill 随主仓库安装，不再复制 `workspace_skill/SKILL.md`。外部 Driver
仍按官方插件机制注册，命令与仿真进程参见
[README](../integrations/lightnav_sonic_emerge/README.md) 和
[SIMULATION](../integrations/lightnav_sonic_emerge/SIMULATION.md)。

workspace 同名 Skill 优先级高于 builtin，用户自定义覆盖仍保持有效。本次实验室
旧副本与历史提交 `9a862bd` 逐字节一致，已备份后移除；备份在：

```text
~/.Emerge/backups/lightnav-sonic-navigation/
52d520749e49dc6e9989007a92fea82d3868d7566664003cc72f8ac7b106d807.md
```

其他 workspace 的自定义副本应先比较/迁移再移除，不能一律覆盖。
在上述实验室默认 workspace 中，新会话将自动发现 builtin 版本。

苹果桌与 ProcTHOR 只是 RGB 传输配置不同，使用同一个 Operational Skill。
苹果桌必须先按原仿真流程释放支撑并确认站稳；不能同时运行独立 control 和
EMERGE Controller。独立 tabletop 的键盘/motion gate/复位/录像评估能力未迁入
Controller，也不声称此次适配与原独立控制器具有相同的运行效果。

## 验证与未验证边界

- 核心离线测试：61 项，含原生 Skill 发现与渐进读取、所有 Python 调用示例
  schema 校验、自定义 workspace、真实 Controller 队列调度、有界后端清理。
- 验证子代理使用正式组装函数和正式 observe/submit 工具；模型为脚本替身。
  覆盖 stop+achieved、stop+not_achieved、预算耗尽+uncertain。
  动作和验证返回均不自动更改 PLAN；脚本扮演 Main Agent 通过 update_plan
  演示已验证推进与保留当前目标/增加重试/压入恢复分支。
- 初始图像、丢帧撤回、关闭撤回、无人刷新时 TTL 失效、visual monitor
  跳过旧图及旧 manifest 兼容性均有回归。
- 苹果桌仿真既有 80 项测试继续回归，不启动真实部署或大型模型。
- 新增/修改的 integration Python 按 Google YAPF 配置格式化，并通过 Ruff。
- 本轮未运行真实 LLM 规划、GPU LightNav 推理或 SONIC 运动，因此没有新增导航
  成功率结论；真实模型的验证质量、恢复决策及苹果桌完整闭环仍需实验验收。

原有 GR00T/VLA 的四项未提交修改保持不变，不纳入本次提交。
