# Multitask Manipulation — 开发指南

## 模块概述

本模块实现了**异构多机器人、多任务**强化学习环境，核心创新是 `MultiTaskRegistry` + `@scatterable` + `scatter_term` 组合，支持在单一场景中并行训练多种机器人执行不同任务，而无需修改 IsaacLab 核心 Managers。

---

## 目录结构

```
multitask/
├── registry.py          # MultiTaskRegistry：动态组装完整 RL 环境配置
├── robots/              # 机器人模块（Franka / OpenArm / UR10）
│   ├── _base.py         # 抽象基类 RobotModuleCfg
│   ├── franka.py
│   ├── openarm.py
│   └── ur10.py
├── tasks/               # 任务模块（reach / lift / cabinet）
│   ├── _base.py         # 抽象基类 TaskModuleCfg
│   ├── reach.py
│   ├── lift.py
│   └── cabinet.py
├── mdp/                 # MDP 组件
│   ├── utils.py         # @scatterable 装饰器、scatter_term 类
│   ├── obs.py           # 观测函数（均为 @scatterable）
│   ├── rewards.py       # 奖励函数（均为 @scatterable）
│   ├── terminations.py  # 终止条件（@scatterable）
│   ├── events.py        # 重置事件（使用 layout.filter_reset_ids）
│   ├── actions.py       # ScatteredActionTerm
│   ├── actions_cfg.py   # ScatteredActionTermCfg
│   ├── commands.py      # PoseCommand（group-aware）
│   └── commands_cfg.py  # PoseCommandCfg / PoseCommandRanges
└── config/demo/         # 示例环境配置
    ├── demo_franka_multi_task_env_cfg.py          # 平铺配置（~700行，仅供参考）
    ├── demo_registry_franka_multi_task_env_cfg.py # 注册式配置（~25行，推荐）
    ├── demo_multi_robot_reach_env_cfg.py
    ├── demo_registry_multi_robot_reach_env_cfg.py
    ├── demo_multi_robot_lift_env_cfg.py
    ├── demo_registry_multi_robot_lift_env_cfg.py
    ├── demo_multi_robot_multi_task_env_cfg.py
    ├── demo_registry_multi_robot_multi_task_env_cfg.py
    └── agents/rsl_rl_ppo_cfg.py
```

---

## 核心架构

### 双索引原则（`env_ids` / `view_ids`）

每个 group-local 资产只克隆到属于它的那部分 env。所有 MDP 函数维护一对索引：

- `env_ids`：在全量 `(num_envs, ...)` 输出 buffer 中的写入位置
- `view_ids`：在该资产自己的局部 buffer 中的读取位置

这对索引由 `SceneEntityCfg.groups` 在 init 时一次性 resolve，运行时零额外开销。

### 数据流

```
CloneCfg + InclusionSet
    ↓ (场景构建时)
EnvLayout（groups → env_ids / view_ids 映射）
    ↓ (init 时 resolve 一次)
SceneEntityCfg.groups → env_ids + view_ids
    ↓ (每步)
@scatterable MDP term → (env_ids, group_local_result) → scatter 到全量 buffer
scatter_term → 汇合多个 @scatterable，共用一块预分配 buffer
```

### Managers 完全无感知

`ObservationManager`、`RewardManager`、`TerminationManager`、`EventManager`、`ActionManager` **不含任何多任务分发代码**。所有异构路由在三层实现：

1. **Config 层** — `CloneCfg` 分区；`SceneEntityCfg.groups` 解析双索引
2. **Layout 层** — `EnvLayout` 提供纯函数索引（frozen dataclass + pure function）
3. **MDP 层** — `@scatterable` + `scatter_term` + `ScatteredActionTerm` 处理局部计算和 scatter

---

## 关键组件说明

### `@scatterable` 装饰器（`mdp/utils.py`）

装饰需要 group-local 计算的函数。被装饰函数需返回 `(env_ids, group_local_result)`，装饰器负责 scatter 到全量 buffer：

```python
@scatterable
def ee_pose(env, asset_cfg: SceneEntityCfg) -> ScatterResult:
    robot = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w[asset_cfg.view_ids, body_idx]  # 局部读
    # ...计算...
    return asset_cfg.env_ids, torch.cat([pos_b, quat_b], dim=-1)
```

- standalone 调用：复用持久零分配 buffer
- 被 `scatter_term` 调用时：通过 `_out=buf` 原地写入共享 buffer

### `scatter_term`（`mdp/utils.py`）

将多个 `@scatterable` 子函数组合为一个 obs term，共享单块预分配 buffer：

```python
ee_pose = ObsTerm(func=mdp.scatter_term, params={"terms": [
    TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"])}),
    TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("ur10_robot",   groups=["ur10_reach"])}),
]})
```

每步：清零 buffer → 各子函数 scatter 写入 → 返回 buffer（零拷贝）。

### `ScatteredActionTerm`（`mdp/actions.py`）

多个机器人共享同一动作列（如三台机器人都用 6D IK）：

```python
arm = mdp.ScatteredActionTermCfg(
    dim=6,  # fallback dim，用于任务禁用时保持动作维度
    terms=[
        DifferentialInverseKinematicsActionCfg(asset_name="franka_robot", ...),
        DifferentialInverseKinematicsActionCfg(asset_name="ur10_robot",   ...),
    ]
)
```

- Policy 输出一份 6D actions，`ScatteredActionTerm` 按 `env_ids` 分发给各子 term
- 所有子 term **必须** 有相同 `action_dim`，否则 init 时抛 `ValueError`
- 子 term 对应资产不在 layout 中时自动跳过（支持任务禁用）

### `MultiTaskRegistry`（`registry.py`）

注册 `(robot, task)` 对并自动组装完整 `ManagerBasedRLEnvCfg`：

```python
env_cfg = (
    MultiTaskRegistry()
    .register(FRANKA_IK, LIFT_TASK,    group_name="franka_lift")
    .register(FRANKA_IK, REACH_TASK,   group_name="franka_reach")
    .register(UR10_IK,   REACH_TASK,   group_name="ur10_reach")
    .build_env_cfg(num_envs=4096, env_spacing=2.5, episode_length_s=8.0)
)
```

Registry 自动处理：
- 场景资产（跨 registration 共用的机器人资产自动提升为全局；任务资产为 group-local）
- Action / Obs / Reward / Termination / Event / Curriculum 的组合
- 全局 penalty（`action_rate`、`joint_vel`）和 curriculum 权重调度

---

## 机器人模块（`RobotModuleCfg`）

| 机器人 | 文件 | 控制模式 | 夹爪 |
|--------|------|----------|------|
| FrankaRobotCfg | franka.py | IK (6D) / relative_joint (7D) | 有 |
| OpenArmRobotCfg | openarm.py | IK (6D) / relative_joint (6D) | 有 |
| UR10RobotCfg | ur10.py | IK (6D) / relative_joint (6D) | 无 |

所有机器人 IK 均使用 Differential IK + DLS（Damped Least Squares）方法。

每个 `RobotModuleCfg` 需实现：
- `name` — 唯一机器人标识符
- `scene_assets(group)` — 机器人 USD 资产 + EE frame sensor
- `action_specs(group)` — 动作列字典（arm / gripper），含维度
- `scatter_obs_terms(group)` — 机器人侧 scatter 观测（通常是 EE pose）
- `reset_events(group)` — 关节复位事件

---

## 任务模块（`TaskModuleCfg`）

| 任务 | 文件 | 目标 | 需要夹爪 |
|------|------|------|----------|
| ReachTaskCfg | reach.py | EE pose tracking | 否 |
| LiftTaskCfg | lift.py | 抓取 cube 到目标位置 | 是 |
| CabinetTaskCfg | cabinet.py | 开抽屉 | 是（Franka/OpenArm） |

每个 `TaskModuleCfg` 需实现：
- `name` — 任务标识符
- `scene_assets(group, robot)` — 任务物体（cube、cabinet 等）
- `command_terms(group, robot)` — 目标指令生成器
- `task_obs_terms(group, robot)` — 任务本地观测（带 group 命名空间）
- `scatter_obs_terms(group, robot)` — 跨 group scatter 观测
- `reward_terms(group, robot)` — 任务奖励函数
- `termination_terms(group, robot)` — 终止条件
- `reset_events(group, robot)` — 物体重置事件

---

## 任务禁用（Play 模式）

Play 配置使用 `apply_task_filter(disabled_tasks)` 在**保持动作/观测维度不变**的前提下禁用任务：

```python
# 在 PLAY 配置类上调用
env_cfg = MultiRobotMultiTaskEnvCfg_PLAY()
env_cfg.apply_task_filter(disabled_tasks=("openarm_lift", "franka_cabinet"))
```

`apply_task_filter` 执行 7 个步骤（完全数据驱动，无 per-task if/else）：
1. clone group `weight=0` → 不克隆资产
2. scene entities → `None`（不生成资产）
3. 涉及禁用 group 的 commands → `None`
4. obs：`MultiTaskObsTerm` → `zero_obs(dim=N)`（维度保持）；`scatter_term` 子项过滤
5. action 子 term 过滤（通过 `ScatteredActionTermCfg.dim` 保持维度）
6. rewards / curriculum → 全部 `None`
7. terminations / events → 选择性 `None`（全局 term 如 `time_out` 保留）

**关键约束**：
- `disabled_tasks` 类型为 `tuple[str, ...]`（OmegaConf 不支持 `set`）
- group-local obs 必须用 `MultiTaskObsTerm(dim=N, ...)` 声明维度
- `ScatteredActionTermCfg` 必须声明 `dim` 字段

---

## 注册环境

Demo 环境通过 gym 注册，在 `config/demo/__init__.py` 中定义：

| 环境名 | 场景描述 |
|--------|----------|
| `Isaac-Flat-Multi-Robot-Reach-v0` | 3机器人同任务（reach） |
| `Isaac-Flat-Multi-Robot-Lift-v0` | 2机器人同任务（lift） |
| `Isaac-Flat-Franka-Multi-Task-v0` | 1机器人3任务 |
| `Isaac-Flat-Multi-Robot-Multi-Task-v0` | 多机器人多任务 |
| `*-Play-v0` 变体 | 对应评估配置 |

---

## 运行命令

```bash
# 训练
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Flat-Multi-Robot-Reach-v0 --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Flat-Multi-Robot-Multi-Task-v0 --headless

# 评估
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Flat-Multi-Robot-Multi-Task-Play-v0 --visualizer kit

# 独立场景演示（无 RL）
./isaaclab.sh -p scripts/demos/heterogeneous_scene.py \
    --visualizer kit --num_envs 24

# 单元测试（EnvLayout）
./isaaclab.sh -p -m pytest source/isaaclab/test/scene/test_env_layout.py
```

---

## 开发注意事项

### 新增机器人

1. 在 `robots/` 创建新文件，继承 `RobotModuleCfg`
2. 实现所有抽象方法（`name`、`scene_assets`、`action_specs`、`scatter_obs_terms`、`reset_events`）
3. 在 `robots/__init__.py` 导出
4. 若任务要求夹爪（如 `CabinetTaskCfg`），需在 `action_specs` 中提供 `"gripper"` 列

### 新增任务

1. 在 `tasks/` 创建新文件，继承 `TaskModuleCfg`
2. 实现所有抽象方法
3. group-local obs 使用 `MultiTaskObsTerm(dim=N, ...)` 而非 `ObsTerm`（支持 play 禁用）
4. 在 `tasks/__init__.py` 导出

### 新增 MDP 函数

- 需要 group-local 计算的函数加 `@scatterable` 装饰器，返回 `(env_ids, result)`
- 全局函数（所有 env 参与）不需要 `@scatterable`
- 奖励/终止函数同样可以 `@scatterable`
- event 函数通过 `env.scene.layout.filter_reset_ids(asset_name, env_ids)` 获取双索引

### 平铺配置 vs 注册式配置

- **注册式**（`demo_registry_*.py`）：推荐，~25 行，自动组装
- **平铺配置**（`demo_multi_robot_*.py`）：保留用于参考，~700 行，手动组装

新配置**始终使用注册式**，不要手动写平铺配置。

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ValueError: All sub-terms must have same action_dim` | `ScatteredActionTermCfg.terms` 维度不一致 | 确保同一列所有子 term 维度相同 |
| `obs dim mismatch` 在 play 模式 | group-local obs 用 `ObsTerm` 而非 `MultiTaskObsTerm` | 改用 `MultiTaskObsTerm(dim=N, ...)` |
| `OmegaConf TypeError` for `disabled_tasks` | 类型为 `set` | 改为 `tuple[str, ...]` |
| `KeyError` in `scatter_term` | 子函数的 `asset_cfg.groups` 匹配不到 layout 中的 group | 检查 group 名称拼写，确认 `CloneCfg` 中已注册 |
| Action dim 在禁用任务后缩小 | `ScatteredActionTermCfg` 缺少 `dim` 字段 | 为每个 `ScatteredActionTermCfg` 声明 `dim` |
