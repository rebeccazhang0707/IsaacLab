# LIBERO 多任务操作环境

[English](./README.en.md)

本包把 LIBERO 的四个套件（`libero_spatial` / `libero_goal` / `libero_object` /
`libero_long`）组合成一个可训练的 `ManagerBasedRLEnv` 多任务强化学习环境。整体设计：

- **单一场景、多任务**：通过 `MultiTaskRegistry` 把每个 LIBERO 任务注册为一个 clone
  group，共享同一台 Franka Panda 机器人（差分 IK 6 维 + 二值夹爪 1 维动作）。
- **harvest + selector 的对象级原型共享**：跨任务出现的相同 USD 模型会被去重为
  同一个 `AssetView`（一次 spawn、一份实例），再按“用到它的所有 env 的并集”克隆进去。
- **按 task id 的 per-env MDP gather**：因为共享的 `AssetView` 会横跨不同任务的 env，
  常规的 `SceneEntityCfg(selector=group)` 分组会串味，所以所有与任务相关的
  reward / termination / observation / reset 都改为在 `mdp/combined.py` 中依据
  确定性的 `sequential` 克隆映射（env `i` 跑 task `i % n_tasks`）逐 env 采集。
- **三种奖励模式**：`goal`（稀疏成功奖励）、`metaworld_dense`（无演示的
  reach→lift→place 稠密塑形 + 成功奖励）、`world_prediction`（需要逐步演示参考轨迹，
  本仓库无此数据源，故降级为仅保留稀疏成功奖励并给出一次警告）。可用环境变量
  `LIBERO_REWARD_MODE` 覆盖。
- **数据驱动布局**：每个任务的物体/夹具布局、成功阈值等从
  `$LIBERO_CONFIG_DIR/<suite>.json` 读取（与上游 benchmark 同源），USD 资产从
  `$LIBERO_ASSETS_DATA_DIR` 加载。

> 说明：本包此前还有一个 `per_task_group`（按任务分组、逐任务命名空间）的场景构建
> 实现，现已被移除。现在磁盘上只保留 `harvest` 这一种实现。两种方案的历史性能对比见
> `scripts/benchmarks/libero_scene_profiling.md`。

## 目录结构

```text
libero/
├── __init__.py                     # 包文档字符串（多任务 LIBERO 环境总述）
├── assets.py                       # 共享资产/目录解析 + Franka 关节体配置
├── config/
│   ├── __init__.py
│   └── franka/
│       ├── __init__.py             # 注册 14 个 gym id（7 基础 + 7 -Play-）
│       ├── libero_all_env_cfg.py   # harvest 工厂 + 各具体 EnvCfg 类
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py   # RSL-RL PPO runner 配置
├── mdp/
│   ├── __init__.py                 # lazy_export 入口
│   ├── __init__.pyi                # 显式再导出（供类型检查/IDE）
│   ├── combined.py                 # per-env 按 task id 采集的任务相关 MDP 项
│   ├── rewards.py                  # LIBERO 专属奖励项（object-to-object 等）
│   └── terminations.py             # LIBERO 专属终止项（到达目标即终止）
├── robots/
│   ├── __init__.py
│   └── franka.py                   # LiberoFrankaRobotCfg / FRANKA
└── tasks/
    ├── __init__.py                 # 包根再导出（build_combined_tasks 等）
    ├── common/                     # 跨实现共享代码
    │   ├── __init__.py
    │   ├── base.py                 # 基础任务 cfg + 资产构建器 + 奖励模式解析
    │   └── suite_loader.py         # 从 JSON 加载每任务布局/目标谓词
    └── harvest/                    # 唯一的场景构建实现（原型共享）
        ├── __init__.py
        ├── prototypes.py           # 把相同 USD 收割为共享 Prototype/绑定
        └── combined.py             # build_combined_tasks / 原型任务 cfg
```

## 各文件职责

### 包根

- **`__init__.py`** — 包级文档字符串，说明 LIBERO 套件如何经 `MultiTaskRegistry`
  组合为一个多任务 RL 环境。
- **`assets.py`** — 共享资产定义与目录解析：`libero_assets_dir()`（读
  `LIBERO_ASSETS_DATA_DIR`）、`libero_config_dir()`（读 `LIBERO_CONFIG_DIR`，否则回退到
  资产目录旁的 `config`）、共享刚体解算属性 `OBJECT_RIGID_PROPS`，以及带 LIBERO PD 增益
  与就绪姿态的 `LIBERO_FRANKA_PANDA_CFG` 关节体配置。

### `config/franka/`

- **`__init__.py`** — 向 gymnasium 注册 14 个环境 id（7 个基础 + 各自的 `-Play-` 变体），
  全部解析到 `libero_all_env_cfg.py` 中的具体 `Libero*EnvCfg` / `_PLAY` 类。
- **`libero_all_env_cfg.py`** — harvest+selector 工厂。核心是
  `make_libero_combined_env_cfg`（任意套件子集 → 一个 `ManagerBasedRLEnvCfg` 子类）与
  `make_libero_combined_play_cfg`；`SUITE_SPECS` 定义可选套件；`_augment` 用 per-env
  task-gather 项替换 registry 的分组任务 MDP（设置 `sequential` 克隆策略、`task_onehot`、
  物体观测、三种奖励模式、成功/掉落终止、按任务重置）。同时导出所有具体 cfg：
  `LiberoAllEnvCfg`、`LiberoSpatialEnvCfg`、`LiberoGoalEnvCfg`、`LiberoObjectEnvCfg`、
  `LiberoLongEnvCfg`、`LiberoSpatialGoalEnvCfg`、`LiberoObjectLongEnvCfg` 及其 `_PLAY`。
- **`agents/__init__.py`** — agents 子包占位。
- **`agents/rsl_rl_ppo_cfg.py`** — RSL-RL PPO 运行配置。基类
  `LiberoMultiTaskPPORunnerCfg`，及按套件命名 `experiment_name` 的子类
  `LiberoObjectPPORunnerCfg` / `LiberoLongPPORunnerCfg` / `LiberoAllPPORunnerCfg` /
  `LiberoGoalPPORunnerCfg` / `LiberoSpatialPPORunnerCfg`。

### `mdp/`

- **`__init__.py`** — 通过 `lazy_export()` 提供惰性再导出入口。
- **`__init__.pyi`** — 显式类型存根：`from ...multitask.mdp import *` 之外，再导出
  `combined.py` 的 per-env 采集项、`rewards.py` 与 `terminations.py` 的 LIBERO 专属项。
- **`combined.py`** — combined 环境按 task id 逐 env 采集的任务相关 MDP。内含缓存
  `_LiberoRuntime`（env→task 映射、成功阈值、抬升/掉落高度、按原型分组的 primary/target
  env）；观测 `libero_task_onehot`、`libero_object_positions`；奖励
  `libero_reach_reward`、`libero_lift_reward`、`libero_place_reward`、`libero_success_bonus`；
  终止 `libero_task_success`、`libero_object_dropped`；重置事件 `reset_libero_prototypes`。
- **`rewards.py`** — LIBERO 专属奖励/掩码：`object_reached_target_mask`（几何成功代理）、
  `object_reached_target_bonus`（稀疏成功奖励）、`object_to_object_distance`（物体到目标的
  tanh 塑形）。
- **`terminations.py`** — LIBERO 专属终止 `object_reached_target`，复用
  `object_reached_target_mask`，在物体到达目标并静止时结束回合。

### `robots/`

- **`__init__.py`** — 再导出 `FRANKA` 与 `LiberoFrankaRobotCfg`。
- **`franka.py`** — `LiberoFrankaRobotCfg`（`RobotModuleCfg`）：场景资产
  （`franka_robot` + `franka_ee_frame` 帧变换器）、动作规格（6 维差分 IK 手臂 + 1 维二值
  夹爪）、机器人侧散布观测（ee_pose / joint_pos / joint_vel）、重置事件；预构建实例
  `FRANKA`。

### `tasks/`

- **`__init__.py`** — 包根再导出：从 `.common` 导出 `LiberoManipulationTaskCfg`、
  `resolve_reward_mode`，从 `.harvest` 导出 `build_combined_tasks`。

#### `tasks/common/`（跨实现共享代码）

- **`__init__.py`** — 再导出 `base.py` 的资产构建器/奖励模式常量与
  `suite_loader.py` 的数据结构与加载函数。
- **`base.py`** — 基础操作任务 cfg `LiberoManipulationTaskCfg`（抽象子类钩子 + 三种
  奖励模式的构建、几何成功代理、终止与重置事件），以及资产构建器
  `kitchen_table`、`rigid_object`、`fixture`、`wooden_cabinet`、`flat_stove`、
  `microwave`、`white_cabinet`；奖励模式解析 `resolve_reward_mode` 与常量
  `REWARD_MODES`。对关系型/关节型目标（`primary_object_key is None`，如开抽屉、开炉灶），
  在 reward / termination / reset 中做了防护（不加针对刚体主物的项、不做 `SceneEntityCfg(name=None)`
  的重置）。
- **`suite_loader.py`** — 数据驱动的套件加载器：从
  `$LIBERO_CONFIG_DIR/<suite>.json` 读取每任务布局与 BDDL 目标谓词，做工作区归一化
  （把各任务的机器人基座平移到共享的厨房基座 `ROBOT_BASE_KITCHEN`）、夹具缩放
  （`_FIXTURE_SCALE_MULTIPLIERS`）、关节体默认状态（`ARTICULATION_TYPES` /
  `ARTICULATION_JOINT_STATES`）。数据结构 `ObjectSpec` / `FixtureSpec` / `TaskLayout`，
  函数 `load_suite`、`suite_is_homogeneous`。`libero_long` 映射到磁盘上的
  `libero_10.json`。

#### `tasks/harvest/`（唯一的场景构建实现）

- **`__init__.py`** — 再导出 harvest 的公开符号。
- **`prototypes.py`** — 把跨任务相同的 USD 模型收割为共享的 `Prototype` / `AssetView`：
  `harvest_libero_prototypes`（按 `LIBERO_SUITES` 顺序收割，返回去重原型 + 每任务
  `TaskBinding`）；数据类 `Prototype`、`ObjectBinding`、`TaskBinding`；套件顺序常量
  `LIBERO_SUITES`。去重按模型（type + scale）+ 同任务内 occurrence；静态夹具再叠加 pose。
- **`combined.py`** — 原型共享的任务模块：`build_combined_tasks`（收割所选套件并为每个
  LIBERO 任务构建一个只贡献其共享原型资产的注册）、`build_prototype_cfgs`（每个原型 →
  一个 fixture/rigid/articulation cfg）、`LiberoPrototypeTaskCfg`（薄任务注册，任务相关
  MDP 全部留空，交由 `mdp/combined.py` 逐 env 补齐）。

## 已注册的 gym 任务 id

以下每个 id 都有一个 `-Play-` 变体（更少 env、关闭观测扰动），共 7 + 7 = 14 个。

| Gym ID | 训练内容 |
| --- | --- |
| `Isaac-Libero-Spatial-Franka-Multi-Task-v0` | `libero_spatial` 单套件（10 个任务） |
| `Isaac-Libero-Goal-Franka-Multi-Task-v0` | `libero_goal` 单套件（10 个任务） |
| `Isaac-Libero-Object-Franka-Multi-Task-v0` | `libero_object` 单套件（10 个任务） |
| `Isaac-Libero-Long-Franka-Multi-Task-v0` | `libero_long`（长程）单套件（10 个任务） |
| `Isaac-Libero-All-Franka-Multi-Task-v0` | 全部四套件（40 个任务）联合训练 |
| `Isaac-Libero-Spatial-Goal-Franka-Multi-Task-v0` | spatial + goal 组合（20 个任务） |
| `Isaac-Libero-Object-Long-Franka-Multi-Task-v0` | object + long 组合（20 个任务） |

## 环境变量

- `LIBERO_ASSETS_DATA_DIR` — LIBERO USD 资产根目录。
- `LIBERO_CONFIG_DIR` — 每任务 JSON 配置目录（未设置时回退到资产目录旁的 `config`）。
- `LIBERO_REWARD_MODE` — 覆盖奖励模式，取值 `goal` / `metaworld_dense` / `world_prediction`。
