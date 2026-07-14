# LIBERO 多任务操作环境

[English](./README.md)

本包把 LIBERO 的四个套件（`libero_spatial` / `libero_goal` / `libero_object` /
`libero_long`）组合成一个可训练的多任务强化学习环境。**主路径**为
**DGPO-on-harvest**：

- **场景栈**：harvest + `CloneCfg` / `MultiTaskRegistry` — 跨任务相同的 USD
  模型去重为共享 `AssetView` 原型，再按确定性 `sequential` 策略克隆（env `i`
  跑 task `i % n_tasks`）。
- **训练 MDP 契约**：OSC 动作（维数 7）与 DGPO 观测组（actor=324 /
  critic=572）。见 `dgpo_layout.py` 与 `envs/dgpo_env_cfg.py`。
- **Demo 可选**：设置 `LIBERO_ASSEMBLED_DATASET_DIR` 时接入 privileged
  EE/关节/物体差分与 demo `initial_state` 重置；否则对应项为零 / 使用作者姿态。
- **数据驱动布局**：每任务物体/夹具布局与成功阈值来自
  `$LIBERO_CONFIG_DIR/<suite>.json`；USD 资产来自 `$LIBERO_ASSETS_DATA_DIR`。
- **四元数标志**：仿真侧数学始终为 XYZW。`policy_quat_order`（默认 `wxyz`）在
  观测边界转换 EE 位姿项；`demo_quat_order`（默认 `wxyz`）在加载时转换 demo
  `obs/ee_states`。任一项设为 `xyzw` 可对齐原生 Isaac Lab 3.x 策略 / demo 约定。

All 套件顺序与 `DGPO_ABC_HARVEST_SUITES` 一致：long → object → spatial → goal
（磁盘上 `libero_long` ≡ `libero_10`）。

## 目录结构

```text
libero/
├── __init__.py                     # 包文档字符串
├── assets.py                       # 资产/配置目录解析 + LIBERO Franka 关节体 cfg
├── dgpo_layout.py                  # 锁定的 DGPO 观测/动作维数 + harvest 套件顺序
├── config/
│   ├── __init__.py
│   └── franka/
│       ├── __init__.py             # gym id（训练 All + play All/子集）
│       ├── libero_dgpo_env_cfg.py  # 惰性 gym 入口 EnvCfg 类
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py   # LiberoAllDgpoPPORunnerCfg（actor 324 / critic 572）
├── envs/
│   ├── __init__.py
│   ├── dgpo_env.py                 # DgpoManagerBasedRLEnv（+ demo command manager）
│   └── dgpo_env_cfg.py             # make_libero_dgpo_env_cfg / play 工厂
├── mdp/
│   ├── __init__.py                 # lazy_export 入口
│   ├── __init__.pyi                # 类型存根 / 再导出
│   ├── observations.py             # DGPO 观测项（policy / proprio / privileged）
│   ├── combined.py                 # per-env 任务采集的成功判定 + 原型重置
│   ├── quat.py                     # policy/demo 四元数顺序辅助
│   ├── rewards.py                  # LIBERO 几何成功 / 塑形辅助
│   ├── terminations.py             # object_reached_target 终止
│   └── demos/                      # 可选 HDF5 demo 命令 + 重置事件
│       ├── __init__.py
│       ├── commands.py             # SourceLiberoCommand / make_dgpo_commands_cfg
│       ├── command_manager.py      # DgpoLiberoCommandManager
│       └── events.py               # demo initial_state 重置
├── robots/
│   ├── __init__.py
│   └── franka_osc.py               # LiberoFrankaOscRobotCfg / FRANKA_OSC（OSC pose_rel）
└── tasks/
    ├── __init__.py
    ├── common/                     # 共享任务 cfg + 套件 JSON 加载
    │   ├── __init__.py
    │   ├── base.py                 # LiberoManipulationTaskCfg + 奖励模式辅助
    │   └── suite_loader.py         # 从 JSON 加载 load_suite / TaskLayout
    └── harvest/                    # 场景构建（原型共享）
        ├── __init__.py
        ├── prototypes.py           # harvest_libero_prototypes / Prototype / TaskBinding
        └── combined.py             # build_combined_tasks / LiberoPrototypeTaskCfg
```

## 各文件职责

### 包根

- **`__init__.py`** — 包级文档字符串。
- **`assets.py`** — `libero_assets_dir()` / `libero_config_dir()`、共享
  `OBJECT_RIGID_PROPS`，以及 `LIBERO_FRANKA_PANDA_CFG`（LIBERO PD 增益与就绪姿态）。
- **`dgpo_layout.py`** — DGPO 维数契约的权威来源（actor=324 / critic=572 /
  action=7）、有序物体缓冲列、共享子任务 multi-hot 标签，以及
  `DGPO_ABC_HARVEST_SUITES`。

### `config/franka/`

- **`__init__.py`** — 将下列 gym id 注册到 `DgpoManagerBasedRLEnv` 与
  `libero_dgpo_env_cfg` 入口。
- **`libero_dgpo_env_cfg.py`** — 惰性 gym 入口类（`LiberoAllDgpoOscEnvCfg`、各
  套件 `_PLAY` 等），由 `make_libero_dgpo_env_cfg` /
  `make_libero_dgpo_play_cfg` 构建。子集**训练** cfg 类仍可程序化使用，但不单独
  注册 gym id。
- **`agents/rsl_rl_ppo_cfg.py`** — `LiberoAllDgpoPPORunnerCfg`，`obs_groups` 为
  actor=`[policy, proprio]` / critic=`[+ privileged_proprio]`。

### `envs/`

- **`dgpo_env.py`** — `DgpoManagerBasedRLEnv`：安装 `DgpoLiberoCommandManager`，
  使 demo 语义命令键按 playground Libero 方式展开。
- **`dgpo_env_cfg.py`** — 一等工厂：harvest 场景 + 单一 OSC 动作 + DGPO 观测组；
  可选 demo 命令/重置；暴露 `policy_quat_order` / `demo_quat_order`。

### `mdp/`

- **`observations.py`** — DGPO 观测项（task multi-hot、位姿缓冲、proprio、
  privileged 差分）。
- **`combined.py`** — `libero_task_success` 与 `reset_libero_prototypes` 的
  per-env 采集（共享 AssetView 不能用按组 selector）。
- **`quat.py`** — 在 policy / demo 边界做 `wxyz` / `xyzw` 转换。
- **`rewards.py` / `terminations.py`** — harvest 任务栈使用的几何成功辅助。
- **`demos/`** — HDF5 demo 加载（`SourceLiberoCommand`）、语义命令管理器，以及
  `reset_libero_scene_to_demo_initial_state`。

### `robots/`

- **`franka_osc.py`** — `LiberoFrankaOscRobotCfg` / `FRANKA_OSC`：OSC
  `pose_rel`（6）+ 二值夹爪（1）；手臂执行器刚度/阻尼置零，由 OSC 接管力矩。

### `tasks/`

- **`common/`** — `LiberoManipulationTaskCfg`、资产构建器，以及 `suite_loader`
  （JSON 布局；`libero_long` → `libero_10.json`）。
- **`harvest/`** — 唯一的场景构建实现：`harvest_libero_prototypes` +
  `build_combined_tasks` 生成供 registry / cloner 使用的原型共享任务注册。

## 已注册的 gym 任务 id

全部为 **DGPO** 路径（OSC + actor=324 / critic=572 / action=7）。
训练只注册全量 All；play 覆盖 All 与各套件子集（每 task 一个 env、关闭观测扰动）。

| Gym ID | 训练 / play 内容 |
| --- | --- |
| `Isaac-Libero-All-Dgpo-Osc-v0` | **训练** 全部四套件（40 个任务） |
| `Isaac-Libero-All-Dgpo-Osc-Play-v0` | play/eval 全部四套件 |
| `Isaac-Libero-Long-Dgpo-Osc-Play-v0` | play `libero_long` / `libero_10`（10 个任务） |
| `Isaac-Libero-Object-Dgpo-Osc-Play-v0` | play `libero_object`（10 个任务） |
| `Isaac-Libero-Spatial-Dgpo-Osc-Play-v0` | play `libero_spatial`（10 个任务） |
| `Isaac-Libero-Goal-Dgpo-Osc-Play-v0` | play `libero_goal`（10 个任务） |
| `Isaac-Libero-Spatial-Goal-Dgpo-Osc-Play-v0` | play spatial + goal（20 个任务） |
| `Isaac-Libero-Object-Long-Dgpo-Osc-Play-v0` | play long + object（20 个任务；DGPO 顺序） |

## 训练 / 评估

```bash
export LIBERO_ASSETS_DATA_DIR=/path/to/libero/USD
export LIBERO_CONFIG_DIR=/path/to/libero/config
# 可选：privileged critic / demo 重置用的 demos
export LIBERO_ASSEMBLED_DATASET_DIR=/path/to/sim2sim_dataset

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Libero-All-Dgpo-Osc-v0 --num_envs 2560 presets=physx

export LIBERO_EVALUATION=1
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Libero-All-Dgpo-Osc-Play-v0 --num_envs 40 \
    --checkpoint /path/to/model.pt presets=physx --headless
```

## 环境变量

- `LIBERO_ASSETS_DATA_DIR` — LIBERO USD 资产根目录。
- `LIBERO_CONFIG_DIR` — 每任务 JSON 配置目录（未设置时回退到资产目录旁的 `config`）。
- `LIBERO_ASSEMBLED_DATASET_DIR` — HDF5 demo 根目录，用于 privileged 差分与 demo
  `initial_state` 重置。详见 `envs/dgpo_env_cfg.py`。
- `LIBERO_EVALUATION=1` — 关闭随机 demo start-timestep 采样。
- `LIBERO_COMPAT_REQUIRE_DEMOS=1` — 缺少 demos 时硬失败。
- `LIBERO_COMPAT_MAX_DEMOS_PER_TASK` — 可选正整数；每任务最多加载 N 条 demo
  （play/debug）。
- `LIBERO_REWARD_MODE` — 在适用时覆盖 harvest 任务奖励模式（`goal` /
  `metaworld_dense` / `world_prediction`）。
- `ROBOT_INIT_NOISE_STD` — demo 重置时机器人手臂关节的可选高斯噪声（默认 `0.0`）。
