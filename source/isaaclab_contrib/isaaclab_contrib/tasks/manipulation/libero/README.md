# LIBERO Multi-Task Manipulation Environment

[中文](./README.md)

This package combines LIBERO's four suites (`libero_spatial` / `libero_goal` /
`libero_object` / `libero_long`) into a single trainable `ManagerBasedRLEnv`
multi-task reinforcement learning environment. Overall design:

- **Single scene, multiple tasks**: via `MultiTaskRegistry`, each LIBERO task is
  registered as a clone group, sharing the same Franka Panda robot (differential
  IK 6-DoF + binary gripper 1-DoF action).
- **Object-level prototype sharing via harvest + selector**: identical USD models
  appearing across tasks are de-duplicated into a single `AssetView` (spawned
  once, one instance), then cloned in following the "union of all envs that use
  it".
- **Per-env MDP gather keyed by task id**: because a shared `AssetView` spans envs
  belonging to different tasks, the usual `SceneEntityCfg(selector=group)`
  grouping would cross wires, so all task-related reward / termination /
  observation / reset logic is instead gathered per env in `mdp/combined.py`
  based on the deterministic `sequential` clone mapping (env `i` runs
  task `i % n_tasks`).
- **Three reward modes**: `goal` (sparse success reward), `metaworld_dense`
  (demonstration-free reach→lift→place dense shaping + success reward), and
  `world_prediction` (requires per-step demonstration reference trajectories,
  which this repo has no data source for, so it degrades to keeping only the
  sparse success reward and emits a one-time warning). Can be overridden with the
  `LIBERO_REWARD_MODE` environment variable.
- **Data-driven layout**: each task's object/fixture layout, success thresholds,
  etc. are read from `$LIBERO_CONFIG_DIR/<suite>.json` (same source as the
  upstream benchmark), and USD assets are loaded from `$LIBERO_ASSETS_DATA_DIR`.

> Note: this package previously also had a `per_task_group` (grouped by task,
> per-task namespaced) scene-building implementation, which has now been removed.
> Only the `harvest` implementation remains on disk. For the historical
> performance comparison between the two approaches, see
> `scripts/benchmarks/libero_scene_profiling.md`.

## Directory Structure

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

## Per-File Responsibilities

### Package root

- **`__init__.py`** — package-level docstring, explaining how the LIBERO suites
  are combined into a single multi-task RL environment via `MultiTaskRegistry`.
- **`assets.py`** — shared asset definitions and directory resolution:
  `libero_assets_dir()` (reads `LIBERO_ASSETS_DATA_DIR`), `libero_config_dir()`
  (reads `LIBERO_CONFIG_DIR`, otherwise falls back to the `config` folder next to
  the assets directory), the shared rigid-body solver properties
  `OBJECT_RIGID_PROPS`, and the `LIBERO_FRANKA_PANDA_CFG` articulation config with
  LIBERO PD gains and a ready pose.

### `config/franka/`

- **`__init__.py`** — registers 14 environment ids with gymnasium (7 base + their
  respective `-Play-` variants), all resolving to the concrete
  `Libero*EnvCfg` / `_PLAY` classes in `libero_all_env_cfg.py`.
- **`libero_all_env_cfg.py`** — the harvest+selector factory. Its core is
  `make_libero_combined_env_cfg` (any subset of suites → one
  `ManagerBasedRLEnvCfg` subclass) and `make_libero_combined_play_cfg`;
  `SUITE_SPECS` defines the selectable suites; `_augment` replaces the registry's
  grouped-task MDP with per-env task-gather items (setting the `sequential` clone
  strategy, `task_onehot`, object observations, the three reward modes,
  success/drop termination, per-task reset). It also exports all concrete cfgs:
  `LiberoAllEnvCfg`, `LiberoSpatialEnvCfg`, `LiberoGoalEnvCfg`,
  `LiberoObjectEnvCfg`, `LiberoLongEnvCfg`, `LiberoSpatialGoalEnvCfg`,
  `LiberoObjectLongEnvCfg`, and their `_PLAY` counterparts.
- **`agents/__init__.py`** — agents sub-package placeholder.
- **`agents/rsl_rl_ppo_cfg.py`** — RSL-RL PPO run configuration. Base class
  `LiberoMultiTaskPPORunnerCfg`, plus per-suite subclasses named by
  `experiment_name`: `LiberoObjectPPORunnerCfg` / `LiberoLongPPORunnerCfg` /
  `LiberoAllPPORunnerCfg` / `LiberoGoalPPORunnerCfg` /
  `LiberoSpatialPPORunnerCfg`.

### `mdp/`

- **`__init__.py`** — provides a lazy re-export entry point via `lazy_export()`.
- **`__init__.pyi`** — explicit type stub: beyond `from ...multitask.mdp import *`,
  it also re-exports the per-env gather items from `combined.py` and the
  LIBERO-specific items from `rewards.py` and `terminations.py`.
- **`combined.py`** — the task-related MDP that the combined environment gathers
  per env keyed by task id. Contains the cached `_LiberoRuntime` (env→task
  mapping, success thresholds, lift/drop heights, primary/target env grouped by
  prototype); observations `libero_task_onehot`, `libero_object_positions`;
  rewards `libero_reach_reward`, `libero_lift_reward`, `libero_place_reward`,
  `libero_success_bonus`; terminations `libero_task_success`,
  `libero_object_dropped`; and the reset event `reset_libero_prototypes`.
- **`rewards.py`** — LIBERO-specific rewards/masks: `object_reached_target_mask`
  (geometric success proxy), `object_reached_target_bonus` (sparse success
  reward), `object_to_object_distance` (tanh shaping of object-to-target
  distance).
- **`terminations.py`** — the LIBERO-specific termination `object_reached_target`,
  reusing `object_reached_target_mask`, which ends the episode when the object
  reaches the target and is at rest.

### `robots/`

- **`__init__.py`** — re-exports `FRANKA` and `LiberoFrankaRobotCfg`.
- **`franka.py`** — `LiberoFrankaRobotCfg` (`RobotModuleCfg`): scene assets
  (`franka_robot` + the `franka_ee_frame` frame transformer), action spec (6-DoF
  differential IK arm + 1-DoF binary gripper), robot-side scatter observations
  (ee_pose / joint_pos / joint_vel), reset events; and the pre-built instance
  `FRANKA`.

### `tasks/`

- **`__init__.py`** — package-root re-exports: from `.common` it exports
  `LiberoManipulationTaskCfg` and `resolve_reward_mode`, and from `.harvest` it
  exports `build_combined_tasks`.

#### `tasks/common/` (cross-implementation shared code)

- **`__init__.py`** — re-exports `base.py`'s asset builders / reward-mode
  constants and `suite_loader.py`'s data structures and loader functions.
- **`base.py`** — the base manipulation task cfg `LiberoManipulationTaskCfg`
  (abstract subclass hooks + construction of the three reward modes, geometric
  success proxy, termination and reset events), plus the asset builders
  `kitchen_table`, `rigid_object`, `fixture`, `wooden_cabinet`, `flat_stove`,
  `microwave`, `white_cabinet`; reward-mode resolution `resolve_reward_mode` and
  the constant `REWARD_MODES`. For relational/articulated goals
  (`primary_object_key is None`, e.g. opening a drawer or turning on a stove), it
  guards the reward / termination / reset (no items targeting a rigid primary
  object, no `SceneEntityCfg(name=None)` reset).
- **`suite_loader.py`** — the data-driven suite loader: reads each task's layout
  and BDDL goal predicates from `$LIBERO_CONFIG_DIR/<suite>.json`, performs
  workspace normalization (translating each task's robot base to the shared
  kitchen base `ROBOT_BASE_KITCHEN`), fixture scaling
  (`_FIXTURE_SCALE_MULTIPLIERS`), and articulation default states
  (`ARTICULATION_TYPES` / `ARTICULATION_JOINT_STATES`). Data structures
  `ObjectSpec` / `FixtureSpec` / `TaskLayout`, functions `load_suite`,
  `suite_is_homogeneous`. `libero_long` maps to `libero_10.json` on disk.

#### `tasks/harvest/` (the sole scene-building implementation)

- **`__init__.py`** — re-exports harvest's public symbols.
- **`prototypes.py`** — harvests USD models that are identical across tasks into
  shared `Prototype` / `AssetView`: `harvest_libero_prototypes` (harvests in
  `LIBERO_SUITES` order, returns de-duplicated prototypes + per-task
  `TaskBinding`); data classes `Prototype`, `ObjectBinding`, `TaskBinding`; the
  suite-order constant `LIBERO_SUITES`. De-duplication is by model (type + scale)
  + occurrence within the same task; static fixtures additionally overlay pose.
- **`combined.py`** — the prototype-sharing task module: `build_combined_tasks`
  (harvests the selected suites and builds one registration per LIBERO task that
  contributes only its shared prototype assets), `build_prototype_cfgs` (each
  prototype → one fixture/rigid/articulation cfg), `LiberoPrototypeTaskCfg` (a
  thin task registration whose task-related MDP is all left empty, to be filled
  in per env by `mdp/combined.py`).

## Registered gym task ids

Each id below has a `-Play-` variant (fewer envs, observation noise disabled),
for a total of 7 + 7 = 14.

| Gym ID | What it trains |
| --- | --- |
| `Isaac-Libero-Spatial-Franka-Multi-Task-v0` | `libero_spatial` single suite (10 tasks) |
| `Isaac-Libero-Goal-Franka-Multi-Task-v0` | `libero_goal` single suite (10 tasks) |
| `Isaac-Libero-Object-Franka-Multi-Task-v0` | `libero_object` single suite (10 tasks) |
| `Isaac-Libero-Long-Franka-Multi-Task-v0` | `libero_long` (long-horizon) single suite (10 tasks) |
| `Isaac-Libero-All-Franka-Multi-Task-v0` | all four suites (40 tasks) trained jointly |
| `Isaac-Libero-Spatial-Goal-Franka-Multi-Task-v0` | spatial + goal combination (20 tasks) |
| `Isaac-Libero-Object-Long-Franka-Multi-Task-v0` | object + long combination (20 tasks) |

## Environment variables

- `LIBERO_ASSETS_DATA_DIR` — root directory of the LIBERO USD assets.
- `LIBERO_CONFIG_DIR` — directory of per-task JSON configs (falls back to the
  `config` folder next to the assets directory when unset).
- `LIBERO_REWARD_MODE` — overrides the reward mode; values `goal` /
  `metaworld_dense` / `world_prediction`.
