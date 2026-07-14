# LIBERO Multi-Task Manipulation Environment

[中文](./README_CN.md)

This package combines LIBERO's four suites (`libero_spatial` / `libero_goal` /
`libero_object` / `libero_long`) into a single trainable `ManagerBasedRLEnv`
multi-task reinforcement learning environment. Overall design:

- **Single scene, multiple tasks**: via `MultiTaskRegistry`, each LIBERO task is
  registered as a clone group, sharing the same Franka Panda robot (OSC
  `pose_rel` 6-DoF + binary gripper 1-DoF action).
- **Object-level prototype sharing via harvest + selector**: identical USD models
  appearing across tasks are de-duplicated into a single `AssetView` (spawned
  once, one instance), then cloned into the union of all envs that use it.
- **Per-env MDP gather keyed by task id**: because a shared `AssetView` spans envs
  belonging to different tasks, success termination and per-task reset are
  gathered per env in `mdp/combined.py` from the deterministic `sequential` clone
  map (env `i` runs task `i % n_tasks`); DGPO observations/rewards are installed
  by `envs/dgpo_env_cfg.py`.
- **DGPO observation contract**: actor=324 / critic=572 / action=7, suite order
  long → object → spatial → goal (see `dgpo_layout.py`).
- **Data-driven layout**: each task's object/fixture layout, success thresholds,
  etc. are read from `$LIBERO_CONFIG_DIR/<suite>.json` (same source as the
  upstream benchmark), and USD assets are loaded from `$LIBERO_ASSETS_DATA_DIR`.

> Note: this package previously also had a `per_task_group` implementation and a
> Diff-IK Multi-Task path, both now removed. Only harvest scene construction +
> the DGPO OSC training path remain. For the historical harvest vs
> `per_task_group` comparison, see `scripts/benchmarks/libero_scene_profiling.md`.

## Directory Structure

```text
libero/
├── __init__.py                     # package docstring
├── assets.py                       # shared assets / dir resolution + Franka articulation
├── dgpo_layout.py                  # DGPO obs/action dim contract + suite order
├── config/
│   ├── __init__.py
│   └── franka/
│       ├── __init__.py             # registers DGPO gym ids
│       ├── libero_dgpo_env_cfg.py  # concrete DGPO EnvCfg classes
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py   # LiberoAllDgpoPPORunnerCfg
├── envs/
│   ├── dgpo_env.py                 # DgpoManagerBasedRLEnv
│   └── dgpo_env_cfg.py             # make_libero_dgpo_env_cfg factory
├── mdp/
│   ├── __init__.py                 # lazy_export entry
│   ├── __init__.pyi                # explicit re-exports (typing / IDE)
│   ├── combined.py                 # per-env success termination + prototype reset
│   ├── observations.py             # DGPO observation terms
│   ├── demos/                      # demo commands + reset
│   ├── rewards.py                  # LIBERO-specific rewards (object-to-object, etc.)
│   └── terminations.py             # LIBERO-specific terminations
├── robots/
│   ├── __init__.py
│   └── franka_osc.py               # LiberoFrankaOscRobotCfg / FRANKA_OSC
└── tasks/
    ├── __init__.py                 # package-root re-exports (build_combined_tasks, etc.)
    ├── common/                     # shared across implementations
    │   ├── __init__.py
    │   ├── base.py                 # asset builders + reward-mode resolution
    │   └── suite_loader.py         # load per-task layouts / goal predicates from JSON
    └── harvest/                    # sole scene-construction impl (prototype sharing)
        ├── __init__.py
        ├── prototypes.py           # harvest identical USDs into shared Prototype/bindings
        └── combined.py             # build_combined_tasks / prototype task cfg
```

## Per-File Responsibilities

### `config/franka/`

- **`__init__.py`** — registers DGPO OSC gym ids (train All + play subsets).
- **`libero_dgpo_env_cfg.py`** — concrete `Libero*DgpoOscEnvCfg` / `_PLAY` classes.
- **`agents/rsl_rl_ppo_cfg.py`** — `LiberoAllDgpoPPORunnerCfg` (and Compat alias).

### `envs/`

- **`dgpo_env_cfg.py`** — `make_libero_dgpo_env_cfg` / `make_libero_dgpo_play_cfg`:
  harvest scene + OSC actions + DGPO obs groups; installs `libero_task_success`
  and `reset_libero_prototypes` (or demo reset when demos are present).

### `mdp/`

- **`combined.py`** — shared per-env success termination `libero_task_success` and
  reset event `reset_libero_prototypes` (used by the DGPO path).
- **`observations.py` / `demos/`** — DGPO multi-hot, pose buffer, privileged diffs,
  demo `initial_state` reset.

### `robots/`

- **`franka_osc.py`** — OSC Franka (action dim 7) for DGPO registration.

### `tasks/harvest/`

- **`prototypes.py` / `combined.py`** — object-level prototype harvest and
  `build_combined_tasks` (required by the DGPO factory).

## Registered gym task ids

All registered ids use the **DGPO** path (OSC + actor=324 / critic=572 / action=7).
Train registers the full All env only; play covers All plus suite subsets
(one env per task, observation noise disabled). Subset train configs still
exist on `libero_dgpo_env_cfg` for programmatic use.

| Gym ID | What it trains / plays |
| --- | --- |
| `Isaac-Libero-All-Dgpo-Osc-v0` | **train** all four suites (40 tasks) |
| `Isaac-Libero-All-Dgpo-Osc-Play-v0` | play/eval all four suites |
| `Isaac-Libero-Long-Dgpo-Osc-Play-v0` | play `libero_long` / `libero_10` (10 tasks) |
| `Isaac-Libero-Object-Dgpo-Osc-Play-v0` | play `libero_object` (10 tasks) |
| `Isaac-Libero-Spatial-Dgpo-Osc-Play-v0` | play `libero_spatial` (10 tasks) |
| `Isaac-Libero-Goal-Dgpo-Osc-Play-v0` | play `libero_goal` (10 tasks) |
| `Isaac-Libero-Spatial-Goal-Dgpo-Osc-Play-v0` | play spatial + goal (20 tasks) |
| `Isaac-Libero-Object-Long-Dgpo-Osc-Play-v0` | play long + object (20 tasks; DGPO order) |

## Environment variables

- `LIBERO_ASSETS_DATA_DIR` — root directory of the LIBERO USD assets.
- `LIBERO_CONFIG_DIR` — directory of per-task JSON configs (falls back to the
  `config` folder next to the assets directory when unset).
- `LIBERO_REWARD_MODE` — overrides the reward mode; values `goal` /
  `metaworld_dense` / `world_prediction`.
- `LIBERO_ASSEMBLED_DATASET_DIR` — HDF5 demo root for privileged diffs + demo
  `initial_state` reset on the DGPO path. See `envs/dgpo_env_cfg.py`.
- `LIBERO_EVALUATION=1` — disable random demo start-timestep sampling (DGPO path).
- `LIBERO_COMPAT_REQUIRE_DEMOS=1` — fail if demos are missing (DGPO path).
