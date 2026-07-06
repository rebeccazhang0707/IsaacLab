# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the LIBERO multitask environments.

Every environment is built by the harvest+selector factory
(:func:`.libero_all_env_cfg.make_libero_combined_env_cfg`): identical object
models are de-duplicated into one shared :class:`~isaaclab.assets.AssetView` and
the task-dependent MDP is gathered per env by task id.  Importing this package
(which happens when ``isaaclab_contrib.tasks`` is imported) registers the ids
below with gymnasium:

* ``Isaac-Libero-Spatial-Franka-Multi-Task-v0`` — spatial suite alone (10 tasks).
* ``Isaac-Libero-Goal-Franka-Multi-Task-v0`` — goal suite alone (10 tasks).
* ``Isaac-Libero-Object-Franka-Multi-Task-v0`` — object suite alone (10 tasks).
* ``Isaac-Libero-Long-Franka-Multi-Task-v0`` — long-horizon suite alone (10 tasks).
* ``Isaac-Libero-All-Franka-Multi-Task-v0`` — all four suites (40 tasks) trained
  together with object-level prototype sharing.
* ``Isaac-Libero-Spatial-Goal-Franka-Multi-Task-v0`` — spatial + goal combo (20 tasks).
* ``Isaac-Libero-Object-Long-Franka-Multi-Task-v0`` — object + long combo (20 tasks).

All ids resolve to :mod:`.libero_all_env_cfg`; arbitrary subsets can be built with
:func:`.libero_all_env_cfg.make_libero_combined_env_cfg`.  Each id has a
``-Play-`` variant with fewer envs and no observation corruption.
"""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Single-suite harvest envs (object-level prototype sharing + per-env task gather)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Libero-Goal-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Goal-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoGoalEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoGoalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Spatial-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoSpatialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoSpatialPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Spatial-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoSpatialEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoSpatialPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# LIBERO-Object suite (object-level prototype sharing, 10 tasks)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Libero-Object-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoObjectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoObjectPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Object-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoObjectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoObjectPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# LIBERO-Long (long-horizon) suite (object-level prototype sharing, 10 tasks)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Libero-Long-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoLongEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoLongPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Long-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoLongEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoLongPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Combined 40-task env (object-level prototype sharing across all four suites)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Libero-All-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoAllEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoAllPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-All-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoAllEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoAllPPORunnerCfg",
    },
)

# Representative cross-suite combos built from the same factory.
gym.register(
    id="Isaac-Libero-Spatial-Goal-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoSpatialGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoMultiTaskPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Spatial-Goal-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoSpatialGoalEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoMultiTaskPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Object-Long-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoObjectLongEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoAllPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Libero-Object-Long-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_all_env_cfg:LiberoObjectLongEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoAllPPORunnerCfg",
    },
)
