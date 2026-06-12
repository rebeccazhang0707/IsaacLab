# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the heterogeneous multi-task demo environments.

Each environment is assembled by :class:`~isaaclab_contrib.tasks.manipulation.multitask.registry.MultiTaskRegistry`
in the sibling ``demo_registry_*_env_cfg`` modules. Importing this package (which happens when
``isaaclab_contrib.tasks`` is imported) registers the ids below with gymnasium.
"""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Multi-robot reach -- OpenArm, Franka, UR10 all doing reach.
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Flat-Multi-Robot-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_reach_env_cfg:RegistryMultiRobotReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Multi-Robot-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_reach_env_cfg:RegistryMultiRobotReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotReachPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Multi-robot lift.
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Flat-Multi-Robot-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_lift_env_cfg:RegistryMultiRobotLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Multi-Robot-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_lift_env_cfg:RegistryMultiRobotLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotLiftPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Franka multi-task -- one robot, several tasks.
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Flat-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_franka_multi_task_env_cfg:RegistryFrankaMultiTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaMultiTaskPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_franka_multi_task_env_cfg:RegistryFrankaMultiTaskEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaMultiTaskPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Multi-robot multi-task -- several robots, several tasks.
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Flat-Multi-Robot-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.demo_registry_multi_robot_multi_task_env_cfg:RegistryMultiRobotMultiTaskEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotMultiTaskPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Multi-Robot-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.demo_registry_multi_robot_multi_task_env_cfg:RegistryMultiRobotMultiTaskEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiRobotMultiTaskPPORunnerCfg",
    },
)
