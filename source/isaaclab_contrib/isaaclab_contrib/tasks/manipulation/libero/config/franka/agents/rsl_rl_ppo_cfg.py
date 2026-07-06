# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiberoMultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 200
    experiment_name = "libero_multi_task"
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiberoObjectPPORunnerCfg(LiberoMultiTaskPPORunnerCfg):
    """PPO runner for the LIBERO-Object per-task multi-task env (10 groups)."""

    experiment_name = "libero_object"


@configclass
class LiberoLongPPORunnerCfg(LiberoMultiTaskPPORunnerCfg):
    """PPO runner for the LIBERO-Long per-task multi-task env (10 groups)."""

    experiment_name = "libero_long"


@configclass
class LiberoAllPPORunnerCfg(LiberoMultiTaskPPORunnerCfg):
    """PPO runner for the combined 40-task LIBERO env (object-level prototype sharing)."""

    experiment_name = "libero_all"


@configclass
class LiberoGoalPPORunnerCfg(LiberoMultiTaskPPORunnerCfg):
    """PPO runner for the goal-only harvest env (10 tasks, object-level prototype sharing)."""

    experiment_name = "libero_goal"


@configclass
class LiberoSpatialPPORunnerCfg(LiberoMultiTaskPPORunnerCfg):
    """PPO runner for the spatial-only harvest env (10 tasks, object-level prototype sharing)."""

    experiment_name = "libero_spatial"
