# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiberoAllDgpoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the DGPO harvest+OSC training path.

    Architecture mirrors RobotLearningLab ``MultiTaskLiberoAdaptiveBCRunnerCfg``:
    ``[512, 256, 128]`` MLPs with ``noise_std_type="log"`` so legacy
    ``model_32500.pt`` (actor 512x324, critic 512x572) still loads under
    rsl-rl >= 4.0, and fresh DGPO training uses the same layout.
    """

    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 200
    experiment_name = "libero_all_dgpo"
    # rsl-rl >= 4.0 uses the "actor" key (not legacy "policy"). Missing "actor"
    # falls back to the env's single "policy" group (277) and breaks ckpt load.
    # Actor: policy(277)+proprio(47)=324; critic: +privileged_proprio(248)=572.
    obs_groups = {
        "actor": ["policy", "proprio"],
        "critic": ["policy", "proprio", "privileged_proprio"],
    }
    run_name = ""
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="log",
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


# Backward-compatible alias (older tests / docs).
LiberoAllCompatPPORunnerCfg = LiberoAllDgpoPPORunnerCfg
