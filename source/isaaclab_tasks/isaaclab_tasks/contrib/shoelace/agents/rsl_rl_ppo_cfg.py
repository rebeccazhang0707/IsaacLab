# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ShoelaceHybridActionDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian arm and Bernoulli gripper distribution configuration."""

    class_name: str = "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution"
    arm_action_scale: float = 0.7
    gripper_logit_scale: float = 2.0


@configclass
class ShoelacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for dual-Franka shoelace untying."""

    num_steps_per_env = 16
    init_at_random_ep_len = False
    max_iterations = 1000
    save_interval = 50
    experiment_name = "shoelace_dual_franka"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=ShoelaceHybridActionDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
