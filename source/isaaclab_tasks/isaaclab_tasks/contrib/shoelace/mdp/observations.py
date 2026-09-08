# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .utils import free_positions, grasp_state, pull_directions, tail_to_tcp_vectors, task_state


def filtered_last_action(
    env,
    left_arm_action_name: str = "left_arm",
    left_gripper_action_name: str = "left_gripper",
    right_arm_action_name: str = "right_arm",
    right_gripper_action_name: str = "right_gripper",
) -> torch.Tensor:
    """Last executed arm commands and raw gripper actions [dimensionless]."""
    action_manager = env.action_manager
    return torch.cat(
        (
            action_manager.get_term(left_arm_action_name).filtered_actions,
            action_manager.get_term(left_gripper_action_name).raw_actions,
            action_manager.get_term(right_arm_action_name).filtered_actions,
            action_manager.get_term(right_gripper_action_name).raw_actions,
        ),
        dim=-1,
    )


def tails_to_tcp(
    env,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tail-to-TCP vectors in the controlling robot root frames [m]."""
    vectors_w = tail_to_tcp_vectors(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    vectors_b = torch.stack(
        [
            math_utils.quat_apply_inverse(env.scene[robot_cfg.name].data.root_quat_w.torch, vectors_w[:, arm])
            for arm, robot_cfg in enumerate((left_robot_cfg, right_robot_cfg))
        ],
        dim=1,
    )
    return vectors_b.flatten(start_dim=1)


def tails_to_knot(
    env,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Tail positions relative to the knot center [m], shape ``(num_envs, 6)``."""
    _, _, knot, tail_positions, _ = task_state(env, asset_cfgs)
    return (tail_positions - knot.unsqueeze(1)).flatten(start_dim=1)


def tail_velocities(
    env,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Free-tail linear velocities [m/s], shape ``(num_envs, 6)``."""
    _, _, _, _, velocities = task_state(env, asset_cfgs)
    return velocities.flatten(start_dim=1)


def reference_pull_directions(env) -> torch.Tensor:
    """Demo pull directions in the controlling robot root frames."""
    directions_w = pull_directions(env.scene.env_origins)
    directions_b = torch.stack(
        [
            math_utils.quat_apply_inverse(
                env.scene[robot_name].data.root_quat_w.torch,
                directions_w[arm].expand(env.num_envs, -1),
            )
            for arm, robot_name in enumerate(("robot_left", "robot_right"))
        ],
        dim=1,
    )
    return directions_b.flatten(start_dim=1)


def inferred_grasp_state(
    env,
    maximum_distance: float,
    maximum_finger_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-tail inferred grasp flags, shape ``(num_envs, 2)``."""
    return grasp_state(
        env,
        maximum_distance,
        maximum_finger_position,
        asset_cfgs,
        left_robot_cfg,
        right_robot_cfg,
    ).float()


def throat_density(
    env,
    throat_radius: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Fraction of free segments within ``throat_radius`` [m] of the knot."""
    positions, _, knot, _, _ = task_state(env, asset_cfgs)
    distances = torch.linalg.vector_norm(free_positions(positions) - knot.unsqueeze(1), dim=-1)
    return (distances < throat_radius).float().mean(dim=1, keepdim=True)


def tail_separation(
    env,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Distance between the two free-tail grasp regions [m]."""
    _, _, _, tail_positions, _ = task_state(env, asset_cfgs)
    return torch.linalg.vector_norm(tail_positions[:, 0] - tail_positions[:, 1], dim=-1, keepdim=True)


def episode_phase(env) -> torch.Tensor:
    """Normalized episode progress, shape ``(num_envs, 1)``."""
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)
