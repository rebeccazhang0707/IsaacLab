# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .utils import free_positions, grasp_state, pull_directions, tail_to_tcp_vectors, task_state


def tails_to_tcp(
    env,
    asset_cfg: SceneEntityCfg,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tail positions relative to their controlling TCPs [m], shape ``(num_envs, 6)``."""
    return tail_to_tcp_vectors(env, asset_cfg, left_robot_cfg, right_robot_cfg).flatten(start_dim=1)


def tails_to_knot(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shoelace"),
) -> torch.Tensor:
    """Tail positions relative to the knot center [m], shape ``(num_envs, 6)``."""
    _, _, knot, tail_positions, _ = task_state(env, asset_cfg)
    return (tail_positions - knot.unsqueeze(1)).flatten(start_dim=1)


def tail_velocities(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shoelace"),
) -> torch.Tensor:
    """Free-tail linear velocities [m/s], shape ``(num_envs, 6)``."""
    _, _, _, _, velocities = task_state(env, asset_cfg)
    return velocities.flatten(start_dim=1)


def reference_pull_directions(env) -> torch.Tensor:
    """Normalized demo pull directions, shape ``(num_envs, 6)``."""
    directions = pull_directions(env.scene.env_origins).flatten().unsqueeze(0)
    return directions.expand(env.num_envs, -1)


def inferred_grasp_state(
    env,
    maximum_distance: float,
    maximum_finger_position: float,
    asset_cfg: SceneEntityCfg,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-tail inferred grasp flags, shape ``(num_envs, 2)``."""
    return grasp_state(
        env,
        maximum_distance,
        maximum_finger_position,
        asset_cfg,
        left_robot_cfg,
        right_robot_cfg,
    ).float()


def throat_density(
    env,
    throat_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shoelace"),
) -> torch.Tensor:
    """Fraction of free segments within ``throat_radius`` [m] of the knot."""
    positions, _, knot, _, _ = task_state(env, asset_cfg)
    distances = torch.linalg.vector_norm(free_positions(positions) - knot.unsqueeze(1), dim=-1)
    return (distances < throat_radius).float().mean(dim=1, keepdim=True)


def tail_separation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shoelace"),
) -> torch.Tensor:
    """Distance between the two free-tail grasp regions [m]."""
    _, _, _, tail_positions, _ = task_state(env, asset_cfg)
    return torch.linalg.vector_norm(tail_positions[:, 0] - tail_positions[:, 1], dim=-1, keepdim=True)


def episode_phase(env) -> torch.Tensor:
    """Normalized episode progress, shape ``(num_envs, 1)``."""
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)
