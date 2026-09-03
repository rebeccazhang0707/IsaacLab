# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from .constants import LEFT_CABLE_SEGMENT_COUNT, PINNED_LAST, REFERENCE_PULL_DIRECTIONS, TAIL_REGIONS, TCP_OFFSET

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, CableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def task_state(
    env: ManagerBasedEnv,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local dynamic-segment, seam, and free-tail state."""
    cables: tuple[CableObject, CableObject] = (
        env.scene[asset_cfgs[0].name],
        env.scene[asset_cfgs[1].name],
    )
    position_chains = tuple(
        cable.data.segment_pose_w.torch[..., :3] - env.scene.env_origins.unsqueeze(1) for cable in cables
    )
    velocity_chains = tuple(cable.data.segment_velocity_w.torch[..., :3] for cable in cables)
    positions = torch.cat(position_chains, dim=1)
    velocities = torch.cat(velocity_chains, dim=1)
    knot = 0.5 * (position_chains[0][:, -1] + position_chains[1][:, 0])
    right_tail_lower = TAIL_REGIONS[0][0] - PINNED_LAST
    right_tail_upper = TAIL_REGIONS[0][1] - PINNED_LAST
    left_tail_lower, left_tail_upper = TAIL_REGIONS[1]
    tail_positions = torch.stack(
        (
            position_chains[1][:, right_tail_lower:right_tail_upper].mean(dim=1),
            position_chains[0][:, left_tail_lower:left_tail_upper].mean(dim=1),
        ),
        dim=1,
    )
    tail_velocities = torch.stack(
        (
            velocity_chains[1][:, right_tail_lower:right_tail_upper].mean(dim=1),
            velocity_chains[0][:, left_tail_lower:left_tail_upper].mean(dim=1),
        ),
        dim=1,
    )
    return positions, velocities, knot, tail_positions, tail_velocities


def free_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return both dynamic cable spans without their fixed seam anchors."""
    return torch.cat(
        (positions[:, : LEFT_CABLE_SEGMENT_COUNT - 1], positions[:, LEFT_CABLE_SEGMENT_COUNT + 1 :]), dim=1
    )


def robot_tcp_position(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one Franka TCP position in the environment frame [m]."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = robot_cfg.body_ids[0]
    hand_position = robot.data.body_pos_w.torch[:, body_id] - env.scene.env_origins
    hand_quaternion = robot.data.body_quat_w.torch[:, body_id]
    offset = hand_position.new_tensor(TCP_OFFSET).expand_as(hand_position)
    return hand_position + math_utils.quat_apply(hand_quaternion, offset)


def tail_to_tcp_vectors(
    env: ManagerBasedEnv,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return each grasped tail position relative to its Franka TCP [m]."""
    _, _, _, tail_positions, _ = task_state(env, asset_cfgs)
    tcp_positions = torch.stack(
        [robot_tcp_position(env, left_robot_cfg), robot_tcp_position(env, right_robot_cfg)],
        dim=1,
    )
    return tail_positions - tcp_positions


def grasp_distances(
    env: ManagerBasedEnv,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return tail-to-TCP grasp distances [m], shape ``(num_envs, 2)``."""
    return torch.linalg.vector_norm(
        tail_to_tcp_vectors(env, asset_cfgs, left_robot_cfg, right_robot_cfg),
        dim=-1,
    )


def gripper_positions(
    env: ManagerBasedEnv,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return driven finger-joint positions [m], shape ``(num_envs, 2)``."""
    robots: tuple[Articulation, Articulation] = (
        env.scene[left_robot_cfg.name],
        env.scene[right_robot_cfg.name],
    )
    robot_cfgs = (left_robot_cfg, right_robot_cfg)
    return torch.stack(
        [robot.data.joint_pos.torch[:, robot_cfg.joint_ids[0]] for robot, robot_cfg in zip(robots, robot_cfgs)],
        dim=1,
    )


def gripper_closed_fraction(
    env: ManagerBasedEnv,
    open_position: float,
    closed_position: float,
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return normalized driven-finger closure in ``[0, 1]``."""
    positions = gripper_positions(env, left_robot_cfg, right_robot_cfg)
    return ((open_position - positions) / (open_position - closed_position)).clamp(0.0, 1.0)


def grasp_state(
    env: ManagerBasedEnv,
    maximum_distance: float,
    maximum_finger_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
    minimum_finger_position: float = 0.0,
) -> torch.Tensor:
    """Infer per-tail grasp state from proximity and contact-sized finger aperture.

    Args:
        env: The task environment.
        maximum_distance: Maximum tail-to-TCP distance [m].
        maximum_finger_position: Maximum driven finger-joint position [m].
        asset_cfgs: Scene entities for the left and right cable chains.
        left_robot_cfg: Left robot hand and finger scene entity.
        right_robot_cfg: Right robot hand and finger scene entity.
        minimum_finger_position: Minimum driven finger-joint position [m].

    Returns:
        Per-tail inferred grasp flags, shape ``(num_envs, 2)``.
    """
    distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    finger_positions = gripper_positions(env, left_robot_cfg, right_robot_cfg)
    valid_aperture = (finger_positions >= minimum_finger_position) & (finger_positions <= maximum_finger_position)
    return (distances <= maximum_distance) & valid_aperture


def pull_directions(reference: torch.Tensor) -> torch.Tensor:
    """Return normalized demo pull directions on the reference tensor's device."""
    directions = reference.new_tensor(REFERENCE_PULL_DIRECTIONS)
    return directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)


def untying_metrics(
    positions: torch.Tensor,
    knot: torch.Tensor,
    tail_positions: torch.Tensor,
    throat_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return throat count, tail-to-knot distances [m], and tail separation [m]."""
    free_segment_positions = free_positions(positions)
    free_distances = torch.linalg.vector_norm(free_segment_positions - knot.unsqueeze(1), dim=-1)
    throat_count = (free_distances < throat_radius).sum(dim=1)
    tail_distances = torch.linalg.vector_norm(tail_positions - knot.unsqueeze(1), dim=-1)
    tail_separation = torch.linalg.vector_norm(tail_positions[:, 0] - tail_positions[:, 1], dim=-1)
    return throat_count, tail_distances, tail_separation


def potential(
    positions: torch.Tensor,
    knot: torch.Tensor,
    tail_positions: torch.Tensor,
    throat_radius: float,
    tail_success_distance: float,
    tail_success_separation: float,
) -> torch.Tensor:
    """Compute the dimensionless dense two-tail untying potential."""
    throat_count, tail_distances, tail_separation = untying_metrics(
        positions,
        knot,
        tail_positions,
        throat_radius,
    )
    free_segment_count = free_positions(positions).shape[1]
    throat_clearance = 1.0 - throat_count.float() / free_segment_count
    return (
        2.0 * throat_clearance
        + (tail_separation / tail_success_separation).clamp(max=2.0)
        + 0.5 * (tail_distances.amin(dim=1) / tail_success_distance).clamp(max=2.0)
    )
