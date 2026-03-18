# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for grouped multitask manipulation environments.

All reward functions accept ``env_ids: torch.Tensor | slice = slice(None)``
so they work in three contexts without code duplication:

* **Global** — called without ``task_group`` on the term config.
  ``env_ids`` defaults to ``slice(None)`` → operates on all envs.
* **task_group** — when assets are scoped to a task group, the
  default ``slice(None)`` selects all local rows; the manager
  handles scattering the result into the full buffer.
* **per_robot** — the manager dispatches once per robot with
  scoped assets; ``env_ids`` stays ``slice(None)`` and the
  scoped asset data is naturally group-sized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


# ===========================================================
# Reach rewards (position / orientation command tracking)
# ===========================================================


def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Position error between the EE and a pose command target [m].

    Uses the first body in ``asset_cfg.body_ids`` as the end-effector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(
        wp.to_torch(asset.data.root_pos_w)[env_ids],
        wp.to_torch(asset.data.root_quat_w)[env_ids],
        des_pos_b,
    )
    curr_pos_w = wp.to_torch(asset.data.body_pos_w)[env_ids, asset_cfg.body_ids[0]]
    return torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reach position reward with tanh kernel."""
    distance = position_command_error(env, command_name, asset_cfg, env_ids)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Orientation error between the EE and a pose command target [rad]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(wp.to_torch(asset.data.root_quat_w)[env_ids], des_quat_b)
    curr_quat_w = wp.to_torch(asset.data.body_quat_w)[env_ids, asset_cfg.body_ids[0]]
    return math_utils.quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reach orientation reward with tanh kernel."""
    quat_error = orientation_command_error(env, command_name, asset_cfg, env_ids)
    return 1 - torch.tanh(quat_error / std)


# ===========================================================
# Lift rewards (object + EE + goal tracking)
# ===========================================================


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("lift_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reaching reward: tanh of distance between EE and object.

    Shared-asset data (e.g. ``ee_frame``) is sliced by ``env_ids``;
    scoped-asset data (e.g. ``object``) is used as-is.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = wp.to_torch(obj.data.root_pos_w)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[env_ids, 0, :]
    distance = torch.linalg.norm(object_pos_w - ee_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("lift_object"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Goal-tracking reward: tanh of distance between object and command target."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = math_utils.combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w)[env_ids],
        wp.to_torch(robot.data.root_quat_w)[env_ids],
        command[:, :3],
    )
    object_pos_w = wp.to_torch(obj.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
    return (object_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# ===========================================================
# Cabinet rewards
# ===========================================================


def cabinet_approach_ee_handle(
    env: ManagerBasedRLEnv,
    threshold: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reward for reaching the cabinet handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[env_ids, 0, :]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def cabinet_align_ee_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reward for aligning with the cabinet handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_quat = wp.to_torch(ee_frame.data.target_quat_w)[env_ids, 0, :]
    handle_quat = wp.to_torch(cabinet_frame.data.target_quat_w)[:, 0, :]
    ee_rot = math_utils.matrix_from_quat(ee_quat)
    handle_rot = math_utils.matrix_from_quat(handle_quat)
    handle_x, handle_y = handle_rot[..., 0], handle_rot[..., 1]
    ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def cabinet_align_grasp_around_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Bonus when fingers straddle the drawer handle correctly."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    fingertips = wp.to_torch(ee_frame.data.target_pos_w)[env_ids, 1:, :]
    left_finger = fingertips[:, 0, :]
    right_finger = fingertips[:, 1, :]
    return (right_finger[:, 2] < handle_pos[:, 2]) & (left_finger[:, 2] > handle_pos[:, 2])


def cabinet_approach_gripper_handle(
    env: ManagerBasedRLEnv,
    offset: float = 0.04,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reward for finger placement around the handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    fingertips = wp.to_torch(ee_frame.data.target_pos_w)[env_ids, 1:, :]
    left_finger = fingertips[:, 0, :]
    right_finger = fingertips[:, 1, :]
    left_dist = torch.abs(left_finger[:, 2] - handle_pos[:, 2])
    right_dist = torch.abs(right_finger[:, 2] - handle_pos[:, 2])
    is_graspable = (right_finger[:, 2] < handle_pos[:, 2]) & (left_finger[:, 2] > handle_pos[:, 2])
    return is_graspable * ((offset - left_dist) + (offset - right_dist))


def cabinet_grasp_handle(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Reward for finger closing once the robot is close to the handle."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[env_ids, 0, :]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    gripper_joint_pos = wp.to_torch(robot.data.joint_pos)[env_ids][:, asset_cfg.joint_ids]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    is_close = distance <= threshold
    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def cabinet_open_drawer_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Drawer opening bonus for the cabinet group."""
    drawer_pos = wp.to_torch(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids[0]]
    is_graspable = cabinet_align_grasp_around_handle(env, ee_frame_cfg, cabinet_frame_cfg, env_ids).float()
    return (is_graspable + 1.0) * drawer_pos


def cabinet_multi_stage_open_drawer(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
    env_ids: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Multi-stage drawer opening bonus for the cabinet group."""
    drawer_pos = wp.to_torch(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids[0]]
    is_graspable = cabinet_align_grasp_around_handle(env, ee_frame_cfg, cabinet_frame_cfg, env_ids).float()
    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable
    return open_easy + open_medium + open_hard
