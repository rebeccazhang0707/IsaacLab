# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for grouped multitask manipulation environments.

With the :class:`~isaaclab.scene.ScopedEnv` proxy, **standard single-task
MDP functions work in multitask configs without modification**.  This
module only keeps terms that have *no* standard equivalent:

* :func:`orientation_command_error_tanh` — missing from the reach task.
* Cabinet-specific rewards with configurable ``ee_frame_cfg`` /
  ``cabinet_frame_cfg`` parameters (the standard cabinet rewards
  hard-code ``"ee_frame"`` / ``"cabinet_frame"``).

Standard reach / lift / cabinet rewards are available via the
``multitask.mdp`` package, which re-exports them through
``__init__.pyi``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import orientation_command_error

# ===========================================================
# Reach reward without a standard equivalent
# ===========================================================


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reach orientation reward with tanh kernel.

    Not available in the standard reach task package, which only
    provides the raw :func:`orientation_command_error`.
    """
    quat_error = orientation_command_error(env, command_name, asset_cfg)
    return 1 - torch.tanh(quat_error / std)


# ===========================================================
# Cabinet rewards (configurable frame names)
# ===========================================================


def cabinet_approach_ee_handle(
    env: ManagerBasedRLEnv,
    threshold: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Reward for reaching the cabinet handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def cabinet_align_ee_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Reward for aligning with the cabinet handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_quat = wp.to_torch(ee_frame.data.target_quat_w)[:, 0, :]
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
) -> torch.Tensor:
    """Bonus when fingers straddle the drawer handle correctly."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    fingertips = wp.to_torch(ee_frame.data.target_pos_w)[:, 1:, :]
    left_finger = fingertips[:, 0, :]
    right_finger = fingertips[:, 1, :]
    return (right_finger[:, 2] < handle_pos[:, 2]) & (left_finger[:, 2] > handle_pos[:, 2])


def cabinet_approach_gripper_handle(
    env: ManagerBasedRLEnv,
    offset: float = 0.04,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Reward for finger placement around the handle."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    fingertips = wp.to_torch(ee_frame.data.target_pos_w)[:, 1:, :]
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
) -> torch.Tensor:
    """Reward for finger closing once the robot is close to the handle."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    handle_pos = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    gripper_joint_pos = wp.to_torch(robot.data.joint_pos)[:, asset_cfg.joint_ids]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    is_close = distance <= threshold
    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def cabinet_open_drawer_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Drawer opening bonus for the cabinet group."""
    drawer_pos = wp.to_torch(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids[0]]
    is_graspable = cabinet_align_grasp_around_handle(env, ee_frame_cfg, cabinet_frame_cfg).float()
    return (is_graspable + 1.0) * drawer_pos


def cabinet_multi_stage_open_drawer(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Multi-stage drawer opening bonus for the cabinet group."""
    drawer_pos = wp.to_torch(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids[0]]
    is_graspable = cabinet_align_grasp_around_handle(env, ee_frame_cfg, cabinet_frame_cfg).float()
    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable
    return open_easy + open_medium + open_hard
