# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for multi-robot / multitask manipulation environments.

For multi-robot environments, prefer the batched observation classes in
``batched_obs.py`` which use the gather-first-compute-once pattern.

With :class:`~isaaclab.scene.ScopedEnv`, standard single-task functions
from the reach / lift / cabinet packages can also be used directly for
``task_group``-scoped terms.  Standard observations are available via the
``multitask.mdp`` package, which re-exports them through ``__init__.pyi``.

This module keeps terms that have **no standard equivalent**:

* :func:`ee_pose_b` — EE pose in the robot root frame.
* :func:`ee_pos_error` — position error between EE and command target.
* :func:`ee_object_pos_error` — position error between object and EE.
* :func:`object_target_pos_error` — target–object error in robot frame.
* :func:`cabinet_rel_ee_drawer_distance` — EE to drawer handle distance
  (configurable frame names, unlike the standard cabinet version).
* :func:`multi_task_onehot` — one-hot task group encoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.sensors import FrameTransformer


# ===========================================================
# Task-space observations (no standard equivalent)
# ===========================================================


def ee_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector pose in the robot root frame [m, -].

    The first body in ``asset_cfg.body_ids`` is treated as the
    end-effector.

    Returns:
        Shape ``(N, 7)`` — ``(pos_x, y, z, quat_w, x, y, z)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids[0]]
    ee_quat_w = wp.to_torch(asset.data.body_quat_w)[:, asset_cfg.body_ids[0]]
    root_pos = wp.to_torch(asset.data.root_pos_w)
    root_quat = wp.to_torch(asset.data.root_quat_w)
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_pos_w,
        ee_quat_w,
    )
    return torch.cat([pos_b, quat_b], dim=-1)


def ee_pos_error(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE position error vector ``(target − current)`` in the root frame [m].

    The first body in ``asset_cfg.body_ids`` is treated as the
    end-effector.  The command is expected to contain the target
    position in its first three columns (body-frame convention).

    Returns:
        Shape ``(N, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ee_pos_w = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids[0]]
    root_pos = wp.to_torch(asset.data.root_pos_w)
    root_quat = wp.to_torch(asset.data.root_quat_w)
    cur_b, _ = math_utils.subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_pos_w,
    )
    return cmd[:, :3] - cur_b


def ee_object_pos_error(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Position error vector ``(object - ee)`` in the robot root frame [m].

    The end-effector position is taken from the first target frame of
    ``ee_frame_cfg`` so per-robot TCP offsets remain consistent across
    heterogeneous robot groups.

    Returns:
        Shape ``(N, 3)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    root_pos = wp.to_torch(robot.data.root_pos_w)
    root_quat = wp.to_torch(robot.data.root_quat_w)
    object_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]

    object_pos_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, object_pos_w)
    ee_pos_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, ee_pos_w)
    return object_pos_b - ee_pos_b


def object_target_pos_error(
    env: ManagerBasedEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position error vector ``(target − object)`` in the robot root frame [m].

    The command is expected to contain the target position in its
    first three columns (body-frame convention).

    Returns:
        Shape ``(N, 3)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    object_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        wp.to_torch(robot.data.root_pos_w),
        wp.to_torch(robot.data.root_quat_w),
        object_pos_w,
    )
    return cmd[:, :3] - object_pos_b


def cabinet_rel_ee_drawer_distance(
    env: ManagerBasedEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cabinet_frame_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_frame"),
) -> torch.Tensor:
    """Drawer-handle minus EE TCP position [m].

    Unlike the standard cabinet version, frame names are configurable
    via ``ee_frame_cfg`` and ``cabinet_frame_cfg``.

    Returns:
        Shape ``(N, 3)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cabinet_frame: FrameTransformer = env.scene[cabinet_frame_cfg.name]
    handle_pos_w = wp.to_torch(cabinet_frame.data.target_pos_w)[:, 0, :]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    return handle_pos_w - ee_pos_w


# ===========================================================
# Robot identity (inherently global)
# ===========================================================


class multi_task_onehot(ManagerTermBase):
    """One-hot encoding of task group for every environment.

    Column *i* is 1.0 for all envs assigned to the *i*-th task group.
    The tensor is built once at init since group assignments are fixed.

    Returns:
        Shape ``(num_envs, num_task_groups)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        groups = layout.group_names
        group_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for i, group in enumerate(groups):
            ids = list(layout.env_ids(group))
            group_idx[ids] = i
        self._onehot = torch.nn.functional.one_hot(group_idx, num_classes=len(groups)).float()

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        return self._onehot
