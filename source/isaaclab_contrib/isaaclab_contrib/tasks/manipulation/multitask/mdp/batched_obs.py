# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched observation terms for multi-robot environments.

Each class uses a **gather-first, compute-once** pattern:

1. A short for-loop scatters per-asset data into pre-allocated
   staging buffers (cheap contiguous memcpy, no math).
2. A single batched call to the expensive math kernel
   (e.g. ``subtract_frame_transforms``) processes all envs at once,
   reducing CUDA kernel launches from ``N_groups × K`` to ``K``.

Variable-width outputs (joint-space terms) are zero-padded to the
maximum across robot groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils

from .utils import BatchedTermBase, CabinetGroupCfg, LiftGroupCfg, ReachGroupCfg, RobotGroupCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.scene import GroupView
    from isaaclab.sensors import FrameTransformer


# ===========================================================
# Task-space observations
# ===========================================================


class batched_ee_pose(BatchedTermBase):
    """EE pose in robot root frame, batched across robot groups [m, -].

    Returns shape ``(num_envs, 7)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, RobotGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups():
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_body_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_body_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._buf = torch.zeros(self._num_envs, 7, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_body_pos[group_view.write] = wp.to_torch(art.data.body_pos_w)[group_view.read, body_idx]
            self._s_body_quat[group_view.write] = wp.to_torch(art.data.body_quat_w)[group_view.read, body_idx]
            self._s_root_pos[group_view.write] = wp.to_torch(art.data.root_pos_w)[group_view.read]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self._s_root_pos, self._s_root_quat, self._s_body_pos, self._s_body_quat
        )
        self._buf[:, :3] = pos_b
        self._buf[:, 3:] = quat_b
        return self._buf


class batched_ee_pos_error(BatchedTermBase):
    """EE position error ``(target - current)`` in root frame, batched [m].

    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_body_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_cmd_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_body_pos[group_view.write] = wp.to_torch(art.data.body_pos_w)[group_view.read, body_idx]
            self._s_root_pos[group_view.write] = wp.to_torch(art.data.root_pos_w)[group_view.read]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
            self._s_cmd_pos[group_view.write] = env.command_manager.get_command(meta.command_name)[group_view.write, :3]
        cur_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_body_pos)
        return self._s_cmd_pos - cur_b


class batched_object_pos_in_robot_frame(BatchedTermBase):
    """Object position in robot root frame, batched [m].
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            robot_view = self._layout[group_key, meta.robot_cfg.name]
            object_view = self._layout[group_key, meta.object_cfg.name]
            self._entries.append((group_key, meta, robot_view, object_view))
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_obj_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for group_key, meta, robot_view, object_view in self._entries:
            robot: Articulation = env.scene[meta.robot_cfg.name]
            obj: RigidObject = env.scene[meta.object_cfg.name]
            self._s_root_pos[robot_view.write] = wp.to_torch(robot.data.root_pos_w)[robot_view.read]
            self._s_root_quat[robot_view.write] = wp.to_torch(robot.data.root_quat_w)[robot_view.read]
            self._s_obj_pos[object_view.write] = wp.to_torch(obj.data.root_pos_w)[object_view.read, :3]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        return obj_b


class batched_ee_object_pos_error(BatchedTermBase):
    """Position error ``(object - ee)`` in robot root frame, batched [m].
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            robot_view = self._layout[group_key, meta.robot_cfg.name]
            object_view = self._layout[group_key, meta.object_cfg.name]
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            self._entries.append((group_key, meta, robot_view, object_view, ee_view))
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_obj_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_ee_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, robot_view, object_view, ee_view in self._entries:
            robot: Articulation = env.scene[meta.robot_cfg.name]
            obj: RigidObject = env.scene[meta.object_cfg.name]
            ee: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            self._s_root_pos[robot_view.write] = wp.to_torch(robot.data.root_pos_w)[robot_view.read]
            self._s_root_quat[robot_view.write] = wp.to_torch(robot.data.root_quat_w)[robot_view.read]
            self._s_obj_pos[object_view.write] = wp.to_torch(obj.data.root_pos_w)[object_view.read, :3]
            self._s_ee_pos[ee_view.write] = wp.to_torch(ee.data.target_pos_w)[ee_view.read, 0, :]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        ee_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_ee_pos)
        return obj_b - ee_b


class batched_object_target_pos_error(BatchedTermBase):
    """Position error ``(target - object)`` in robot root frame, batched [m].
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            robot_view = self._layout[group_key, meta.robot_cfg.name]
            object_view = self._layout[group_key, meta.object_cfg.name]
            self._entries.append((group_key, meta, robot_view, object_view))
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_obj_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_cmd_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, robot_view, object_view in self._entries:
            robot: Articulation = env.scene[meta.robot_cfg.name]
            obj: RigidObject = env.scene[meta.object_cfg.name]
            self._s_root_pos[robot_view.write] = wp.to_torch(robot.data.root_pos_w)[robot_view.read]
            self._s_root_quat[robot_view.write] = wp.to_torch(robot.data.root_quat_w)[robot_view.read]
            self._s_obj_pos[object_view.write] = wp.to_torch(obj.data.root_pos_w)[object_view.read, :3]
            self._s_cmd_pos[robot_view.write] = env.command_manager.get_command(meta.command_name)[robot_view.write, :3]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        return self._s_cmd_pos - obj_b


# ===========================================================
# Joint-space observations
# ===========================================================


class batched_joint_pos_rel(BatchedTermBase):
    """Relative joint positions, batched across robot groups [rad or m].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, RobotGroupCfg, GroupView, list[int] | slice, int]] = []
        max_j = 0
        for group_key, meta in self._iter_groups():
            art: Articulation = env.scene[meta.asset_cfg.name]
            jids = meta.asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(art.data.joint_pos).shape[1]
            max_j = max(max_j, nj)
            group_view = self._layout[group_key, meta.asset_cfg.name]
            self._entries.append((group_key, meta, group_view, jids, nj))
        self._s_pos = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)
        self._s_default = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_pos.zero_()
        self._s_default.zero_()
        for _, meta, group_view, jids, nj in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_pos[group_view.write, :nj] = wp.to_torch(art.data.joint_pos)[group_view.read, :][:, jids]
            self._s_default[group_view.write, :nj] = wp.to_torch(art.data.default_joint_pos)[group_view.read, :][
                :, jids
            ]
        return self._s_pos - self._s_default


class batched_joint_vel(BatchedTermBase):
    """Joint velocities, batched across robot groups [rad/s or m/s].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, RobotGroupCfg, GroupView, list[int] | slice, int]] = []
        max_j = 0
        for group_key, meta in self._iter_groups():
            art: Articulation = env.scene[meta.asset_cfg.name]
            jids = meta.asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(art.data.joint_vel).shape[1]
            max_j = max(max_j, nj)
            group_view = self._layout[group_key, meta.asset_cfg.name]
            self._entries.append((group_key, meta, group_view, jids, nj))
        self._buf = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._buf.zero_()
        for _, meta, group_view, jids, nj in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._buf[group_view.write, :nj] = wp.to_torch(art.data.joint_vel)[group_view.read, :][:, jids]
        return self._buf


# ===========================================================
# Command observations
# ===========================================================


class batched_generated_commands(BatchedTermBase):
    """Generated commands, batched across robot groups.
    Returns shape ``(num_envs, max_cmd_dim)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        max_dim = 0
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            cmd = env.command_manager.get_command(meta.command_name)
            cdim = cmd.shape[-1] if cmd.dim() > 1 else 1
            max_dim = max(max_dim, cdim)
            group_view = self._layout[group_key]
            self._entries.append((group_key, meta, group_view, cdim))
        self._buf = torch.zeros(self._num_envs, max(max_dim, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._buf.zero_()
        for group_key, meta, group_view, cdim in self._entries:
            cmd = env.command_manager.get_command(meta.command_name)
            if cmd.dim() == 1:
                self._buf[group_view.write, 0] = cmd[group_view.write]
            else:
                self._buf[group_view.write, :cdim] = cmd[group_view.write, :cdim]
        return self._buf


# ===========================================================
# Cabinet observations
# ===========================================================


class batched_cabinet_joint_pos(BatchedTermBase):
    """Relative cabinet joint positions, batched across cabinet groups [rad or m].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, list[int] | slice, int]] = []
        max_j = 0
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            cab: Articulation = env.scene[meta.cabinet_asset_cfg.name]
            jids = meta.cabinet_asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(cab.data.joint_pos).shape[1]
            max_j = max(max_j, nj)
            group_view = self._layout[group_key, meta.cabinet_asset_cfg.name]
            self._entries.append((group_key, meta, group_view, jids, nj))
        self._s_pos = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)
        self._s_default = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_pos.zero_()
        self._s_default.zero_()
        for group_key, meta, group_view, jids, nj in self._entries:
            cab: Articulation = env.scene[meta.cabinet_asset_cfg.name]
            self._s_pos[group_view.write, :nj] = wp.to_torch(cab.data.joint_pos)[group_view.read, :][:, jids]
            self._s_default[group_view.write, :nj] = wp.to_torch(cab.data.default_joint_pos)[group_view.read, :][
                :, jids
            ]
        return self._s_pos - self._s_default


class batched_cabinet_joint_vel(BatchedTermBase):
    """Cabinet joint velocities, batched across cabinet groups [rad/s or m/s].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, list[int] | slice, int]] = []
        max_j = 0
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            cab: Articulation = env.scene[meta.cabinet_asset_cfg.name]
            jids = meta.cabinet_asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(cab.data.joint_vel).shape[1]
            max_j = max(max_j, nj)
            group_view = self._layout[group_key, meta.cabinet_asset_cfg.name]
            self._entries.append((group_key, meta, group_view, jids, nj))
        self._buf = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._buf.zero_()
        for group_key, meta, group_view, jids, nj in self._entries:
            cab: Articulation = env.scene[meta.cabinet_asset_cfg.name]
            self._buf[group_view.write, :nj] = wp.to_torch(cab.data.joint_vel)[group_view.read, :][:, jids]
        return self._buf


class batched_cabinet_rel_ee_drawer_distance(BatchedTermBase):
    """Drawer-handle minus EE TCP position, batched across cabinet groups [m].
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, ee_view, cabinet_view))
        self._s_ee_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_ee_pos.zero_()
        self._s_handle_pos.zero_()
        for _, meta, ee_view, cabinet_view in self._entries:
            ee: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_ee_pos[ee_view.write] = wp.to_torch(ee.data.target_pos_w)[ee_view.read, 0, :]
            self._s_handle_pos[cabinet_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[cabinet_view.read, 0, :]
        return self._s_handle_pos - self._s_ee_pos


# ===========================================================
# Robot identity
# ===========================================================


class multi_task_onehot(BatchedTermBase):
    """One-hot encoding of task group for every environment.
    Returns shape ``(num_envs, num_groups)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        groups = self._layout.group_names
        group_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        for i, name in enumerate(groups):
            group_idx[self._layout[name].env_ids] = i
        self._onehot = torch.nn.functional.one_hot(group_idx, num_classes=len(groups)).float()

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        return self._onehot
