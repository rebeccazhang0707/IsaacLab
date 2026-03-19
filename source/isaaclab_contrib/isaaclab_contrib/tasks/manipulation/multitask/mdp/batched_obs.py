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
from isaaclab.managers import ManagerTermBase

from .utils import resolve_scene_entity_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ManagerTermBaseCfg


# ===========================================================
# Task-space observations  (gather-first, compute-once)
# ===========================================================


class batched_ee_pose(ManagerTermBase):
    """EE pose in robot root frame, batched across robot groups [m, -].
    Returns shape ``(num_envs, 7)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice, int]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None:
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            self._entries.append((asset_name, layout.env_slice(gk), meta.asset_cfg.body_ids[0]))
        N, dev = env.num_envs, env.device
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_body_quat = torch.zeros(N, 4, device=dev)
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._buf = torch.zeros(N, 7, device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        for name, sl, bidx in self._entries:
            a = env.scene[name]
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[:, bidx]
            self._s_body_quat[sl] = wp.to_torch(a.data.body_quat_w)[:, bidx]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self._s_root_pos, self._s_root_quat, self._s_body_pos, self._s_body_quat
        )
        self._buf[:, :3] = pos_b
        self._buf[:, 3:] = quat_b
        return self._buf


class batched_ee_pos_error(ManagerTermBase):
    """EE position error ``(target - current)`` in root frame, batched [m].

    Reads ``command_name`` and ``asset_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice, int, str]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.command_name is None:
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            self._entries.append((asset_name, layout.env_slice(gk), meta.asset_cfg.body_ids[0], meta.command_name))
        N, dev = env.num_envs, env.device
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        for name, sl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[:, bidx]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        cur_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_body_pos)
        return self._s_cmd_pos - cur_b


class batched_object_pos_in_robot_frame(ManagerTermBase):
    """Object position in robot root frame, batched [m].

    Reads ``robot_cfg`` and ``object_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.robot_cfg is None or meta.object_cfg is None:
                continue
            self._entries.append((meta.robot_cfg.name, meta.object_cfg.name, layout.env_slice(gk)))
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_obj_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        for robot_name, obj_name, sl in self._entries:
            robot = env.scene[robot_name]
            self._s_root_pos[sl] = wp.to_torch(robot.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(robot.data.root_quat_w)
            self._s_obj_pos[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, :3]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        return obj_b


class batched_ee_object_pos_error(ManagerTermBase):
    """Position error ``(object - ee)`` in robot root frame, batched [m].

    Reads ``robot_cfg``, ``object_cfg``, ``ee_frame_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.robot_cfg is None or meta.object_cfg is None or meta.ee_frame_cfg is None:
                continue
            self._entries.append(
                (meta.robot_cfg.name, meta.object_cfg.name, meta.ee_frame_cfg.name, layout.env_slice(gk))
            )
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_obj_pos = torch.zeros(N, 3, device=dev)
        self._s_ee_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        for robot_name, obj_name, ee_name, sl in self._entries:
            robot = env.scene[robot_name]
            self._s_root_pos[sl] = wp.to_torch(robot.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(robot.data.root_quat_w)
            self._s_obj_pos[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, :3]
            self._s_ee_pos[sl] = wp.to_torch(env.scene[ee_name].data.target_pos_w)[:, 0, :]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        ee_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_ee_pos)
        return obj_b - ee_b


class batched_object_target_pos_error(ManagerTermBase):
    """Position error ``(target - object)`` in robot root frame, batched [m].

    Reads ``command_name``, ``robot_cfg``, ``object_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs, 3)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.command_name is None or meta.robot_cfg is None or meta.object_cfg is None:
                continue
            self._entries.append((meta.robot_cfg.name, meta.object_cfg.name, meta.command_name, layout.env_slice(gk)))
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_obj_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        for robot_name, obj_name, cmd_name, sl in self._entries:
            robot = env.scene[robot_name]
            self._s_root_pos[sl] = wp.to_torch(robot.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(robot.data.root_quat_w)
            self._s_obj_pos[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, :3]
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        obj_b, _ = math_utils.subtract_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_obj_pos)
        return self._s_cmd_pos - obj_b


# ===========================================================
# Joint-space observations  (gather-first, compute-once)
# ===========================================================


class batched_joint_pos_rel(ManagerTermBase):
    """Relative joint positions, batched across robot groups [rad or m].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice, list | slice, int]] = []
        max_j = 0
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None:
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            jids = meta.asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(env.scene[asset_name].data.joint_pos).shape[1]
            max_j = max(max_j, nj)
            self._entries.append((asset_name, layout.env_slice(gk), jids, nj))
        N, dev = env.num_envs, env.device
        self._s_pos = torch.zeros(N, max(max_j, 1), device=dev)
        self._s_default = torch.zeros(N, max(max_j, 1), device=dev)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        self._s_pos.zero_()
        self._s_default.zero_()
        for name, sl, jids, nj in self._entries:
            a = env.scene[name]
            self._s_pos[sl, :nj] = wp.to_torch(a.data.joint_pos)[:, jids]
            self._s_default[sl, :nj] = wp.to_torch(a.data.default_joint_pos)[:, jids]
        return self._s_pos - self._s_default


class batched_joint_vel(ManagerTermBase):
    """Joint velocities, batched across robot groups [rad/s or m/s].
    Returns shape ``(num_envs, max_joints)`` with zero-padding.

    Pure data scatter — no compute to hoist out of the loop.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice, list | slice, int]] = []
        max_j = 0
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None:
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            jids = meta.asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(env.scene[asset_name].data.joint_vel).shape[1]
            max_j = max(max_j, nj)
            self._entries.append((asset_name, layout.env_slice(gk), jids, nj))
        self._buf = torch.zeros(env.num_envs, max(max_j, 1), device=env.device)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        self._buf.zero_()
        for name, sl, jids, nj in self._entries:
            self._buf[sl, :nj] = wp.to_torch(env.scene[name].data.joint_vel)[:, jids]
        return self._buf


# ===========================================================
# Command observations  (pure scatter, no compute to batch)
# ===========================================================


class batched_generated_commands(ManagerTermBase):
    """Generated commands, batched across robot groups.

    Reads ``command_name`` from ``robot_meta``.
    Returns shape ``(num_envs, max_cmd_dim)`` with zero-padding.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[slice, str, int]] = []
        max_dim = 0
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.command_name is None:
                continue
            cmd = env.command_manager.get_command(meta.command_name)
            cdim = cmd.shape[-1] if cmd.dim() > 1 else 1
            max_dim = max(max_dim, cdim)
            self._entries.append((layout.env_slice(gk), meta.command_name, cdim))
        self._buf = torch.zeros(env.num_envs, max(max_dim, 1), device=env.device)

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        self._buf.zero_()
        for sl, cmd_name, cdim in self._entries:
            cmd = env.command_manager.get_command(cmd_name)
            if cmd.dim() == 1:
                self._buf[sl, 0] = cmd
            else:
                self._buf[sl, :cdim] = cmd
        return self._buf
