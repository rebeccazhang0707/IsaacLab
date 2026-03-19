# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched reward terms for multi-robot environments.

Each class uses a **gather-first, compute-once** pattern:

1. A short for-loop scatters per-asset data into pre-allocated
   staging buffers (cheap contiguous memcpy, no math).
2. A single batched call to the expensive math kernel
   (e.g. ``combine_frame_transforms``, ``quat_mul``) processes
   all envs at once, reducing CUDA kernel launches from
   ``N_groups × K`` to ``K``.

All terms return shape ``(num_envs,)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase

from .utils import resolve_scene_entity_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


# ===========================================================
# Joint-space rewards  (gather-first, compute-once)
# ===========================================================


class batched_joint_vel_l2(ManagerTermBase):
    """Joint velocity L2 penalty, batched across robot groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
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
        self._s_vel = torch.zeros(env.num_envs, max(max_j, 1), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._s_vel.zero_()
        for name, sl, jids, nj in self._entries:
            self._s_vel[sl, :nj] = wp.to_torch(env.scene[name].data.joint_vel)[:, jids]
        return torch.sum(self._s_vel.square(), dim=-1)


# ===========================================================
# Position tracking rewards  (gather-first, compute-once)
# ===========================================================


class batched_position_command_error(ManagerTermBase):
    """Position command tracking L2 error, batched across robot groups [m].
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
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
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        for name, sl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[:, bidx]
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        return torch.linalg.norm(self._s_body_pos - des_pos_w, dim=1)


class batched_position_command_error_tanh(ManagerTermBase):
    """Position command tracking with tanh kernel, batched across robot groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
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
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
        for name, sl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[:, bidx]
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        dist = torch.linalg.norm(self._s_body_pos - des_pos_w, dim=1)
        return 1.0 - torch.tanh(dist / std)


# ===========================================================
# Orientation tracking rewards  (gather-first, compute-once)
# ===========================================================


class batched_orientation_command_error(ManagerTermBase):
    """Orientation command tracking error (shortest path), batched across robot groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
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
        self._s_root_quat = torch.zeros(N, 4, device=dev)
        self._s_body_quat = torch.zeros(N, 4, device=dev)
        self._s_cmd_quat = torch.zeros(N, 4, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        for name, sl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)
            self._s_body_quat[sl] = wp.to_torch(a.data.body_quat_w)[:, bidx]
            self._s_cmd_quat[sl] = env.command_manager.get_command(cmd_name)[:, 3:7]
        des_quat_w = math_utils.quat_mul(self._s_root_quat, self._s_cmd_quat)
        return math_utils.quat_error_magnitude(self._s_body_quat, des_quat_w)


# ===========================================================
# Object manipulation rewards  (gather-first, compute-once)
# ===========================================================


class batched_object_ee_distance(ManagerTermBase):
    """Object-to-EE distance with tanh kernel, batched across robot groups.

    Reads ``object_cfg`` and ``ee_frame_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.object_cfg is None or meta.ee_frame_cfg is None:
                continue
            self._entries.append((meta.object_cfg.name, meta.ee_frame_cfg.name, layout.env_slice(gk)))
        N, dev = env.num_envs, env.device
        self._s_obj_pos = torch.zeros(N, 3, device=dev)
        self._s_ee_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
        for obj_name, ee_name, sl in self._entries:
            self._s_obj_pos[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)
            self._s_ee_pos[sl] = wp.to_torch(env.scene[ee_name].data.target_pos_w)[:, 0, :]
        dist = torch.linalg.norm(self._s_obj_pos - self._s_ee_pos, dim=1)
        return 1.0 - torch.tanh(dist / std)


class batched_object_is_lifted(ManagerTermBase):
    """Object lifted above threshold, batched across robot groups.

    Reads ``object_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None or meta.object_cfg is None:
                continue
            self._entries.append((meta.object_cfg.name, layout.env_slice(gk)))
        self._s_height = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, minimal_height: float = 0.04) -> torch.Tensor:
        for obj_name, sl in self._entries:
            self._s_height[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, 2]
        return torch.where(self._s_height > minimal_height, 1.0, 0.0)


class batched_object_goal_distance(ManagerTermBase):
    """Object-to-goal distance with tanh kernel, batched across robot groups.

    Reads ``command_name``, ``robot_cfg``, ``object_cfg`` from ``robot_meta``.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
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

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.3, minimal_height: float = 0.04) -> torch.Tensor:
        for robot_name, obj_name, cmd_name, sl in self._entries:
            robot = env.scene[robot_name]
            obj = env.scene[obj_name]
            self._s_root_pos[sl] = wp.to_torch(robot.data.root_pos_w)
            self._s_root_quat[sl] = wp.to_torch(robot.data.root_quat_w)
            self._s_obj_pos[sl] = wp.to_torch(obj.data.root_pos_w)
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        dist = torch.linalg.norm(des_pos_w - self._s_obj_pos, dim=1)
        return (self._s_obj_pos[:, 2] > minimal_height) * (1.0 - torch.tanh(dist / std))
