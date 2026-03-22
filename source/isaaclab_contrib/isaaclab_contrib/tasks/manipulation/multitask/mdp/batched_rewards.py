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

``robot_meta`` is keyed by **task-group name** (not asset name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase

from .utils import CabinetGroupCfg, LiftGroupCfg, ReachGroupCfg, resolve_scene_entity_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.sensors import FrameTransformer

_Sl = slice | torch.Tensor


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
        self._entries: list[tuple[str, _Sl, _Sl, list | slice, int]] = []
        max_j = 0
        for group_key, meta in robot_meta.items():
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            jids = meta.asset_cfg.joint_ids
            nj = len(jids) if isinstance(jids, list) else wp.to_torch(env.scene[meta.asset_cfg.name].data.joint_vel).shape[1]
            max_j = max(max_j, nj)
            data_sl = layout.cross_slice(group_key, meta.asset_cfg.name)
            self._entries.append((meta.asset_cfg.name, layout.env_slice(group_key), data_sl, jids, nj))
        self._s_vel = torch.zeros(env.num_envs, max(max_j, 1), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._s_vel.zero_()
        for name, sl, dsl, jids, nj in self._entries:
            self._s_vel[sl, :nj] = wp.to_torch(env.scene[name].data.joint_vel)[dsl, :][:, jids]
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
        self._entries: list[tuple[str, _Sl, _Sl, int, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, (ReachGroupCfg, LiftGroupCfg)):
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            data_sl = layout.cross_slice(group_key, meta.asset_cfg.name)
            self._entries.append((meta.asset_cfg.name, layout.env_slice(group_key), data_sl, meta.asset_cfg.body_ids[0], meta.command_name))
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        for name, sl, dsl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)[dsl]
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)[dsl]
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[dsl, bidx]
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
        self._entries: list[tuple[str, _Sl, _Sl, int, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, (ReachGroupCfg, LiftGroupCfg)):
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            data_sl = layout.cross_slice(group_key, meta.asset_cfg.name)
            self._entries.append((meta.asset_cfg.name, layout.env_slice(group_key), data_sl, meta.asset_cfg.body_ids[0], meta.command_name))
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_body_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
        for name, sl, dsl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_pos[sl] = wp.to_torch(a.data.root_pos_w)[dsl]
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)[dsl]
            self._s_body_pos[sl] = wp.to_torch(a.data.body_pos_w)[dsl, bidx]
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
        self._entries: list[tuple[str, _Sl, _Sl, int, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, (ReachGroupCfg, LiftGroupCfg)):
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            data_sl = layout.cross_slice(group_key, meta.asset_cfg.name)
            self._entries.append((meta.asset_cfg.name, layout.env_slice(group_key), data_sl, meta.asset_cfg.body_ids[0], meta.command_name))
        N, dev = env.num_envs, env.device
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_body_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_cmd_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        for name, sl, dsl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)[dsl]
            self._s_body_quat[sl] = wp.to_torch(a.data.body_quat_w)[dsl, bidx]
            self._s_cmd_quat[sl] = env.command_manager.get_command(cmd_name)[:, 3:7]
        des_quat_w = math_utils.quat_mul(self._s_root_quat, self._s_cmd_quat)
        return math_utils.quat_error_magnitude(self._s_body_quat, des_quat_w)


class batched_orientation_command_error_tanh(ManagerTermBase):
    """Orientation command tracking with tanh kernel, batched across robot groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, _Sl, _Sl, int, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, (ReachGroupCfg, LiftGroupCfg)):
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            data_sl = layout.cross_slice(group_key, meta.asset_cfg.name)
            self._entries.append((meta.asset_cfg.name, layout.env_slice(group_key), data_sl, meta.asset_cfg.body_ids[0], meta.command_name))
        N, dev = env.num_envs, env.device
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_body_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_cmd_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
        for name, sl, dsl, bidx, cmd_name in self._entries:
            a = env.scene[name]
            self._s_root_quat[sl] = wp.to_torch(a.data.root_quat_w)[dsl]
            self._s_body_quat[sl] = wp.to_torch(a.data.body_quat_w)[dsl, bidx]
            self._s_cmd_quat[sl] = env.command_manager.get_command(cmd_name)[:, 3:7]
        des_quat_w = math_utils.quat_mul(self._s_root_quat, self._s_cmd_quat)
        err = math_utils.quat_error_magnitude(self._s_body_quat, des_quat_w)
        return 1.0 - torch.tanh(err / std)


# ===========================================================
# Object manipulation rewards  (gather-first, compute-once)
# ===========================================================


class batched_object_ee_distance(ManagerTermBase):
    """Object-to-EE distance with tanh kernel, batched across lift groups.

    Reads ``object_cfg`` and ``ee_frame_cfg`` from :class:`LiftGroupCfg` entries.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, LiftGroupCfg):
                continue
            o_dsl = layout.cross_slice(group_key, meta.object_cfg.name)
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            self._entries.append((meta.object_cfg.name, meta.ee_frame_cfg.name, layout.env_slice(group_key), o_dsl, ee_dsl))
        N, dev = env.num_envs, env.device
        self._s_obj_pos = torch.zeros(N, 3, device=dev)
        self._s_ee_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
        for obj_name, ee_name, sl, o_dsl, ee_dsl in self._entries:
            self._s_obj_pos[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[o_dsl]
            self._s_ee_pos[sl] = wp.to_torch(env.scene[ee_name].data.target_pos_w)[ee_dsl, 0, :]
        dist = torch.linalg.norm(self._s_obj_pos - self._s_ee_pos, dim=1)
        return 1.0 - torch.tanh(dist / std)


class batched_object_is_lifted(ManagerTermBase):
    """Object lifted above threshold, batched across lift groups.

    Reads ``object_cfg`` from :class:`LiftGroupCfg` entries.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, LiftGroupCfg):
                continue
            o_dsl = layout.cross_slice(group_key, meta.object_cfg.name)
            self._entries.append((meta.object_cfg.name, layout.env_slice(group_key), o_dsl))
        self._s_height = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, minimal_height: float = 0.04) -> torch.Tensor:
        for obj_name, sl, o_dsl in self._entries:
            self._s_height[sl] = wp.to_torch(env.scene[obj_name].data.root_pos_w)[o_dsl, 2]
        return torch.where(self._s_height > minimal_height, 1.0, 0.0)


class batched_object_goal_distance(ManagerTermBase):
    """Object-to-goal distance with tanh kernel, batched across lift groups.

    Reads ``command_name``, ``robot_cfg``, ``object_cfg`` from
    :class:`LiftGroupCfg` entries.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, LiftGroupCfg):
                continue
            r_dsl = layout.cross_slice(group_key, meta.robot_cfg.name)
            o_dsl = layout.cross_slice(group_key, meta.object_cfg.name)
            self._entries.append((meta.robot_cfg.name, meta.object_cfg.name, meta.command_name, layout.env_slice(group_key), r_dsl, o_dsl))
        N, dev = env.num_envs, env.device
        self._s_root_pos = torch.zeros(N, 3, device=dev)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_obj_pos = torch.zeros(N, 3, device=dev)
        self._s_cmd_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.3, minimal_height: float = 0.04) -> torch.Tensor:
        for robot_name, obj_name, cmd_name, sl, r_dsl, o_dsl in self._entries:
            robot = env.scene[robot_name]
            obj = env.scene[obj_name]
            self._s_root_pos[sl] = wp.to_torch(robot.data.root_pos_w)[r_dsl]
            self._s_root_quat[sl] = wp.to_torch(robot.data.root_quat_w)[r_dsl]
            self._s_obj_pos[sl] = wp.to_torch(obj.data.root_pos_w)[o_dsl]
            self._s_cmd_pos[sl] = env.command_manager.get_command(cmd_name)[:, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        dist = torch.linalg.norm(des_pos_w - self._s_obj_pos, dim=1)
        return (self._s_obj_pos[:, 2] > minimal_height) * (1.0 - torch.tanh(dist / std))


# ===========================================================
# Cabinet rewards  (gather-first, compute-once)
# ===========================================================


def _cabinet_ee_handle_data(
    env: ManagerBasedRLEnv,
    entries: list[tuple[str, str, _Sl, _Sl, _Sl]],
    s_ee_pos: torch.Tensor,
    s_handle_pos: torch.Tensor,
) -> None:
    """Scatter EE and handle positions for cabinet groups."""
    for ee_name, cab_name, sl, ee_dsl, cab_dsl in entries:
        ee_frame: FrameTransformer = env.scene[ee_name]
        cab_frame: FrameTransformer = env.scene[cab_name]
        s_ee_pos[sl] = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 0, :]
        s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[cab_dsl, 0, :]


class batched_cabinet_approach_ee_handle(ManagerTermBase):
    """Reward for reaching the cabinet handle, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((meta.ee_frame_cfg.name, meta.cabinet_frame_cfg.name, layout.env_slice(group_key), ee_dsl, cab_dsl))
        N, dev = env.num_envs, env.device
        self._s_ee_pos = torch.zeros(N, 3, device=dev)
        self._s_handle_pos = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.2) -> torch.Tensor:
        _cabinet_ee_handle_data(env, self._entries, self._s_ee_pos, self._s_handle_pos)
        distance = torch.linalg.norm(self._s_handle_pos - self._s_ee_pos, dim=-1, ord=2)
        reward = torch.pow(1.0 / (1.0 + distance**2), 2)
        return torch.where(distance <= threshold, 2 * reward, reward)


class batched_cabinet_align_ee_handle(ManagerTermBase):
    """Reward for aligning with the cabinet handle, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((meta.ee_frame_cfg.name, meta.cabinet_frame_cfg.name, layout.env_slice(group_key), ee_dsl, cab_dsl))
        N, dev = env.num_envs, env.device
        self._s_ee_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._s_handle_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, -1).clone()
        self._buf = torch.zeros(N, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._buf.zero_()
        for ee_name, cab_name, sl, ee_dsl, cab_dsl in self._entries:
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[cab_name]
            self._s_ee_quat[sl] = wp.to_torch(ee_frame.data.target_quat_w)[ee_dsl, 0, :]
            self._s_handle_quat[sl] = wp.to_torch(cab_frame.data.target_quat_w)[cab_dsl, 0, :]
        ee_rot = math_utils.matrix_from_quat(self._s_ee_quat)
        handle_rot = math_utils.matrix_from_quat(self._s_handle_quat)
        handle_x, handle_y = handle_rot[..., 0], handle_rot[..., 1]
        ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
        align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


class batched_cabinet_align_grasp_around_handle(ManagerTermBase):
    """Bonus when fingers straddle the drawer handle, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((meta.ee_frame_cfg.name, meta.cabinet_frame_cfg.name, layout.env_slice(group_key), ee_dsl, cab_dsl))
        N, dev = env.num_envs, env.device
        self._s_handle_pos = torch.zeros(N, 3, device=dev)
        self._s_left = torch.zeros(N, 3, device=dev)
        self._s_right = torch.zeros(N, 3, device=dev)
        self._buf = torch.zeros(N, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._buf.zero_()
        for ee_name, cab_name, sl, ee_dsl, cab_dsl in self._entries:
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[cab_name]
            self._s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[cab_dsl, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 1:, :]
            self._s_left[sl] = fingertips[:, 0, :]
            self._s_right[sl] = fingertips[:, 1, :]
        self._buf = (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])
        return self._buf.float()


class batched_cabinet_approach_gripper_handle(ManagerTermBase):
    """Reward for finger placement around the handle, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((meta.ee_frame_cfg.name, meta.cabinet_frame_cfg.name, layout.env_slice(group_key), ee_dsl, cab_dsl))
        N, dev = env.num_envs, env.device
        self._s_handle_pos = torch.zeros(N, 3, device=dev)
        self._s_left = torch.zeros(N, 3, device=dev)
        self._s_right = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
        for ee_name, cab_name, sl, ee_dsl, cab_dsl in self._entries:
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[cab_name]
            self._s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[cab_dsl, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 1:, :]
            self._s_left[sl] = fingertips[:, 0, :]
            self._s_right[sl] = fingertips[:, 1, :]
        left_dist = torch.abs(self._s_left[:, 2] - self._s_handle_pos[:, 2])
        right_dist = torch.abs(self._s_right[:, 2] - self._s_handle_pos[:, 2])
        is_graspable = (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])
        return is_graspable * ((offset - left_dist) + (offset - right_dist))


class batched_cabinet_grasp_handle(ManagerTermBase):
    """Reward for closing fingers near the handle, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, str, str, list | slice, _Sl, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            r_dsl = layout.cross_slice(group_key, meta.asset_cfg.name)
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((
                meta.asset_cfg.name,
                meta.ee_frame_cfg.name,
                meta.cabinet_frame_cfg.name,
                meta.asset_cfg.joint_ids,
                layout.env_slice(group_key),
                r_dsl,
                ee_dsl,
                cab_dsl,
            ))
        N, dev = env.num_envs, env.device
        self._s_ee_pos = torch.zeros(N, 3, device=dev)
        self._s_handle_pos = torch.zeros(N, 3, device=dev)
        self._s_gripper = torch.zeros(N, device=dev)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.03, open_joint_pos: float = 0.04) -> torch.Tensor:
        self._s_gripper.zero_()
        for robot_name, ee_name, cab_name, jids, sl, r_dsl, ee_dsl, cab_dsl in self._entries:
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[cab_name]
            robot = env.scene[robot_name]
            self._s_ee_pos[sl] = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 0, :]
            self._s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[cab_dsl, 0, :]
            self._s_gripper[sl] = torch.sum(open_joint_pos - wp.to_torch(robot.data.joint_pos)[r_dsl, :][:, jids], dim=-1)
        distance = torch.linalg.norm(self._s_handle_pos - self._s_ee_pos, dim=-1, ord=2)
        is_close = distance <= threshold
        return is_close * self._s_gripper


class batched_cabinet_open_drawer_bonus(ManagerTermBase):
    """Drawer opening bonus, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, list | slice, str, str, _Sl, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            resolve_scene_entity_cfg(env, meta.cabinet_asset_cfg)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_asset_cfg.name)
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            frame_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((
                meta.cabinet_asset_cfg.name,
                meta.cabinet_asset_cfg.joint_ids,
                meta.ee_frame_cfg.name,
                meta.cabinet_frame_cfg.name,
                layout.env_slice(group_key),
                cab_dsl,
                ee_dsl,
                frame_dsl,
            ))
        N, dev = env.num_envs, env.device
        self._s_drawer_pos = torch.zeros(N, device=dev)
        self._s_handle_pos = torch.zeros(N, 3, device=dev)
        self._s_left = torch.zeros(N, 3, device=dev)
        self._s_right = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._s_drawer_pos.zero_()
        for cab_name, jids, ee_name, frame_name, sl, cab_dsl, ee_dsl, frame_dsl in self._entries:
            self._s_drawer_pos[sl] = wp.to_torch(env.scene[cab_name].data.joint_pos)[cab_dsl, jids[0]]
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[frame_name]
            self._s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[frame_dsl, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 1:, :]
            self._s_left[sl] = fingertips[:, 0, :]
            self._s_right[sl] = fingertips[:, 1, :]
        is_graspable = ((self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])).float()
        return (is_graspable + 1.0) * self._s_drawer_pos


class batched_cabinet_multi_stage_open_drawer(ManagerTermBase):
    """Multi-stage drawer opening bonus, batched across cabinet groups.
    Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, list | slice, str, str, _Sl, _Sl, _Sl, _Sl]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            resolve_scene_entity_cfg(env, meta.cabinet_asset_cfg)
            cab_dsl = layout.cross_slice(group_key, meta.cabinet_asset_cfg.name)
            ee_dsl = layout.cross_slice(group_key, meta.ee_frame_cfg.name)
            frame_dsl = layout.cross_slice(group_key, meta.cabinet_frame_cfg.name)
            self._entries.append((
                meta.cabinet_asset_cfg.name,
                meta.cabinet_asset_cfg.joint_ids,
                meta.ee_frame_cfg.name,
                meta.cabinet_frame_cfg.name,
                layout.env_slice(group_key),
                cab_dsl,
                ee_dsl,
                frame_dsl,
            ))
        N, dev = env.num_envs, env.device
        self._s_drawer_pos = torch.zeros(N, device=dev)
        self._s_handle_pos = torch.zeros(N, 3, device=dev)
        self._s_left = torch.zeros(N, 3, device=dev)
        self._s_right = torch.zeros(N, 3, device=dev)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._s_drawer_pos.zero_()
        for cab_name, jids, ee_name, frame_name, sl, cab_dsl, ee_dsl, frame_dsl in self._entries:
            self._s_drawer_pos[sl] = wp.to_torch(env.scene[cab_name].data.joint_pos)[cab_dsl, jids[0]]
            ee_frame: FrameTransformer = env.scene[ee_name]
            cab_frame: FrameTransformer = env.scene[frame_name]
            self._s_handle_pos[sl] = wp.to_torch(cab_frame.data.target_pos_w)[frame_dsl, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_dsl, 1:, :]
            self._s_left[sl] = fingertips[:, 0, :]
            self._s_right[sl] = fingertips[:, 1, :]
        is_graspable = ((self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])).float()
        open_easy = (self._s_drawer_pos > 0.01) * 0.5
        open_medium = (self._s_drawer_pos > 0.2) * is_graspable
        open_hard = (self._s_drawer_pos > 0.3) * is_graspable
        return open_easy + open_medium + open_hard
