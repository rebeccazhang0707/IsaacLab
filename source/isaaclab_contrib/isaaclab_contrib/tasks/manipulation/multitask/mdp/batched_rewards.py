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

from .utils import BatchedTermBase, CabinetGroupCfg, LiftGroupCfg, ReachGroupCfg, RobotGroupCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.scene import GroupView
    from isaaclab.sensors import FrameTransformer


# ===========================================================
# Joint-space rewards
# ===========================================================


class batched_joint_vel_l2(BatchedTermBase):
    """Joint velocity L2 penalty, batched across robot groups."""

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
        self._s_vel = torch.zeros(self._num_envs, max(max_j, 1), device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_vel.zero_()
        for _, meta, group_view, jids, nj in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_vel[group_view.write, :nj] = wp.to_torch(art.data.joint_vel)[group_view.read, :][:, jids]
        return torch.sum(self._s_vel.square(), dim=-1)


# ===========================================================
# Position tracking rewards
# ===========================================================


class batched_position_command_error(BatchedTermBase):
    """Position command tracking L2 error, batched across robot groups [m]."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_body_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_cmd_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_root_pos[group_view.write] = wp.to_torch(art.data.root_pos_w)[group_view.read]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
            self._s_body_pos[group_view.write] = wp.to_torch(art.data.body_pos_w)[group_view.read, body_idx]
            self._s_cmd_pos[group_view.write] = env.command_manager.get_command(meta.command_name)[group_view.write, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        return torch.linalg.norm(self._s_body_pos - des_pos_w, dim=1)


class batched_position_command_error_tanh(BatchedTermBase):
    """Position command tracking with tanh kernel, batched across robot groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_root_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_body_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_cmd_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1, robot_meta: dict | None = None) -> torch.Tensor:
        for group_key, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_root_pos[group_view.write] = wp.to_torch(art.data.root_pos_w)[group_view.read]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
            self._s_body_pos[group_view.write] = wp.to_torch(art.data.body_pos_w)[group_view.read, body_idx]
            self._s_cmd_pos[group_view.write] = env.command_manager.get_command(meta.command_name)[group_view.write, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        dist = torch.linalg.norm(self._s_body_pos - des_pos_w, dim=1)
        return 1.0 - torch.tanh(dist / std)


# ===========================================================
# Orientation tracking rewards
# ===========================================================


class batched_orientation_command_error(BatchedTermBase):
    """Orientation command tracking error (shortest path), batched across robot groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_body_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_cmd_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for group_key, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
            self._s_body_quat[group_view.write] = wp.to_torch(art.data.body_quat_w)[group_view.read, body_idx]
            self._s_cmd_quat[group_view.write] = env.command_manager.get_command(meta.command_name)[
                group_view.write, 3:7
            ]
        des_quat_w = math_utils.quat_mul(self._s_root_quat, self._s_cmd_quat)
        return math_utils.quat_error_magnitude(self._s_body_quat, des_quat_w)


class batched_orientation_command_error_tanh(BatchedTermBase):
    """Orientation command tracking with tanh kernel, batched across robot groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, ReachGroupCfg | LiftGroupCfg, GroupView, int]] = []
        for group_key, meta in self._iter_groups(ReachGroupCfg, LiftGroupCfg):
            group_view = self._layout[group_key, meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            self._entries.append((group_key, meta, group_view, body_idx))
        self._s_root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_body_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_cmd_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, group_view, body_idx in self._entries:
            art: Articulation = env.scene[meta.asset_cfg.name]
            self._s_root_quat[group_view.write] = wp.to_torch(art.data.root_quat_w)[group_view.read]
            self._s_body_quat[group_view.write] = wp.to_torch(art.data.body_quat_w)[group_view.read, body_idx]
            self._s_cmd_quat[group_view.write] = env.command_manager.get_command(meta.command_name)[
                group_view.write, 3:7
            ]
        des_quat_w = math_utils.quat_mul(self._s_root_quat, self._s_cmd_quat)
        err = math_utils.quat_error_magnitude(self._s_body_quat, des_quat_w)
        return 1.0 - torch.tanh(err / std)


# ===========================================================
# Object manipulation rewards
# ===========================================================


class batched_object_ee_distance(BatchedTermBase):
    """Object-to-EE distance with tanh kernel, batched across lift groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            object_view = self._layout[group_key, meta.object_cfg.name]
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            self._entries.append((group_key, meta, object_view, ee_view))
        self._s_obj_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_ee_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.1, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, object_view, ee_view in self._entries:
            self._s_obj_pos[object_view.write] = wp.to_torch(env.scene[meta.object_cfg.name].data.root_pos_w)[
                object_view.read
            ]
            self._s_ee_pos[ee_view.write] = wp.to_torch(env.scene[meta.ee_frame_cfg.name].data.target_pos_w)[
                ee_view.read, 0, :
            ]
        dist = torch.linalg.norm(self._s_obj_pos - self._s_ee_pos, dim=1)
        return 1.0 - torch.tanh(dist / std)


class batched_object_is_lifted(BatchedTermBase):
    """Object lifted above threshold, batched across lift groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            object_view = self._layout[group_key, meta.object_cfg.name]
            self._entries.append((group_key, meta, object_view))
        self._s_height = torch.zeros(self._num_envs, device=self._device)

    def __call__(
        self, env: ManagerBasedRLEnv, minimal_height: float = 0.04, robot_meta: dict | None = None
    ) -> torch.Tensor:
        for _, meta, object_view in self._entries:
            self._s_height[object_view.write] = wp.to_torch(env.scene[meta.object_cfg.name].data.root_pos_w)[
                object_view.read, 2
            ]
        return torch.where(self._s_height > minimal_height, 1.0, 0.0)


class batched_object_goal_distance(BatchedTermBase):
    """Object-to-goal distance with tanh kernel, batched across lift groups."""

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

    def __call__(
        self, env: ManagerBasedRLEnv, std: float = 0.3, minimal_height: float = 0.04, robot_meta: dict | None = None
    ) -> torch.Tensor:
        for _, meta, robot_view, object_view in self._entries:
            robot: Articulation = env.scene[meta.robot_cfg.name]
            obj: RigidObject = env.scene[meta.object_cfg.name]
            self._s_root_pos[robot_view.write] = wp.to_torch(robot.data.root_pos_w)[robot_view.read]
            self._s_root_quat[robot_view.write] = wp.to_torch(robot.data.root_quat_w)[robot_view.read]
            self._s_obj_pos[object_view.write] = wp.to_torch(obj.data.root_pos_w)[object_view.read]
            self._s_cmd_pos[robot_view.write] = env.command_manager.get_command(meta.command_name)[robot_view.write, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(self._s_root_pos, self._s_root_quat, self._s_cmd_pos)
        dist = torch.linalg.norm(des_pos_w - self._s_obj_pos, dim=1)
        return (self._s_obj_pos[:, 2] > minimal_height) * (1.0 - torch.tanh(dist / std))


# ===========================================================
# Cabinet rewards
# ===========================================================


class batched_cabinet_approach_ee_handle(BatchedTermBase):
    """Reward for reaching the cabinet handle, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, ee_view, cabinet_view))
        self._s_ee_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.2, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, ee_view, cabinet_view in self._entries:
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_ee_pos[ee_view.write] = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 0, :]
            self._s_handle_pos[cabinet_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[cabinet_view.read, 0, :]
        distance = torch.linalg.norm(self._s_handle_pos - self._s_ee_pos, dim=-1, ord=2)
        reward = torch.pow(1.0 / (1.0 + distance**2), 2)
        return torch.where(distance <= threshold, 2 * reward, reward)


class batched_cabinet_align_ee_handle(BatchedTermBase):
    """Reward for aligning with the cabinet handle, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, ee_view, cabinet_view))
        self._s_ee_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()
        self._s_handle_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).expand(self._num_envs, -1).clone()

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, ee_view, cabinet_view in self._entries:
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_ee_quat[ee_view.write] = wp.to_torch(ee_frame.data.target_quat_w)[ee_view.read, 0, :]
            self._s_handle_quat[cabinet_view.write] = wp.to_torch(cab_frame.data.target_quat_w)[cabinet_view.read, 0, :]
        ee_rot = math_utils.matrix_from_quat(self._s_ee_quat)
        handle_rot = math_utils.matrix_from_quat(self._s_handle_quat)
        handle_x, handle_y = handle_rot[..., 0], handle_rot[..., 1]
        ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
        align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


class batched_cabinet_align_grasp_around_handle(BatchedTermBase):
    """Bonus when fingers straddle the drawer handle, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, ee_view, cabinet_view))
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_left = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_right = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        for group_key, meta, ee_view, cabinet_view in self._entries:
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_handle_pos[cabinet_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[cabinet_view.read, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 1:, :]
            self._s_left[ee_view.write] = fingertips[:, 0, :]
            self._s_right[ee_view.write] = fingertips[:, 1, :]
        is_grasp = (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])
        return is_grasp.float()


class batched_cabinet_approach_gripper_handle(BatchedTermBase):
    """Reward for finger placement around the handle, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, ee_view, cabinet_view))
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_left = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_right = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, offset: float = 0.04, robot_meta: dict | None = None) -> torch.Tensor:
        for _, meta, ee_view, cabinet_view in self._entries:
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_handle_pos[cabinet_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[cabinet_view.read, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 1:, :]
            self._s_left[ee_view.write] = fingertips[:, 0, :]
            self._s_right[ee_view.write] = fingertips[:, 1, :]
        left_dist = torch.abs(self._s_left[:, 2] - self._s_handle_pos[:, 2])
        right_dist = torch.abs(self._s_right[:, 2] - self._s_handle_pos[:, 2])
        is_graspable = (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (
            self._s_left[:, 2] > self._s_handle_pos[:, 2]
        )
        return is_graspable * ((offset - left_dist) + (offset - right_dist))


class batched_cabinet_grasp_handle(BatchedTermBase):
    """Reward for closing fingers near the handle, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            robot_view = self._layout[group_key, meta.asset_cfg.name]
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            cabinet_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, robot_view, ee_view, cabinet_view))
        self._s_ee_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_gripper = torch.zeros(self._num_envs, device=self._device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float = 0.03,
        open_joint_pos: float = 0.04,
        robot_meta: dict | None = None,
    ) -> torch.Tensor:
        self._s_gripper.zero_()
        for _, meta, robot_view, ee_view, cabinet_view in self._entries:
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            robot: Articulation = env.scene[meta.asset_cfg.name]
            jids = meta.asset_cfg.joint_ids
            self._s_ee_pos[ee_view.write] = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 0, :]
            self._s_handle_pos[cabinet_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[cabinet_view.read, 0, :]
            self._s_gripper[robot_view.write] = torch.sum(
                open_joint_pos - wp.to_torch(robot.data.joint_pos)[robot_view.read, :][:, jids], dim=-1
            )
        distance = torch.linalg.norm(self._s_handle_pos - self._s_ee_pos, dim=-1, ord=2)
        return (distance <= threshold) * self._s_gripper


class batched_cabinet_open_drawer_bonus(BatchedTermBase):
    """Drawer opening bonus, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            cabinet_view = self._layout[group_key, meta.cabinet_asset_cfg.name]
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            frame_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, cabinet_view, ee_view, frame_view))
        self._s_drawer_pos = torch.zeros(self._num_envs, device=self._device)
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_left = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_right = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_drawer_pos.zero_()
        for _, meta, cabinet_view, ee_view, frame_view in self._entries:
            joint_pos = wp.to_torch(env.scene[meta.cabinet_asset_cfg.name].data.joint_pos)
            self._s_drawer_pos[cabinet_view.write] = joint_pos[
                cabinet_view.read, meta.cabinet_asset_cfg.joint_ids
            ].squeeze(-1)
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_handle_pos[frame_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[frame_view.read, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 1:, :]
            self._s_left[ee_view.write] = fingertips[:, 0, :]
            self._s_right[ee_view.write] = fingertips[:, 1, :]
        is_graspable = (
            (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])
        ).float()
        return (is_graspable + 1.0) * self._s_drawer_pos


class batched_cabinet_multi_stage_open_drawer(BatchedTermBase):
    """Multi-stage drawer opening bonus, batched across cabinet groups."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView, GroupView, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            cabinet_view = self._layout[group_key, meta.cabinet_asset_cfg.name]
            ee_view = self._layout[group_key, meta.ee_frame_cfg.name]
            frame_view = self._layout[group_key, meta.cabinet_frame_cfg.name]
            self._entries.append((group_key, meta, cabinet_view, ee_view, frame_view))
        self._s_drawer_pos = torch.zeros(self._num_envs, device=self._device)
        self._s_handle_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_left = torch.zeros(self._num_envs, 3, device=self._device)
        self._s_right = torch.zeros(self._num_envs, 3, device=self._device)

    def __call__(self, env: ManagerBasedRLEnv, robot_meta: dict | None = None) -> torch.Tensor:
        self._s_drawer_pos.zero_()
        for _, meta, cabinet_view, ee_view, frame_view in self._entries:
            joint_pos = wp.to_torch(env.scene[meta.cabinet_asset_cfg.name].data.joint_pos)
            self._s_drawer_pos[cabinet_view.write] = joint_pos[
                cabinet_view.read, meta.cabinet_asset_cfg.joint_ids
            ].squeeze(-1)
            ee_frame: FrameTransformer = env.scene[meta.ee_frame_cfg.name]
            cab_frame: FrameTransformer = env.scene[meta.cabinet_frame_cfg.name]
            self._s_handle_pos[frame_view.write] = wp.to_torch(cab_frame.data.target_pos_w)[frame_view.read, 0, :]
            fingertips = wp.to_torch(ee_frame.data.target_pos_w)[ee_view.read, 1:, :]
            self._s_left[ee_view.write] = fingertips[:, 0, :]
            self._s_right[ee_view.write] = fingertips[:, 1, :]
        is_graspable = (
            (self._s_right[:, 2] < self._s_handle_pos[:, 2]) & (self._s_left[:, 2] > self._s_handle_pos[:, 2])
        ).float()
        open_easy = (self._s_drawer_pos > 0.01) * 0.5
        open_medium = (self._s_drawer_pos > 0.2) * is_graspable
        open_hard = (self._s_drawer_pos > 0.3) * is_graspable
        return open_easy + open_medium + open_hard
