# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched event terms for heterogeneous multi-robot environments.

These handle the dual-indexing needed when assets don't span all envs:
- ``global_ids``: for env_origins lookup
- ``local_ids``: for asset data indexing

Usage::

    reset_joints = EventTerm(
        func=batched_reset_joints,
        mode="reset",
        params={"robot_meta": ROBOT_META, "position_range": (0.5, 1.5)},
    )
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import math as math_utils

from .utils import RobotGroupCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ManagerTermBaseCfg


def _iter_groups(
    env: ManagerBasedEnv, env_ids: torch.Tensor, robot_meta: dict[str, RobotGroupCfg], asset_key: str = "asset_cfg"
) -> Iterator[tuple[str, RobotGroupCfg, torch.Tensor, torch.Tensor, Articulation | RigidObject]]:
    """Iterate over groups, yielding (group_key, meta, global_ids, local_ids, asset).

    Yields:
        group_key: Group key string.
        meta: Group config (e.g., LiftGroupCfg).
        global_ids: Global env indices for this group (for env_origins).
        local_ids: Local asset indices (for asset data).
        asset: The asset object (Articulation or RigidObject).
    """
    layout = env.scene.layout
    for group_key, meta in robot_meta.items():
        cfg = getattr(meta, asset_key, None)
        if cfg is None:
            continue
        group_view = layout[group_key]
        _, global_ids = group_view.filter(env_ids)
        if global_ids.numel() == 0:
            continue
        # Local indices if asset exclusive to group
        asset_groups = layout.assets.get(cfg.name)
        if asset_groups and len(asset_groups) == 1 and asset_groups[0] == group_key:
            local_ids = group_view.to_local(global_ids)
        else:
            local_ids = global_ids
        asset: Articulation | RigidObject = env.scene[cfg.name]
        yield group_key, meta, global_ids, local_ids, asset


class batched_reset_to_default(ManagerTermBase):
    """Reset robots and objects to default state."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot_meta = cfg.params.get("robot_meta") or {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_joint_targets: bool = False,
        robot_meta: dict | None = None,
    ):
        for group_key, meta, global_ids, local_ids, art in _iter_groups(env, env_ids, self._robot_meta):
            # Root state
            pose = wp.to_torch(art.data.default_root_pose)[local_ids].clone()
            vel = wp.to_torch(art.data.default_root_vel)[local_ids].clone()
            pose[:, :3] += env.scene.env_origins[global_ids]
            art.write_root_pose_to_sim_index(root_pose=pose, env_ids=local_ids)
            art.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=local_ids)
            # Joint state
            jpos = wp.to_torch(art.data.default_joint_pos)[local_ids].clone()
            jvel = wp.to_torch(art.data.default_joint_vel)[local_ids].clone()
            art.write_joint_position_to_sim_index(position=jpos, env_ids=local_ids)
            art.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=local_ids)
            if reset_joint_targets:
                art.set_joint_position_target_index(target=jpos, env_ids=local_ids)
                art.set_joint_velocity_target_index(target=jvel, env_ids=local_ids)
            # Object if present
            obj_cfg = getattr(meta, "object_cfg", None)
            if obj_cfg is not None:
                obj: RigidObject = env.scene[obj_cfg.name]
                asset_groups = env.scene.layout.assets.get(obj_cfg.name)
                group_view = env.scene.layout[group_key]
                if asset_groups and len(asset_groups) == 1 and asset_groups[0] == group_key:
                    obj_local_ids = group_view.to_local(global_ids)
                else:
                    obj_local_ids = global_ids
                obj_pose = wp.to_torch(obj.data.default_root_pose)[obj_local_ids].clone()
                obj_vel = wp.to_torch(obj.data.default_root_vel)[obj_local_ids].clone()
                obj_pose[:, :3] += env.scene.env_origins[global_ids]
                obj.write_root_pose_to_sim_index(root_pose=obj_pose, env_ids=obj_local_ids)
                obj.write_root_velocity_to_sim_index(root_velocity=obj_vel, env_ids=obj_local_ids)


class batched_reset_joints(ManagerTermBase):
    """Reset robot joints by scaling default positions."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot_meta = cfg.params.get("robot_meta") or {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: tuple[float, float] = (1.0, 1.0),
        velocity_range: tuple[float, float] = (0.0, 0.0),
        robot_meta: dict | None = None,
    ):
        for _, meta, _, local_ids, art in _iter_groups(env, env_ids, self._robot_meta):
            cfg = meta.asset_cfg
            jids = cfg.joint_ids if cfg.joint_ids != slice(None) else slice(None)
            idx = local_ids[:, None] if jids != slice(None) else local_ids
            jpos = wp.to_torch(art.data.default_joint_pos)[idx, jids].clone()
            jvel = wp.to_torch(art.data.default_joint_vel)[idx, jids].clone()
            jpos *= math_utils.sample_uniform(*position_range, jpos.shape, jpos.device)
            jvel *= math_utils.sample_uniform(*velocity_range, jvel.shape, jvel.device)
            limits = wp.to_torch(art.data.soft_joint_pos_limits)[idx, jids]
            jpos = jpos.clamp_(limits[..., 0], limits[..., 1])
            vlim = wp.to_torch(art.data.soft_joint_vel_limits)[idx, jids]
            jvel = jvel.clamp_(-vlim, vlim)
            art.write_joint_position_to_sim_index(position=jpos, joint_ids=jids, env_ids=local_ids)
            art.write_joint_velocity_to_sim_index(velocity=jvel, joint_ids=jids, env_ids=local_ids)


class batched_reset_object_uniform(ManagerTermBase):
    """Reset objects with random position/velocity offsets."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot_meta = cfg.params.get("robot_meta") or {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]] | None = None,
        velocity_range: dict[str, tuple[float, float]] | None = None,
        robot_meta: dict | None = None,
    ):
        pose_range = pose_range or {}
        velocity_range = velocity_range or {}
        layout = env.scene.layout

        for group_key, meta, global_ids, _, _ in _iter_groups(env, env_ids, self._robot_meta):
            obj_cfg = getattr(meta, "object_cfg", None)
            if obj_cfg is None:
                continue
            obj: RigidObject = env.scene[obj_cfg.name]
            group_view = layout[group_key]
            asset_groups = layout.assets.get(obj_cfg.name)
            if asset_groups and len(asset_groups) == 1 and asset_groups[0] == group_key:
                local_ids = group_view.to_local(global_ids)
            else:
                local_ids = global_ids

            # Pose
            pose = wp.to_torch(obj.data.default_root_pose)[local_ids].clone()
            vel = wp.to_torch(obj.data.default_root_vel)[local_ids].clone()
            ranges = torch.tensor(
                [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=obj.device
            )
            samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(local_ids), 6), device=obj.device)
            pose[:, :3] += env.scene.env_origins[global_ids] + samples[:, :3]
            qd = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
            pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], qd)
            # Velocity
            vranges = torch.tensor(
                [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=obj.device
            )
            vsamples = math_utils.sample_uniform(vranges[:, 0], vranges[:, 1], (len(local_ids), 6), device=obj.device)
            vel += vsamples

            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=local_ids)
            obj.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=local_ids)


class batched_reset_cabinet(ManagerTermBase):
    """Reset cabinet articulation to default state."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot_meta = cfg.params.get("robot_meta") or {}

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, robot_meta: dict | None = None):
        layout = env.scene.layout
        for group_key, meta in self._robot_meta.items():
            cab_cfg = getattr(meta, "cabinet_asset_cfg", None)
            if cab_cfg is None:
                continue
            group_view = layout[group_key]
            _, global_ids = group_view.filter(env_ids)
            if global_ids.numel() == 0:
                continue
            cab: Articulation = env.scene[cab_cfg.name]
            asset_groups = layout.assets.get(cab_cfg.name)
            if asset_groups and len(asset_groups) == 1 and asset_groups[0] == group_key:
                local_ids = group_view.to_local(global_ids)
            else:
                local_ids = global_ids
            # Root state
            pose = wp.to_torch(cab.data.default_root_pose)[local_ids].clone()
            vel = wp.to_torch(cab.data.default_root_vel)[local_ids].clone()
            pose[:, :3] += env.scene.env_origins[global_ids]
            cab.write_root_pose_to_sim_index(root_pose=pose, env_ids=local_ids)
            cab.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=local_ids)
            # Joint state
            jpos = wp.to_torch(cab.data.default_joint_pos)[local_ids].clone()
            jvel = wp.to_torch(cab.data.default_joint_vel)[local_ids].clone()
            cab.write_joint_position_to_sim_index(position=jpos, env_ids=local_ids)
            cab.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=local_ids)
