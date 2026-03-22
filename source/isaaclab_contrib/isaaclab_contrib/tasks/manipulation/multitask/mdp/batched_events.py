# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched event terms for multi-robot environments.

Each class iterates ``robot_meta`` to discover robot groups and
dispatches the underlying reset logic per group with properly
filtered ``env_ids``.  This replaces the former ``per_robot=True``
dispatch mechanism with explicit, self-contained classes.

``robot_meta`` is keyed by **task-group name** (not asset name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .utils import CabinetGroupCfg, LiftGroupCfg, asset_env_ids, filter_env_ids, resolve_scene_entity_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ManagerTermBaseCfg


class batched_reset_to_default(ManagerTermBase):
    """Reset all robot groups (and their objects) to default state.

    Iterates ``robot_meta`` and resets each robot's articulation root
    pose, root velocity, and joint state.  When the metadata includes
    ``object_cfg`` (:class:`LiftGroupCfg`), the corresponding rigid
    object is also reset.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[SceneEntityCfg, SceneEntityCfg | None, str]] = []
        for group_key, meta in robot_meta.items():
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            obj_cfg = getattr(meta, "object_cfg", None)
            if obj_cfg is not None:
                resolve_scene_entity_cfg(env, obj_cfg)
            self._entries.append((meta.asset_cfg, obj_cfg, group_key))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        reset_joint_targets: bool = False,
    ) -> None:
        layout = env.scene.layout
        for asset_cfg, object_cfg, gk in self._entries:
            group_env_ids, skip = filter_env_ids(layout, gk, env_ids)
            if skip:
                continue
            a_ids = asset_env_ids(layout, gk, asset_cfg.name, group_env_ids)

            art = env.scene[asset_cfg.name]
            default_pose = wp.to_torch(art.data.default_root_pose)[a_ids].clone()
            default_vel = wp.to_torch(art.data.default_root_vel)[a_ids].clone()
            default_pose[:, :3] += env.scene.env_origins[group_env_ids]
            art.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=a_ids)
            art.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=a_ids)

            default_jpos = wp.to_torch(art.data.default_joint_pos)[a_ids].clone()
            default_jvel = wp.to_torch(art.data.default_joint_vel)[a_ids].clone()
            art.write_joint_position_to_sim_index(position=default_jpos, env_ids=a_ids)
            art.write_joint_velocity_to_sim_index(velocity=default_jvel, env_ids=a_ids)
            if reset_joint_targets:
                art.set_joint_position_target_index(target=default_jpos, env_ids=a_ids)
                art.set_joint_velocity_target_index(target=default_jvel, env_ids=a_ids)

            if object_cfg is not None:
                o_ids = asset_env_ids(layout, gk, object_cfg.name, group_env_ids)
                obj = env.scene[object_cfg.name]
                obj_pose = wp.to_torch(obj.data.default_root_pose)[o_ids].clone()
                obj_vel = wp.to_torch(obj.data.default_root_vel)[o_ids].clone()
                obj_pose[:, :3] += env.scene.env_origins[group_env_ids]
                obj.write_root_pose_to_sim_index(root_pose=obj_pose, env_ids=o_ids)
                obj.write_root_velocity_to_sim_index(root_velocity=obj_vel, env_ids=o_ids)


class batched_reset_joints_by_scale(ManagerTermBase):
    """Reset joint positions/velocities by scaling defaults, batched across robot groups.

    Iterates ``robot_meta`` and randomizes joint positions within
    ``[default * position_range[0], default * position_range[1]]``
    and velocities within ``velocity_range``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[SceneEntityCfg, list | slice, str]] = []
        for group_key, meta in robot_meta.items():
            resolve_scene_entity_cfg(env, meta.asset_cfg)
            self._entries.append((meta.asset_cfg, meta.asset_cfg.joint_ids, group_key))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        position_range: tuple[float, float] = (0.5, 1.5),
        velocity_range: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        layout = env.scene.layout
        for asset_cfg, jids, gk in self._entries:
            group_env_ids, skip = filter_env_ids(layout, gk, env_ids)
            if skip:
                continue
            a_ids = asset_env_ids(layout, gk, asset_cfg.name, group_env_ids)

            art = env.scene[asset_cfg.name]
            default_jpos = wp.to_torch(art.data.default_joint_pos)[a_ids]
            default_jvel = wp.to_torch(art.data.default_joint_vel)[a_ids]

            jpos = default_jpos.clone()
            jvel = default_jvel.clone()
            jpos[:, jids] *= torch.empty_like(jpos[:, jids]).uniform_(*position_range)
            jvel[:, jids] = torch.empty_like(jvel[:, jids]).uniform_(*velocity_range)

            limits = wp.to_torch(art.data.soft_joint_pos_limits)[a_ids]
            jpos = jpos.clamp(limits[..., 0], limits[..., 1])

            art.write_joint_position_to_sim_index(position=jpos, env_ids=a_ids)
            art.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=a_ids)
            art.set_joint_position_target_index(target=jpos, env_ids=a_ids)
            art.set_joint_velocity_target_index(target=jvel, env_ids=a_ids)


class batched_reset_object_state_uniform(ManagerTermBase):
    """Reset object root states with uniform randomization, batched across lift groups.

    Iterates :class:`LiftGroupCfg` entries that have ``object_cfg``
    and randomizes the object's root pose and velocity relative to
    its default state.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[SceneEntityCfg, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, LiftGroupCfg):
                continue
            self._entries.append((meta.object_cfg, group_key))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        velocity_range: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if pose_range is None:
            pose_range = {}
        if velocity_range is None:
            velocity_range = {}

        layout = env.scene.layout
        for obj_cfg, gk in self._entries:
            group_env_ids, skip = filter_env_ids(layout, gk, env_ids)
            if skip:
                continue
            o_ids = asset_env_ids(layout, gk, obj_cfg.name, group_env_ids)

            asset = env.scene[obj_cfg.name]
            default_root_pose = wp.to_torch(asset.data.default_root_pose)[o_ids].clone()
            default_root_vel = wp.to_torch(asset.data.default_root_vel)[o_ids].clone()

            n = len(o_ids) if o_ids is not None else asset.num_instances
            range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=asset.device)
            rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)

            positions = default_root_pose[:, :3] + env.scene.env_origins[group_env_ids] + rand_samples[:, :3]
            ori_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            orientations = math_utils.quat_mul(default_root_pose[:, 3:7], ori_delta)

            range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=asset.device)
            rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
            velocities = default_root_vel + rand_samples

            asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=o_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=o_ids)


class batched_reset_cabinet_to_default(ManagerTermBase):
    """Reset cabinet articulations to default state, batched across cabinet groups.

    Iterates :class:`CabinetGroupCfg` entries and resets the cabinet's
    root pose, root velocity, and joint state.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[SceneEntityCfg, str]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            resolve_scene_entity_cfg(env, meta.cabinet_asset_cfg)
            self._entries.append((meta.cabinet_asset_cfg, group_key))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
    ) -> None:
        layout = env.scene.layout
        for cab_cfg, gk in self._entries:
            group_env_ids, skip = filter_env_ids(layout, gk, env_ids)
            if skip:
                continue
            c_ids = asset_env_ids(layout, gk, cab_cfg.name, group_env_ids)

            art = env.scene[cab_cfg.name]
            default_pose = wp.to_torch(art.data.default_root_pose)[c_ids].clone()
            default_vel = wp.to_torch(art.data.default_root_vel)[c_ids].clone()
            default_pose[:, :3] += env.scene.env_origins[group_env_ids]
            art.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=c_ids)
            art.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=c_ids)

            default_jpos = wp.to_torch(art.data.default_joint_pos)[c_ids].clone()
            default_jvel = wp.to_torch(art.data.default_joint_vel)[c_ids].clone()
            art.write_joint_position_to_sim_index(position=default_jpos, env_ids=c_ids)
            art.write_joint_velocity_to_sim_index(velocity=default_jvel, env_ids=c_ids)
