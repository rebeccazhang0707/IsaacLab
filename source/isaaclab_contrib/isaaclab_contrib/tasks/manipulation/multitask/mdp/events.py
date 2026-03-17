# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot event terms for heterogeneous scenes.

**Per-asset functions** (use with ``per_robot=True``):
    Accept ``asset_cfg: SceneEntityCfg`` (auto-injected by the
    manager from ``robot_meta``) and group-local ``env_ids``.

**Scatter-based functions** (self-dispatching):
    Iterate ``robot_meta`` entries and map env-ids internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ===========================================================
# Per-asset event functions  (use with per_robot=True)
# ===========================================================


def _reset_root_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    global_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset root pose and velocity of a single asset to its defaults."""
    asset = env.scene[asset_cfg.name]
    default_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
    default_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()
    default_pose[:, :3] += env.scene.env_origins[global_ids]
    asset.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=env_ids)


def reset_asset_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg | None = None,
    reset_joint_targets: bool = False,
) -> None:
    """Reset robot (and optionally object) to default state.

    Resets the articulation's root pose, root velocity, and joint
    state.  When ``object_cfg`` is provided (auto-injected via
    ``per_robot``), also resets the rigid object's root state.

    ``env_ids`` are group-local (0-based) when dispatched via
    ``per_robot=True``.
    """
    layout = env.scene.layout
    group_key = layout.group_for_asset(asset_cfg.name)
    global_ids = layout.local_to_global(group_key, env_ids)

    _reset_root_to_default(env, env_ids, global_ids, asset_cfg)

    art = env.scene[asset_cfg.name]
    default_jpos = wp.to_torch(art.data.default_joint_pos)[env_ids].clone()
    default_jvel = wp.to_torch(art.data.default_joint_vel)[env_ids].clone()
    art.write_joint_position_to_sim_index(position=default_jpos, env_ids=env_ids)
    art.write_joint_velocity_to_sim_index(velocity=default_jvel, env_ids=env_ids)
    if reset_joint_targets:
        art.set_joint_position_target_index(target=default_jpos, env_ids=env_ids)
        art.set_joint_velocity_target_index(target=default_jvel, env_ids=env_ids)

    if object_cfg is not None:
        _reset_root_to_default(env, env_ids, global_ids, object_cfg)


def reset_object_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset an object's root state with uniform randomization.

    Wrapper around the upstream ``reset_root_state_uniform`` logic that:

    * accepts ``object_cfg`` so ``per_robot`` auto-injection from
      ``robot_meta`` works (upstream uses ``asset_cfg``).
    * translates group-local ``env_ids`` to global indices for
      ``env_origins``.
    """
    layout = env.scene.layout
    group_key = layout.group_for_asset(object_cfg.name)
    global_ids = layout.local_to_global(group_key, env_ids)

    asset = env.scene[object_cfg.name]
    default_root_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
    default_root_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = default_root_pose[:, 0:3] + env.scene.env_origins[global_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = default_root_vel + rand_samples

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
