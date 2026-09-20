# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset randomization for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation, CableObject, RigidObject
from isaaclab.envs.mdp.events import reset_joints_by_offset
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

from .. import shoelace_constants as physics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def install_settled_default_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Install the offline gravity-settled cable poses as defaults at startup.

    Args:
        env: Environment containing both shoelace cables.
        env_ids: Unused; startup installs defaults for every environment.
    """
    names = ("shoelace_left", "shoelace_right")
    with np.load(physics.ASSET_DIR / "settled_tail_clear_segment_poses.npz", allow_pickle=False) as state:
        if tuple(state.files) != names:
            raise ValueError(f"Expected settled shoelace arrays {names}, got {tuple(state.files)}")
        for name in names:
            cable: CableObject = env.scene[name]
            default_pose = cable.data.default_segment_pose_w.torch
            local_pose = np.asarray(state[name], dtype=np.float32)
            if local_pose.shape != default_pose.shape[1:] or not np.isfinite(local_pose).all():
                raise ValueError(f"Invalid settled {name} poses: expected finite {tuple(default_pose.shape[1:])}")
            if not np.allclose(np.linalg.norm(local_pose[:, 3:], axis=1), 1.0, atol=2.0e-5):
                raise ValueError(f"Settled {name} poses contain non-unit quaternions")
            default_pose.copy_(default_pose.new_tensor(local_pose))
            default_pose[..., :3] += env.scene.env_origins.unsqueeze(1)
            velocity = cable.data.default_segment_velocity_w.torch
            velocity.zero_()
            cable.write_segment_pose_to_sim_index(segment_pose=default_pose)
            cable.write_segment_velocity_to_sim_index(segment_velocity=velocity)


def reset_arm_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Perturb arm joints about their defaults and synchronize position targets.

    Args:
        env: Environment containing the arm.
        env_ids: Environments to reset.
        position_range: Uniform joint position offset bounds [rad].
        asset_cfg: Arm articulation and joints to randomize; exclude finger joints.
    """
    reset_joints_by_offset(env, env_ids, position_range, (0.0, 0.0), asset_cfg)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch[env_ids][:, asset_cfg.joint_ids]
    robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_shoe_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
) -> None:
    """Translate the shoe and both laces together about their default positions.

    The pinned lace mesh is a child of the shoe; cable segment poses include the
    fixed anchors. Sampling from defaults avoids accumulating offsets across resets.
    Run after ``reset_scene_to_default`` to restore default velocities as well.

    Args:
        env: Shoelace environment.
        env_ids: Environments to reset.
        position_range: Uniform translation offset bounds [m], keyed by ``x``, ``y``,
            or ``z``. Omitted axes have zero offset.
    """
    shoe: RigidObject = env.scene["shoe"]
    bounds = torch.tensor([position_range.get(axis, (0.0, 0.0)) for axis in "xyz"], device=env.device)
    offset = sample_uniform(bounds[:, 0], bounds[:, 1], (len(env_ids), 3), env.device)
    root_pose = shoe.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids] + offset
    shoe.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    for name in ("shoelace_left", "shoelace_right"):
        cable: CableObject = env.scene[name]
        segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
        segment_pose[..., :3] += offset.unsqueeze(1)
        cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
