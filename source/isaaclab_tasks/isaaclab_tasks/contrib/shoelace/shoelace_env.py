# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime setup for the dual-Franka Newton shoelace RL environment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_newton.physics import NewtonCfg

from isaaclab.envs import ManagerBasedRLEnv

from . import shoelace_physics as physics
from .shoelace_model import TAIL_BEND_DAMPING as TAIL_BEND_DAMPING
from .shoelace_model import TAIL_BEND_STIFFNESS as TAIL_BEND_STIFFNESS
from .shoelace_model import TAIL_STIFF_CORE_LENGTH as TAIL_STIFF_CORE_LENGTH
from .shoelace_model import TAIL_STIFF_TRANSITION_LENGTH as TAIL_STIFF_TRANSITION_LENGTH
from .shoelace_model import _ShoelacePhysics

if TYPE_CHECKING:
    from .shoelace_env_cfg import ShoelaceEnvCfg


def _load_settled_segment_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load validated gravity-settled cable segment poses [m, quaternion x-y-z-w]."""
    names = ("shoelace_left", "shoelace_right")
    with np.load(path, allow_pickle=False) as state:
        if tuple(state.files) != names:
            raise ValueError(f"Expected settled shoelace arrays {names}, got {tuple(state.files)}")
        poses = tuple(np.asarray(state[name], dtype=np.float32) for name in names)
    expected_shapes = (
        (physics.PINNED_FIRST + 1, 7),
        (physics.SHOELACE_SEGMENT_COUNT - physics.PINNED_LAST, 7),
    )
    for name, pose, expected_shape in zip(names, poses, expected_shapes, strict=True):
        if pose.shape != expected_shape or not np.isfinite(pose).all():
            raise ValueError(f"Invalid settled {name} poses: expected finite {expected_shape}, got {pose.shape}")
        if not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1.0, atol=2.0e-5):
            raise ValueError(f"Settled {name} poses contain non-unit quaternions")
    return poses


class ShoelaceEnv(ManagerBasedRLEnv):
    """Expose the dual-Franka shoelace scene through the manager-based RL interface."""

    cfg: ShoelaceEnvCfg

    def __init__(self, cfg: ShoelaceEnvCfg, render_mode: str | None = None, **kwargs: object) -> None:
        self._centerline, self._cable_radius, authored_mean_segment_length = physics.load_shoelace()
        settled_state_asset = physics.ASSET_DIR / "settled_tail_clear_segment_poses.npz"
        self._settled_segment_poses = _load_settled_segment_poses(settled_state_asset)
        mean_segment_length = float(np.linalg.norm(np.diff(self._centerline, axis=0), axis=1).mean())
        self._configure_runtime_cfg(cfg, authored_mean_segment_length)
        self._physics = _ShoelacePhysics(
            cfg,
            self._centerline,
            self._cable_radius,
            authored_mean_segment_length / mean_segment_length,
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._install_settled_default_state()

    def _configure_runtime_cfg(self, cfg: ShoelaceEnvCfg, authored_mean_segment_length: float) -> None:
        """Fill asset geometry and environment-count-dependent capacities."""
        cfg.scene.shoelace_pinned_visual.spawn = physics.pinned_shoelace_spawner(self._centerline, self._cable_radius)
        cable_normals = physics.parallel_transport_normals(self._centerline)
        cable_centerlines = (
            self._centerline[: physics.PINNED_FIRST + 2],
            self._centerline[physics.PINNED_LAST :],
        )
        cable_normal_chains = (
            cable_normals[: physics.PINNED_FIRST + 2],
            cable_normals[physics.PINNED_LAST :],
        )
        for cable_cfg, centerline, normals in zip(
            (cfg.scene.shoelace_left.spawn, cfg.scene.shoelace_right.spawn),
            cable_centerlines,
            cable_normal_chains,
            strict=True,
        ):
            physics.configure_cable_spawn(
                cable_cfg,
                centerline,
                normals,
                self._cable_radius,
                authored_mean_segment_length,
            )

        ground_size = physics.ground_size(cfg.scene.num_envs, cfg.scene.env_spacing)
        cfg.scene.ground.spawn.size = (ground_size, ground_size)
        physics_cfg = cfg.sim.physics
        if not isinstance(physics_cfg, NewtonCfg):
            raise TypeError("The dual-Franka shoelace task requires Newton physics")
        physics_cfg.collision_cfg.rigid_contact_max = physics.CONTACTS_PER_ENV * cfg.scene.num_envs
        physics_cfg.collision_cfg.max_triangle_pairs = max(
            physics_cfg.collision_cfg.max_triangle_pairs,
            physics.MIN_TRIANGLE_PAIRS,
            physics.TRIANGLE_PAIRS_PER_ENV * cfg.scene.num_envs,
        )
        solver_cfg = physics_cfg.solver_cfg
        solver_cfg.contact_max_triangle_pairs = max(
            solver_cfg.contact_max_triangle_pairs or 0, physics.MIN_TRIANGLE_PAIRS
        )
        # Contact matching requires a triangle budget below 2**20 in Newton 1.6.
        # Scale only the reduction hashtable to retain matching at large batch sizes.
        solver_cfg.contact_reduction_hashtable_size_factor = max(
            solver_cfg.contact_reduction_hashtable_size_factor or 0.25,
            0.25 * physics.TRIANGLE_PAIRS_PER_ENV * cfg.scene.num_envs / solver_cfg.contact_max_triangle_pairs,
        )

    def _install_settled_default_state(self) -> None:
        """Install the offline gravity-settled cable poses as reset defaults."""
        for cable_name, local_pose in zip(
            ("shoelace_left", "shoelace_right"),
            self._settled_segment_poses,
            strict=True,
        ):
            cable = self.scene[cable_name]
            default_pose = cable.data.default_segment_pose_w.torch
            segment_pose = default_pose.new_tensor(local_pose).unsqueeze(0).expand_as(default_pose).clone()
            segment_pose[..., :3] += self.scene.env_origins.unsqueeze(1)
            segment_velocity = torch.zeros_like(cable.data.default_segment_velocity_w.torch)
            default_pose.copy_(segment_pose)
            cable.data.default_segment_velocity_w.torch.copy_(segment_velocity)
            cable.write_segment_pose_to_sim_index(segment_pose=segment_pose)
            cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity)

    def _init_sim(self) -> None:
        """Register the shoe material and Newton callback before model creation."""
        physics.install_shoe_material_reference()
        self._physics.register()
        super()._init_sim()

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset task state and invalidate cached contact observations."""
        super()._reset_idx(env_ids)
        self._physics.contacts.reset(env_ids)
