# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared state helpers for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def tail_state(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return three-segment mean tail positions [m] and velocities [m/s]."""
    left_cable = env.scene[cable_cfgs[0].name]
    right_cable = env.scene[cable_cfgs[1].name]
    left_positions = left_cable.data.segment_pose_w.torch[:, :3, :3].mean(dim=1)
    right_positions = right_cable.data.segment_pose_w.torch[:, -3:, :3].mean(dim=1)
    left_velocities = left_cable.data.segment_velocity_w.torch[:, :3, :3].mean(dim=1)
    right_velocities = right_cable.data.segment_velocity_w.torch[:, -3:, :3].mean(dim=1)
    return torch.stack((right_positions, left_positions), dim=1), torch.stack(
        (right_velocities, left_velocities), dim=1
    )


def tail_x_separation(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return the absolute two-tail X-position separation [m]."""
    tail_positions, _ = tail_state(env, cable_cfgs)
    return torch.abs(tail_positions[:, 1, 0] - tail_positions[:, 0, 0])
