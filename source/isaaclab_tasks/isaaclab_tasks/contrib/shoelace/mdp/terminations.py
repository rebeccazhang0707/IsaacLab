# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .utils import tail_x_separation, untying_metrics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def tail_x_separation_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return success when absolute two-tail X separation reaches the threshold [m]."""
    separation = tail_x_separation(env, cable_cfgs)
    return torch.isfinite(separation) & (separation >= threshold)


def shoelace_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    throat_radius: float,
    maximum_throat_segments: int,
    tail_success_distance: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Require a cleared knot throat and two tails separated away from the fixed anchors.

    This geometric criterion follows ``reb/newton_shoelace_demo`` and adds the current task's
    X-separation requirement. Grasp quality affects dense rewards, not geometric success.

    Args:
        env: Shoelace environment.
        threshold: Minimum absolute two-tail X separation [m].
        throat_radius: Radius around the fixed seam midpoint [m].
        maximum_throat_segments: Maximum number of free segments inside the throat at success.
        tail_success_distance: Minimum distance of each tail from the seam midpoint [m].
        cable_cfgs: Left and right cable scene entities.

    Returns:
        Success flags for finite cable states satisfying all geometric criteria, shape [N].
    """
    throat_count, tail_distances, separation, finite = untying_metrics(env, cable_cfgs, throat_radius)
    return (
        finite
        & (throat_count <= maximum_throat_segments)
        & (tail_distances.amin(dim=1) >= tail_success_distance)
        & (separation >= threshold)
    )
