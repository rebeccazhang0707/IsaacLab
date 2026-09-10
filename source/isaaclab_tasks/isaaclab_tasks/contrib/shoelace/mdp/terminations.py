# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .utils import tail_x_separation

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
