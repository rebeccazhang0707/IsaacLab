# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for multi-robot tasks.

These wrappers re-expose standard termination functions with parameter
names that match the ``robot_meta`` convention (``object_cfg`` instead
of ``asset_cfg``), so that ``per_robot=True`` auto-injection works
correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def object_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object's root height is below the minimum height.

    Thin wrapper around the height check that accepts ``object_cfg``
    so ``per_robot`` auto-injection from ``robot_meta`` works.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return wp.to_torch(asset.data.root_pos_w)[:, 2] < minimum_height
