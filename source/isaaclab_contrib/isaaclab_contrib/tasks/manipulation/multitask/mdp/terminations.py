# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for multi-robot tasks.

These functions accept ``object_cfg`` for consistency with the
``robot_meta`` convention.  Use them with ``task_group`` scoping
or from batched termination classes in ``batched_terminations.py``.
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
    """Terminate when the object's root height is below the minimum height [m].

    Accepts ``object_cfg`` for consistency with the ``robot_meta`` convention.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    return wp.to_torch(asset.data.root_pos_w)[:, 2] < minimum_height


def cabinet_drawer_opened(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate cabinet episodes once the drawer is sufficiently open."""
    drawer_pos = wp.to_torch(env.scene[asset_cfg.name].data.joint_pos)[:, asset_cfg.joint_ids[0]]
    return drawer_pos > threshold
