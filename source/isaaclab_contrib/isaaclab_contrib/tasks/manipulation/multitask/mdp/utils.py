# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers and config types for batched MDP term classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene.env_layout import EnvLayout


@configclass
class RobotGroupCfg:
    """Typed metadata for a robot group in heterogeneous multi-robot environments.

    Declares the scene entities and command association for a single
    robot type.  Environment configs store a ``robot_meta`` dict
    mapping asset names to instances of this class so that batched
    MDP term classes have typed, IDE-discoverable fields instead of
    an opaque ``dict[str, Any]``.

    If a task requires additional per-robot metadata beyond these
    common fields, subclass this configclass and add extra fields.

    Example::

        robot_meta = {
            "franka_robot": RobotGroupCfg(
                asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"]),
                command_name="franka_ee_pose",
            ),
        }
    """

    asset_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg identifying the robot articulation.

    Typically includes ``body_names`` for the end-effector and
    ``joint_names`` for the arm (and optionally gripper) joints.
    """

    command_name: str | None = None
    """Name of the command term that generates targets for this robot."""

    robot_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the robot articulation root.

    Used by reward/observation functions that need the root pose
    separately from the EE body (e.g. ``object_ee_distance``).
    When ``None``, functions that accept ``robot_cfg`` will not
    receive an auto-injected value.
    """

    object_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the manipulation object (e.g. a cube to lift)."""

    ee_frame_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the end-effector FrameTransformer sensor."""


def resolve_scene_entity_cfg(env: ManagerBasedEnv, cfg: SceneEntityCfg) -> None:
    """Resolve body/joint ids on a :class:`SceneEntityCfg` if not already resolved.

    Batched MDP term classes call this during ``__init__`` for each
    :class:`SceneEntityCfg` found in ``robot_meta``.  The guard avoids
    redundant resolution when the same cfg object is shared.
    """
    already = isinstance(cfg.body_ids, list) or isinstance(cfg.joint_ids, list)
    if not already:
        cfg.resolve(env.scene)


def filter_env_ids(
    layout: EnvLayout,
    group_key: str,
    env_ids: torch.Tensor | None,
) -> tuple[torch.Tensor | None, bool]:
    """Filter global ``env_ids`` down to a single task group.

    Args:
        layout: The environment layout.
        group_key: The task-group key to filter for.
        env_ids: Global env indices, or ``None`` for all envs.

    Returns:
        A tuple ``(local_ids, skip)``.  When *skip* is ``True`` the
        caller should ``continue`` to the next group (no envs matched).
        When *env_ids* is ``None``, *local_ids* is also ``None``
        (meaning "all envs in this group").
    """
    if env_ids is None:
        return None, False
    local, _ = layout.filter_and_split(group_key, env_ids)
    if local.numel() == 0:
        return local, True
    return local, False
