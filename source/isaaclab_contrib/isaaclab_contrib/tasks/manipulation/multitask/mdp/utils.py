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
    """Base metadata for a robot/task group in multi-robot environments.

    Environment configs store a ``robot_meta`` dict mapping **task-group
    names** to instances of this class (or its subclasses) so that
    batched MDP term classes have typed, IDE-discoverable fields.

    Subclass for specific task domains:

    * :class:`ReachGroupCfg` -- reach tasks
    * :class:`LiftGroupCfg` -- lift tasks
    * :class:`CabinetGroupCfg` -- cabinet tasks

    Example::

        robot_meta = {
            "franka_reach": ReachGroupCfg(
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


@configclass
class ReachGroupCfg(RobotGroupCfg):
    """Metadata for a reach task group.

    Reach tasks require a command target but no object or cabinet.
    """

    command_name: str = MISSING
    """Name of the command term that generates the reach target."""


@configclass
class LiftGroupCfg(RobotGroupCfg):
    """Metadata for a lift task group.

    Lift tasks require a command target, a robot root reference,
    a manipulation object, and an EE frame sensor.
    """

    command_name: str = MISSING
    """Name of the command term that generates the object goal pose."""

    robot_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the robot articulation root (used for frame transforms)."""

    object_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the rigid object to lift."""

    ee_frame_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the end-effector FrameTransformer sensor."""


@configclass
class CabinetGroupCfg(RobotGroupCfg):
    """Metadata for a cabinet task group.

    Cabinet tasks require an EE frame, a cabinet handle frame,
    and the cabinet articulation for reading drawer joint state.
    """

    ee_frame_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the end-effector FrameTransformer sensor."""

    cabinet_frame_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the cabinet handle FrameTransformer sensor."""

    cabinet_asset_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg for the cabinet articulation (joint names for the drawer)."""


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
    """Filter ``env_ids`` to only those belonging to *group_key*.

    Args:
        layout: The environment layout.
        group_key: The task-group key to filter for.
        env_ids: Env indices to filter, or ``None`` for all envs.

    Returns:
        A tuple ``(env_ids, skip)``.  *env_ids* contains only the
        subset that belong to this group.  When *skip* is ``True``
        no envs matched and the caller should ``continue`` to the
        next group.  When the input *env_ids* is ``None``, the output
        is also ``None`` (meaning "all envs in this group").
    """
    if env_ids is None:
        return None, False
    _, matched = layout.filter_and_split(group_key, env_ids)
    if matched.numel() == 0:
        return matched, True
    return matched, False


def asset_env_ids(
    layout: EnvLayout,
    group_key: str,
    asset_name: str,
    env_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the correct env indices for writing to an asset's sim buffers.

    Shared assets use the same indices as the caller.  Group-specific
    assets need 0-based indices, which this function handles internally.

    Args:
        layout: The environment layout.
        group_key: The task-group key the caller is iterating over.
        asset_name: Scene entity name of the asset being written to.
        env_ids: Env indices from :func:`filter_env_ids`, or ``None``
            for all envs in the group.

    Returns:
        Indices suitable for ``write_*_to_sim_index`` calls, or ``None``
        when all envs in the asset's partition should be written.
    """
    asset_group = layout._asset_groups.get(asset_name)
    if asset_group is None or asset_group != group_key:
        return env_ids
    return layout.global_to_local(group_key, env_ids) if env_ids is not None else None
