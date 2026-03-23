# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config types and base class for batched MDP terms."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar, overload

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.scene import EnvLayout

_GroupT = TypeVar("_GroupT", bound="RobotGroupCfg")


class BatchedTermBase(ManagerTermBase):
    """Base class for batched MDP terms that iterate over robot_meta groups.

    Provides common setup and helper methods to reduce boilerplate in
    batched observation, reward, termination, and event terms.

    ``robot_meta`` is read from ``cfg.params["robot_meta"]``, where all
    nested ``SceneEntityCfg`` instances are auto-resolved by the manager.

    Subclasses should:
    1. Call ``super().__init__(cfg, env)``
    2. Use ``self._iter_groups(*types)`` to iterate filtered groups
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._layout: EnvLayout = env.scene.layout
        # Read from params (auto-resolved by manager)
        self._robot_meta: dict = cfg.params.get("robot_meta") or {}
        self._num_envs = env.num_envs
        self._device = env.device

    @overload
    def _iter_groups(self) -> Iterator[tuple[str, RobotGroupCfg]]: ...

    @overload
    def _iter_groups(self, group_type: type[_GroupT], /) -> Iterator[tuple[str, _GroupT]]: ...

    @overload
    def _iter_groups(
        self, group_type1: type[RobotGroupCfg], group_type2: type[RobotGroupCfg], /, *more: type[RobotGroupCfg]
    ) -> Iterator[tuple[str, RobotGroupCfg]]: ...

    def _iter_groups(self, *group_types: type[RobotGroupCfg]) -> Iterator[tuple[str, RobotGroupCfg]]:
        """Iterate robot_meta entries, optionally filtering by group type.

        Args:
            *group_types: If provided, only yield entries that are instances
                of one of these types. If empty, yield all entries.

        Yields:
            (group_key, meta) tuples with proper typing for autocomplete.
        """
        for group_key, meta in self._robot_meta.items():
            if group_types and not isinstance(meta, group_types):
                continue
            yield group_key, meta


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
class PoseCommandRanges:
    """Uniform sampling ranges for a pose command [m, rad]."""

    pos_x: tuple[float, float] = (0.0, 0.0)
    """Min/max for X position [m]."""
    pos_y: tuple[float, float] = (0.0, 0.0)
    """Min/max for Y position [m]."""
    pos_z: tuple[float, float] = (0.0, 0.0)
    """Min/max for Z position [m]."""
    roll: tuple[float, float] = (0.0, 0.0)
    """Min/max for roll angle [rad]."""
    pitch: tuple[float, float] = (0.0, 0.0)
    """Min/max for pitch angle [rad]."""
    yaw: tuple[float, float] = (0.0, 0.0)
    """Min/max for yaw angle [rad]."""


@configclass
class ReachGroupCfg(RobotGroupCfg):
    """Metadata for a reach task group.

    Reach tasks require a command target but no object or cabinet.
    """

    command_name: str = MISSING
    """Name of the command term that generates the reach target."""

    command_ranges: PoseCommandRanges = MISSING
    """Sampling ranges for the pose command target."""


@configclass
class LiftGroupCfg(RobotGroupCfg):
    """Metadata for a lift task group.

    Lift tasks require a command target, a robot root reference,
    a manipulation object, and an EE frame sensor.
    """

    command_name: str = MISSING
    """Name of the command term that generates the object goal pose."""

    command_ranges: PoseCommandRanges = MISSING
    """Sampling ranges for the pose command target."""

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
