# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for batched command terms."""

from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class BatchedPoseCommandCfg(CommandTermCfg):
    """Configuration for :class:`BatchedPoseCommand`.

    Per-group ranges and robot references are specified via
    :attr:`robot_meta` (keyed by task-group name).
    """

    class_type: type | str = "{DIR}.batched_commands:BatchedPoseCommand"

    robot_meta: dict = {}
    """Mapping from group key to group config (e.g., ReachGroupCfg, LiftGroupCfg)."""

    make_quat_unique: bool = True
    """Whether to ensure the quaternion has a positive real part."""
