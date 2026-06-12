# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot module definitions for multitask environments."""

from ._base import RobotModuleCfg
from .franka import FRANKA_IK, FRANKA_JOINT, FrankaRobotCfg
from .openarm import OPENARM_IK, OPENARM_JOINT, OpenArmRobotCfg
from .ur10 import UR10_IK, UR10_JOINT, UR10RobotCfg
