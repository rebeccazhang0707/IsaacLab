# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task module definitions for multitask environments."""

from ._base import TaskModuleCfg
from .cabinet import CABINET_TASK, CabinetTaskCfg
from .lift import LIFT_TASK, LIFT_TASK_OPENARM, LiftTaskCfg
from .reach import REACH_TASK, REACH_TASK_OPENARM, ReachTaskCfg
