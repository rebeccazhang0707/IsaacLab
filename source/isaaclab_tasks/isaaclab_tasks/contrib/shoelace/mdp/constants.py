# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-specific shoelace topology and robot-grasp constants."""

SHOELACE_SEGMENT_COUNT = 360
PINNED_FIRST = 76
PINNED_LAST = 284
LEFT_CABLE_SEGMENT_COUNT = PINNED_FIRST + 1
RIGHT_CABLE_SEGMENT_COUNT = SHOELACE_SEGMENT_COUNT - PINNED_LAST
DYNAMIC_SEGMENT_COUNT = LEFT_CABLE_SEGMENT_COUNT + RIGHT_CABLE_SEGMENT_COUNT
TAIL_REGIONS = ((357, 360), (0, 3))
TCP_OFFSET = (0.0, 0.0, 0.107)
REFERENCE_PULL_DIRECTIONS = ((-1.0, -0.35, 0.15), (1.0, -0.35, 0.15))
