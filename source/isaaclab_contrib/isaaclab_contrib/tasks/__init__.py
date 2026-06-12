# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Externally contributed Isaac Lab task environments.

Importing this package registers the contributed gym environments. ``train.py`` /
``play.py`` import ``isaaclab_contrib.tasks`` for exactly this side effect.
"""

# Importing each leaf config package runs its ``gym.register`` calls.
from .manipulation.multitask.config import demo  # noqa: F401
