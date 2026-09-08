# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for shoelace action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import EMADifferentialInverseKinematicsAction


@configclass
class EMADifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`EMADifferentialInverseKinematicsAction`."""

    alpha: float = 1.0
    """Weight of the newest Cartesian command; one preserves the unfiltered action."""

    warmup_steps: int = 0
    """Number of control steps after reset that hold the arm command at zero."""

    warmup_steps_by_curriculum_level: tuple[int, ...] | None = None
    """Optional warm-up duration indexed by the active discrete curriculum level."""

    warmup_curriculum_term_name: str = "pull_to_grasp"
    """Curriculum term that provides levels for ``warmup_steps_by_curriculum_level``."""

    class_type: type[EMADifferentialInverseKinematicsAction] | str = (
        "{DIR}.actions:EMADifferentialInverseKinematicsAction"
    )
