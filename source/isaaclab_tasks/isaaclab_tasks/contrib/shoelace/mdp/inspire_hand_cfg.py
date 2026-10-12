# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for Inspire finger targets and a moving pinch frame."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .inspire_hand import InspireFingerAction, InspirePinchIKAction


@configclass
class InspireFingerActionCfg(BinaryJointPositionActionCfg):
    """Binary targets for coupled finger chains, with a control-step angular-rate bound."""

    class_type: type[InspireFingerAction] | str = "{DIR}.inspire_hand:InspireFingerAction"
    max_target_velocity: float = 1.0
    """Maximum joint-target change rate [rad/s]; actual speed depends on the drive and contacts."""


@configclass
class InspirePinchIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Cartesian control of the moving thumb/index surface midpoint, with wrist orientation."""

    class_type: type[InspirePinchIKAction] | str = "{DIR}.inspire_hand:InspirePinchIKAction"
    pad_body_names: tuple[str, str] = MISSING
    """Thumb distal and index intermediate body names, in that order."""
    pad_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    """Approximate fingertip contact-surface centres in each finger body's local frame [m]."""
