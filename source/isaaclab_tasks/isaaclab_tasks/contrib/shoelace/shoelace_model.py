# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton model lifecycle configuration for the coupled shoelace scene."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsEvent

from . import shoelace_physics as physics
from .shoelace_contacts import _FingerTailContacts

if TYPE_CHECKING:
    from .shoelace_env_cfg import ShoelaceEnvCfg


TAIL_BEND_STIFFNESS = 100.0
TAIL_BEND_DAMPING = 2.0
TAIL_STIFF_CORE_LENGTH = 0.050
TAIL_STIFF_TRANSITION_LENGTH = 0.012


def _tail_joint_blend_weights(
    joint_count: int,
    segment_length: float,
    free_end_at_start: bool,
) -> np.ndarray:
    """Return smooth weights for the gravity-resistant distal tail material."""
    distances = segment_length * np.arange(1, joint_count + 1, dtype=np.float64)
    if TAIL_STIFF_TRANSITION_LENGTH > 0.0:
        weights = np.clip(
            (TAIL_STIFF_CORE_LENGTH + TAIL_STIFF_TRANSITION_LENGTH - distances) / TAIL_STIFF_TRANSITION_LENGTH,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)
    else:
        weights = (distances <= TAIL_STIFF_CORE_LENGTH).astype(np.float64)
    return weights if free_end_at_start else weights[::-1].copy()


class _ShoelacePhysics:
    """Configure cable materials, topology, anchors, and contacts before solver construction."""

    def __init__(
        self, cfg: ShoelaceEnvCfg, centerline: np.ndarray, cable_radius: float, stiffness_scale: float
    ) -> None:
        self.cfg = cfg
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.joint_stiffness_scale = stiffness_scale

    @property
    def finger_tail_signed_distance(self) -> torch.Tensor | None:
        """Current four finger-tail surface distances [m], shape [N, 4]."""
        return self.contacts.signed_distance

    def register(self) -> None:
        """Register the callback that finalizes the Newton builder."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_task_builder")

    def _configure_builder(self, _: object) -> None:
        """Finalize cable materials, topology, anchors, and collision filters."""
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton builder is unavailable during MODEL_INIT")
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp.values = {} if gravcomp.values is None else gravcomp.values
        for body, label in enumerate(builder.body_label):
            if "/RobotLeft/" in label or "/RobotRight/" in label:
                gravcomp.values[body] = 1.0
        build = physics.configure_shoelace_builder(
            builder,
            self.centerline,
            self.cable_radius,
            self.cfg.scene.num_envs,
            lace_mu=self.cfg.lace_mu,
            shoe_mu=self.cfg.shoe_mu,
            cable_inertia_regularization=self.cfg.cable_inertia_regularization,
        )
        self.cable_joints = [left + right for left, right in build.joint_chains]
        self.contacts = _FingerTailContacts(builder, build)
        NewtonManager.register_callback(
            self.contacts.initialize, PhysicsEvent.PHYSICS_READY, name="shoelace_task_contact_observer"
        )
        for shape, label in enumerate(builder.shape_label):
            if "panda_leftfinger" in label or "panda_rightfinger" in label:
                builder.shape_material_mu[shape] = self.cfg.finger_mu

        for world_joints in build.joint_chains:
            for joints, free_end_at_start in zip(world_joints, (True, False), strict=True):
                weights = _tail_joint_blend_weights(len(joints), build.mean_segment_length, free_end_at_start)
                stiffness: list[float] = []
                damping: list[float] = []
                for weight in weights:
                    bend_stiffness = physics.BEND_STIFFNESS + weight * (TAIL_BEND_STIFFNESS - physics.BEND_STIFFNESS)
                    bend_damping = physics.BEND_DAMPING + weight * (TAIL_BEND_DAMPING - physics.BEND_DAMPING)
                    stiffness.extend((physics.STRETCH_STIFFNESS,) * 2 + (bend_stiffness,) * 2)
                    damping.extend((physics.STRETCH_DAMPING,) * 2 + (bend_damping,) * 2)
                dof_slice = physics.cable_joint_dof_slice(builder, joints)
                builder.joint_target_ke[dof_slice] = tuple(value * self.joint_stiffness_scale for value in stiffness)
                builder.joint_target_kd[dof_slice] = tuple(value * self.joint_stiffness_scale for value in damping)
        physics.configure_dahl(builder, self.cable_joints)
