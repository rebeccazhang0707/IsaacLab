# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda robot module for the LIBERO multitask environment.

Wraps the LIBERO-specific Franka articulation (see
:data:`~...assets.LIBERO_FRANKA_PANDA_CFG`) in the
:class:`~...multitask.robots._base.RobotModuleCfg` interface so it can be
composed by the :class:`~...multitask.registry.MultiTaskRegistry` alongside the
LIBERO task modules.  Provides a 6-D differential-IK arm column and a 1-D binary
gripper column.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.robots._base import RobotModuleCfg

from ..assets import LIBERO_FRANKA_PANDA_CFG

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@dataclass
class LiberoFrankaRobotCfg(RobotModuleCfg):
    """Franka Panda robot module for the LIBERO suites.

    Uses differential IK (6-D pose delta) for the arm and a binary open/close
    action (1-D) for the gripper.  The robot articulation carries LIBERO's PD
    gains and ready-pose joints; gravity is disabled on its rigid bodies to
    match the LIBERO reference environment.
    """

    ik_scale: float = 0.5
    """Scale factor applied to IK pose-delta commands [--]."""

    @property
    def name(self) -> str:
        return "franka"

    @property
    def all_joint_names(self) -> list[str]:
        return ["panda_joint.*", "panda_finger.*"]

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str) -> dict[str, object]:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = f"/Visuals/{group}_FrankaEEFrame"

        return {
            "franka_robot": LIBERO_FRANKA_PANDA_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
            ),
            "franka_ee_frame": FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                        name="ee_tcp",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                        name="tool_leftfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                    ),
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                        name="tool_rightfinger",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                    ),
                ],
            ),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, object]]:
        arm = DifferentialInverseKinematicsActionCfg(
            asset_name="franka_robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=_IK_CTRL,
            scale=self.ik_scale,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        gripper = BinaryJointPositionActionCfg(
            asset_name="franka_robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        return {
            "arm": (6, arm),
            "gripper": (1, gripper),
        }

    # ------------------------------------------------------------------
    # Observations (robot-side scatter contributions)
    # ------------------------------------------------------------------

    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, TermCfg]]:
        return {
            "ee_pose": (
                7,
                TermCfg(
                    func=mdp.ee_pose,
                    params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], selector=group)},
                ),
            ),
            "joint_pos": (
                9,
                TermCfg(
                    func=mdp.joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("franka_robot", selector=group)},
                ),
            ),
            "joint_vel": (
                9,
                TermCfg(
                    func=mdp.joint_vel,
                    params={"asset_cfg": SceneEntityCfg("franka_robot", selector=group)},
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        # Keys are namespaced by group so both LIBERO registrations keep their
        # own reset term (the Franka is a global/shared asset, so each group's
        # term resets only the environments that belong to it).
        return {
            f"{group}_franka_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("franka_robot", selector=group)],
                },
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built instance
# ---------------------------------------------------------------------------

FRANKA = LiberoFrankaRobotCfg()
"""Franka Panda with DiffIK arm control (6-D pose delta) and a binary gripper."""
