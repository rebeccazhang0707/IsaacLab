# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_shoelace_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] = (
        SceneEntityCfg("shoelace_left"),
        SceneEntityCfg("shoelace_right"),
    ),
) -> None:
    """Restore selected cables to their authored knotted pose with zero velocity."""
    physics = getattr(env, "_physics", None)
    reset_grasp_assist = getattr(physics, "reset_grasp_assist", None)
    if reset_grasp_assist is not None:
        reset_grasp_assist(env_ids)
    for asset_cfg in asset_cfgs:
        cable = env.scene[asset_cfg.name]
        segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
        segment_velocity = cable.data.default_segment_velocity_w.torch[env_ids].clone()
        cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
        cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)


class ResetShoelaceCurriculum(ManagerTermBase):
    """Keep the authored shoelace fixed while staging robot grasp distance."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        difficulty_term_name: str,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        arm_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        gripper_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        grasp_joint_positions: tuple[tuple[float, ...], tuple[float, ...]],
        open_position: float,
        closed_position: float,
        gripper_open_phase_fraction: float,
        approach_phase_exponent: float = 1.0,
    ) -> None:
        """Apply one curriculum reset.

        The cable is restored to its authored pose at every level. At zero
        difficulty, both arms start with their TCPs centered on the authored
        tails and both grippers closed. Early levels open the grippers while
        keeping the TCPs at the tails. Later levels move the open grippers
        toward their configured pregrasp joint positions.

        Args:
            env: The learning environment.
            env_ids: Environments to reset.
            difficulty_term_name: Curriculum term that owns reset difficulty.
            asset_cfgs: Left and right cable configurations.
            arm_cfgs: Left and right robot arm configurations.
            gripper_cfgs: Left and right robot gripper configurations.
            grasp_joint_positions: Arm joint positions at zero difficulty [rad].
            open_position: Driven finger position for an open gripper [m].
            closed_position: Driven finger position for a closed gripper [m].
            gripper_open_phase_fraction: Fraction of curriculum difficulty reserved for opening the gripper at the
                tail before increasing approach distance.
            approach_phase_exponent: Exponent applied to normalized approach difficulty. Values above one allocate
                finer levels near the grasp pose.
        """
        difficulty_term = getattr(env.curriculum_manager.cfg, difficulty_term_name).func
        difficulty = difficulty_term.difficulty[env_ids].unsqueeze(-1)
        arm_difficulty, gripper_difficulty = self._phase_difficulties(
            difficulty, gripper_open_phase_fraction, approach_phase_exponent
        )
        reset_shoelace_state(env, env_ids, asset_cfgs)

        for arm_cfg, grasp_position in zip(arm_cfgs, grasp_joint_positions, strict=True):
            robot = env.scene[arm_cfg.name]
            default_position = robot.data.default_joint_pos.torch[env_ids][:, arm_cfg.joint_ids]
            grasp_position_tensor = default_position.new_tensor(grasp_position).expand_as(default_position)
            joint_position = self._interpolate_joint_positions(grasp_position_tensor, default_position, arm_difficulty)
            joint_velocity = torch.zeros_like(joint_position)
            robot.write_joint_position_to_sim_index(
                position=joint_position, joint_ids=arm_cfg.joint_ids, env_ids=env_ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=joint_velocity, joint_ids=arm_cfg.joint_ids, env_ids=env_ids
            )
            robot.set_joint_position_target_index(target=joint_position, joint_ids=arm_cfg.joint_ids, env_ids=env_ids)

        finger_position = closed_position + gripper_difficulty.squeeze(-1) * (open_position - closed_position)
        for gripper_cfg in gripper_cfgs:
            robot = env.scene[gripper_cfg.name]
            joint_position = finger_position.unsqueeze(-1).expand(-1, len(gripper_cfg.joint_ids))
            joint_velocity = torch.zeros_like(joint_position)
            robot.write_joint_position_to_sim_index(
                position=joint_position, joint_ids=gripper_cfg.joint_ids, env_ids=env_ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=joint_velocity, joint_ids=gripper_cfg.joint_ids, env_ids=env_ids
            )
            robot.set_joint_position_target_index(
                target=joint_position, joint_ids=gripper_cfg.joint_ids, env_ids=env_ids
            )

    @staticmethod
    def _interpolate_joint_positions(
        grasp_position: torch.Tensor,
        pregrasp_position: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate arm positions from grasp to pregrasp over reset difficulty."""
        return torch.lerp(grasp_position, pregrasp_position, difficulty)

    @staticmethod
    def _phase_difficulties(
        difficulty: torch.Tensor,
        gripper_open_phase_fraction: float,
        approach_phase_exponent: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split reset difficulty into sequential gripper and arm phases."""
        if not 0.0 < gripper_open_phase_fraction < 1.0:
            raise ValueError("gripper_open_phase_fraction must lie in (0, 1).")
        if not math.isfinite(approach_phase_exponent) or approach_phase_exponent <= 0.0:
            raise ValueError("approach_phase_exponent must be finite and positive.")
        bounded_difficulty = difficulty.clamp(0.0, 1.0)
        gripper_difficulty = (bounded_difficulty / gripper_open_phase_fraction).clamp(max=1.0)
        normalized_arm_difficulty = (
            (bounded_difficulty - gripper_open_phase_fraction) / (1.0 - gripper_open_phase_fraction)
        ).clamp(0.0, 1.0)
        arm_difficulty = normalized_arm_difficulty.pow(approach_phase_exponent)
        return arm_difficulty, gripper_difficulty
