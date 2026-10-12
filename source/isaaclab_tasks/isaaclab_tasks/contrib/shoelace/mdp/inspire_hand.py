# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent Inspire finger commands with bounded joint-target changes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms

from .observations import tcp_position_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

    from .inspire_hand_cfg import InspireFingerActionCfg, InspirePinchIKActionCfg


class InspireFingerAction(BinaryJointPositionAction):
    """Select open/close finger targets and advance toward them at a bounded angular rate.

    The binary action controls the configured finger chains together. Joint targets advance once per
    control step, independently of whether the physics backend handles decimation internally.
    """

    cfg: InspireFingerActionCfg

    def __init__(self, cfg: InspireFingerActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if not math.isfinite(cfg.max_target_velocity) or cfg.max_target_velocity <= 0.0:
            raise ValueError("max_target_velocity must be finite and positive")
        self._joint_indices = wp.to_torch(self._joint_ids).long()
        self._delta = self._close_command - self._open_command
        self._delta_norm_squared = float(self._delta.square().sum().item())
        if self._delta_norm_squared == 0.0:
            raise ValueError("Finger open and close commands must differ")
        self._maximum_progress_rate = cfg.max_target_velocity / float(self._delta.abs().max().item())
        self._progress = torch.zeros((self.num_envs, 1), device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        desired = ((self._processed_actions - self._open_command) * self._delta).sum(dim=1, keepdim=True)
        desired.div_(self._delta_norm_squared)
        maximum_change = self._maximum_progress_rate * self._env.step_dt
        self._progress.add_((desired - self._progress).clamp(-maximum_change, maximum_change))
        # A shared interpolation fraction preserves authored mimic ratios throughout the movement.
        self._processed_actions.copy_(self._open_command + self._progress * self._delta)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        positions = self._asset.data.joint_pos.torch[indices][:, self._joint_indices]
        progress = ((positions - self._open_command) * self._delta).sum(dim=1, keepdim=True)
        self._progress[indices] = (progress / self._delta_norm_squared).clamp(0.0, 1.0)
        self._processed_actions[indices] = self._open_command + self._progress[indices] * self._delta


class InspirePinchIKAction(DifferentialInverseKinematicsAction):
    """Control the live thumb/index surface midpoint using arm joints and wrist orientation."""

    cfg: InspirePinchIKActionCfg

    def __init__(self, cfg: InspirePinchIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.body_offset is not None and (
            cfg.body_offset.pos != (0.0, 0.0, 0.0) or cfg.body_offset.rot != (0.0, 0.0, 0.0, 1.0)
        ):
            raise ValueError("The moving pinch frame requires an identity wrist offset")
        self._pad_body_ids, _ = self._asset.find_bodies(cfg.pad_body_names, preserve_order=True)
        if len(self._pad_body_ids) != 2:
            raise ValueError("The pinch frame requires exactly one thumb and one index body")
        self._pad_jacobian_ids = [body - int(self._asset.is_fixed_base) for body in self._pad_body_ids]
        self._pad_offsets = torch.tensor(cfg.pad_offsets, device=self.device)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = tcp_position_w(self._asset, self._pad_body_ids, self._pad_offsets)
        return subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            position,
            self._asset.data.body_quat_w.torch[:, self._body_idx],
        )

    def _compute_frame_jacobian(self) -> torch.Tensor:
        # Arm-column derivatives of the moving surface midpoint, not a fixed wrist offset.
        body_jacobians = self._asset.data.body_link_jacobian_w.torch
        pad_jacobians = body_jacobians[:, self._pad_jacobian_ids]
        pad_jacobians = pad_jacobians[..., self._jacobi_joint_ids]
        quaternions = self._asset.data.body_quat_w.torch[:, self._pad_body_ids]
        offsets = self._pad_offsets.expand(self.num_envs, -1, -1)
        levers = quat_apply(quaternions, offsets)
        linear = pad_jacobians[:, :, :3] + torch.linalg.cross(pad_jacobians[:, :, 3:], levers[..., None], dim=2)
        rotation = matrix_from_quat(quat_inv(self._asset.data.root_quat_w.torch))
        self._jacobian_b[:, :3] = torch.bmm(rotation, linear.mean(dim=1))
        wrist_angular = body_jacobians[:, self._jacobi_body_idx, 3:][..., self._jacobi_joint_ids]
        self._jacobian_b[:, 3:] = torch.bmm(rotation, wrist_angular)
        return self._jacobian_b


def inspire_finger_close_error(
    env: ManagerBasedEnv,
    finger_cfgs: tuple[SceneEntityCfg, ...],
    motion_ranges: tuple[float, ...],
) -> torch.Tensor:
    """Return signed target errors for each thumb/index primary joint, normalized by its stroke."""
    errors = []
    for cfg, stroke in zip(finger_cfgs, motion_ranges, strict=True):
        robot = env.scene[cfg.name]
        joint = cfg.joint_ids[0]
        errors.append((robot.data.joint_pos_target.torch[:, joint] - robot.data.joint_pos.torch[:, joint]) / stroke)
    return torch.stack(errors, dim=1)
