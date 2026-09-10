# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the dual-Franka shoelace task."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .observations import finger_tail_signed_distance, tail_tcp_relative_speed, tails_to_tcp
from .utils import tail_x_separation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg


class dense_task_reward(ManagerTermBase):
    """Reward acquiring both tails and pulling them apart through one potential difference.

    For each arm, acquisition combines TCP proximity with contact-aware grasp quality using a Hamacher
    soft-AND. Partial approach credit keeps acquisition learnable before contact, and averaging the two
    arms rewards acquiring either tail first. Pulling combines both grasps with reset-relative X separation.

    The term stores filtered grasp qualities, the initial separation, and one previous potential per
    environment. Invalid samples return zero without advancing this state. The first valid evaluation
    after reset seeds it and returns zero.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_per_gripper_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._baseline_x_separation = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._warned_legacy_params = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear episode state so the next valid evaluation gives no reset credit.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        selected = slice(None) if env_ids is None else env_ids
        self._filtered_per_gripper_grasp[selected] = torch.nan
        self._baseline_x_separation[selected] = torch.nan
        self._previous_potential[selected] = torch.nan

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_std: float,
        contact_std: float,
        relative_speed_std: float,
        grasp_filter_time_constant: float,
        open_position: float,
        closed_position: float,
        success_x_separation: float,
        maximum_progress_rate: float | None = None,
        approach_weight: float | None = None,
        grasp_weight: float | None = None,
        task_weight: float | None = None,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        *,
        acquisition_weight: float = 0.3,
        approach_fraction: float = 0.3,
    ) -> torch.Tensor:
        """Return the signed rate of acquisition-and-pull progress [1/s].

        With ``H`` denoting Hamacher soft-AND, ``A`` per-arm approach, ``G`` filtered per-arm grasp quality,
        and ``X`` normalized separation progress, the potential is::

            acquire = mean(H(A, approach_fraction + (1 - approach_fraction) * G))
            pull = H(H(G_left, G_right), X)
            potential = acquisition_weight * acquire + (1 - acquisition_weight) * pull

        Both phases remain active: losing grasp lowers acquisition and suppresses pulling. The separation
        progress starts at the first valid post-reset separation and reaches one at ``success_x_separation``.
        Each grasp soft-ANDs two finger-contact qualities, actual closure, and low tail-TCP slip. Grasp
        filtering smooths contact flicker, so release reduces the score over the filter time constant.

        The return value is ``(potential - previous_potential) / env.step_dt``. It is not clipped, so gains
        and losses cancel over a closed cycle of the complete reward state in the undiscounted sum. Once
        the filtered state settles, holding still gives zero. The reward manager applies the term weight
        and multiplies by ``env.step_dt``. This is a progress objective, not a guarantee of policy-invariant
        shaping for a discounted MDP.

        Args:
            env: The task environment.
            reach_std: Tail-to-TCP approach-distance width [m].
            contact_std: Signed-distance width around zero contact [m]. Gaps and deep penetrations both
                lower the grasp quality.
            relative_speed_std: Tail-TCP slip-speed width [m/s].
            grasp_filter_time_constant: Grasp-quality low-pass time constant [s].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            success_x_separation: Successful absolute two-tail X separation [m].
            maximum_progress_rate: Deprecated rate limit [1/s]. Accepted but ignored; remove it from configs.
            approach_weight: Deprecated approach budget. Legacy budgets override the new stage fractions;
                migrate as described in the task README.
            grasp_weight: Deprecated grasp budget, folded into acquisition with ``approach_weight``.
            task_weight: Deprecated pull budget. The sum of legacy budgets preserves the potential scale.
            cable_cfgs: Required left and right cable scene entities.
            robot_cfgs: Required left and right robot hand and finger scene entities.
            acquisition_weight: Fraction of the potential allocated to acquisition in ``[0, 1]``.
                Pulling receives the remainder.
            approach_fraction: Fraction of acquisition available before grasping in ``[0, 1]``.
                A value strictly between zero and one gives feedback for both approaching and grasping.

        Returns:
            Reward rates [1/s], shape [N]. With the new parameters, each potential lies in ``[0, 1]``.

        Raises:
            ValueError: If scene entities are omitted or stage budgets are invalid.
        """
        if cable_cfgs is None or robot_cfgs is None:
            raise ValueError("dense_task_reward requires cable_cfgs and robot_cfgs")
        potential_scale = 1.0
        legacy_weights = (approach_weight, grasp_weight, task_weight)
        if maximum_progress_rate is not None or any(weight is not None for weight in legacy_weights):
            if not self._warned_legacy_params:
                warnings.warn(
                    "dense_task_reward: maximum_progress_rate is ignored and the three legacy weights are"
                    " deprecated. Use acquisition_weight and approach_fraction; see the shoelace README.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._warned_legacy_params = True
            if any(weight is not None for weight in legacy_weights):
                approach_budget, grasp_budget, pull_budget = (
                    default if weight is None else weight
                    for weight, default in zip(legacy_weights, (0.15, 0.35, 1.0), strict=True)
                )
                budgets = (approach_budget, grasp_budget, pull_budget)
                potential_scale = sum(budgets)
                if not all(math.isfinite(budget) and budget >= 0.0 for budget in budgets) or potential_scale <= 0.0:
                    raise ValueError("Legacy reward budgets must be finite, nonnegative, and have a positive sum")
                acquisition_budget = approach_budget + grasp_budget
                acquisition_weight = acquisition_budget / potential_scale
                approach_fraction = approach_budget / acquisition_budget if acquisition_budget > 0.0 else 0.0
        if not 0.0 <= acquisition_weight <= 1.0 or not 0.0 <= approach_fraction <= 1.0:
            raise ValueError("acquisition_weight and approach_fraction must be in [0, 1]")

        tail_vectors = tails_to_tcp(env, cable_cfgs, robot_cfgs).reshape(env.num_envs, 2, 3)
        tail_distances = torch.linalg.vector_norm(tail_vectors, dim=-1)
        signed_distance = finger_tail_signed_distance(env).reshape(env.num_envs, 2, 2)
        relative_speed = tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs)
        closure = _gripper_closed_fraction(env, robot_cfgs, open_position, closed_position)
        x_separation = tail_x_separation(env, cable_cfgs)
        finite = (
            torch.isfinite(tail_distances).all(dim=1)
            & torch.isfinite(signed_distance).all(dim=(1, 2))
            & torch.isfinite(relative_speed).all(dim=1)
            & torch.isfinite(closure).all(dim=1)
            & torch.isfinite(x_separation)
        )

        approach = 1.0 - torch.tanh(tail_distances / max(reach_std, 1.0e-6))
        finger_contact = torch.exp(-torch.square(signed_distance / max(contact_std, 1.0e-6)))
        bilateral_finger_contact = _hamacher_product(finger_contact[:, :, 0], finger_contact[:, :, 1])
        per_gripper_grasp = _hamacher_product(bilateral_finger_contact, closure)
        motion_match = 1.0 - torch.tanh(relative_speed / max(relative_speed_std, 1.0e-6))
        per_gripper_grasp = _hamacher_product(per_gripper_grasp, motion_match)

        filter_time = max(grasp_filter_time_constant, 1.0e-6)
        filter_alpha = 1.0 - math.exp(-env.step_dt / filter_time)
        unseeded_grasp = ~torch.isfinite(self._filtered_per_gripper_grasp)
        filtered_per_gripper_grasp = torch.where(
            unseeded_grasp,
            per_gripper_grasp,
            self._filtered_per_gripper_grasp + filter_alpha * (per_gripper_grasp - self._filtered_per_gripper_grasp),
        )
        self._filtered_per_gripper_grasp.copy_(
            torch.where(finite.unsqueeze(1), filtered_per_gripper_grasp, self._filtered_per_gripper_grasp)
        )
        bilateral_grasp = _hamacher_product(filtered_per_gripper_grasp[:, 0], filtered_per_gripper_grasp[:, 1])
        acquire = _hamacher_product(
            approach, approach_fraction + (1.0 - approach_fraction) * filtered_per_gripper_grasp
        ).mean(dim=1)

        unseeded_baseline = ~torch.isfinite(self._baseline_x_separation)
        self._baseline_x_separation.copy_(
            torch.where(finite & unseeded_baseline, x_separation, self._baseline_x_separation)
        )
        x_progress_range = (success_x_separation - self._baseline_x_separation).clamp_min(1.0e-6)
        x_progress = ((x_separation - self._baseline_x_separation) / x_progress_range).clamp(0.0, 1.0)
        pull = _hamacher_product(bilateral_grasp, x_progress)
        potential = potential_scale * (acquisition_weight * acquire + (1.0 - acquisition_weight) * pull)

        valid = finite & torch.isfinite(self._previous_potential)
        progress = torch.where(valid, potential - self._previous_potential, torch.zeros_like(potential))
        self._previous_potential.copy_(torch.where(finite, potential, self._previous_potential))
        return progress / env.step_dt


def _gripper_closed_fraction(
    env: ManagerBasedRLEnv,
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    open_position: float,
    closed_position: float,
) -> torch.Tensor:
    """Return normalized actual gripper closure in ``[0, 1]``."""
    positions = torch.stack(
        [env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids[0]] for robot_cfg in robot_cfgs],
        dim=1,
    )
    return ((open_position - positions) / max(open_position - closed_position, 1.0e-6)).clamp(0.0, 1.0)


def _hamacher_product(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the Hamacher soft-AND of two values in ``[0, 1]``."""
    return (a * b) / (a + b - a * b + eps)
