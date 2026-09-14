# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the dual-Franka shoelace task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .observations import finger_tail_signed_distance, tail_tcp_relative_speed, tails_to_tcp
from .utils import tail_outward_x, tail_x_separation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg


def arm_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared normalized arm commands, excluding binary gripper commands.

    Args:
        env: Task environment with ``left_arm`` and ``right_arm`` action terms.

    Returns:
        Sum of squared arm commands before Cartesian scaling, shape [N].
    """
    return _arm_action_squared_sum(env, env.action_manager.action)


def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared changes in normalized arm commands, excluding gripper changes.

    Args:
        env: Task environment with ``left_arm`` and ``right_arm`` action terms.

    Returns:
        Sum of squared differences from the previous policy step, shape [N]. This is not divided by
        the step duration. The action manager clears history to zero on reset.
    """
    delta = env.action_manager.action - env.action_manager.prev_action
    return _arm_action_squared_sum(env, delta)


def shoelace_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the geometric success flag as an event rate [1/s].

    The reward manager cancels ``step_dt``, so its weight is the reward per successful episode.
    The environment resets after success; timeout alone earns no completion reward.

    Args:
        env: Task environment with a ``success`` termination term.

    Returns:
        Success event rates [1/s], shape [N].
    """
    return env.termination_manager.get_term("success").float() / env.step_dt


class dense_task_reward(ManagerTermBase):
    """Stateful progress reward for acquiring and pulling the two free tails.

    Maintains per-environment grasp filters, initial tail offsets, and the previous potential.

    Notes:
        - Phase metrics under ``Metrics/shoelace/`` average only environments with finite reward inputs.
        - ``valid_fraction`` measures numerical input validity, not grasp quality or task success.
        - ``success_rate`` averages the latest completed result per environment, excluding those with
          no completed episode. It is zero until the first completion.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_per_gripper_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._baseline_outward_x = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._last_episode_success = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._metrics: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear episode state so the next valid evaluation gives no reset credit.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        selected = slice(None) if env_ids is None else env_ids
        self._filtered_per_gripper_grasp[selected] = torch.nan
        self._baseline_outward_x[selected] = torch.nan
        self._previous_potential[selected] = torch.nan
        # ManagerBasedRLEnv replaces the log dictionary before resetting reward terms.
        self._env.extras.setdefault("log", {}).update(self._metrics)

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
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        *,
        acquisition_weight: float = 0.6,
        approach_fraction: float = 0.5,
        bilateral_pull_fraction: float = 0.8,
        bilateral_approach_fraction: float = 0.5,
        bilateral_grasp_fraction: float = 0.7,
    ) -> torch.Tensor:
        """Return the signed rate of acquisition-and-pull progress [1/s].

        Reward composition:
            1. Acquisition: independent approach and grasp credit plus bilateral bonuses. Each grasp
               combines two-finger contact, actual closure, and low tail-TCP slip, independently of proximity.
            2. Pulling: each tail's outward progress gated by its own grasp, plus a bilateral bonus.
            3. Feedback: signed potential difference divided by ``step_dt``; holding still pays zero
               once filters settle. The reward manager multiplies by ``step_dt`` and the term weight.

        Notes:
            - Reset: the first finite sample seeds state and returns zero; regrasping never resets references.
            - Pull progress: starts at 0.5 and changes smoothly on either side of the initial tail position.
              This also gives grasp credit at the initial position. Equations are in the task README.
            - Invalid inputs: return zero and preserve reward history; excluded from phase metric averages.
            - Cycles: signed gains and losses cancel over a full reward-state cycle without discounting.
              This does not guarantee policy invariance under discounting.

        Args:
            env: The task environment.
            reach_std: Tail-to-TCP approach-distance width [m].
            contact_std: Contact-distance width [m]; gaps and deep penetration reduce grasp quality.
            relative_speed_std: Tail-TCP slip-speed width [m/s].
            grasp_filter_time_constant: Grasp-quality low-pass time constant [s].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            success_x_separation: X-separation target for the pull scale [m]; success is checked separately.
            cable_cfgs: Required left and right cable scene entities.
            robot_cfgs: Required left and right robot hand and finger scene entities.
            acquisition_weight: Acquisition fraction in [0, 1]; pulling receives the remainder.
            approach_fraction: Approach share of acquisition in [0, 1]; grasps receive the remainder.
            bilateral_pull_fraction: Bilateral share of pulling in [0, 1]; zero disables the bonus.
            bilateral_approach_fraction: Bilateral share of approach in [0, 1]; zero gives the per-arm mean.
            bilateral_grasp_fraction: Bilateral share of grasp acquisition in [0, 1]; zero gives the per-arm mean.

        Returns:
            Signed reward rates [1/s], shape [N]. The potential is bounded in [0, 1].

        Raises:
            ValueError: If scene entities, reward budgets, or the separation target are invalid.
        """
        if cable_cfgs is None or robot_cfgs is None:
            raise ValueError("dense_task_reward requires cable_cfgs and robot_cfgs")
        if not 0.0 <= acquisition_weight <= 1.0 or not 0.0 <= approach_fraction <= 1.0:
            raise ValueError("acquisition_weight and approach_fraction must be in [0, 1]")
        if not all(
            0.0 <= value <= 1.0
            for value in (bilateral_pull_fraction, bilateral_approach_fraction, bilateral_grasp_fraction)
        ):
            raise ValueError("Bilateral approach, grasp, and pull fractions must be in [0, 1]")
        if not math.isfinite(success_x_separation) or success_x_separation <= 0.0:
            raise ValueError("success_x_separation must be finite and positive")

        # Per-arm tensors follow robot order (left, right), which is opposite to the cable naming.
        tail_vectors = tails_to_tcp(env, cable_cfgs, robot_cfgs).reshape(env.num_envs, 2, 3)
        tail_distances = torch.linalg.vector_norm(tail_vectors, dim=-1)
        per_gripper_grasp, grasp_finite = _grasp_quality(
            env, contact_std, relative_speed_std, open_position, closed_position, cable_cfgs, robot_cfgs
        )
        x_separation = tail_x_separation(env, cable_cfgs)
        outward_x = tail_outward_x(env, cable_cfgs)
        # Per-environment numerical mask for reward inputs; this does not assess grasp or task success.
        finite = (
            torch.isfinite(tail_distances).all(dim=1)
            & grasp_finite
            & torch.isfinite(x_separation)
            & torch.isfinite(outward_x).all(dim=1)
        )

        approach = 1.0 - torch.tanh(tail_distances / max(reach_std, 1.0e-6))
        filtered_per_gripper_grasp = _filter_grasps(
            self._filtered_per_gripper_grasp, per_gripper_grasp, finite, env.step_dt, grasp_filter_time_constant
        )
        bilateral_grasp = _hamacher_product(filtered_per_gripper_grasp[:, 0], filtered_per_gripper_grasp[:, 1])
        # Independent credit starts either arm; cooperation makes acquiring the other arm more valuable.
        approach_score = _bilateral_score(approach, bilateral_approach_fraction)
        grasp_score = _bilateral_score(filtered_per_gripper_grasp, bilateral_grasp_fraction)
        acquire = approach_fraction * approach_score + (1.0 - approach_fraction) * grasp_score

        # Keep each reference fixed until episode reset; releasing/regrasping must not renew pull credit.
        unseeded_baseline = ~torch.isfinite(self._baseline_outward_x)
        self._baseline_outward_x.copy_(
            torch.where(finite.unsqueeze(1) & unseeded_baseline, outward_x, self._baseline_outward_x)
        )
        initial_separation = self._baseline_outward_x.sum(dim=1).abs()
        # Split the remaining target separation between arms; the 1 cm floor avoids a near-zero scale.
        pull_scale = (0.5 * (success_x_separation - initial_separation)).clamp_min(0.01)
        # Start at 0.5 so outward motion below the reset baseline still changes the potential smoothly.
        per_arm_progress = 0.5 * (1.0 + torch.tanh((outward_x - self._baseline_outward_x) / pull_scale.unsqueeze(1)))
        # Gate each tail by its own grasp; the bilateral term is a bonus, not a prerequisite for pulling.
        per_arm_pull = _hamacher_product(filtered_per_gripper_grasp, per_arm_progress)
        pull = _bilateral_score(per_arm_pull, bilateral_pull_fraction)
        potential = acquisition_weight * acquire + (1.0 - acquisition_weight) * pull

        metric_values = {
            "approach_distance_m": tail_distances.mean(dim=1),
            "grasp_left": filtered_per_gripper_grasp[:, 0],
            "grasp_right": filtered_per_gripper_grasp[:, 1],
            "grasp_both": bilateral_grasp,
            "pull_x_separation_m": x_separation,
            "pull_left": per_arm_pull[:, 0],
            "pull_right": per_arm_pull[:, 1],
        }
        samples = torch.stack(tuple(metric_values.values()), dim=-1)
        means = torch.where(finite.unsqueeze(1), samples, 0.0).sum(dim=0) / finite.sum().clamp_min(1)
        self._metrics = {
            f"Metrics/shoelace/{name}": value.detach() for name, value in zip(metric_values, means, strict=True)
        }
        # Fraction of environments passing the NaN/Inf check this step; normally 1.0.
        self._metrics["Metrics/shoelace/valid_fraction"] = finite.float().mean()
        # Retain each environment's latest completed result; exclude environments with no finished episode.
        self._last_episode_success.copy_(
            torch.where(
                env.termination_manager.dones,
                env.termination_manager.get_term("success").float(),
                self._last_episode_success,
            )
        )
        self._metrics["Metrics/shoelace/success_rate"] = self._last_episode_success.nan_to_num().sum() / (
            torch.isfinite(self._last_episode_success).sum().clamp_min(1)
        )
        # RSL-RL retains each step's dictionary until logging the training iteration.
        env.extras["log"] = {**env.extras.get("log", {}), **self._metrics}

        # Seed without reset credit, and keep negative changes so full-state cycles cancel without discounting.
        valid = finite & torch.isfinite(self._previous_potential)
        progress = torch.where(valid, potential - self._previous_potential, torch.zeros_like(potential))
        self._previous_potential.copy_(torch.where(finite, potential, self._previous_potential))
        # RewardManager multiplies by step_dt, leaving the weighted potential difference per policy step.
        return progress / env.step_dt


class grasp_hold_reward(ManagerTermBase):
    """Reward retained physical grasps, with an independent filter for hold-only ablations.

    Unlike the dense potential difference, this score remains positive during stable grasping.
    The manager multiplies it by ``step_dt`` and its weight, making the weight a maximum reward
    per second. Invalid contact, closure, or slip inputs earn zero without advancing the filter.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_per_gripper_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the grasp filter for the selected environments.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        self._filtered_per_gripper_grasp[slice(None) if env_ids is None else env_ids] = torch.nan

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_std: float,
        relative_speed_std: float,
        grasp_filter_time_constant: float,
        open_position: float,
        closed_position: float,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        bilateral_grasp_fraction: float = 0.8,
    ) -> torch.Tensor:
        """Return sustained grasp quality with a bilateral bonus.

        Args:
            env: Task environment.
            contact_std: Finger-tail contact-distance width [m].
            relative_speed_std: Tail-TCP slip-speed width [m/s].
            grasp_filter_time_constant: Grasp-quality low-pass time constant [s].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            cable_cfgs: Left and right cable scene entities.
            robot_cfgs: Left and right hand and finger scene entities.
            bilateral_grasp_fraction: Bilateral share in [0, 1]; the remainder rewards each arm independently.

        Returns:
            Hold scores in [0, 1], shape [N]. The first finite sample seeds the filter without a ramp.

        Raises:
            ValueError: If the bilateral fraction is outside [0, 1].
        """
        if not 0.0 <= bilateral_grasp_fraction <= 1.0:
            raise ValueError("bilateral_grasp_fraction must be in [0, 1]")
        grasp, finite = _grasp_quality(
            env, contact_std, relative_speed_std, open_position, closed_position, cable_cfgs, robot_cfgs
        )
        filtered = _filter_grasps(
            self._filtered_per_gripper_grasp, grasp, finite, env.step_dt, grasp_filter_time_constant
        )
        return torch.where(finite, _bilateral_score(filtered, bilateral_grasp_fraction), 0.0)


def _arm_action_squared_sum(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    """Select arm commands by name so action reordering cannot include grippers."""
    term_slices: dict[str, slice] = {}
    start = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        term_slices[name] = slice(start, start + dim)
        start += dim
    return actions[:, term_slices["left_arm"]].square().sum(dim=-1) + actions[:, term_slices["right_arm"]].square().sum(
        dim=-1
    )


def _grasp_quality(
    env: ManagerBasedRLEnv,
    contact_std: float,
    relative_speed_std: float,
    open_position: float,
    closed_position: float,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share physical grasp criteria between acquisition and independently evaluated retention."""
    signed_distance = finger_tail_signed_distance(env).reshape(env.num_envs, 2, 2)
    relative_speed = tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs)
    closure = _gripper_closed_fraction(env, robot_cfgs, open_position, closed_position)
    finite = (
        torch.isfinite(signed_distance).all(dim=(1, 2))
        & torch.isfinite(relative_speed).all(dim=1)
        & torch.isfinite(closure).all(dim=1)
    )
    # Both gaps and deep penetration reduce quality; closure alone cannot establish a grasp.
    finger_contact = torch.exp(-torch.square(signed_distance / max(contact_std, 1.0e-6)))
    contact = _hamacher_product(finger_contact[:, :, 0], finger_contact[:, :, 1])
    grasp = _hamacher_product(contact, closure)
    motion_match = 1.0 - torch.tanh(relative_speed / max(relative_speed_std, 1.0e-6))
    return _hamacher_product(grasp, motion_match), finite


def _filter_grasps(
    previous: torch.Tensor, grasp: torch.Tensor, finite: torch.Tensor, step_dt: float, time_constant: float
) -> torch.Tensor:
    """Update the filter in place, seeding from finite samples and preserving invalid environments."""
    alpha = 1.0 - math.exp(-step_dt / max(time_constant, 1.0e-6))
    filtered = torch.where(torch.isfinite(previous), previous + alpha * (grasp - previous), grasp)
    previous.copy_(torch.where(finite.unsqueeze(1), filtered, previous))
    return previous


def _bilateral_score(per_arm_score: torch.Tensor, bilateral_fraction: float) -> torch.Tensor:
    """Combine independent credit with a symmetric cooperation bonus."""
    bilateral = _hamacher_product(per_arm_score[:, 0], per_arm_score[:, 1])
    return (1.0 - bilateral_fraction) * per_arm_score.mean(dim=1) + bilateral_fraction * bilateral


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


def _hamacher_product(a: torch.Tensor, b: torch.Tensor | float, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the Hamacher soft-AND of two values in ``[0, 1]``."""
    return (a * b) / (a + b - a * b + eps)
