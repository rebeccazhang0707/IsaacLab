# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.rewards import is_terminated_term
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .utils import (
    grasp_distances,
    grasp_state,
    gripper_closed_fraction,
    potential,
    pull_directions,
    robot_tcp_position,
    task_state,
    untying_metrics,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class grasp_acquisition_event(ManagerTermBase):
    """Reward each strict per-tail grasp once per episode."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._candidate_steps = torch.zeros((env.num_envs, 2), dtype=torch.int64, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._acquired[selected] = False
        self._candidate_steps[selected] = 0

    def __call__(
        self,
        env,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        side_weights: tuple[float, float],
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
        minimum_finger_position: float = 0.0,
        confirmation_steps: int = 1,
    ) -> torch.Tensor:
        """Return the weighted rate of first-time strict tail acquisitions.

        Args:
            env: The task environment.
            maximum_grasp_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            side_weights: Relative acquisition weights for the left and right robots.
            asset_cfgs: Scene entities for the left and right cable chains.
            left_robot_cfg: Left robot hand and finger scene entity.
            right_robot_cfg: Right robot hand and finger scene entity.
            minimum_finger_position: Minimum driven finger-joint position for acquisition [m].
            confirmation_steps: Consecutive control steps required to confirm acquisition.

        Returns:
            Per-environment weighted acquisition-event rate.
        """
        grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
            minimum_finger_position,
        )
        self._candidate_steps.copy_(
            torch.where(grasped, self._candidate_steps + 1, torch.zeros_like(self._candidate_steps))
        )
        confirmed = self._candidate_steps >= max(int(confirmation_steps), 1)
        newly_acquired = confirmed & (~self._acquired)
        self._acquired |= confirmed
        weights = newly_acquired.new_tensor(side_weights, dtype=torch.float)
        return (
            torch.sum(newly_acquired.float() * weights, dim=1)
            / max(sum(side_weights), 1.0e-6)
            / max(env.step_dt, 1.0e-6)
        )


def second_tail_coordination(
    env: ManagerBasedRLEnv,
    reach_std: float,
    grasp_std: float,
    open_position: float,
    closed_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward approaching either tail while the other tail remains acquired.

    Args:
        env: The task environment.
        reach_std: Approach-distance tanh width [m].
        grasp_std: Grasp-proximity tanh width [m].
        open_position: Driven finger-joint position when open [m].
        closed_position: Driven finger-joint position when closed [m].
        asset_cfgs: Scene entities for the left and right cable chains.
        left_robot_cfg: Left robot hand and finger scene entity.
        right_robot_cfg: Right robot hand and finger scene entity.

    Returns:
        Per-environment coordination score in the interval ``[0, 1]``.
    """
    distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    closure = gripper_closed_fraction(
        env,
        open_position,
        closed_position,
        left_robot_cfg,
        right_robot_cfg,
    )
    reach = 1.0 - torch.tanh(distances / max(reach_std, 1.0e-6))
    grasp_proximity = 1.0 - torch.tanh(distances / max(grasp_std, 1.0e-6))
    acquisition = _hamacher_product(grasp_proximity, closure)
    cross_tail_scores = torch.stack(
        (
            _hamacher_product(reach[:, 0], acquisition[:, 1]),
            _hamacher_product(reach[:, 1], acquisition[:, 0]),
        ),
        dim=1,
    )
    score = cross_tail_scores.amax(dim=1)
    finite = torch.isfinite(distances).all(dim=1) & torch.isfinite(closure).all(dim=1)
    return torch.where(finite, score, torch.zeros_like(score))


class second_tail_approach_progress(ManagerTermBase):
    """Reward approaching the remaining tail after one strict acquisition."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_distances = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous_distances[selected] = torch.nan
        self._acquired[selected] = False

    def __call__(
        self,
        env,
        std: float,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        open_position: float,
        closed_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return normalized progress toward the unacquired tail.

        Args:
            env: The task environment.
            std: Distance-improvement normalization width [m].
            maximum_grasp_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            asset_cfgs: Scene entities for the left and right cable chains.
            left_robot_cfg: Left robot hand and finger scene entity.
            right_robot_cfg: Right robot hand and finger scene entity.

        Returns:
            Per-environment progress toward the remaining tail after exactly one acquisition.
        """
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        open_fraction = 1.0 - gripper_closed_fraction(
            env,
            open_position,
            closed_position,
            left_robot_cfg,
            right_robot_cfg,
        )
        grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        )
        self._acquired |= grasped

        finite = torch.isfinite(distances) & torch.isfinite(open_fraction)
        seeded = torch.isfinite(self._previous_distances)
        progress = torch.where(
            finite & seeded,
            self._previous_distances - distances,
            torch.zeros_like(distances),
        )
        self._previous_distances.copy_(torch.where(torch.isfinite(distances), distances, self._previous_distances))
        remaining_tail = torch.stack(
            (
                self._acquired[:, 1] & (~self._acquired[:, 0]),
                self._acquired[:, 0] & (~self._acquired[:, 1]),
            ),
            dim=1,
        )
        normalized_progress = progress / max(std, 1.0e-6) * open_fraction
        return torch.sum(torch.where(finite & remaining_tail, normalized_progress, 0.0), dim=1)


class remaining_tail_grasping(ManagerTermBase):
    """Reward closing and retaining every gripper after its strict tail acquisition."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._ready = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._acquired[selected] = False
        self._ready[selected] = False

    def __call__(
        self,
        env,
        std: float,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        open_position: float,
        closed_position: float,
        command_temperature: float,
        command_weight: float,
        gripper_action_names: tuple[str, str],
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return normalized close-command retention for acquired grippers.

        Args:
            env: The task environment.
            std: Grasp-proximity exponential width [m].
            maximum_grasp_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            command_temperature: Binary-action margin scale used for smooth closure shaping.
            command_weight: Relative weight of the current close-command margin in the grasp score.
            gripper_action_names: Names of the left and right gripper action terms.
            asset_cfgs: Scene entities for the left and right cable chains.
            left_robot_cfg: Left robot hand and finger scene entity.
            right_robot_cfg: Right robot hand and finger scene entity.

        Returns:
            Per-environment acquired-gripper retention score in the interval ``[0, 1]``.
        """
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        closure = gripper_closed_fraction(
            env,
            open_position,
            closed_position,
            left_robot_cfg,
            right_robot_cfg,
        )
        raw_actions = torch.cat(
            tuple(env.action_manager.get_term(name).raw_actions for name in gripper_action_names),
            dim=1,
        )
        command_closure = torch.sigmoid(-raw_actions / max(command_temperature, 1.0e-6))
        grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        )
        self._acquired |= grasped
        finite = torch.isfinite(distances) & torch.isfinite(closure) & torch.isfinite(command_closure)
        self._ready |= finite & grasped
        proximity = torch.exp(-torch.square(distances / max(std, 1.0e-6)))
        retained_proximity = torch.maximum(proximity, grasped.float())
        bounded_command_weight = min(max(command_weight, 0.0), 1.0)
        grasp_score = (1.0 - bounded_command_weight) * closure + bounded_command_weight * command_closure
        per_gripper_score = torch.where(finite & self._ready, retained_proximity * grasp_score, 0.0)
        ready_count = self._ready.sum(dim=1).clamp_min(1)
        score = torch.sum(per_gripper_score, dim=1) / ready_count
        return torch.where(finite.all(dim=1), score, torch.zeros_like(score))


class untying_progress(ManagerTermBase):
    """Reward the rate of net two-tail separation and knot-throat clearance after acquisition."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._acquired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous_potential[selected] = torch.nan
        self._acquired[selected] = False

    def __call__(
        self,
        env,
        throat_radius: float,
        tail_success_distance: float,
        tail_success_separation: float,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        maximum_progress_rate: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return finite untying-potential improvement per second while both tails are grasped."""
        positions, _, knot, tail_positions, _ = task_state(env, asset_cfgs)
        current = potential(
            positions,
            knot,
            tail_positions,
            throat_radius,
            tail_success_distance,
            tail_success_separation,
        )
        finite = torch.isfinite(current)
        unseeded = ~torch.isfinite(self._previous_potential)
        progress = torch.where(finite & (~unseeded), current - self._previous_potential, torch.zeros_like(current))
        self._previous_potential.copy_(torch.where(finite, current, self._previous_potential))
        both_grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        ).all(dim=1)
        self._acquired |= finite & both_grasped
        progress_rate = progress / max(env.step_dt, 1.0e-6)
        bounded_maximum_rate = max(maximum_progress_rate, 0.0)
        progress_rate = progress_rate.clamp(min=-bounded_maximum_rate, max=bounded_maximum_rate)
        return progress_rate * self._acquired


class acquired_grasp_retention(ManagerTermBase):
    """Reward retaining each strict grasp after its acquisition event."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._acquired[selected] = False

    def __call__(
        self,
        env,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return the fraction of previously acquired tails that remain strictly grasped.

        Args:
            env: The task environment.
            maximum_grasp_distance: Maximum tail-to-TCP grasp distance [m].
            maximum_finger_position: Maximum driven finger-joint position for grasping [m].
            asset_cfgs: Scene entities for the left and right cable chains.
            left_robot_cfg: Left robot hand and finger scene entity.
            right_robot_cfg: Right robot hand and finger scene entity.

        Returns:
            Per-environment retained-grasp fraction in the interval ``[0, 1]``.
        """
        grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        )
        retention = (grasped & self._acquired).float().mean(dim=1)
        self._acquired |= grasped
        return retention


class tail_approach_progress(ManagerTermBase):
    """Reward stepwise reductions in open-gripper tail distance."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_distances = torch.full((env.num_envs, 2), torch.nan, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous_distances[selected] = torch.nan

    def __call__(
        self,
        env,
        std: float,
        open_position: float,
        closed_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
        side_weights: tuple[float, float] = (1.0, 1.0),
    ) -> torch.Tensor:
        """Return weighted, normalized tail-distance improvement for open grippers."""
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        open_fraction = 1.0 - gripper_closed_fraction(
            env,
            open_position,
            closed_position,
            left_robot_cfg,
            right_robot_cfg,
        )
        finite = torch.isfinite(distances) & torch.isfinite(open_fraction)
        unseeded = ~torch.isfinite(self._previous_distances)
        progress = torch.where(
            finite & (~unseeded),
            self._previous_distances - distances,
            torch.zeros_like(distances),
        )
        self._previous_distances.copy_(torch.where(torch.isfinite(distances), distances, self._previous_distances))
        normalized_progress = torch.where(
            finite,
            progress / max(std, 1.0e-6) * open_fraction,
            torch.zeros_like(progress),
        )
        weights = normalized_progress.new_tensor(side_weights)
        return torch.sum(normalized_progress * weights, dim=1) / max(sum(side_weights), 1.0e-6)


class premature_close_event(ManagerTermBase):
    """Penalize closing motion away from a tail before its first acquisition."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_closure = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous_closure[selected] = torch.nan
        self._acquired[selected] = False

    def __call__(
        self,
        env,
        acquisition_distance: float,
        maximum_finger_position: float,
        open_position: float,
        closed_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return closure increments made away from a tail before acquisition.

        Args:
            env: The task environment.
            acquisition_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            asset_cfgs: Scene entities for the left and right cable chains.
            left_robot_cfg: Left robot hand and finger scene entity.
            right_robot_cfg: Right robot hand and finger scene entity.

        Returns:
            Per-environment premature closure increment averaged over both grippers.
        """
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        closure = gripper_closed_fraction(
            env,
            open_position,
            closed_position,
            left_robot_cfg,
            right_robot_cfg,
        )
        self._acquired |= grasp_state(
            env,
            acquisition_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        )
        unseeded = torch.isnan(self._previous_closure)
        closing = torch.where(
            unseeded,
            torch.zeros_like(closure),
            (closure - self._previous_closure).clamp(min=0.0),
        )
        self._previous_closure.copy_(closure)
        excess_distance = (distances / max(acquisition_distance, 1.0e-6) - 1.0).clamp(0.0, 1.0)
        penalty = (closing * excess_distance * (~self._acquired)).mean(dim=1)
        finite = torch.isfinite(distances).all(dim=1) & torch.isfinite(closure).all(dim=1)
        return torch.where(finite, penalty, torch.zeros_like(penalty))


class termination_event_reward(is_terminated_term):
    """Express selected non-timeout termination rewards as per-event impulses."""

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        """Return selected event counts divided by the environment step interval."""
        return super().__call__(env, term_keys) / env.step_dt


def finite_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    maximum_penalty: float = 100.0,
) -> torch.Tensor:
    """Penalize squared joint velocities without propagating terminal-state outliers.

    Args:
        env: The task environment.
        asset_cfg: Articulation and joint selection to penalize.
        maximum_penalty: Maximum per-step squared-velocity penalty.

    Returns:
        Bounded per-environment squared joint-velocity penalty.

    The joint velocities use [m/s or rad/s, depending on joint type]. The penalty has units
    [(m/s)^2 or (rad/s)^2, depending on joint type].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)
    bounded_penalty = penalty.clamp_max(max(maximum_penalty, 0.0))
    return torch.where(torch.isfinite(penalty), bounded_penalty, torch.zeros_like(penalty))


def shoelace_dense_reward(
    env: ManagerBasedRLEnv,
    reach_std: float,
    grasp_std: float,
    throat_radius: float,
    maximum_throat_segments: int,
    tail_success_distance: float,
    tail_success_separation: float,
    target_speed: float,
    open_position: float,
    closed_position: float,
    approach_weight: float,
    acquisition_weight: float,
    task_weight: float,
    pull_weight: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return bounded approach, acquisition, and untying reward composed by soft-AND.

    Args:
        env: The task environment.
        reach_std: Approach-distance tanh width [m].
        grasp_std: Grasp-proximity tanh width [m].
        throat_radius: Radius used to classify segments inside the knot throat [m].
        maximum_throat_segments: Maximum throat segment count at success.
        tail_success_distance: Per-tail distance from the knot at success [m].
        tail_success_separation: Tail-to-tail separation at success [m].
        target_speed: Pull speed that sets the directional bonus scale [m/s].
        open_position: Driven finger-joint position when open [m].
        closed_position: Driven finger-joint position when closed [m].
        approach_weight: Relative weight of partial approach credit.
        acquisition_weight: Relative weight of partial tail-acquisition credit.
        task_weight: Relative weight of the grasp-gated geometric goal.
        pull_weight: Relative weight of positive directional pulling.
        asset_cfgs: Scene entities for the left and right cable chains.
        left_robot_cfg: Left robot hand and finger scene entity.
        right_robot_cfg: Right robot hand and finger scene entity.

    Returns:
        Per-environment dense reward bounded by the sum of the four non-negative weights.
    """
    positions, _, knot, tail_positions, tail_velocities = task_state(env, asset_cfgs)
    tcp_positions = torch.stack(
        (
            robot_tcp_position(env, left_robot_cfg),
            robot_tcp_position(env, right_robot_cfg),
        ),
        dim=1,
    )
    distances = torch.linalg.vector_norm(tail_positions - tcp_positions, dim=-1)
    closure = gripper_closed_fraction(
        env,
        open_position,
        closed_position,
        left_robot_cfg,
        right_robot_cfg,
    )
    finite = (
        torch.isfinite(positions).all(dim=(1, 2))
        & torch.isfinite(tail_positions).all(dim=(1, 2))
        & torch.isfinite(tail_velocities).all(dim=(1, 2))
        & torch.isfinite(tcp_positions).all(dim=(1, 2))
        & torch.isfinite(closure).all(dim=1)
    )

    reach = 1.0 - torch.tanh(distances / max(reach_std, 1.0e-6))
    grasp_proximity = 1.0 - torch.tanh(distances / max(grasp_std, 1.0e-6))
    acquisition = _hamacher_product(grasp_proximity, closure)
    approach_score = reach.mean(dim=1)
    per_tail_acquisition = _hamacher_product(reach, acquisition)
    acquisition_score = _hamacher_product(per_tail_acquisition[:, 0], per_tail_acquisition[:, 1])
    both_acquired = _hamacher_product(acquisition[:, 0], acquisition[:, 1])

    throat_count, tail_distances, tail_separation = untying_metrics(
        positions,
        knot,
        tail_positions,
        throat_radius,
    )
    free_segment_count = positions.shape[1] - 2
    throat_range = max(free_segment_count - maximum_throat_segments, 1)
    throat_goal = ((free_segment_count - throat_count.float()) / throat_range).clamp(0.0, 1.0)
    tail_distance_goal = (tail_distances / max(tail_success_distance, 1.0e-6)).clamp(0.0, 1.0)
    both_tails_distant = _hamacher_product(tail_distance_goal[:, 0], tail_distance_goal[:, 1])
    separation_goal = (tail_separation / max(tail_success_separation, 1.0e-6)).clamp(0.0, 1.0)
    tail_goal = _hamacher_product(both_tails_distant, separation_goal)
    geometry_goal = _hamacher_product(throat_goal, tail_goal)
    task_score = _hamacher_product(both_acquired, geometry_goal)

    projected_speed = torch.sum(tail_velocities * pull_directions(tail_velocities), dim=-1)
    positive_pull = torch.tanh(projected_speed / max(target_speed, 1.0e-6)).clamp(min=0.0)
    per_tail_pull = positive_pull * acquisition
    pull_score = _hamacher_product(per_tail_pull[:, 0], per_tail_pull[:, 1])

    reward = (
        approach_weight * approach_score
        + acquisition_weight * acquisition_score
        + task_weight * task_score
        + pull_weight * pull_score
    )
    # Terminations are evaluated before rewards and resets, so invalid terminal state can reach this term once.
    return torch.where(finite, reward, torch.zeros_like(reward))


def tail_reaching(
    env,
    std: float,
    open_position: float,
    closed_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward approaching each tail while its assigned gripper remains open."""
    distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    open_fraction = 1.0 - gripper_closed_fraction(
        env,
        open_position,
        closed_position,
        left_robot_cfg,
        right_robot_cfg,
    )
    return (torch.exp(-torch.square(distances / std)) * open_fraction).mean(dim=1)


def tail_grasping(
    env,
    std: float,
    maximum_distance: float,
    maximum_finger_position: float,
    open_position: float,
    closed_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward closing near each tail and bonus completed inferred grasps."""
    distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    closure = gripper_closed_fraction(
        env,
        open_position,
        closed_position,
        left_robot_cfg,
        right_robot_cfg,
    )
    proximity = torch.exp(-torch.square(distances / std))
    grasped = grasp_state(
        env,
        maximum_distance,
        maximum_finger_position,
        asset_cfgs,
        left_robot_cfg,
        right_robot_cfg,
    )
    return (0.5 * proximity * closure + 0.5 * grasped.float()).mean(dim=1)


def closing_away_from_tails(
    env,
    acquisition_distance: float,
    open_position: float,
    closed_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize closing a gripper before its TCP reaches the assigned tail."""
    distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
    closure = gripper_closed_fraction(
        env,
        open_position,
        closed_position,
        left_robot_cfg,
        right_robot_cfg,
    )
    excess_distance = (distances / acquisition_distance - 1.0).clamp(0.0, 1.0)
    return (closure * excess_distance).mean(dim=1)


def directional_tail_pull(
    env,
    target_speed: float,
    maximum_grasp_distance: float,
    maximum_finger_position: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward grasped-tail velocity projected along the demo force directions."""
    _, _, _, _, tail_velocities = task_state(env, asset_cfgs)
    projected_speed = torch.sum(tail_velocities * pull_directions(tail_velocities), dim=-1)
    grasped = grasp_state(
        env,
        maximum_grasp_distance,
        maximum_finger_position,
        asset_cfgs,
        left_robot_cfg,
        right_robot_cfg,
    )
    return (torch.tanh(projected_speed / target_speed) * grasped).mean(dim=1)


def _hamacher_product(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the Hamacher soft-AND of two values in ``[0, 1]``."""
    return (a * b) / (a + b - a * b + eps)
