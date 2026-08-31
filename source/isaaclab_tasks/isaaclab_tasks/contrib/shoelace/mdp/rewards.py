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
    from isaaclab.envs import ManagerBasedRLEnv


class untying_progress(ManagerTermBase):
    """Reward increases in two-tail separation and knot-throat clearance."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous_potential[selected] = torch.nan

    def __call__(
        self,
        env,
        throat_radius: float,
        tail_success_distance: float,
        tail_success_separation: float,
        maximum_grasp_distance: float,
        maximum_finger_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return untying-potential improvement while both tails are grasped."""
        positions, _, knot, tail_positions, _ = task_state(env, asset_cfgs)
        current = potential(
            positions,
            knot,
            tail_positions,
            throat_radius,
            tail_success_distance,
            tail_success_separation,
        )
        unseeded = torch.isnan(self._previous_potential)
        progress = torch.where(unseeded, torch.zeros_like(current), current - self._previous_potential)
        self._previous_potential.copy_(current)
        both_grasped = grasp_state(
            env,
            maximum_grasp_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        ).all(dim=1)
        return progress * both_grasped


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
    ) -> torch.Tensor:
        """Return normalized tail-distance improvement for open grippers."""
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        unseeded = torch.isnan(self._previous_distances)
        progress = torch.where(unseeded, torch.zeros_like(distances), self._previous_distances - distances)
        self._previous_distances.copy_(distances)
        open_fraction = 1.0 - gripper_closed_fraction(
            env,
            open_position,
            closed_position,
            left_robot_cfg,
            right_robot_cfg,
        )
        return (progress / std * open_fraction).mean(dim=1)


class termination_event_reward(is_terminated_term):
    """Express selected non-timeout termination rewards as per-event impulses."""

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        """Return selected event counts divided by the environment step interval."""
        return super().__call__(env, term_keys) / env.step_dt


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

    reach = 1.0 - torch.tanh(distances / max(reach_std, 1.0e-6))
    grasp_proximity = 1.0 - torch.tanh(distances / max(grasp_std, 1.0e-6))
    acquisition = _hamacher_product(grasp_proximity, closure)
    approach_score = reach.mean(dim=1)
    acquisition_score = _hamacher_product(reach, acquisition).mean(dim=1)
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
    pull_score = (positive_pull * acquisition).mean(dim=1)

    return (
        approach_weight * approach_score
        + acquisition_weight * acquisition_score
        + task_weight * task_score
        + pull_weight * pull_score
    )


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
