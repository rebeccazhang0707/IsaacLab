# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .utils import (
    grasp_distances,
    grasp_state,
    gripper_closed_fraction,
    potential,
    pull_directions,
    task_state,
)


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
