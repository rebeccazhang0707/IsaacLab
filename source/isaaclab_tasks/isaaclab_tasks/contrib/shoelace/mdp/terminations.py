# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .utils import grasp_distances, grasp_state, task_state, untying_metrics


def shoelace_unsafe(
    env,
    minimum_lace_height: float,
    maximum_lace_spread: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Terminate non-finite cables and cables outside the safe workspace."""
    positions, _, _, _, _ = task_state(env, asset_cfgs)
    finite = torch.isfinite(positions).all(dim=(1, 2))
    spread = torch.linalg.vector_norm(positions - positions.mean(dim=1, keepdim=True), dim=-1).amax(dim=1)
    return (~finite) | (positions[..., 2].amin(dim=1) < minimum_lace_height) | (spread > maximum_lace_spread)


class lost_grasp(ManagerTermBase):
    """Terminate when a previously acquired tail leaves its assigned TCP."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._acquired[selected] = False

    def __call__(
        self,
        env,
        acquisition_distance: float,
        maximum_finger_position: float,
        maximum_grasp_distance: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return per-environment grasp-loss flags after acquisition."""
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        self._acquired |= grasp_state(
            env,
            acquisition_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
        )
        return (~torch.isfinite(distances).all(dim=1)) | (self._acquired & (distances > maximum_grasp_distance)).any(
            dim=1
        )


def shoelace_success(
    env,
    throat_radius: float,
    maximum_throat_segments: int,
    tail_success_distance: float,
    tail_success_separation: float,
    maximum_success_grasp_distance: float,
    maximum_finger_position: float,
    minimum_lace_height: float,
    maximum_lace_spread: float,
    asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    left_robot_cfg: SceneEntityCfg,
    right_robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Detect a cleared knot throat while both robots retain and separate the free tails."""
    positions, _, knot, tail_positions, _ = task_state(env, asset_cfgs)
    throat_count, tail_distances, tail_separation = untying_metrics(
        positions,
        knot,
        tail_positions,
        throat_radius,
    )
    grasped = grasp_state(
        env,
        maximum_success_grasp_distance,
        maximum_finger_position,
        asset_cfgs,
        left_robot_cfg,
        right_robot_cfg,
    ).all(dim=1)
    unsafe = shoelace_unsafe(env, minimum_lace_height, maximum_lace_spread, asset_cfgs)
    return (
        (throat_count <= maximum_throat_segments)
        & (tail_distances.amin(dim=1) >= tail_success_distance)
        & (tail_separation >= tail_success_separation)
        & grasped
        & (~unsafe)
    )
