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
    left_robot_cfg: SceneEntityCfg | None = None,
    right_robot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate non-finite task state and cables outside the safe workspace."""
    positions, velocities, _, _, _ = task_state(env, asset_cfgs)
    finite = torch.isfinite(positions).all(dim=(1, 2)) & torch.isfinite(velocities).all(dim=(1, 2))
    for robot_cfg in (left_robot_cfg, right_robot_cfg):
        if robot_cfg is None:
            continue
        robot = env.scene[robot_cfg.name]
        for state in (
            robot.data.joint_pos.torch,
            robot.data.joint_vel.torch,
            robot.data.body_pos_w.torch,
            robot.data.body_quat_w.torch,
        ):
            finite &= torch.isfinite(state).flatten(start_dim=1).all(dim=1)
    spread = torch.linalg.vector_norm(positions - positions.mean(dim=1, keepdim=True), dim=-1).amax(dim=1)
    return (~finite) | (positions[..., 2].amin(dim=1) < minimum_lace_height) | (spread > maximum_lace_spread)


class lost_grasp(ManagerTermBase):
    """Terminate after either retained grasp remains lost for a confirmation window."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._bilaterally_acquired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._candidate_steps = torch.zeros((env.num_envs, 2), dtype=torch.int64, device=env.device)
        self._release_candidate_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        initial_grasp = _configured_grasp_state(self, "acquisition_distance")
        if initial_grasp is None:
            self._acquired[selected] = False
            self._bilaterally_acquired[selected] = False
        else:
            self._acquired[selected] = initial_grasp[selected]
            self._bilaterally_acquired[selected] = initial_grasp[selected].all(dim=1)
        self._candidate_steps[selected] = 0
        self._release_candidate_steps[selected] = 0

    def __call__(
        self,
        env,
        acquisition_distance: float,
        maximum_finger_position: float,
        maximum_grasp_distance: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
        minimum_finger_position: float = 0.0,
        confirmation_steps: int = 1,
        release_confirmation_steps: int = 1,
    ) -> torch.Tensor:
        """Return per-environment grasp-loss flags after acquisition."""
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        physics = getattr(env, "_physics", None)
        retained = getattr(physics, "grasp_assist_active", None)
        if not getattr(physics, "grasp_assist_enabled", retained is not None):
            retained = None
        if retained is None:
            acquired = grasp_state(
                env,
                acquisition_distance,
                maximum_finger_position,
                asset_cfgs,
                left_robot_cfg,
                right_robot_cfg,
                minimum_finger_position,
            )
            self._candidate_steps.copy_(
                torch.where(acquired, self._candidate_steps + 1, torch.zeros_like(self._candidate_steps))
            )
            acquired = self._candidate_steps >= max(int(confirmation_steps), 1)
            retained = grasp_state(
                env,
                maximum_grasp_distance,
                maximum_finger_position,
                asset_cfgs,
                left_robot_cfg,
                right_robot_cfg,
                minimum_finger_position,
            )
        else:
            acquired = retained = retained.bool()
        self._acquired |= acquired
        self._bilaterally_acquired |= acquired.all(dim=1)
        release_candidate = self._bilaterally_acquired & (self._acquired & (~retained)).any(dim=1)
        self._release_candidate_steps.copy_(
            torch.where(
                release_candidate,
                self._release_candidate_steps + 1,
                torch.zeros_like(self._release_candidate_steps),
            )
        )
        confirmed_release = self._release_candidate_steps >= max(int(release_confirmation_steps), 1)
        return (~torch.isfinite(distances).all(dim=1)) | confirmed_release


def _configured_grasp_state(term: ManagerTermBase, distance_key: str) -> torch.Tensor | None:
    """Return strict grasp state using a manager term's configured parameters."""
    params = getattr(term.cfg, "params", None)
    required = (distance_key, "maximum_finger_position", "asset_cfgs", "left_robot_cfg", "right_robot_cfg")
    if not isinstance(params, dict) or any(name not in params for name in required):
        return None
    return grasp_state(
        term._env,
        params[distance_key],
        params["maximum_finger_position"],
        params["asset_cfgs"],
        params["left_robot_cfg"],
        params["right_robot_cfg"],
        params.get("minimum_finger_position", 0.0),
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
    minimum_finger_position: float = 0.0,
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
        minimum_finger_position,
    ).all(dim=1)
    unsafe = shoelace_unsafe(
        env,
        minimum_lace_height,
        maximum_lace_spread,
        asset_cfgs,
        left_robot_cfg,
        right_robot_cfg,
    )
    return (
        (throat_count <= maximum_throat_segments)
        & (tail_distances.amin(dim=1) >= tail_success_distance)
        & (tail_separation >= tail_success_separation)
        & grasped
        & (~unsafe)
    )
