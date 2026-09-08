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
        socket_targets: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        maximum_socket_error: float | None = None,
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
                socket_targets,
                maximum_socket_error,
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


class missed_grasp_acquisition(ManagerTermBase):
    """Terminate when both tails are reachable but bilateral acquisition stalls."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._bilaterally_acquired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._acquisition_started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._candidate_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset acquisition history and seed it from the configured task state."""
        selected = slice(None) if env_ids is None else env_ids
        initial_grasp = _configured_grasp_state(self, "acquisition_distance")
        if initial_grasp is None:
            self._bilaterally_acquired[selected] = False
        else:
            self._bilaterally_acquired[selected] = initial_grasp[selected].all(dim=1)
        self._acquisition_started[selected] = self._bilaterally_acquired[selected]
        self._candidate_steps[selected] = 0

    def __call__(
        self,
        env,
        acquisition_distance: float,
        maximum_finger_position: float,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
        minimum_finger_position: float = 0.0,
        deadline_steps: int = 1,
        socket_targets: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        maximum_socket_error: float | None = None,
    ) -> torch.Tensor:
        """Return failures after acquisition stalls while both tails are within reach.

        Args:
            env: The task environment.
            acquisition_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            asset_cfgs: Shoelace scene entities.
            left_robot_cfg: Left robot scene entity.
            right_robot_cfg: Right robot scene entity.
            minimum_finger_position: Minimum driven finger-joint position for acquisition [m].
            deadline_steps: Consecutive reachable control steps allowed before termination.
            socket_targets: Target tail-to-TCP vectors in the controlling hand frames [m].
            maximum_socket_error: Maximum Euclidean error from each contact socket target [m].

        Returns:
            Per-environment missed-acquisition flags.
        """
        distances = grasp_distances(env, asset_cfgs, left_robot_cfg, right_robot_cfg)
        acquired = grasp_state(
            env,
            acquisition_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
            minimum_finger_position,
            socket_targets,
            maximum_socket_error,
        ).all(dim=1)
        self._bilaterally_acquired |= acquired
        reachable = torch.isfinite(distances).all(dim=1) & (distances <= acquisition_distance).all(dim=1)
        self._acquisition_started |= reachable
        candidate = self._acquisition_started & (~self._bilaterally_acquired)
        self._candidate_steps.copy_(
            torch.where(candidate, self._candidate_steps + 1, torch.zeros_like(self._candidate_steps))
        )
        return self._candidate_steps >= max(int(deadline_steps), 1)


class insufficient_separation_progress(ManagerTermBase):
    """Terminate acquired trajectories that miss a reset-relative tail-separation deadline."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._baseline_tail_separation = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._bilaterally_acquired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._steps_since_acquisition = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset progress history and capture the configured reset geometry."""
        selected = slice(None) if env_ids is None else env_ids
        self._baseline_tail_separation[selected] = torch.nan
        initial_grasp = _configured_grasp_state(self, "acquisition_distance")
        if initial_grasp is None:
            self._bilaterally_acquired[selected] = False
        else:
            self._bilaterally_acquired[selected] = initial_grasp[selected].all(dim=1)
        self._steps_since_acquisition[selected] = 0

        params = getattr(self.cfg, "params", None)
        if not isinstance(params, dict) or "asset_cfgs" not in params:
            return
        _, _, _, tail_positions, _ = task_state(self._env, params["asset_cfgs"])
        tail_separation = torch.linalg.vector_norm(tail_positions[:, 0] - tail_positions[:, 1], dim=1)
        selected_separation = tail_separation[selected]
        self._baseline_tail_separation[selected] = torch.where(
            torch.isfinite(selected_separation),
            selected_separation,
            torch.full_like(selected_separation, torch.nan),
        )

    def __call__(
        self,
        env,
        acquisition_distance: float,
        maximum_finger_position: float,
        target_tail_separation: float,
        minimum_progress_fraction: float,
        deadline_steps: int,
        asset_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        left_robot_cfg: SceneEntityCfg,
        right_robot_cfg: SceneEntityCfg,
        minimum_finger_position: float = 0.0,
        socket_targets: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        maximum_socket_error: float | None = None,
    ) -> torch.Tensor:
        """Return failures that do not maintain enough separation after acquisition.

        Args:
            env: The task environment.
            acquisition_distance: Maximum tail-to-TCP acquisition distance [m].
            maximum_finger_position: Maximum driven finger-joint position for acquisition [m].
            target_tail_separation: Tail separation used by the success target [m].
            minimum_progress_fraction: Required fraction of reset-to-target separation progress.
            deadline_steps: Acquired control steps allowed before enforcing the requirement.
            asset_cfgs: Shoelace scene entities.
            left_robot_cfg: Left robot scene entity.
            right_robot_cfg: Right robot scene entity.
            minimum_finger_position: Minimum driven finger-joint position for acquisition [m].
            socket_targets: Target tail-to-TCP vectors in the controlling hand frames [m].
            maximum_socket_error: Maximum Euclidean error from each contact socket target [m].

        Returns:
            Per-environment insufficient-separation flags.
        """
        _, _, _, tail_positions, _ = task_state(env, asset_cfgs)
        tail_separation = torch.linalg.vector_norm(tail_positions[:, 0] - tail_positions[:, 1], dim=1)
        unseeded = ~torch.isfinite(self._baseline_tail_separation)
        self._baseline_tail_separation.copy_(
            torch.where(unseeded & torch.isfinite(tail_separation), tail_separation, self._baseline_tail_separation)
        )
        acquired = grasp_state(
            env,
            acquisition_distance,
            maximum_finger_position,
            asset_cfgs,
            left_robot_cfg,
            right_robot_cfg,
            minimum_finger_position,
            socket_targets,
            maximum_socket_error,
        ).all(dim=1)
        self._bilaterally_acquired |= acquired
        self._steps_since_acquisition.copy_(
            torch.where(
                self._bilaterally_acquired,
                self._steps_since_acquisition + 1,
                torch.zeros_like(self._steps_since_acquisition),
            )
        )
        progress_fraction = min(max(float(minimum_progress_fraction), 0.0), 1.0)
        remaining_separation = (target_tail_separation - self._baseline_tail_separation).clamp_min(0.0)
        required_separation = self._baseline_tail_separation + progress_fraction * remaining_separation
        finite = torch.isfinite(tail_separation) & torch.isfinite(required_separation)
        return (
            self._bilaterally_acquired
            & (self._steps_since_acquisition >= max(int(deadline_steps), 1))
            & finite
            & (tail_separation < required_separation)
        )


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
        params.get("socket_targets"),
        params.get("maximum_socket_error"),
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
