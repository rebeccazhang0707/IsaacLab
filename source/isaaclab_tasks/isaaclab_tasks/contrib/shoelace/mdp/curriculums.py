# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for dual-Franka shoelace untying."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PullToGraspCurriculum(ManagerTermBase):
    """Promote robot-at-tail resets toward the complete approach-and-grasp task.

    The scheduler evaluates fixed-size windows at the current level. Successful
    windows unlock the next level, while a small fraction of resets remains at
    the preceding level to reduce forgetting.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        level_count = int(cfg.params["level_count"])
        initial_level = int(cfg.params.get("initial_level", 0))
        if level_count < 2:
            raise ValueError("level_count must be at least two.")
        if not 0 <= initial_level < level_count:
            raise ValueError("initial_level must lie within the curriculum levels.")

        self._level_count = level_count
        self._current_level = initial_level
        self._levels = torch.full((env.num_envs,), initial_level, dtype=torch.long, device=env.device)
        self._window_episode_count = 0
        self._window_success_count = 0
        self._last_success_rate = 0.0

    @property
    def difficulty(self) -> torch.Tensor:
        """Per-environment reset difficulty in ``[0, 1]``."""
        return self._levels.float() / (self._level_count - 1)

    @property
    def levels(self) -> torch.Tensor:
        """Per-environment discrete reset levels."""
        return self._levels

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        level_count: int,
        success_term_name: str,
        promotion_success_rate: float,
        minimum_episodes: int,
        current_level_fraction: float,
        initial_level: int = 0,
    ) -> dict[str, torch.Tensor | float]:
        """Credit completed episodes, promote difficulty, and sample reset levels.

        Args:
            env: The learning environment.
            env_ids: Environments being reset.
            level_count: Number of reset difficulty levels.
            success_term_name: Termination term used to credit success.
            promotion_success_rate: Window success rate required for promotion.
            minimum_episodes: Completed current-level episodes per promotion window.
            current_level_fraction: Fraction of resets sampled at the current level.
            initial_level: Initial curriculum level.

        Returns:
            Scalar curriculum metrics for logging.
        """
        del level_count, initial_level
        if minimum_episodes <= 0:
            raise ValueError("minimum_episodes must be positive.")
        if not 0.0 <= promotion_success_rate <= 1.0:
            raise ValueError("promotion_success_rate must lie in [0, 1].")
        if not 0.0 <= current_level_fraction <= 1.0:
            raise ValueError("current_level_fraction must lie in [0, 1].")

        env_ids = self._resolve_env_ids(env_ids)
        if env.common_step_counter > 0 and env_ids.numel() > 0:
            completed_levels = self._levels[env_ids]
            current_mask = completed_levels == self._current_level
            current_episode_count = int(current_mask.sum().item())
            if current_episode_count > 0:
                successes = env.termination_manager.get_term(success_term_name)[env_ids].bool()
                self._window_episode_count += current_episode_count
                self._window_success_count += int((successes & current_mask).sum().item())

        if self._window_episode_count >= minimum_episodes:
            self._last_success_rate = self._window_success_count / self._window_episode_count
            if self._last_success_rate >= promotion_success_rate:
                self._current_level = min(self._current_level + 1, self._level_count - 1)
            self._window_episode_count = 0
            self._window_success_count = 0

        if env_ids.numel() > 0:
            self._levels[env_ids] = self._sample_levels(env_ids.numel(), current_level_fraction)

        return {
            "current_level": float(self._current_level),
            "mean_sampled_level": self._levels.float().mean(),
            "mean_difficulty": self.difficulty.mean(),
            "full_task_fraction": (self._levels == self._level_count - 1).float().mean(),
            "last_window_success_rate": self._last_success_rate,
        }

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        """Return environment indices as a device tensor."""
        if isinstance(env_ids, slice):
            return torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)

    def _sample_levels(self, count: int, current_level_fraction: float) -> torch.Tensor:
        """Sample the current level with a replay fraction from its predecessor."""
        levels = torch.full((count,), self._current_level, dtype=torch.long, device=self._env.device)
        if self._current_level > 0 and current_level_fraction < 1.0:
            replay = torch.rand(count, device=self._env.device) >= current_level_fraction
            levels[replay] -= 1
        return levels
