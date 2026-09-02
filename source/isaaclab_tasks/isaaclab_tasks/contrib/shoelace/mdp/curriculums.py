# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for dual-Franka shoelace untying."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PullToGraspCurriculum(ManagerTermBase):
    """Promote robot-at-tail resets toward the complete approach-and-grasp task.

    The scheduler evaluates fixed-size windows at the current level. Successful
    windows first increase the share of stable environment slots assigned to the
    current level and then unlock the next level. Remaining slots replay preceding
    levels to prevent long current-level episodes from crowding replay transitions
    out of the policy batch.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        level_count = int(cfg.params["level_count"])
        legacy_approach_level_count = cfg.params.get("approach_level_count")
        if legacy_approach_level_count is not None or "grasp_assist_strengths" in cfg.params:
            warnings.warn(
                "approach_level_count and grasp_assist_strengths are deprecated; use level_count for the reset "
                "curriculum and grasp_assist_enabled to enable or disable grasp assistance.",
                FutureWarning,
                stacklevel=2,
            )
        if legacy_approach_level_count is not None:
            legacy_approach_level_count = int(legacy_approach_level_count)
            if not 2 <= legacy_approach_level_count <= level_count:
                raise ValueError("approach_level_count must lie between two and level_count.")
            level_count = legacy_approach_level_count
        initial_level = int(cfg.params.get("initial_level", 0))
        if legacy_approach_level_count is not None and initial_level >= level_count:
            initial_level = level_count - 1
        if level_count < 2:
            raise ValueError("level_count must be at least two.")
        if not 0 <= initial_level < level_count:
            raise ValueError("initial_level must lie within the curriculum levels.")

        self._level_count = level_count
        physics = getattr(env, "_physics", None)
        self._grasp_assist_enabled = bool(getattr(physics, "grasp_assist_enabled", False))
        self._current_level = initial_level
        self._levels = torch.full((env.num_envs,), initial_level, dtype=torch.long, device=env.device)
        self._slot_quantiles = (torch.arange(env.num_envs, device=env.device, dtype=torch.float32) + 0.5) / max(
            env.num_envs, 1
        )
        self._current_level_fractions = self._validate_fraction_schedule(
            cfg.params["current_level_fraction"], cfg.params.get("current_level_fraction_schedule", ())
        )
        self._terminal_level_fractions = self._validate_fraction_schedule(
            cfg.params.get("terminal_level_fraction", cfg.params["current_level_fraction"]),
            cfg.params.get("terminal_level_fraction_schedule", cfg.params.get("current_level_fraction_schedule", ())),
        )
        self._current_level_fraction_index = len(self._active_fraction_schedule()) - 1
        self._window_episode_count = 0
        self._window_success_count = 0
        self._last_success_rate = 0.0
        self._successful_window_count = 0

    @property
    def difficulty(self) -> torch.Tensor:
        """Per-environment reset difficulty in ``[0, 1]``."""
        return self._levels.float() / (self._level_count - 1)

    @property
    def grasp_assist_scale(self) -> torch.Tensor:
        """Per-environment indicator that fixed-strength grasp assistance is enabled.

        This compatibility property no longer varies by curriculum level.
        """
        return torch.full(
            self._levels.shape,
            float(self._grasp_assist_enabled),
            dtype=torch.float32,
            device=self._env.device,
        )

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
        replay_level_weights: Sequence[float] = (1.0,),
        current_level_fraction_schedule: Sequence[float] = (),
        fraction_increase_success_rate: float = 0.5,
        fraction_backoff_success_rate: float = 0.1,
        promotion_window_count: int = 1,
        approach_level_count: int | None = None,
        grasp_assist_strengths: Sequence[float] = (1.0,),
        terminal_level_fraction: float | None = None,
        terminal_level_fraction_schedule: Sequence[float] = (),
    ) -> dict[str, torch.Tensor | float]:
        """Credit completed episodes, promote difficulty, and sample reset levels.

        Args:
            env: The learning environment.
            env_ids: Environments being reset.
            level_count: Number of reset difficulty levels.
            success_term_name: Termination term used to credit success.
            promotion_success_rate: Window success rate required for promotion.
            minimum_episodes: Completed current-level episodes per promotion window.
            current_level_fraction: Maximum fraction of environment slots assigned to the current level.
            initial_level: Initial curriculum level.
            replay_level_weights: Relative sampling weights for preceding levels, ordered from the immediately
                preceding level backward. Weights for unavailable levels are discarded and the remainder is
                normalized.
            current_level_fraction_schedule: Increasing current-level slot fractions. An empty schedule uses only
                ``current_level_fraction``. The last value must equal ``current_level_fraction``.
            fraction_increase_success_rate: Window success rate required to increase current-level exposure.
            fraction_backoff_success_rate: Success rate below which current-level exposure decreases one stage.
            promotion_window_count: Consecutive successful windows required for promotion at maximum exposure.
            approach_level_count: Deprecated legacy reset-level count. New configurations should omit it.
            grasp_assist_strengths: Deprecated and ignored. Configure grasp assistance through the environment.
            terminal_level_fraction: Maximum current-level fraction at the final curriculum level.
            terminal_level_fraction_schedule: Increasing current-level fractions used only at the final level.

        Returns:
            Scalar curriculum metrics for logging.
        """
        del (
            level_count,
            initial_level,
            current_level_fraction_schedule,
            approach_level_count,
            grasp_assist_strengths,
            terminal_level_fraction,
            terminal_level_fraction_schedule,
        )
        if minimum_episodes <= 0:
            raise ValueError("minimum_episodes must be positive.")
        if not 0.0 <= promotion_success_rate <= 1.0:
            raise ValueError("promotion_success_rate must lie in [0, 1].")
        if not 0.0 <= current_level_fraction <= 1.0:
            raise ValueError("current_level_fraction must lie in [0, 1].")
        if not 0.0 <= fraction_backoff_success_rate < fraction_increase_success_rate <= promotion_success_rate:
            raise ValueError("Curriculum success rates must satisfy 0 <= backoff < increase <= promotion <= 1.")
        if promotion_window_count <= 0:
            raise ValueError("promotion_window_count must be positive.")
        replay_level_weights = tuple(float(weight) for weight in replay_level_weights)
        if not replay_level_weights:
            raise ValueError("replay_level_weights must contain at least one weight.")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in replay_level_weights):
            raise ValueError("replay_level_weights must contain finite non-negative weights.")
        if replay_level_weights[0] <= 0.0:
            raise ValueError("The immediately preceding replay level must have a positive weight.")

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
            active_fractions = self._active_fraction_schedule()
            maximum_exposure = self._current_level_fraction_index == len(active_fractions) - 1
            if self._last_success_rate >= promotion_success_rate and maximum_exposure:
                self._successful_window_count += 1
                if self._successful_window_count >= promotion_window_count:
                    if self._current_level < self._level_count - 1:
                        self._current_level += 1
                        self._current_level_fraction_index = 0
                    self._successful_window_count = 0
            elif self._last_success_rate >= fraction_increase_success_rate:
                self._current_level_fraction_index = min(
                    self._current_level_fraction_index + 1, len(active_fractions) - 1
                )
                self._successful_window_count = 0
            elif self._last_success_rate < fraction_backoff_success_rate:
                self._current_level_fraction_index = max(self._current_level_fraction_index - 1, 0)
                self._successful_window_count = 0
            else:
                self._successful_window_count = 0
            self._window_episode_count = 0
            self._window_success_count = 0

        if env_ids.numel() > 0:
            active_fractions = self._active_fraction_schedule()
            self._levels[env_ids] = self._sample_levels(
                env_ids, active_fractions[self._current_level_fraction_index], replay_level_weights
            )

        active_fractions = self._active_fraction_schedule()
        return {
            "current_level": float(self._current_level),
            "current_level_fraction": active_fractions[self._current_level_fraction_index],
            "mean_sampled_level": self._levels.float().mean(),
            "mean_difficulty": self.difficulty.mean(),
            "full_task_fraction": (self._levels == self._level_count - 1).float().mean(),
            "mean_grasp_assist_scale": self.grasp_assist_scale.mean(),
            "unassisted_fraction": (self.grasp_assist_scale == 0.0).float().mean(),
            "last_window_success_rate": self._last_success_rate,
            "successful_window_count": float(self._successful_window_count),
        }

    def _active_fraction_schedule(self) -> tuple[float, ...]:
        """Return the exposure schedule for the current frontier level."""
        if self._current_level == self._level_count - 1:
            return self._terminal_level_fractions
        return self._current_level_fractions

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        """Return environment indices as a device tensor."""
        if isinstance(env_ids, slice):
            return torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)

    def _sample_levels(
        self,
        env_ids: torch.Tensor,
        current_level_fraction: float,
        replay_level_weights: tuple[float, ...],
    ) -> torch.Tensor:
        """Assign stable environment strata to the current and replay levels."""
        levels = torch.full((env_ids.numel(),), self._current_level, dtype=torch.long, device=self._env.device)
        if self._current_level > 0 and current_level_fraction < 1.0:
            quantiles = self._slot_quantiles[env_ids]
            replay = quantiles >= current_level_fraction
            replay_count = int(replay.sum().item())
            if replay_count > 0:
                available_depth = min(self._current_level, len(replay_level_weights))
                weights = levels.new_tensor(replay_level_weights[:available_depth], dtype=torch.float32)
                weights /= weights.sum()
                replay_quantiles = (quantiles[replay] - current_level_fraction) / (1.0 - current_level_fraction)
                replay_depths = torch.searchsorted(weights.cumsum(dim=0), replay_quantiles, right=True) + 1
                levels[replay] -= replay_depths
        return levels

    @staticmethod
    def _validate_fraction_schedule(
        current_level_fraction: float, current_level_fraction_schedule: Sequence[float]
    ) -> tuple[float, ...]:
        """Validate and return the ordered current-level exposure schedule."""
        fractions = tuple(float(fraction) for fraction in current_level_fraction_schedule)
        if not fractions:
            fractions = (float(current_level_fraction),)
        if any(not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("current_level_fraction_schedule must contain finite values in [0, 1].")
        if any(left >= right for left, right in zip(fractions, fractions[1:], strict=False)):
            raise ValueError("current_level_fraction_schedule must be strictly increasing.")
        if not math.isclose(fractions[-1], current_level_fraction):
            raise ValueError("The last current_level_fraction_schedule value must equal current_level_fraction.")
        return fractions
