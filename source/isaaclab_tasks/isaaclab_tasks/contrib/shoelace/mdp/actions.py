# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filtered Cartesian arm actions for the shoelace task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import EMADifferentialInverseKinematicsActionCfg


class EMADifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Warm up and low-pass filter Cartesian commands while retaining their raw policy input."""

    cfg: EMADifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: EMADifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._alpha = float(cfg.alpha)
        if not math.isfinite(self._alpha) or not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must be finite and lie in (0, 1], got {self._alpha}.")
        self._warmup_steps = int(cfg.warmup_steps)
        if self._warmup_steps < 0:
            raise ValueError(f"Warm-up steps must be non-negative, got {self._warmup_steps}.")
        self._warmup_steps_by_level = None
        if cfg.warmup_steps_by_curriculum_level is not None:
            self._warmup_steps_by_level = torch.tensor(
                cfg.warmup_steps_by_curriculum_level, dtype=torch.int64, device=self.device
            )
            if self._warmup_steps_by_level.ndim != 1 or self._warmup_steps_by_level.numel() == 0:
                raise ValueError("Curriculum warm-up steps must be a non-empty one-dimensional sequence.")
            if torch.any(self._warmup_steps_by_level < 0):
                raise ValueError("Curriculum warm-up steps must be non-negative.")
            if not cfg.warmup_curriculum_term_name:
                raise ValueError("A curriculum term name is required for level-dependent warm-up steps.")
        self._filtered_actions = torch.zeros_like(self._raw_actions)
        self._history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._warmup_steps_by_env = torch.full_like(self._steps_since_reset, self._warmup_steps)

    @property
    def filtered_actions(self) -> torch.Tensor:
        """Last filtered policy-space actions [dimensionless]."""
        return self._filtered_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Hold the reset warm-up, then filter subsequent Cartesian commands."""
        warming_up = self._steps_since_reset < self._warmup_steps_by_env
        effective_actions = torch.where(warming_up.unsqueeze(-1), 0.0, actions)
        newest_weight = torch.where(self._history_valid, self._alpha, 1.0).unsqueeze(-1)
        self._filtered_actions.lerp_(effective_actions, newest_weight)
        self._history_valid.fill_(True)
        self._steps_since_reset.add_(1)
        super().process_actions(self._filtered_actions)
        self._raw_actions.copy_(actions)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear the selected warm-up counters and filter histories."""
        selected = slice(None) if env_ids is None else env_ids
        super().reset(selected)
        self._processed_actions[selected] = 0.0
        self._filtered_actions[selected] = 0.0
        self._history_valid[selected] = False
        self._steps_since_reset[selected] = 0
        self._warmup_steps_by_env[selected] = self._warmup_steps
        if self._warmup_steps_by_level is not None:
            curriculum_cfg = getattr(self._env.curriculum_manager.cfg, self.cfg.warmup_curriculum_term_name)
            levels = curriculum_cfg.func.levels[selected]
            if levels.numel() > 0 and int(levels.max().item()) >= self._warmup_steps_by_level.numel():
                raise ValueError("Curriculum warm-up steps do not cover every sampled curriculum level.")
            self._warmup_steps_by_env[selected] = self._warmup_steps_by_level[levels]
