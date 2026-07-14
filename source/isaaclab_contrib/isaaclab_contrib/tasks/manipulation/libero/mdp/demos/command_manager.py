# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command manager that registers one demo term under multiple semantic names.

Playground uses the same pattern so ``get_command("ee_pose")`` /
``get_command("joint_state")`` / ``get_command("source_action")`` share one
trajectory resample / frame advance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandManager, CommandTerm, CommandTermCfg


class DgpoLiberoCommandManager(CommandManager):
    """Expand ``semantic_keys`` on a command cfg into multiple registered names.

    ``compute`` / ``reset`` run once per unique term instance so demo frame
    counters advance only once per step.
    """

    def __init__(self, cfg: object, env):
        self._semantic_keys: set[str] = set()
        super().__init__(cfg, env)

    def _prepare_terms(self):
        self._semantic_keys = set()
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, CommandTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CommandTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            semantic_keys = getattr(term_cfg, "semantic_keys", None)
            if semantic_keys and isinstance(semantic_keys, (list, tuple)):
                term = term_cfg.class_type(term_cfg, self._env)
                if not isinstance(term, CommandTerm):
                    raise TypeError(f"Returned object for the term '{term_name}' is not of type CommandTerm.")
                for key in semantic_keys:
                    self._terms[key] = term
                    self._semantic_keys.add(key)
            else:
                term = term_cfg.class_type(term_cfg, self._env)
                if not isinstance(term, CommandTerm):
                    raise TypeError(f"Returned object for the term '{term_name}' is not of type CommandTerm.")
                self._terms[term_name] = term

    def get_command(self, name: str) -> torch.Tensor:
        term = self._terms[name]
        if name in self._semantic_keys and hasattr(term, "get_command_view"):
            return term.get_command_view(name)
        return term.command

    def compute(self, dt: float):
        seen: set[int] = set()
        for term in self._terms.values():
            tid = id(term)
            if tid in seen:
                continue
            seen.add(tid)
            term.compute(dt)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        extras: dict[str, torch.Tensor] = {}
        term_to_names: dict[int, list[str]] = {}
        for name, term in self._terms.items():
            term_to_names.setdefault(id(term), []).append(name)
        for names in term_to_names.values():
            term = self._terms[names[0]]
            metrics = term.reset(env_ids=env_ids)
            if metrics is None:
                continue
            log_name = names[0]
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{log_name}/{metric_name}"] = metric_value
        return extras


# Short-lived alias for gym / older imports.
CompatLiberoCommandManager = DgpoLiberoCommandManager
