# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched termination terms for multi-robot environments.

Each class iterates ``robot_meta`` to discover robot groups and
scatters per-group termination signals into a single
``(num_envs,)`` boolean tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


class batched_object_height_below_minimum(ManagerTermBase):
    """Terminate when any robot group's object falls below a minimum height.

    Iterates ``robot_meta`` entries that have ``object_cfg`` and checks
    each object's root height.  Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice]] = []
        for asset_name, meta in robot_meta.items():
            gk = layout.group_for_asset(asset_name)
            if gk is None:
                continue
            obj_cfg = getattr(meta, "object_cfg", None)
            if obj_cfg is None:
                continue
            self._entries.append((obj_cfg.name, layout.env_slice(gk)))
        self._buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRLEnv, minimum_height: float = -0.05) -> torch.Tensor:
        self._buf.zero_()
        for obj_name, sl in self._entries:
            height = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, 2]
            self._buf[sl] = height < minimum_height
        return self._buf
