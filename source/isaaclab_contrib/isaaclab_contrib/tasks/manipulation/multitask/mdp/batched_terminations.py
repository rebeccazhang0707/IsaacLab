# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched termination terms for multi-robot environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from .utils import BatchedTermBase, CabinetGroupCfg, LiftGroupCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.scene import GroupView


class batched_object_height_below_minimum(BatchedTermBase):
    """Terminate when any lift group's object falls below a minimum height."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, LiftGroupCfg, GroupView]] = []
        for group_key, meta in self._iter_groups(LiftGroupCfg):
            self._entries.append((group_key, meta, self._layout[group_key]))
        self._buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

    def __call__(
        self, env: ManagerBasedRLEnv, minimum_height: float = -0.05, robot_meta: dict | None = None
    ) -> torch.Tensor:
        self._buf.zero_()
        for _, meta, group_view in self._entries:
            obj: RigidObject = env.scene[meta.object_cfg.name]
            height = wp.to_torch(obj.data.root_pos_w)[:, 2]
            self._buf[group_view.write] = height < minimum_height
        return self._buf


class batched_cabinet_drawer_opened(BatchedTermBase):
    """Terminate cabinet episodes once the drawer is sufficiently open."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._entries: list[tuple[str, CabinetGroupCfg, GroupView]] = []
        for group_key, meta in self._iter_groups(CabinetGroupCfg):
            self._entries.append((group_key, meta, self._layout[group_key]))
        self._buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.39, robot_meta: dict | None = None) -> torch.Tensor:
        self._buf.zero_()
        for _, meta, group_view in self._entries:
            cabinet: Articulation = env.scene[meta.cabinet_asset_cfg.name]
            drawer_pos = wp.to_torch(cabinet.data.joint_pos)[:, meta.cabinet_asset_cfg.joint_ids].squeeze(-1)
            self._buf[group_view.write] = drawer_pos > threshold
        return self._buf
