# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched termination terms for multi-robot environments.

Each class iterates ``robot_meta`` to discover robot groups and
scatters per-group termination signals into a single
``(num_envs,)`` boolean tensor.

``robot_meta`` is keyed by **task-group name** (not asset name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase

from .utils import CabinetGroupCfg, LiftGroupCfg, resolve_scene_entity_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


class batched_object_height_below_minimum(ManagerTermBase):
    """Terminate when any lift group's object falls below a minimum height.

    Iterates :class:`LiftGroupCfg` entries that have ``object_cfg`` and
    checks each object's root height.  Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, slice]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, LiftGroupCfg):
                continue
            self._entries.append((meta.object_cfg.name, layout.env_slice(group_key)))
        self._buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRLEnv, minimum_height: float = -0.05) -> torch.Tensor:
        self._buf.zero_()
        for obj_name, sl in self._entries:
            height = wp.to_torch(env.scene[obj_name].data.root_pos_w)[:, 2]
            self._buf[sl] = height < minimum_height
        return self._buf


class batched_cabinet_drawer_opened(ManagerTermBase):
    """Terminate cabinet episodes once the drawer is sufficiently open.

    Iterates :class:`CabinetGroupCfg` entries and checks the cabinet
    joint position.  Returns shape ``(num_envs,)``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        layout = env.scene.layout
        robot_meta = getattr(env.cfg, "robot_meta", None) or {}
        self._entries: list[tuple[str, list | slice, slice]] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, CabinetGroupCfg):
                continue
            resolve_scene_entity_cfg(env, meta.cabinet_asset_cfg)
            self._entries.append((meta.cabinet_asset_cfg.name, meta.cabinet_asset_cfg.joint_ids, layout.env_slice(group_key)))
        self._buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRLEnv, threshold: float = 0.39) -> torch.Tensor:
        self._buf.zero_()
        for cab_name, jids, sl in self._entries:
            drawer_pos = wp.to_torch(env.scene[cab_name].data.joint_pos)[:, jids[0]]
            self._buf[sl] = drawer_pos > threshold
        return self._buf
