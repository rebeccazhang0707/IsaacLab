# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched command terms for multi-robot environments.

Each class iterates ``robot_meta`` to discover robot groups and
manages per-group command buffers internally, routing resampling
and metrics through :class:`EnvLayout`.

``robot_meta`` is keyed by **clone-group name** (not asset name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm
from isaaclab.scene.env_layout import GroupView
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from .batched_commands_cfg import BatchedPoseCommandCfg
from .utils import LiftGroupCfg, ReachGroupCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene.env_layout import EnvLayout


class _PoseGroup:
    """Per-group bookkeeping for :class:`BatchedPoseCommand`."""

    __slots__ = ("key", "meta", "asset", "body_idx", "gv")

    def __init__(
        self,
        key: str,
        meta: ReachGroupCfg | LiftGroupCfg,
        asset: Articulation,
        body_idx: int,
        gv: GroupView,
    ):
        self.key = key
        self.meta = meta
        self.asset = asset
        self.body_idx = body_idx
        self.gv = gv


class BatchedPoseCommand(CommandTerm):
    """Batched pose command for multi-robot / multi-task environments.

    Iterates ``robot_meta`` (looking for :class:`ReachGroupCfg` and
    :class:`LiftGroupCfg` entries) to discover which groups need pose
    commands.  Maintains a single ``(num_envs, 7)`` command buffer and
    fills each group's rows using :class:`EnvLayout`.

    Registered as a **single** command term with the
    :class:`CommandManager`.
    """

    cfg: BatchedPoseCommandCfg

    def __init__(self, cfg: BatchedPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        layout: EnvLayout = env.scene.layout
        robot_meta: dict = cfg.robot_meta or {}

        self._groups: list[_PoseGroup] = []
        for group_key, meta in robot_meta.items():
            if not isinstance(meta, (ReachGroupCfg, LiftGroupCfg)):
                continue
            if not hasattr(meta, "command_ranges"):
                continue
            # Resolve SceneEntityCfg if not already resolved (idempotent)
            meta.asset_cfg.resolve(env.scene)
            asset: Articulation = env.scene[meta.asset_cfg.name]
            body_ids = meta.asset_cfg.body_ids
            body_idx = body_ids[0] if isinstance(body_ids, list) else 0
            gv = layout[group_key, meta.asset_cfg.name]
            self._groups.append(_PoseGroup(group_key, meta, asset, body_idx, gv))

        N = self.num_envs
        dev = self.device

        self.pose_command_b = torch.zeros(N, 7, device=dev)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.metrics["position_error"] = torch.zeros(N, device=dev)
        self.metrics["orientation_error"] = torch.zeros(N, device=dev)

    def __str__(self) -> str:
        msg = "BatchedPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tGroups: {[g.key for g in self._groups]}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command [m, -]. Shape is (num_envs, 7)."""
        return self.pose_command_b

    def _update_metrics(self):
        for g in self._groups:
            w = g.gv.write
            r = g.gv.read
            root_pos = wp.to_torch(g.asset.data.root_pos_w)[r]
            root_quat = wp.to_torch(g.asset.data.root_quat_w)[r]

            self.pose_command_w[w, :3], self.pose_command_w[w, 3:] = combine_frame_transforms(
                root_pos,
                root_quat,
                self.pose_command_b[w, :3],
                self.pose_command_b[w, 3:],
            )
            pos_error, rot_error = compute_pose_error(
                self.pose_command_w[w, :3],
                self.pose_command_w[w, 3:],
                wp.to_torch(g.asset.data.body_pos_w)[r, g.body_idx],
                wp.to_torch(g.asset.data.body_quat_w)[r, g.body_idx],
            )
            self.metrics["position_error"][w] = torch.linalg.norm(pos_error, dim=-1)
            self.metrics["orientation_error"][w] = torch.linalg.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor):
        layout = self._env.scene.layout
        for g in self._groups:
            _, matched = layout[g.key].filter(env_ids)
            if matched.numel() == 0:
                continue
            r = g.meta.command_ranges
            n = matched.numel()
            rand = torch.empty(n, device=self.device)
            self.pose_command_b[matched, 0] = rand.uniform_(*r.pos_x)
            self.pose_command_b[matched, 1] = rand.uniform_(*r.pos_y)
            self.pose_command_b[matched, 2] = rand.uniform_(*r.pos_z)

            euler = torch.zeros(n, 3, device=self.device)
            euler[:, 0].uniform_(*r.roll)
            euler[:, 1].uniform_(*r.pitch)
            euler[:, 2].uniform_(*r.yaw)
            quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            self.pose_command_b[matched, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG

                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/BatchedPoseCommand/goal"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        for g in self._groups:
            if not g.asset.is_initialized:
                return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
