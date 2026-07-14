# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env task-gather MDP for the harvest (object-level prototype-sharing) LIBERO env.

The combined ``Isaac-Libero-*-Dgpo-Osc-*`` scene shares identical object models as a
single :class:`~isaaclab.assets.AssetView` across every task that uses them (see
:mod:`..tasks.harvest.prototypes`).  A shared AssetView therefore spans envs belonging to
*different* tasks, so the usual per-clone-group ``SceneEntityCfg(selector=group)``
scoping does **not** work here: a group's selector env set overlaps every other
group that shares an asset, which would corrupt task-dependent rewards/goals.

Instead every task-dependent term gathers **per env by task id**.  The scene is
cloned with the deterministic :func:`~isaaclab.cloner.sequential` strategy, so env
``i`` runs task ``i % n_tasks``.  From that map we build, once per env instance:

* per-env success tolerances,
* for each *distinct* prototype, the global envs for which it is the primary /
  target object -- so a single loop over prototypes scatters each env's own
  goal object pose into a full ``(num_envs, ...)`` buffer.

Assets are always located with ``selector.filter_reset_ids(proto_name, env_ids)``
(clean per-asset ``(global_env_ids, view_ids)``), mirroring the demo's
``scene.selector.filter_reset_ids``.  Quaternions are ``(x, y, z, w)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from ..tasks.harvest.prototypes import TaskBinding


# ---------------------------------------------------------------------------
# Per-env runtime (built lazily, cached on the env instance)
# ---------------------------------------------------------------------------


class _LiberoRuntime:
    """Cached per-env task assignment and prototype gather indices.

    Built once from the deterministic ``env i -> task i % n_tasks`` clone map and
    the harvested task bindings; reused by every task-dependent MDP term.
    """

    def __init__(self, env: ManagerBasedEnv, tasks: list[TaskBinding]) -> None:
        device = env.device
        self.n_tasks = len(tasks)
        n = env.num_envs
        self.env_task = torch.arange(n, device=device, dtype=torch.long) % self.n_tasks

        xy = torch.tensor([t.success_xy_threshold for t in tasks], device=device)
        hz = torch.tensor([t.success_height_threshold for t in tasks], device=device)
        self.xy_threshold = xy[self.env_task]
        self.height_threshold = hz[self.env_task]

        # prototype name -> global env ids for which it is the primary / target object
        self.primary_envs = self._group_by_proto(tasks, "primary_proto", device)
        self.target_envs = self._group_by_proto(tasks, "target_proto", device)

    def _group_by_proto(self, tasks: list[TaskBinding], attr: str, device: str) -> dict[str, torch.Tensor]:
        out: dict[str, list[torch.Tensor]] = {}
        for task_idx, task in enumerate(tasks):
            name = getattr(task, attr)
            if name is None:
                continue
            envs = (self.env_task == task_idx).nonzero(as_tuple=True)[0]
            out.setdefault(name, []).append(envs)
        return {name: torch.cat(envs).sort().values for name, envs in out.items()}


def _runtime(env: ManagerBasedEnv, tasks: list[TaskBinding]) -> _LiberoRuntime:
    """Return the cached :class:`_LiberoRuntime`, building it on first use."""
    rt = getattr(env, "_libero_runtime", None)
    if rt is None:
        rt = _LiberoRuntime(env, tasks)
        env._libero_runtime = rt
    return rt


def _gather_object_state(
    env: ManagerBasedEnv, envs_by_proto: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter each env's assigned object world pose + speed into full buffers.

    Args:
        env: The environment instance.
        envs_by_proto: Prototype name -> global env ids that use it as this role.

    Returns:
        ``(pos_w, speed)`` where ``pos_w`` is ``(num_envs, 3)`` world position [m]
        and ``speed`` is ``(num_envs,)`` linear speed [m/s]; rows without an
        assigned object stay zero.
    """
    selector = env.scene.selector
    pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    speed = torch.zeros(env.num_envs, device=env.device)
    for name, envs in envs_by_proto.items():
        global_ids, view_ids = selector.filter_reset_ids(name, envs)
        if global_ids.numel() == 0:
            continue
        asset = env.scene[name]
        pos_w[global_ids] = wp.to_torch(asset.data.root_pos_w)[view_ids, :3]
        speed[global_ids] = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_w)[view_ids, :3], dim=-1)
    return pos_w, speed


def _success_mask(env: ManagerBasedRLEnv, rt: _LiberoRuntime, speed_threshold: float | None) -> torch.Tensor:
    """Per-env geometric success: primary near target (xy + height) and near rest."""
    primary_pos, primary_speed = _gather_object_state(env, rt.primary_envs)
    target_pos, _ = _gather_object_state(env, rt.target_envs)
    xy_dist = torch.linalg.norm(primary_pos[:, :2] - target_pos[:, :2], dim=-1)
    height_dist = torch.abs(primary_pos[:, 2] - target_pos[:, 2])
    reached = (xy_dist < rt.xy_threshold) & (height_dist < rt.height_threshold)
    if speed_threshold is not None:
        reached &= primary_speed < speed_threshold
    return reached


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def libero_task_success(
    env: ManagerBasedRLEnv, tasks: list[TaskBinding], speed_threshold: float | None = 0.4
) -> torch.Tensor:
    """Episode ends when the per-env geometric success proxy is met."""
    rt = _runtime(env, tasks)
    return _success_mask(env, rt, speed_threshold)


# ---------------------------------------------------------------------------
# Reset event
# ---------------------------------------------------------------------------


def reset_libero_prototypes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    tasks: list[TaskBinding],
    pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Write each task's authored object poses into its own envs, with jitter.

    Mirrors the demo's per-task reset (``CloneEngine.reset`` /
    ``_write_init_state``): for every task, restrict the reset envs to that task's
    envs (``env % n_tasks == task_idx``), then for each object the task uses locate
    its shared prototype view with :meth:`filter_reset_ids` and write this task's
    ``(x, y, z, w)`` pose (plus a small uniform position jitter).
    """
    selector = env.scene.selector
    n_tasks = len(tasks)
    device = env.device
    ranges = torch.tensor([(pose_range or {}).get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=device)
    for task_idx, task in enumerate(tasks):
        task_envs = env_ids[(env_ids % n_tasks) == task_idx]
        if task_envs.numel() == 0:
            continue
        for binding in task.object_bindings:
            global_ids, view_ids = selector.filter_reset_ids(binding.proto_name, task_envs)
            if global_ids.numel() == 0:
                continue
            asset = env.scene[binding.proto_name]
            count = global_ids.shape[0]
            jitter = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (count, 3), device=device)
            pose = torch.zeros((count, 7), device=device)
            pose[:, :3] = torch.tensor(binding.pos, device=device) + env.scene.env_origins[global_ids] + jitter
            pose[:, 3:7] = torch.tensor(binding.rot, device=device)
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_ids)
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((count, 6), device=device), env_ids=view_ids
            )
            if binding.is_articulation:
                joint_pos = wp.to_torch(asset.data.default_joint_pos)[view_ids].clone()
                joint_vel = wp.to_torch(asset.data.default_joint_vel)[view_ids].clone()
                asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=view_ids)
                asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=view_ids)
