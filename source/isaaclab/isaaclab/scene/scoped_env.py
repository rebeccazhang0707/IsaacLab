# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped environment proxy for transparent task-group data slicing.

Inspired by Newton's :class:`ArticulationView`, these lightweight proxies
intercept attribute access on the environment, scene, assets, and their
data buffers so that every tensor returned is pre-sliced to a task group's
environment subset.  MDP term functions that receive a :class:`ScopedEnv`
instead of the real environment see *only* the rows that belong to their
task group — without any manual indexing.

Since :func:`partition_env_ids` always produces contiguous ranges,
``group_slice`` is typically ``slice(a, b)``, giving **zero-copy** views
on both :class:`warp.array` and :class:`torch.Tensor` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import InteractiveScene
    from isaaclab.scene.env_layout import EnvLayout


class ScopedData:
    """Proxy that slices every tensor/wp.array attribute along dim-0.

    Wraps an asset's ``.data`` object (e.g. ``ArticulationData``,
    ``RigidObjectData``, ``FrameTransformerData``) and automatically
    applies ``group_slice`` to any attribute that is a tensor with at
    least one dimension.

    Non-tensor attributes (scalars, methods, strings, etc.) are returned
    unchanged via delegation.
    """

    __slots__ = ("_data", "_group_slice")

    def __init__(self, data: Any, group_slice: slice | torch.Tensor):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_group_slice", group_slice)

    def __getattr__(self, name: str) -> Any:
        val = getattr(self._data, name)
        if isinstance(val, wp.array) and val.ndim >= 1:
            return val[self._group_slice]
        if isinstance(val, torch.Tensor) and val.ndim >= 1:
            return val[self._group_slice]
        return val


class ScopedAsset:
    """Proxy that intercepts ``.data`` on an asset to return :class:`ScopedData`.

    All other attributes (``num_instances``, ``cfg``, ``find_bodies``,
    etc.) are delegated to the real asset unchanged.
    """

    __slots__ = ("_asset", "_data_proxy")

    def __init__(self, asset: Any, group_slice: slice | torch.Tensor):
        object.__setattr__(self, "_asset", asset)
        object.__setattr__(self, "_data_proxy", ScopedData(asset.data, group_slice))

    @property
    def data(self) -> ScopedData:
        # Access the real asset's .data to trigger lazy sensor updates
        # (e.g. SensorBase._update_outdated_buffers) before returning
        # the scoped proxy.  The proxy's _data reference points to the
        # same mutable object, so updates are visible immediately.
        self._asset.data
        return self._data_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._asset, name)


class ScopedScene:
    """Proxy for :class:`InteractiveScene` that wraps shared assets.

    When ``scene[key]`` is called:

    * If the asset is already scoped to the same task group (its data is
      already ``(group_envs, ...)``), the original asset is returned.
    * If the asset is shared (global, ``(num_envs, ...)``), it is wrapped
      in :class:`ScopedAsset` so that ``.data`` tensors are pre-sliced.

    The ``env_origins`` property is also sliced to ``(group_envs, 3)``.
    """

    __slots__ = ("_scene", "_group_slice", "_task_group", "_asset_cache")

    def __init__(
        self,
        scene: InteractiveScene,
        group_slice: slice | torch.Tensor,
        task_group: str,
    ):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_group_slice", group_slice)
        object.__setattr__(self, "_task_group", task_group)
        object.__setattr__(self, "_asset_cache", {})

    def __getitem__(self, key: str) -> Any:
        cached = self._asset_cache.get(key)
        if cached is not None:
            return cached

        asset = self._scene[key]
        layout: EnvLayout = self._scene.layout
        asset_group = layout.group_for_asset(key)
        if asset_group is not None:
            self._asset_cache[key] = asset
            return asset
        proxy = ScopedAsset(asset, self._group_slice)
        self._asset_cache[key] = proxy
        return proxy

    @property
    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins[self._group_slice]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)


class ScopedCommandManager:
    """Proxy for the command manager that slices shared command tensors.

    Commands scoped to the same task group already return
    ``(group_envs, ...)``, so they are returned as-is.  Commands that
    are not scoped (or scoped to a different group) are sliced using
    ``group_slice``.
    """

    __slots__ = ("_cmd_mgr", "_group_slice", "_task_group")

    def __init__(self, cmd_mgr: Any, group_slice: slice | torch.Tensor, task_group: str):
        object.__setattr__(self, "_cmd_mgr", cmd_mgr)
        object.__setattr__(self, "_group_slice", group_slice)
        object.__setattr__(self, "_task_group", task_group)

    def get_command(self, name: str) -> torch.Tensor:
        cmd = self._cmd_mgr.get_command(name)
        term = self._cmd_mgr._terms[name]
        cmd_group = term.group_key
        if cmd_group == self._task_group:
            return cmd
        return cmd[self._group_slice]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cmd_mgr, name)


class ScopedEnv:
    """Top-level proxy that wraps :class:`ManagerBasedEnv` for a task group.

    Key proxied attributes:

    * ``scene`` → :class:`ScopedScene`
    * ``num_envs`` → number of environments in the task group
    * ``command_manager`` → :class:`ScopedCommandManager`

    Everything else (``sim``, ``cfg``, ``device``, ``action_manager``,
    etc.) is delegated to the real environment.

    Constructed once per task-group term during ``_prepare_terms()`` and
    reused every simulation step.
    """

    __slots__ = ("_env", "_group_slice", "_group_size", "_scene_proxy", "_cmd_proxy")

    def __init__(
        self,
        env: ManagerBasedEnv,
        group_slice: slice | torch.Tensor,
        group_size: int,
        task_group: str,
    ):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_group_slice", group_slice)
        object.__setattr__(self, "_group_size", group_size)
        object.__setattr__(self, "_scene_proxy", ScopedScene(env.scene, group_slice, task_group))
        object.__setattr__(
            self,
            "_cmd_proxy",
            ScopedCommandManager(env.command_manager, group_slice, task_group),
        )

    @property
    def num_envs(self) -> int:
        return self._group_size

    @property
    def scene(self) -> ScopedScene:
        return self._scene_proxy

    @property
    def command_manager(self) -> ScopedCommandManager:
        return self._cmd_proxy

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._env, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
