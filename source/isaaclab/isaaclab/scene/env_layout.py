# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized environment layout for heterogeneous multi-task scenes.

Design follows Newton's pattern: frozen dataclasses + pure functions + minimal state.

Architecture:
- Layer 1: Pure functions (filter_to_group, to_local, to_global, etc.)
- Layer 2: GroupView wrapper (delegates to pure functions)
- Layer 3: EnvLayout - minimal state, exposes raw data for composition
"""

from __future__ import annotations

__all__ = [
    # Pure functions
    "is_contiguous_slice",
    "filter_to_group",
    "to_local",
    "to_global",
    "get_env_ids",
    "compute_read_index",
    "build_layout",
    "resolve_asset_env_ids",
    # Data classes
    "GroupLayout",
    "GroupView",
    # Main class
    "EnvLayout",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .clone_cfg import CloneCfg


# ============================================================================
# Layer 1: Pure Functions (Newton-like core)
# ============================================================================


def is_contiguous_slice(indices: list[int]) -> bool:
    """Check if indices form a contiguous sequence."""
    n = len(indices)
    if n > 1:
        for i in range(1, n):
            if indices[i] != indices[i - 1] + 1:
                return False
    return True


@dataclass(frozen=True)
class GroupLayout:
    """Immutable layout data for a group - like Newton's FrequencyLayout.

    Either ``slice`` or ``indices`` is set, never both.
    """

    offset: int
    """First env index in this group."""

    count: int
    """Number of envs in this group."""

    slice: slice | None = None
    """Set if envs are contiguous (zero-copy indexing)."""

    indices: torch.Tensor | None = None
    """Set if envs are non-contiguous (requires gather)."""

    device: str = "cpu"
    """Device for tensor operations."""

    @property
    def is_contiguous(self) -> bool:
        """Whether this group's envs are contiguous."""
        return self.slice is not None


def filter_to_group(layout: GroupLayout, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter env_ids to those in this group. Returns (local_ids, matched_global_ids)."""
    if layout.slice is not None:
        start, stop = layout.slice.start, layout.slice.stop
        mask = (env_ids >= start) & (env_ids < stop)
        matched = env_ids[mask]
        return matched - start, matched
    else:
        indices = layout.indices
        positions = torch.searchsorted(indices, env_ids)
        valid_mask = (positions < layout.count) & (indices[positions.clamp(max=layout.count - 1)] == env_ids)
        return positions[valid_mask], env_ids[valid_mask]


def to_local(layout: GroupLayout, env_ids: torch.Tensor) -> torch.Tensor:
    """Convert global env_ids to 0-based local indices. Non-matching IDs are dropped."""
    if layout.slice is not None:
        start, stop = layout.slice.start, layout.slice.stop
        mask = (env_ids >= start) & (env_ids < stop)
        return env_ids[mask] - start
    else:
        indices = layout.indices
        positions = torch.searchsorted(indices, env_ids)
        valid_mask = (positions < layout.count) & (indices[positions.clamp(max=layout.count - 1)] == env_ids)
        return positions[valid_mask]


def to_global(layout: GroupLayout, local_ids: torch.Tensor) -> torch.Tensor:
    """Convert local indices back to global env_ids."""
    if layout.slice is not None:
        return local_ids + layout.slice.start
    else:
        return layout.indices[local_ids]


def get_env_ids(layout: GroupLayout) -> torch.Tensor:
    """Get all env IDs for this group."""
    if layout.slice is not None:
        return torch.arange(layout.slice.start, layout.slice.stop, dtype=torch.long, device=layout.device)
    else:
        return layout.indices


def compute_read_index(
    group_layout: GroupLayout,
    asset_env_ids: torch.Tensor | None,
    group_idx: int,
    asset_group_idxs: tuple[int, ...] | None,
) -> slice | torch.Tensor:
    """Compute read index into asset buffer."""
    if asset_group_idxs is None:
        return group_layout.slice if group_layout.slice is not None else group_layout.indices
    if len(asset_group_idxs) == 1 and asset_group_idxs[0] == group_idx:
        return slice(None)
    if group_layout.slice is not None:
        start = torch.searchsorted(asset_env_ids, group_layout.slice.start).item()
        return slice(start, start + group_layout.count)
    else:
        return torch.searchsorted(asset_env_ids, group_layout.indices)


def build_layout(env_ids: list[int], device: str) -> GroupLayout:
    """Build a GroupLayout from env_ids list."""
    if len(env_ids) == 0:
        return GroupLayout(offset=0, count=0, slice=slice(0, 0), indices=None, device=device)
    elif is_contiguous_slice(env_ids):
        return GroupLayout(
            offset=env_ids[0],
            count=len(env_ids),
            slice=slice(env_ids[0], env_ids[-1] + 1),
            indices=None,
            device=device,
        )
    else:
        return GroupLayout(
            offset=env_ids[0],
            count=len(env_ids),
            slice=None,
            indices=torch.tensor(env_ids, dtype=torch.long, device=device),
            device=device,
        )


def resolve_asset_env_ids(
    layout: EnvLayout,
    asset_name: str,
    env_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Map global env_ids to local asset indices for heterogeneous layouts.

    Pure function: composes from layout's raw data.

    Args:
        layout: The EnvLayout instance.
        asset_name: Name of the asset.
        env_ids: Global env indices, or None for all envs.

    Returns:
        Local indices for this asset, or None if no envs apply.
    """
    asset_groups = layout.assets.get(asset_name)

    # Homogeneous or unregistered asset: spans all envs
    if not asset_groups and not layout.group_names:
        return env_ids

    # No groups registered for this asset but layout is heterogeneous
    if not asset_groups:
        return env_ids

    # Heterogeneous: filter to asset's groups and collect local indices
    if env_ids is None:
        env_ids = torch.arange(layout.num_envs, dtype=torch.long, device=layout._device)

    local_ids = []
    for group_name in asset_groups:
        local, _ = layout[group_name].filter(env_ids)
        if local.numel() > 0:
            local_ids.append(local)

    if not local_ids:
        return None
    return torch.cat(local_ids) if len(local_ids) > 1 else local_ids[0]


# ============================================================================
# Layer 2: GroupView Wrapper (delegates to pure functions)
# ============================================================================


@dataclass
class GroupView:
    """Ergonomic wrapper around GroupLayout - delegates to pure functions.

    ``write`` indexes into a full-env output buffer (shape ``(num_envs, ...)``).
    ``read`` indexes into the asset's data buffer.
    """

    write: slice | torch.Tensor
    """Index into a full-env ``(num_envs, ...)`` output buffer."""

    read: slice | torch.Tensor
    """Index into the asset's data buffer."""

    layout: GroupLayout = field(repr=False)
    """The underlying GroupLayout - exposed for pure function composition."""

    def filter(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter env_ids to this group. Returns (local_ids, matched_global_ids)."""
        return filter_to_group(self.layout, env_ids)

    def to_local(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Convert global env_ids to 0-based local indices."""
        return to_local(self.layout, env_ids)

    def to_global(self, local_ids: torch.Tensor) -> torch.Tensor:
        """Convert local indices back to global env_ids."""
        return to_global(self.layout, local_ids)

    @property
    def env_ids(self) -> torch.Tensor:
        """All env IDs in this group."""
        return get_env_ids(self.layout)

    @property
    def count(self) -> int:
        """Number of envs in this group."""
        return self.layout.count


# ============================================================================
# Layer 3: EnvLayout (minimal state, exposes raw data for composition)
# ============================================================================


_HOMOGENEOUS_LAYOUT = GroupLayout(offset=0, count=0, slice=slice(None), indices=None, device="cpu")


class EnvLayout:
    """Centralized environment partitioning for heterogeneous multi-task scenes.

    Minimal API - exposes raw data for caller composition.

    Example::

        # Indexing
        gv = layout["lift", "robot"]
        output[gv.write] = robot.data[gv.read]

        # Composition from raw data
        if (g := layout.assets.get("robot")) and len(g) == 1:
            ...  # asset is exclusive to one group

        # Pure function usage
        local, matched = filter_to_group(gv.layout, env_ids)
    """

    def __init__(self, num_envs: int, device: str = "cuda:0"):
        self._num_envs = num_envs
        self._device = device
        self._layouts: dict[str, GroupLayout] = {}
        self._group_names: tuple[str, ...] = ()
        self._assets: dict[str, tuple[str, ...]] = {}
        self._terms: dict[str, str] = {}
        self._clone_cfg: CloneCfg | None = None

    # ── fundamental properties ────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        """Total number of environments."""
        return self._num_envs

    @property
    def group_names(self) -> tuple[str, ...]:
        """Names of all registered groups (immutable)."""
        return self._group_names

    # ── raw data access for composition ───────────────────────────────────

    @property
    def layouts(self) -> Mapping[str, GroupLayout]:
        """Read-only mapping of group name → GroupLayout."""
        return MappingProxyType(self._layouts)

    @property
    def assets(self) -> Mapping[str, tuple[str, ...]]:
        """Read-only mapping of asset name → tuple of group names."""
        return MappingProxyType(self._assets)

    @property
    def terms(self) -> Mapping[str, str]:
        """Read-only mapping of term ID → group name."""
        return MappingProxyType(self._terms)

    # ── setup ─────────────────────────────────────────────────────────────

    def apply_clone_cfg(self, clone_cfg: CloneCfg) -> None:
        """Store clone config for later use."""
        if self._clone_cfg is not None:
            raise RuntimeError("Clone config has already been applied.")
        self._clone_cfg = clone_cfg

    def apply_assignment(
        self,
        assignment: torch.Tensor,
        group_names: tuple[str, ...],
        group_assets: dict[str, list[str]] | None = None,
    ) -> None:
        """Populate layout from assignment tensor."""
        assignment = assignment.to(device=self._device, dtype=torch.long)

        if group_assets is None and self._clone_cfg is not None:
            group_assets = {name: inc.assets for name, inc in self._clone_cfg.clone_groups.items()}

        self._group_names = group_names
        self._layouts = {}

        for idx, name in enumerate(group_names):
            env_ids = (assignment == idx).nonzero(as_tuple=True)[0].tolist()
            self._layouts[name] = build_layout(env_ids, self._device)

            if group_assets and name in group_assets:
                for asset_name in group_assets[name]:
                    self._add_asset_to_group(asset_name, name)

    def register(self, key: str, env_ids: torch.Tensor) -> None:
        """Register a named environment partition."""
        env_ids_list = env_ids.tolist()

        if any(i < 0 or i >= self._num_envs for i in env_ids_list):
            raise ValueError(f"env_ids for '{key}' out of range [0, {self._num_envs})")
        if len(set(env_ids_list)) != len(env_ids_list):
            raise ValueError(f"env_ids for '{key}' contain duplicates")

        self._layouts[key] = build_layout(env_ids_list, self._device)
        if key not in self._group_names:
            self._group_names = (*self._group_names, key)

    def register_asset(self, asset_name: str, group_key: str) -> None:
        """Register an asset to a group."""
        if group_key in self._layouts:
            self._add_asset_to_group(asset_name, group_key)

    def _add_asset_to_group(self, asset_name: str, group_key: str) -> None:
        current = self._assets.get(asset_name, ())
        if group_key not in current:
            self._assets[asset_name] = (*current, group_key)

    def register_term(self, term_id: str, group_key: str) -> None:
        """Register a manager-term to a group."""
        if group_key in self._layouts:
            self._terms[term_id] = group_key

    # ── single entry point ────────────────────────────────────────────────

    def __getitem__(self, key: str | tuple[str, str | None]) -> GroupView:
        """Get GroupView: layout["lift"] or layout["lift", "robot"]."""
        if isinstance(key, str):
            return self._make_view(key, None)
        return self._make_view(key[0], key[1])

    def group_view(self, group_key: str | None, asset_name: str | None = None) -> GroupView:
        """Get GroupView for a group/asset pair. Alias for __getitem__."""
        return self._make_view(group_key, asset_name)

    def _make_view(self, group_key: str | None, asset_name: str | None) -> GroupView:
        if group_key is None:
            return GroupView(write=slice(None), read=slice(None), layout=_HOMOGENEOUS_LAYOUT)

        layout = self._layouts.get(group_key)
        if layout is None:
            raise KeyError(f"unregistered group '{group_key}'. Available: {list(self._group_names)}")

        write = layout.slice if layout.slice is not None else layout.indices
        read = self._compute_read(group_key, asset_name, layout)
        return GroupView(write=write, read=read, layout=layout)

    def _compute_read(self, group_key: str, asset_name: str | None, layout: GroupLayout) -> slice | torch.Tensor:
        if asset_name is None:
            return slice(None)

        asset_groups = self._assets.get(asset_name)
        if asset_groups is None:
            return layout.slice if layout.slice is not None else layout.indices

        group_idx = self._group_names.index(group_key)
        asset_group_idxs = tuple(self._group_names.index(g) for g in asset_groups)

        if len(asset_group_idxs) == 1 and asset_group_idxs[0] == group_idx:
            return slice(None)

        asset_env_ids = self._get_asset_env_ids(asset_name)
        return compute_read_index(layout, asset_env_ids, group_idx, asset_group_idxs)

    def _get_asset_env_ids(self, asset_name: str) -> torch.Tensor:
        asset_groups = self._assets.get(asset_name, ())
        if not asset_groups:
            return torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        tensors = [get_env_ids(self._layouts[g]) for g in asset_groups]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors).unique().sort().values

    def __repr__(self) -> str:
        groups = ", ".join(f"{n}: {self._layouts[n].count} envs" for n in self._group_names)
        return f"EnvLayout(num_envs={self._num_envs}, groups=[{groups or 'homogeneous'}])"
