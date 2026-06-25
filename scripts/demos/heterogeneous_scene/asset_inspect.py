# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure asset-cfg inspectors for the heterogeneous-scene demo.

Every function here is read-only ``AssetBaseCfg -> value``: it digs into a cfg's
``spawn`` to answer one question (its de-dup signature, display label, unique prim name,
or DOF count). Kept together so the single "read a spawn" idea lives in one place rather
than being re-derived across the harvest pipeline.

Import after :class:`~isaaclab.app.AppLauncher` has started (the isaaclab/pxr imports
require the running app).
"""

from __future__ import annotations

import re

from log import note_log

from isaaclab.assets import ArticulationCfg, AssetBaseCfg


def spawn_signature(cfg: AssetBaseCfg) -> str:
    """String identifying the physical model a spawn produces (USD set + scale)."""
    spawn = getattr(cfg, "spawn", None)
    if spawn is None:
        return f"{type(cfg).__name__}:no_spawn"
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd is not None:
        body = usd
    elif sub:
        body = "+".join(sorted(getattr(s, "usd_path", None) or type(s).__name__ for s in sub))
    else:
        body = type(spawn).__name__
    return f"{type(cfg).__name__}|{body}|scale={getattr(spawn, 'scale', None)}"


def model_label(cfg: AssetBaseCfg) -> str:
    """Short human-readable model name for reports (USD file, shape class, or variant set)."""
    spawn = getattr(cfg, "spawn", None)
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd:
        return usd.split("/")[-1]
    if sub:
        labels = [(getattr(s, "usd_path", None) or "").split("/")[-1] or type(s).__name__ for s in sub]
        return "{" + ", ".join(labels) + "}"
    return type(spawn).__name__ if spawn else "-"


def unique_prim_name(cfg: AssetBaseCfg, taken: set[str]) -> str:
    """A unique scene-field name (and prim leaf) derived from the asset's model."""
    spawn = getattr(cfg, "spawn", None)
    usd = getattr(spawn, "usd_path", None)
    sub = getattr(spawn, "assets_cfg", None)
    if usd:
        stem = usd.split("/")[-1].rsplit(".", 1)[0]
    elif sub:
        first_usd = getattr(sub[0], "usd_path", None)
        stem = (first_usd.split("/")[-1].rsplit(".", 1)[0] + "_multi") if first_usd else "multi_asset"
    elif spawn is not None:
        stem = type(spawn).__name__.replace("Cfg", "")
    else:
        stem = type(cfg).__name__
    base = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower() or "asset"
    name, i = base, 1
    while name in taken:
        i += 1
        name = f"{base}_{i}"
    taken.add(name)
    return name


# DOF count per spawn USD path, so a model shared by many tasks is inspected only once.
_DOF_CACHE: dict[str, int] = {}


def _spawn_usd_path(cfg: AssetBaseCfg) -> str | None:
    """First USD path of an asset's spawn (handles multi-asset / multi-USD spawners)."""
    spawn = getattr(cfg, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    if usd_path is None:  # multi-asset spawner: inspect the first variant that has a USD
        sub = getattr(spawn, "assets_cfg", None) or []
        usd_path = next((getattr(s, "usd_path", None) for s in sub if getattr(s, "usd_path", None)), None)
    if isinstance(usd_path, (list, tuple)):  # MultiUsdFileCfg may hold a list of paths
        usd_path = usd_path[0] if usd_path else None
    return usd_path or None


def articulation_dof_exceeds(cfg: AssetBaseCfg, max_dof: int) -> bool:
    """Whether an articulation cfg has more than ``max_dof`` DOFs (revolute + prismatic).

    Fast best-effort check for filtering heavy robots. It first opens the spawn USD with
    ``LoadNone`` (deferring payloads so the slow geometry load is skipped), stops counting
    the moment the threshold is crossed, and caches the per-USD result so a model shared by
    many tasks is inspected only once. If the fast pass finds *no* joints -- which happens
    when an asset (e.g. G1, GR1) authors its kinematic structure behind a payload -- it
    loads the payloads on the same stage and recounts, so such robots are still detected.
    Returns ``False`` for non-articulations or USDs that cannot be inspected, so an
    undeterminable asset is never filtered out.
    """
    if not isinstance(cfg, ArticulationCfg):
        return False
    usd_path = _spawn_usd_path(cfg)
    if not usd_path:
        return False

    count = _DOF_CACHE.get(usd_path)
    if count is None:
        try:
            from pxr import Usd, UsdPhysics

            def count_dofs(stage) -> int:
                """Count revolute + prismatic joints, stopping once ``max_dof`` is passed."""
                n = 0
                for prim in stage.Traverse():
                    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                        n += 1
                        if n > max_dof:  # a lower bound is enough; max_dof is fixed for the run
                            break
                return n

            stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadNone)
            count = 0
            if stage is not None:
                count = count_dofs(stage)
                if count == 0:  # joints likely behind deferred payloads -> load fully, recount once
                    stage.Load()
                    count = count_dofs(stage)
        except Exception as exc:  # noqa: BLE001 - an uninspectable asset must not abort discovery
            note_log(f"[note] could not inspect DOFs for {usd_path}: {type(exc).__name__}: {exc}")
            count = 0
        _DOF_CACHE[usd_path] = count
    return count > max_dof
