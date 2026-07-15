# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections.abc import Callable

from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.cloner import InclusionSet
from isaaclab.sensors import SensorBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import find_unique_string_name
from isaaclab.utils.configclass import configclass

from .interactive_scene_cfg import InteractiveSceneCfg

_ENV_ROOT = "{ENV_REGEX_NS}/"
_SLOT_METADATA = "_scene_add_slots"


def scene_add(
    scene_cfg1: InteractiveSceneCfg,
    scene_cfg2: InteractiveSceneCfg,
    *,
    asset_skip: Callable[[AssetBaseCfg], bool] | None = None,
) -> InteractiveSceneCfg:
    """Add one scene's environment assets to another as a new clone combination.

    The first operand owns the scene execution settings and its existing clone
    combinations; the second contributes one new combination. Both operands
    contribute spawned physics-backed asset fields at literal
    ``{ENV_REGEX_NS}/Leaf`` roots. Field names are logical slots: an equal
    definition in an existing slot reuses that binding, while a different
    definition receives a unique field name and prim path. Sensors, terrain
    importers, and spawnless assets are ignored. Global assets are not
    composed: skip them with :paramref:`asset_skip` and attach shared world
    assets to the returned scene afterwards.

    Args:
        scene_cfg1: Scene containing the existing clone combinations.
        scene_cfg2: Scene whose environment assets are added. It must not
            declare clone combinations of its own.
        asset_skip: Optional predicate called with each spawned asset
            configuration. An asset is omitted when the predicate returns
            :data:`True`. The predicate must not mutate the configuration.

    Returns:
        A new scene configuration containing both operands' clone combinations.

    Raises:
        TypeError: If an operand or a spawned scene field type is unsupported.
        ValueError: If an operand has no environment asset, contains a global
            or malformed asset root or a rigid-object collection, or if
            existing clone combinations reference unknown assets.
    """
    for name, scene_cfg in (("scene_cfg1", scene_cfg1), ("scene_cfg2", scene_cfg2)):
        if not isinstance(scene_cfg, InteractiveSceneCfg):
            raise TypeError(f"{name} must be an InteractiveSceneCfg, got {type(scene_cfg).__name__}.")
        scene_cfg.validate()
    if scene_cfg2.clone_cfg.clone_combinations:
        raise ValueError("scene_cfg2 must not declare clone combinations; fold it as scene_cfg1 instead.")

    left = _scene_assets(scene_cfg1, asset_skip)
    right = _scene_assets(scene_cfg2, asset_skip)
    for name, assets in (("scene_cfg1", left), ("scene_cfg2", right)):
        if not assets:
            raise ValueError(f"{name} must contain at least one spawned environment asset.")

    entities = {name: cfg for name, _, cfg in left}
    definitions = {name: _asset_definition(cfg) for name, _, cfg in left}
    slots = {name: slot for name, slot, _ in left}
    paths = {cfg.prim_path for cfg in entities.values()}
    if len(paths) != len(entities):
        raise ValueError("scene_cfg1 contains duplicate asset roots.")

    used_names = set(dir(InteractiveSceneCfg)) | set(entities)
    right_name_map: dict[str, str] = {}
    for source_name, slot, cfg in right:
        definition = _asset_definition(cfg)
        target_name = next(
            (
                name
                for name, existing in definitions.items()
                if slots[name] == slot and type(existing) is type(definition) and existing == definition
            ),
            None,
        )
        if target_name is None:
            target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
            cfg.prim_path = find_unique_string_name(cfg.prim_path, lambda path: path not in paths)
            entities[target_name] = cfg
            definitions[target_name] = definition
            slots[target_name] = slot
            used_names.add(target_name)
            paths.add(cfg.prim_path)
        right_name_map[source_name] = target_name

    rows = _scene_rows(scene_cfg1, [name for name, _, _ in left])
    rows.append(InclusionSet(assets=list(dict.fromkeys(right_name_map.values()))))
    return _make_scene(scene_cfg1, entities, slots, rows)


def _scene_assets(
    scene_cfg: InteractiveSceneCfg,
    asset_skip: Callable[[AssetBaseCfg], bool] | None,
) -> list[tuple[str, str, AssetBaseCfg]]:
    """Copy the environment-scoped spawned assets of one scene."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    slots = getattr(type(scene_cfg), _SLOT_METADATA, {})
    assets = []
    for name, value in vars(scene_cfg).items():
        if name in base_fields or value is None or isinstance(value, (SensorBaseCfg, TerrainImporterCfg)):
            continue
        if isinstance(value, RigidObjectCollectionCfg):
            raise ValueError(f"scene_add does not support rigid-object collection field {name!r}.")
        if not isinstance(value, AssetBaseCfg):
            raise TypeError(f"scene_add does not support scene field {name!r} of type {type(value).__name__}.")
        if value.spawn is None or (asset_skip is not None and asset_skip(value)):
            continue
        if type(value) is AssetBaseCfg:
            raise TypeError(
                f"scene_add does not support env-scoped plain AssetBaseCfg field {name!r}. Use a physics-backed "
                "configuration such as RigidObjectCfg or ArticulationCfg, or skip global static assets with "
                "asset_skip and attach them to the composed scene afterwards."
            )

        prim_path = value.prim_path
        if not (
            isinstance(prim_path, str)
            and prim_path.startswith(_ENV_ROOT)
            and prim_path[len(_ENV_ROOT) :].isidentifier()
        ):
            raise ValueError(
                f"scene_add composes assets at literal '{{ENV_REGEX_NS}}/Leaf' roots; {name!r} uses {prim_path!r}."
                " Skip global assets with asset_skip and attach shared world assets to the composed scene."
            )
        assets.append((name, slots.get(name, name), copy.deepcopy(value)))
    return assets


def _asset_definition(cfg: AssetBaseCfg) -> AssetBaseCfg:
    """Remove the output binding before comparing two asset configs."""
    definition = copy.deepcopy(cfg)
    definition.prim_path = "{SCENE_SLOT}"
    return definition


def _scene_rows(scene_cfg: InteractiveSceneCfg, asset_names: list[str]) -> list[InclusionSet]:
    """Return one operand's existing clone rows, or one row covering all assets."""
    combinations = scene_cfg.clone_cfg.clone_combinations
    if not combinations:
        return [InclusionSet(assets=asset_names)]

    unknown = sorted({name for row in combinations for name in row.assets} - set(asset_names))
    if unknown:
        raise ValueError(f"Clone combinations reference unknown or skipped scene fields: {unknown}.")
    return [InclusionSet(assets=list(row.assets), weight=row.weight) for row in combinations]


def _make_scene(
    source: InteractiveSceneCfg,
    entities: dict[str, AssetBaseCfg],
    slots: dict[str, str],
    rows: list[InclusionSet],
) -> InteractiveSceneCfg:
    """Build a configclass that InteractiveScene can parse normally."""
    clone_cfg = copy.deepcopy(source.clone_cfg)
    clone_cfg.clone_combinations = rows
    values = {name: copy.deepcopy(getattr(source, name)) for name in InteractiveSceneCfg.__dataclass_fields__}
    values["clone_cfg"] = clone_cfg
    namespace = {"__module__": __name__, "__doc__": "Scene configuration produced by scene_add.", **entities}
    scene_type = configclass(type("_AddedSceneCfg", (InteractiveSceneCfg,), namespace))
    scene_cfg = scene_type(**values)
    setattr(scene_type, _SLOT_METADATA, slots)
    return scene_cfg
