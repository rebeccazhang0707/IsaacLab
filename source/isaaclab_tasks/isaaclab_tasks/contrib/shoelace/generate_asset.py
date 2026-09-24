# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the shoe/cable USD offline: ``uv run python -m isaaclab_tasks.contrib.shoelace.generate_asset``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import newton
from isaaclab_newton.sim.schemas import NewtonCollisionCfg

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import spawn_physics_material

from . import asset_authoring as physics
from . import shoelace_assets as assets
from . import shoelace_constants as constants


def generate_asset(output: Path) -> None:
    """Bake the validated cable model into a portable, environment-independent USD asset.

    Args:
        output: Destination USD file, normally ``data/shoelace.usda``.
    """
    stage = sim_utils.create_new_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = "/World/envs/env_0"
    UsdGeom.Xform.Define(stage, root)
    centerline, radius, reference_length = physics.load_shoelace()
    assets = [
        physics.shoe_asset_cfg(visible=True),
        physics.shoe_collider_asset_cfg(),
        physics.shoe_visual_asset_cfg(f"{root}/Materials/shoes"),
        physics.tongue_upper_asset_cfg(),
    ]
    materials = UsdGeom.Scope.Define(stage, f"{root}/Materials").GetPrim()
    materials.GetReferences().AddReference(str(constants.MODEL_ASSET), "/mat")
    for asset in assets:
        path = asset.prim_path.replace("{ENV_REGEX_NS}", root)
        asset.spawn.func(path, asset.spawn, translation=asset.init_state.pos, orientation=asset.init_state.rot)
    pinned = physics.pinned_shoelace_spawner(centerline, radius)
    pinned.func(f"{root}/Shoe/ShoelacePinned", pinned)
    for side, section in (
        ("Left", slice(None, constants.PINNED_FIRST + 2)),
        ("Right", slice(constants.PINNED_LAST, None)),
    ):
        spawn = physics.cable_spawn_cfg(
            centerline[section], cable_radius=radius, reference_segment_length=reference_length
        )
        spawn.func(f"{root}/Shoelace{side}", spawn)

    builder = newton.ModelBuilder(up_axis="Z")
    builder.begin_world()
    result = {"path_shape_map": {}, "path_cable_map": {}}
    for name in ("Shoe", "ShoelaceLeft", "ShoelaceRight"):
        imported = builder.add_usd(stage, root_path=f"{root}/{name}", return_deformable_results=True)
        for key in result:
            result[key].update(imported[key])
    builder.end_world()
    build = physics.configure_shoelace_builder(builder, centerline, radius)
    for side, bodies in zip(("Left", "Right"), build.body_chains, strict=True):
        prim = stage.GetPrimAtPath(f"{root}/Shoelace{side}/geometry/mesh")
        masses = [
            builder.body_mass[body] or builder.body_mass[bodies[1 if index == 0 else index - 1]]
            for index, body in enumerate(bodies)
        ]
        prim.CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
        prim.CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set("segment")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        _contact_material(material.GetPrim(), constants.LACE_MU)
        sim_utils.apply_namespaced(NewtonCollisionCfg(contact_gap=constants.CONTACT_GAP), str(prim.GetPath()), stage)

    # Keep the pinned span visible using a separate visual mesh, as with standard rigid assets.
    pinned_path = f"{root}/Shoe/ShoelacePinned/geometry/mesh"
    Sdf.CopySpec(stage.GetRootLayer(), pinned_path, stage.GetRootLayer(), pinned_path + "_visual")
    stage.GetPrimAtPath(pinned_path + "_visual").RemoveAPI(UsdPhysics.CollisionAPI)
    for path, friction in (
        (f"{root}/Shoe/Collider", constants.SHOE_MU),
        (f"{root}/Shoe/TongueUpper/geometry/mesh", constants.SHOE_MU),
        (pinned_path, constants.LACE_MU),
    ):
        prim = stage.GetPrimAtPath(path)
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        material = UsdShade.Material.Define(stage, path + "/ContactMaterial")
        _contact_material(material.GetPrim(), friction)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        sim_utils.apply_namespaced(NewtonCollisionCfg(contact_gap=constants.CONTACT_GAP), str(prim.GetPath()), stage)

    # Group all filters for each source pair into one native element-filter prim.
    sources = {shape: (path, None) for path, shape in result["path_shape_map"].items()}
    for path, (bodies, _) in result["path_cable_map"].items():
        sources.update(
            {shape: (path, index) for index, body in enumerate(bodies) for shape in builder.body_shapes[body]}
        )
    groups = {}
    for first, second in sorted(builder.shape_collision_filter_pairs):
        path0, index0 = sources[first]
        path1, index1 = sources[second]
        groups.setdefault((path0, path1), []).append((index0, index1))
    for number, ((path0, path1), pairs) in enumerate(sorted(groups.items())):
        prim = stage.DefinePrim(f"{root}/CollisionFilters/Filter_{number}", "PhysicsElementCollisionFilter")
        for side, path in enumerate((path0, path1)):
            indices = [pair[side] for pair in pairs]
            prim.CreateRelationship(f"physics:src{side}").SetTargets([path])
            prim.CreateAttribute(f"physics:groupElemCounts{side}", Sdf.ValueTypeNames.IntArray).Set(
                [0 if index is None else 1 for index in indices]
            )
            prim.CreateAttribute(f"physics:groupElemIndices{side}", Sdf.ValueTypeNames.IntArray).Set(
                [index for index in indices if index is not None]
            )

    # Flatten only this small composite asset; rebase relationships and texture paths for portability.
    output.parent.mkdir(parents=True, exist_ok=True)
    target = Usd.Stage.CreateInMemory()
    Sdf.CopySpec(stage.Flatten(), root, target.GetRootLayer(), "/Shoelace")
    target.SetDefaultPrim(target.GetPrimAtPath("/Shoelace"))
    UsdGeom.SetStageUpAxis(target, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(target, 1.0)
    for prim in target.Traverse():
        for relationship in prim.GetRelationships():
            relationship.SetTargets([path.ReplacePrefix(root, "/Shoelace") for path in relationship.GetTargets()])
        for attribute in prim.GetAttributes():
            if attribute.GetConnections():
                attribute.SetConnections([path.ReplacePrefix(root, "/Shoelace") for path in attribute.GetConnections()])
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset and (value := attribute.Get()):
                path = value.resolvedPath or value.path
                if os.path.isabs(path):
                    attribute.Set(Sdf.AssetPath(os.path.relpath(path, output.parent)))
    target.GetRootLayer().Export(str(output))
    if output.suffix == ".usda":
        output.write_text(output.read_text().rstrip() + "\n")


def _contact_material(prim: Usd.Prim, friction: float) -> None:
    spawn_physics_material(
        str(prim.GetPath()), assets.rigid_material(friction, constants.CONTACT_KD), stage=prim.GetStage()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=constants.ASSET_DIR / "shoelace.usda")
    generate_asset(parser.parse_args().output)
