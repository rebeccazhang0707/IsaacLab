# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from isaaclab_newton.sim.schemas import (
    NewtonCablePropertiesCfg,
    NewtonCollisionCfg,
    apply_newton_cable_properties,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import CableMaterialCfg, UsdPhysicsRigidBodyMaterialCfg, spawn_physics_material
from isaaclab.sim.spawners.shapes import CableCfg

# Cable joint degrees of freedom, in the order Newton lays them out.
_STRETCH, _SHEAR, _BEND, _TWIST = range(4)


def _import_cable_joint_stiffness(**material_kwargs) -> list[float]:
    """Spawn a cable, import it into Newton, and return its first joint's per-DOF stiffness."""
    sim_utils.create_new_stage()
    stage = sim_utils.get_current_stage()
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0)],
        physics_material=CableMaterialCfg(
            thickness=0.02, density=1000.0, stretch_stiffness=1.0e6, bend_stiffness=2.0e4, **material_kwargs
        ),
    )
    cfg.func("/World/Cable", cfg)

    builder = newton.ModelBuilder()
    builder.add_usd(stage, root_path="/World/Cable")
    dof0 = builder.joint_qd_start[0]
    return [builder.joint_target_ke[dof0 + offset] for offset in range(4)]


def test_newton_cable_shear_and_twist_fall_back_when_unset():
    """Test that unset shear/twist reuse the stretch/bend stiffness."""
    stiffness = _import_cable_joint_stiffness()

    assert stiffness[_SHEAR] == pytest.approx(stiffness[_STRETCH])
    assert stiffness[_TWIST] == pytest.approx(stiffness[_BEND])


def test_newton_cable_authored_shear_and_twist_override_fallbacks():
    """Test that authored moduli decouple shear from stretch and twist from bend."""
    fallback = _import_cable_joint_stiffness()
    stiffness = _import_cable_joint_stiffness(shear_stiffness=9.0e9, twist_stiffness=7.0e7)

    # Stretch and bend are untouched; only the newly authored degrees of freedom move.
    assert stiffness[_STRETCH] == pytest.approx(fallback[_STRETCH])
    assert stiffness[_BEND] == pytest.approx(fallback[_BEND])
    assert stiffness[_SHEAR] > 100.0 * stiffness[_STRETCH]
    assert stiffness[_TWIST] > 100.0 * stiffness[_BEND]


def test_newton_cable_partial_and_zero_shear_twist():
    """Test that one authored modulus leaves the other's fallback intact and an authored zero is kept."""
    only_shear = _import_cable_joint_stiffness(shear_stiffness=9.0e9)
    assert only_shear[_SHEAR] > 100.0 * only_shear[_STRETCH]
    assert only_shear[_TWIST] == pytest.approx(only_shear[_BEND])

    zero_twist = _import_cable_joint_stiffness(twist_stiffness=0.0)
    assert zero_twist[_TWIST] == pytest.approx(0.0)
    assert zero_twist[_BEND] > 0.0


def test_newton_imports_cable_without_registry():
    """Test that Newton imports the authored cable directly from USD."""
    stage = sim_utils.create_new_stage()
    material = CableMaterialCfg(
        thickness=0.02,
        density=1234.0,
        stretch_stiffness=2.5e6,
        bend_stiffness=7.5e4,
    )
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.1, 0.0), (0.4, 0.0, 0.0)],
        physics_material=material,
    )
    cfg.func("/World/Cable", cfg)

    cable_path = "/World/Cable/geometry/mesh"
    cable_prim = stage.GetPrimAtPath(cable_path)
    assert not cable_prim.HasAttribute("connections")

    builder = newton.ModelBuilder()
    import_result = builder.add_usd(stage, root_path="/World/Cable", return_deformable_results=True)
    cable_attrs = import_result["path_cable_attrs"][cable_path]

    assert builder.body_count == 2
    assert builder.joint_count == 1
    assert builder.shape_count == 2
    assert import_result["path_cable_map"][cable_path] == ([0, 1], [0])
    assert cable_attrs["closed"] is False
    assert cable_attrs["material"]["thickness"] == pytest.approx(material.thickness)
    assert cable_attrs["material"]["density"] == pytest.approx(material.density)
    assert cable_attrs["material"]["stretchStiffness"] == pytest.approx(material.stretch_stiffness)
    assert cable_attrs["material"]["bendStiffness"] == pytest.approx(material.bend_stiffness)
    # Shear/twist are left unauthored when unset.
    assert "shearStiffness" not in cable_attrs["material"]
    assert "twistStiffness" not in cable_attrs["material"]


def test_newton_imports_cable_with_adjacent_collision_filtering():
    """Test that only connected cable segments are collision-filtered."""
    stage = sim_utils.create_new_stage()
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0)],
        physics_material=CableMaterialCfg(),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )
    cfg.func("/World/Cable", cfg)

    builder = newton.ModelBuilder()
    builder.add_usd(stage, root_path="/World/Cable")
    filter_pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}

    assert all(flag & newton.ShapeFlags.COLLIDE_SHAPES for flag in builder.shape_flags)
    assert filter_pairs == {(0, 1), (1, 2)}


def _cable_stage() -> Usd.Stage:
    stage = sim_utils.create_new_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cfg = CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0)],
        physics_material=CableMaterialCfg(),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )
    cfg.func("/World/Cable", cfg)
    return stage


def _source_builder(stage: Usd.Stage) -> newton.ModelBuilder:
    return build_source_builders(stage, ["/World/Cable"], newton.ModelBuilder, [])["/World/Cable"]


def test_cable_dahl_and_native_contact_properties_survive_usd_roundtrip_and_replication(tmp_path):
    """Preserve Dahl arrays and native masses, contacts, and filters through replication."""
    stage = _cable_stage()
    path = "/World/Cable/geometry/mesh"
    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([0.1, 0.2, 0.3])
    prim.CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set("segment")
    native = newton.ModelBuilder()
    native.add_usd(stage, root_path="/World/Cable")
    inertias = np.asarray(native.body_inertia).reshape(3, 3, 3)
    apply_newton_cable_properties(NewtonCablePropertiesCfg(dahl_max_strains=[0.2, 0.3]), path, stage)
    # Partial fragment overrides keep the other Dahl array and all native properties.
    apply_newton_cable_properties(NewtonCablePropertiesCfg(dahl_decay=[0.4, 0.5]), path, stage)
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    spawn_physics_material(
        str(material.GetPath()),
        [
            UsdPhysicsRigidBodyMaterialCfg(dynamic_friction=0.7),
            NewtonMaterialCfg(contact_stiffness=123.0, contact_damping=4.0),
        ],
        stage=stage,
    )
    sim_utils.apply_namespaced(NewtonCollisionCfg(contact_gap=0.001), path, stage)
    filter_prim = stage.DefinePrim("/World/Cable/Filter", "PhysicsElementCollisionFilter")
    for side, segment in ((0, 0), (1, 2)):
        filter_prim.CreateRelationship(f"physics:src{side}").SetTargets([path])
        filter_prim.CreateAttribute(f"physics:groupElemCounts{side}", Sdf.ValueTypeNames.IntArray).Set([1])
        filter_prim.CreateAttribute(f"physics:groupElemIndices{side}", Sdf.ValueTypeNames.IntArray).Set([segment])
    asset = tmp_path / "cable.usda"
    stage.GetRootLayer().Export(str(asset))
    source = _source_builder(Usd.Stage.Open(str(asset)))
    native_contacts = newton.ModelBuilder()
    native_contacts.add_usd(stage, root_path="/World/Cable")
    builder = newton.ModelBuilder()
    builder.replicate(source, 2, xforms=[wp.transform(), wp.transform(wp.vec3(1, 2, 3), wp.quat(0, 0, 1, 0))])

    np.testing.assert_allclose(builder.body_mass, [0.1, 0.2, 0.3] * 2)
    np.testing.assert_allclose(
        np.asarray(builder.body_inertia).reshape(6, 3, 3),
        np.tile(inertias, (2, 1, 1)),
    )
    np.testing.assert_allclose(builder.joint_target_ke, np.tile(native.joint_target_ke, 2))
    np.testing.assert_allclose(builder.joint_target_kd, np.tile(native.joint_target_kd, 2))
    # Generic import follows Newton's contact semantics without an Isaac Lab override.
    for field in ("shape_material_mu", "shape_material_ke", "shape_material_kd", "shape_gap"):
        np.testing.assert_allclose(getattr(builder, field), np.tile(getattr(native_contacts, field), 2))
    model = builder.finalize(device="cpu")
    np.testing.assert_allclose(model.vbd.dahl_eps_max.numpy(), [0.2, 0.3] * 2)
    np.testing.assert_allclose(model.vbd.dahl_tau.numpy(), [0.4, 0.5] * 2)
    assert set(builder.shape_collision_filter_pairs) == {(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)}


@pytest.mark.parametrize(
    ("name", "usd_type", "value"),
    [
        ("dahlMaxStrains", Sdf.ValueTypeNames.FloatArray, [0.1]),
        ("dahlMaxStrains", Sdf.ValueTypeNames.FloatArray, [-0.1, 0.2]),
        ("dahlDecay", Sdf.ValueTypeNames.FloatArray, [0.1, float("nan")]),
    ],
)
def test_invalid_cable_physics_arrays_fail_import(name, usd_type, value):
    stage = _cable_stage()
    stage.GetPrimAtPath("/World/Cable/geometry/mesh").CreateAttribute(f"isaaclab:cable:{name}", usd_type).Set(value)
    with pytest.raises(ValueError, match=name):
        _source_builder(stage)


@pytest.mark.parametrize("legacy_overrides", [False, True])
def test_cable_without_dahl_preserves_native_import(legacy_overrides):
    """The importer no longer consumes task-specific body and joint USD overrides."""
    stage = _cable_stage()
    if legacy_overrides:
        prim = stage.GetPrimAtPath("/World/Cable/geometry/mesh")
        for name, usd_type, value in (
            ("fixedSegments", Sdf.ValueTypeNames.IntArray, [0]),
            ("inertiaRegularization", Sdf.ValueTypeNames.Double, 0.001),
            ("segmentOrientations", Sdf.ValueTypeNames.Double4Array, [Gf.Vec4d(0, 0, 0, 1)] * 3),
            ("jointStiffnesses", Sdf.ValueTypeNames.Double4Array, [Gf.Vec4d(123)] * 2),
            ("jointDampings", Sdf.ValueTypeNames.Double4Array, [Gf.Vec4d(456)] * 2),
        ):
            prim.CreateAttribute(f"isaaclab:cable:{name}", usd_type).Set(value)
    native = newton.ModelBuilder()
    native.add_usd(stage, root_path="/World/Cable")
    imported = _source_builder(stage)
    for name in ("body_mass", "body_inertia", "body_q", "joint_target_ke", "joint_target_kd", "shape_gap"):
        np.testing.assert_array_equal(getattr(imported, name), getattr(native, name))


@pytest.mark.parametrize(
    "name", ["fixed_segments", "inertia_regularization", "segment_orientations", "joint_stiffnesses", "joint_dampings"]
)
def test_legacy_cable_fragment_fields_require_startup_events(name):
    stage = _cable_stage()
    cfg = NewtonCablePropertiesCfg(**{name: 0.001 if name == "inertia_regularization" else []})
    with pytest.raises(ValueError, match="use a startup event"):
        apply_newton_cable_properties(cfg, "/World/Cable/geometry/mesh", stage)


def test_legacy_fixed_fixture_attribute_does_not_override_native_body():
    stage = _cable_stage()
    cube = UsdGeom.Cube.Define(stage, "/World/Cable/Fixture").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(cube).CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(cube)
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(1.0)
    cube.CreateAttribute("isaaclab:physics:fixed", Sdf.ValueTypeNames.Bool).Set(True)
    builder = _source_builder(stage)
    body = builder.body_label.index(str(cube.GetPath()))
    native = newton.ModelBuilder()
    native.add_usd(stage, root_path="/World/Cable")
    native_body = native.body_label.index(str(cube.GetPath()))
    for name in ("body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia"):
        np.testing.assert_array_equal(getattr(builder, name)[body], getattr(native, name)[native_body])
    assert builder.body_shapes[body]


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_cable_dahl_accepts_rotation_and_rejects_unbaked_scale(scale):
    stage = _cable_stage()
    root = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cable"))
    root.AddRotateZOp().Set(180.0)
    root.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3d(scale))
    curve = stage.GetPrimAtPath("/World/Cable/geometry/mesh")
    curve.CreateAttribute("isaaclab:cable:dahlMaxStrains", Sdf.ValueTypeNames.FloatArray).Set([0.2, 0.3])
    if scale != 1.0:
        with pytest.raises(ValueError, match="bake scale/shear"):
            _source_builder(stage)
    else:
        builder = _source_builder(stage)
        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.vbd.dahl_eps_max.numpy(), [0.2, 0.3])
