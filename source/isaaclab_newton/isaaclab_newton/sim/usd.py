# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import optional Isaac Lab physics attributes before Newton asset replication.

The USD contract is documented in ``docs/source/how-to/newton_cable_assets.rst``.
Indices are local to the imported curve, never to a replicated Newton model.
"""

from __future__ import annotations

from typing import Any

import newton
import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, UsdShade


def _add_usd(builder: newton.ModelBuilder, stage: Usd.Stage, **kwargs: Any) -> dict:
    """Import USD and apply asset-local physics properties to the source builder."""
    result = builder.add_usd(stage, return_deformable_results=True, **kwargs)
    for path, (bodies, joints) in result.get("path_cable_map", {}).items():
        prim = stage.GetPrimAtPath(path)
        if prim:
            _apply_cable_properties(builder, prim, bodies, joints)
    for path, body in result.get("path_body_map", {}).items():
        if stage.GetPrimAtPath(path).GetAttribute("isaaclab:physics:fixed").Get():
            _fix_body(builder, body)
    return result


def _array(prim: Usd.Prim, name: str, shape: tuple[int, ...], nonnegative: bool = False) -> np.ndarray | None:
    value = prim.GetAttribute(f"isaaclab:cable:{name}").Get()
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all() or (nonnegative and np.any(array < 0.0)):
        raise ValueError(
            f"{prim.GetPath()}: {name} must have finite {'nonnegative ' if nonnegative else ''}shape {shape}"
        )
    return array


def _apply_cable_properties(builder: newton.ModelBuilder, prim: Usd.Prim, bodies: list[int], joints: list[int]) -> None:
    # Validate all authored arrays before changing this curve's model properties.
    count = len(bodies)
    fixed = prim.GetAttribute("isaaclab:cable:fixedSegments").Get()
    fixed = [] if fixed is None else list(fixed)
    if any(not isinstance(index, int) or not 0 <= index < count for index in fixed):
        raise ValueError(f"{prim.GetPath()}: fixedSegments must contain segment indices in [0, {count})")
    regularization = _array(prim, "inertiaRegularization", (), nonnegative=True)
    orientations = _array(prim, "segmentOrientations", (count, 4))
    if orientations is not None and not np.allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=2.0e-5):
        raise ValueError(f"{prim.GetPath()}: segmentOrientations must contain unit xyzw quaternions")
    stiffness = _array(prim, "jointStiffnesses", (len(joints), 4), nonnegative=True)
    damping = _array(prim, "jointDampings", (len(joints), 4), nonnegative=True)
    dahl = {
        "vbd:dahl_eps_max": _array(prim, "dahlMaxStrains", (len(joints),), nonnegative=True),
        "vbd:dahl_tau": _array(prim, "dahlDecay", (len(joints),), nonnegative=True),
    }
    has_arrays = any(array is not None for array in (orientations, regularization, stiffness, damping, *dahl.values()))
    if has_arrays or fixed:
        curves = UsdGeom.BasisCurves(prim)
        if list(curves.GetCurveVertexCountsAttr().Get()) != [count + 1] or len(joints) != count - 1:
            raise ValueError(f"{prim.GetPath()}: cable physics arrays require one standalone open curve")
    if has_arrays:
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        linear = np.asarray(transform)[:3, :3]
        if not np.allclose(linear @ linear.T, np.eye(3), atol=1.0e-6) or np.linalg.det(linear) < 0.0:
            raise ValueError(f"{prim.GetPath()}: bake scale/shear before authoring cable physics arrays")
        if UsdGeom.GetStageMetersPerUnit(prim.GetStage()) != 1.0:
            raise ValueError(f"{prim.GetPath()}: cable physics arrays require metersPerUnit=1")
        if any(builder.joint_type[joint] != newton.JointType.ROD for joint in joints):
            raise ValueError(f"{prim.GetPath()}: cable physics arrays require ROD joints")

    for index, body in enumerate(bodies):
        if regularization is not None and index not in fixed:
            builder.body_inertia[body] += wp.mat33(np.eye(3, dtype=np.float32) * float(regularization))
            builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])
        if orientations is not None:
            rotation = transform.ExtractRotationQuat()
            world_rotation = wp.quat(*rotation.GetImaginary(), rotation.GetReal())
            pose = builder.body_q[body]
            builder.body_q[body] = wp.transform(
                wp.transform_get_translation(pose), world_rotation * wp.quat(*orientations[index])
            )
        if index in fixed:
            _fix_body(builder, body)
    for index, joint in enumerate(joints):
        start = builder.joint_qd_start[joint]
        if stiffness is not None:
            builder.joint_target_ke[start : start + 4] = stiffness[index].tolist()
        if damping is not None:
            builder.joint_target_kd[start : start + 4] = damping[index].tolist()
    if any(value is not None for value in dahl.values()):
        if "vbd:dahl_eps_max" not in builder.custom_attributes:
            newton.solvers.SolverVBD.register_custom_attributes(builder)
        for name, values in dahl.items():
            if values is not None:
                builder.custom_attributes[name].values.update(zip(joints, values.tolist(), strict=True))

    _apply_cable_contact_properties(builder, prim, bodies)


def _apply_cable_contact_properties(builder: newton.ModelBuilder, prim: Usd.Prim, bodies: list[int]) -> None:
    # The curve importer creates capsule shapes but currently omits rigid-contact material fields.
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    if material:
        for usd_name, field in (
            ("physics:dynamicFriction", "shape_material_mu"),
            ("newton:contactStiffness", "shape_material_ke"),
            ("newton:contactDamping", "shape_material_kd"),
        ):
            attribute = material.GetPrim().GetAttribute(usd_name)
            if attribute.HasAuthoredValueOpinion():
                value = float(attribute.Get())
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{prim.GetPath()}: {usd_name} must be finite and nonnegative")
                for body in bodies:
                    for shape in builder.body_shapes[body]:
                        getattr(builder, field)[shape] = value
    gap_attr = prim.GetAttribute("newton:contactGap")
    gap = gap_attr.Get() if gap_attr.HasAuthoredValueOpinion() else None
    if gap is not None and gap != float("-inf"):
        if not np.isfinite(gap) or gap < 0.0:
            raise ValueError(f"{prim.GetPath()}: newton:contactGap must be nonnegative or -inf")
        for body in bodies:
            for shape in builder.body_shapes[body]:
                builder.shape_gap[shape] = float(gap)


def _fix_body(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = builder.body_inv_inertia[body] = wp.mat33()
