# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import construction-time Dahl attributes before Newton asset replication.

Task-specific body and joint overrides belong in manager-based startup events.
The remaining USD contract is documented in ``docs/source/how-to/newton_cable_assets.rst``.
"""

from __future__ import annotations

from typing import Any

import newton
import numpy as np

from pxr import Usd, UsdGeom


def _add_usd(builder: newton.ModelBuilder, stage: Usd.Stage, **kwargs: Any) -> dict:
    """Import native USD properties and populate Dahl attributes before solver construction."""
    result = builder.add_usd(stage, return_deformable_results=True, **kwargs)
    for path, (bodies, joints) in result.get("path_cable_map", {}).items():
        prim = stage.GetPrimAtPath(path)
        if prim:
            _apply_cable_dahl_properties(builder, prim, bodies, joints)
    return result


def _apply_cable_dahl_properties(
    builder: newton.ModelBuilder, prim: Usd.Prim, bodies: list[int], joints: list[int]
) -> None:
    values_by_name = {}
    for usd_name, name in (("dahlMaxStrains", "vbd:dahl_eps_max"), ("dahlDecay", "vbd:dahl_tau")):
        value = prim.GetAttribute(f"isaaclab:cable:{usd_name}").Get()
        if value is None:
            continue
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (len(joints),) or not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{prim.GetPath()}: {usd_name} must have finite nonnegative shape {(len(joints),)}")
        values_by_name[name] = values
    if not values_by_name:
        return

    curves = UsdGeom.BasisCurves(prim)
    if list(curves.GetCurveVertexCountsAttr().Get()) != [len(bodies) + 1] or len(joints) != len(bodies) - 1:
        raise ValueError(f"{prim.GetPath()}: Dahl arrays require one standalone open curve")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    linear = np.asarray(transform)[:3, :3]
    if not np.allclose(linear @ linear.T, np.eye(3), atol=1.0e-6) or np.linalg.det(linear) < 0.0:
        raise ValueError(f"{prim.GetPath()}: bake scale/shear before authoring Dahl arrays")
    if UsdGeom.GetStageMetersPerUnit(prim.GetStage()) != 1.0:
        raise ValueError(f"{prim.GetPath()}: Dahl arrays require metersPerUnit=1")
    if any(builder.joint_type[joint] != newton.JointType.ROD for joint in joints):
        raise ValueError(f"{prim.GetPath()}: Dahl arrays require ROD joints")

    # Native curve import does not populate these joint attributes. Startup is too late:
    # VBD decides whether to enable Dahl when constructed, before startup events run.
    if "vbd:dahl_eps_max" not in builder.custom_attributes:
        newton.solvers.SolverVBD.register_custom_attributes(builder)
    for name, values in values_by_name.items():
        builder.custom_attributes[name].values.update(zip(joints, values.tolist(), strict=True))
