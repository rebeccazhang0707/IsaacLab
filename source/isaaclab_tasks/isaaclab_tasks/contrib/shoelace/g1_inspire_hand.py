# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Straight Inspire fingers and invisible pinch pads for the G1 shoelace task."""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils

from .shoelace_assets import ShoelaceUsdCfg, spawn_shoelace_usd

# Calibrated in wrist coordinates with downstream finger joints held at zero.
INDEX_CLOSED = (1.225962774, 1.348051220)
THUMB_PITCH = (0.4999, 0.408370270)
THUMB_YAW = (1.233697987, 1.297461830)
TCP_BODY_OFFSETS = (((0.193129098, -0.063929696, 0.029342429),), ((0.186283631, 0.070964776, 0.024288309),))
PAD_OFFSETS = (
    (0.014079444, -0.015321529, -0.006426900),
    (-0.004842526, -0.035955938, -0.002620689),
    (0.011255558, 0.013833699, -0.010036920),
    (-0.004698405, 0.041790697, -0.006952743),
)
STRAIGHT_JOINT_NAMES = tuple(
    f"{side}_{part}_joint"
    for side in ("L", "R")
    for part in ("thumb_intermediate", "thumb_distal", "index_intermediate")
)
# Body, box centre [m], xyzw orientation. Opposing faces have a nominal 2.8 mm closed gap.
_PINCH_BOXES = (
    (
        "L_thumb_distal",
        (0.014210405, -0.013827257, -0.006426735),
        (0.702162359, -0.083801980, 0.022175560, 0.706720238),
    ),
    (
        "L_index_intermediate",
        (-0.003395612, -0.035604817, -0.002802767),
        (0.581729113, -0.471470177, 0.401984128, -0.526987544),
    ),
    (
        "R_thumb_distal",
        (0.011382208, 0.012385734, -0.009666304),
        (0.789644536, 0.000454291, -0.053111143, -0.611261407),
    ),
    (
        "R_index_intermediate",
        (-0.003287404, 0.041405200, -0.006620379),
        (0.507093701, 0.363411431, 0.492804199, 0.606574094),
    ),
)


@sim_utils.clone
def spawn_g1_shoelace(
    prim_path: str,
    cfg: ShoelaceUsdCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn G1 with task-local straight joints and 16 x 10 x 3 mm collision pads.

    Terminal thumb/index mesh colliders are replaced by invisible boxes bound to the same
    physics materials. Downstream mimic constraints are removed and their revolute limits
    restricted to +/-0.001 degrees. The source asset, masses and inertias remain unchanged.
    All modifications precede environment cloning and Newton import.
    """
    root = spawn_shoelace_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    straight_joints = set()
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in STRAIGHT_JOINT_NAMES:
            for schema in list(prim.GetAppliedSchemas()):
                if "Mimic" in schema:
                    prim.RemoveAppliedSchema(schema)
            for relationship in prim.GetRelationships():
                if "mimic" in relationship.GetName().lower():
                    relationship.SetTargets([])
            joint = UsdPhysics.RevoluteJoint(prim)
            joint.CreateLowerLimitAttr(-0.001)
            joint.CreateUpperLimitAttr(0.001)
            straight_joints.add(prim.GetName())
    if straight_joints != set(STRAIGHT_JOINT_NAMES):
        raise ValueError(f"G1 straight-finger joints missing: {set(STRAIGHT_JOINT_NAMES) - straight_joints}")

    for body_name, position, quaternion in _PINCH_BOXES:
        body = stage.GetPrimAtPath(root.GetPath().AppendChild(body_name))
        if not body:
            raise ValueError(f"G1 pinch-pad body missing: {body_name}")
        material = None
        for prim in Usd.PrimRange(body):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = UsdPhysics.CollisionAPI(prim)
                if collision.GetCollisionEnabledAttr().Get():
                    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                    if bound:
                        material = bound
                    collision.CreateCollisionEnabledAttr(False)
        if not material:
            raise ValueError(f"G1 pinch-pad body has no enabled collider with a physics material: {body_name}")
        box = UsdGeom.Cube.Define(stage, body.GetPath().AppendChild("pinch_box"))
        box.CreateSizeAttr(1.0)
        transform = UsdGeom.Xformable(box)
        transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        transform.AddOrientOp().Set(Gf.Quatf(quaternion[3], Gf.Vec3f(*quaternion[:3])))
        transform.AddScaleOp().Set(Gf.Vec3f(0.016, 0.010, 0.003))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material, materialPurpose="physics")
        box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return root
