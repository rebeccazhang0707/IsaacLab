# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline geometry and model calculations used to generate the shoelace USD asset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from . import shoelace_constants as constants
from .shoelace_assets import rigid_material
from .shoelace_contacts import _find_chains

if TYPE_CHECKING:
    from newton import ModelBuilder


@dataclass
class ShoelaceBuild:
    """Ordered indices for the offline shoe/cable prototype.

    Attributes:
        body_chains: Left and right cable body indices in the prototype.
        joint_chains: Left and right cable joint indices in the prototype.
        mean_segment_length: Mean cable segment length [m].
    """

    body_chains: tuple[list[int], list[int]]
    joint_chains: tuple[list[int], list[int]]
    mean_segment_length: float


def load_usd_curve(path: Path, prim_path: str) -> tuple[np.ndarray, float]:
    """Load one world-space cable centerline [m] and its minimum authored radius [m]."""
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Missing prim {prim_path} in {path}")
    points = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).reshape(4, 4)
    centerline = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
    curves = UsdGeom.BasisCurves(prim)
    counts = list(curves.GetCurveVertexCountsAttr().Get())
    if counts != [len(centerline)]:
        raise ValueError(f"Expected one curve in {path}, got counts {counts}")
    widths = curves.GetWidthsAttr().Get()
    if not widths:
        raise ValueError(f"Curve in {path} has no widths attribute")
    return centerline, 0.5 * float(min(widths))


def resample_centerline(centerline: np.ndarray, num_segments: int) -> np.ndarray:
    """Resample a polyline [m] to uniformly spaced arc-length segments."""
    lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Shoelace centerline must not contain coincident adjacent points")
    arc_length = np.r_[0.0, np.cumsum(lengths)]
    samples = np.linspace(0.0, arc_length[-1], num_segments + 1)
    return np.column_stack([np.interp(samples, arc_length, centerline[:, axis]) for axis in range(3)])


def load_shoelace() -> tuple[np.ndarray, float, float]:
    """Load and resample the shared shoelace centerline.

    Returns:
        Resampled centerline [m], cable radius [m], and authored mean segment length [m].
    """
    authored_centerline, cable_radius = load_usd_curve(constants.CURVE_ASSET, "/World/Curve")
    authored_lengths = np.linalg.norm(np.diff(authored_centerline, axis=0), axis=1)
    if len(authored_lengths) != constants.AUTHORED_SEGMENT_COUNT:
        raise ValueError(f"Expected {constants.AUTHORED_SEGMENT_COUNT} authored segments, got {len(authored_lengths)}")
    centerline = resample_centerline(authored_centerline, constants.SHOELACE_SEGMENT_COUNT)
    return centerline, cable_radius, float(authored_lengths.mean())


def parallel_transport_normals(centerline: np.ndarray) -> list[tuple[float, float, float]]:
    """Build one normal per centerline point using Newton parallel-transport frames."""
    quaternions = newton.utils.create_parallel_transport_cable_quaternions([wp.vec3(*p) for p in centerline])
    normals = [wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0)) for quaternion in quaternions]
    normals.append(normals[-1])
    return [(float(normal[0]), float(normal[1]), float(normal[2])) for normal in normals]


def tube_mesh(centerline: np.ndarray, radius: float, sides: int = constants.PINNED_TUBE_SIDES) -> newton.Mesh:
    """Build an open triangulated tube around a fixed centerline [m]."""
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.asarray(parallel_transport_normals(centerline))
    normals -= np.sum(normals * tangents, axis=1, keepdims=True) * tangents
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)
    angles = np.arange(sides) * (2.0 * math.pi / sides)
    radial = np.cos(angles)[None, :, None] * normals[:, None, :] + np.sin(angles)[None, :, None] * binormals[:, None, :]
    vertices = (centerline[:, None, :] + radius * radial).reshape(-1, 3)
    first = np.arange((len(centerline) - 1) * sides).reshape(-1, sides)
    next_side = np.roll(first, -1, axis=1)
    indices = np.stack((first, next_side + sides, first + sides, first, next_side, next_side + sides), axis=-1).ravel()
    return newton.Mesh(
        vertices=vertices,
        indices=indices,
        normals=radial.reshape(-1, 3),
        compute_inertia=False,
        is_solid=False,
    )


def pinned_shoelace_spawner(centerline: np.ndarray, cable_radius: float) -> sim_utils.SpawnerCfg:
    """Build a cloned spawner for the fixed eyelet-held shoelace span."""
    pinned_centerline = centerline[constants.PINNED_FIRST + 1 : constants.PINNED_LAST + 1]

    @sim_utils.clone
    def spawn_pinned(
        prim_path: str,
        _: sim_utils.SpawnerCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: object,
    ) -> Usd.Prim:
        del kwargs
        root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
        tube = tube_mesh(pinned_centerline, cable_radius)
        mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), f"{prim_path}/geometry/mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in tube.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(tube.indices) // 3))
        mesh.CreateFaceVertexIndicesAttr(tube.indices.tolist())
        mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, normal)) for normal in tube.normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*constants.CABLE_COLOR)])
        collision_cfg = sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
        if not sim_utils.apply_collision_properties(str(mesh.GetPath()), [collision_cfg], create_if_missing=True):
            raise RuntimeError(f"Failed to enable pinned collision mesh at {mesh.GetPath()}")
        return root

    return sim_utils.SpawnerCfg(func=spawn_pinned)


def configure_shoe_collider(
    prim_path: str,
    _: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Enable and hide the collider prim already provided by the shoe USD."""
    del translation, orientation, kwargs
    collision_cfg = sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
    if not sim_utils.apply_collision_properties(prim_path, [collision_cfg], create_if_missing=True):
        raise RuntimeError(f"Failed to enable the shoe collider at {prim_path}")
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    sim_utils.set_prim_visibility(prim, False)
    return prim


def shoe_asset_cfg(visible: bool = True) -> RigidObjectCfg:
    """Build the environment-local kinematic shoe with a resettable root pose."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shoe",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(constants.COLLIDER_ASSET),
            visible=visible,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )


def shoe_collider_asset_cfg() -> AssetBaseCfg:
    """Build the hidden collision-mesh asset nested under the shoe."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/Collider",
        spawn=sim_utils.SpawnerCfg(func=configure_shoe_collider),
    )


def shoe_visual_asset_cfg(material_path: str) -> AssetBaseCfg:
    """Build the environment-local visual shoe asset."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(constants.MODEL_ASSET),
            visual_material_bindings={"Model": material_path},
        ),
    )


def tongue_upper_asset_cfg() -> AssetBaseCfg:
    """Build the hidden upper-tongue collision panel."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/TongueUpper",
        spawn=sim_utils.CuboidCfg(
            size=constants.TONGUE_UPPER_SIZE,
            visible=False,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            physics_material=rigid_material(constants.SHOE_MU, constants.CONTACT_KD),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=constants.TONGUE_UPPER_CENTER,
            rot=(
                math.cos(0.5 * constants.TONGUE_UPPER_Y_ROTATION) * math.sin(0.5 * constants.TONGUE_UPPER_PITCH),
                math.sin(0.5 * constants.TONGUE_UPPER_Y_ROTATION) * math.cos(0.5 * constants.TONGUE_UPPER_PITCH),
                -math.sin(0.5 * constants.TONGUE_UPPER_Y_ROTATION) * math.sin(0.5 * constants.TONGUE_UPPER_PITCH),
                math.cos(0.5 * constants.TONGUE_UPPER_Y_ROTATION) * math.cos(0.5 * constants.TONGUE_UPPER_PITCH),
            ),
        ),
    )


def cable_spawn_cfg(centerline: np.ndarray, cable_radius: float, reference_segment_length: float) -> sim_utils.CableCfg:
    """Build the prototype cable from its centerline, radius, and reference segment length [m]."""
    return sim_utils.CableCfg(
        positions=[tuple(point) for point in centerline],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=constants.CABLE_COLOR, roughness=0.8),
        physics_material=sim_utils.CableMaterialCfg(
            thickness=2.0 * cable_radius,
            density=constants.CABLE_DENSITY,
            stretch_stiffness=constants.STRETCH_STIFFNESS * reference_segment_length / (math.pi * cable_radius**2),
            bend_stiffness=constants.BEND_STIFFNESS * reference_segment_length / (0.25 * math.pi * cable_radius**4),
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
    )


def configure_shoelace_builder(builder: ModelBuilder, centerline: np.ndarray, cable_radius: float) -> ShoelaceBuild:
    """Configure one offline prototype's frames, capsule masses, anchors, and filters.

    Args:
        builder: Newton builder containing the prototype in world zero.
        centerline: Resampled cable centerline [m], shape [N, 3].
        cable_radius: Cable radius [m].

    Returns:
        Ordered cable body/joint indices and the mean segment length.
    """
    if len(centerline) - 1 != constants.SHOELACE_SEGMENT_COUNT:
        raise ValueError(f"Expected {constants.SHOELACE_SEGMENT_COUNT} cable segments, got {len(centerline) - 1}")
    body_chains = _find_chains(builder.body_label, builder.body_world, 1, "edge_body")[0]
    joint_chains = _find_chains(builder.joint_label, builder.joint_world, 1, "cable")[0]
    expected_counts = (constants.PINNED_FIRST + 1, len(centerline) - 1 - constants.PINNED_LAST)
    observed = tuple((len(bodies), len(joints)) for bodies, joints in zip(body_chains, joint_chains, strict=True))
    expected = tuple((count, count - 1) for count in expected_counts)
    if observed != expected:
        raise RuntimeError(f"Unexpected prototype cable topology: {observed}, expected {expected}")

    body_by_label = _index_unique_labels(builder.body_label)
    shape_by_label = _index_unique_labels(builder.shape_label)
    root = "/World/envs/env_0/Shoe"
    shoe_shapes = [
        _label_index(shape_by_label, root + suffix) for suffix in ("/Collider", "/TongueUpper/geometry/mesh")
    ]
    pinned_shape = _label_index(shape_by_label, f"{root}/ShoelacePinned/geometry/mesh")
    _make_body_static(builder, _label_index(body_by_label, root))

    quaternions = newton.utils.create_parallel_transport_cable_quaternions([wp.vec3(*point) for point in centerline])
    segment_ranges = (range(0, constants.PINNED_FIRST + 1), range(constants.PINNED_LAST, len(centerline) - 1))
    neighbor_window = _neighbor_filter_window(centerline, cable_radius)
    chain_shapes = []
    for bodies, segment_indices in zip(body_chains, segment_ranges, strict=True):
        shapes = []
        for body, segment in zip(bodies, segment_indices, strict=True):
            builder.body_q[body] = wp.transform(
                wp.transform_get_translation(builder.body_q[body]), quaternions[segment]
            )
            shape = int(builder.body_shapes[body][0])
            shapes.append(shape)
            _correct_capsule_mass(builder, body, shape)
        _filter_rod_neighbors(builder, shapes, neighbor_window)
        chain_shapes.append(shapes)

    for body, shape in ((body_chains[0][-1], chain_shapes[0][-1]), (body_chains[1][0], chain_shapes[1][0])):
        _make_body_static(builder, body)
        for shoe_shape in shoe_shapes:
            builder.add_shape_collision_filter_pair(shape, shoe_shape)
    for shoe_shape in shoe_shapes:
        builder.add_shape_collision_filter_pair(pinned_shape, shoe_shape)
    for shape in (*chain_shapes[0][-neighbor_window:], *chain_shapes[1][:neighbor_window]):
        builder.add_shape_collision_filter_pair(shape, pinned_shape)

    mean_segment_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
    return ShoelaceBuild(body_chains, joint_chains, mean_segment_length)


def _index_unique_labels(labels: list[str]) -> dict[str, int]:
    result = {label: index for index, label in enumerate(labels)}
    if len(result) != len(labels):
        raise RuntimeError("Newton labels must be unique in the asset prototype")
    return result


def _label_index(label_indices: dict[str, int], expected: str) -> int:
    try:
        return label_indices[expected]
    except KeyError as exc:
        raise RuntimeError(f"Missing Newton label {expected!r}") from exc


def _neighbor_filter_window(points: np.ndarray, radius: float) -> int:
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    return max(1, int(np.floor((2.0 * radius + constants.CONTACT_GAP) / minimum_length + 1.0e-9)) + 1)


def _filter_rod_neighbors(builder: ModelBuilder, shapes: list[int], window: int) -> None:
    for first, first_shape in enumerate(shapes):
        for second_shape in shapes[first + 1 : first + 1 + window]:
            builder.add_shape_collision_filter_pair(first_shape, second_shape)


def _make_body_static(builder: ModelBuilder, body: int) -> None:
    builder.body_mass[body] = builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = builder.body_inv_inertia[body] = wp.mat33()


def _correct_capsule_mass(builder: ModelBuilder, body: int, shape: int) -> None:
    radius = float(builder.shape_scale[shape][0])
    segment_length = 2.0 * float(builder.shape_scale[shape][1])
    mass_scale = 1.0 + 4.0 * radius / (3.0 * segment_length)
    builder.body_mass[body] *= mass_scale
    builder.body_inertia[body] *= mass_scale
    builder.body_inv_mass[body] = 1.0 / builder.body_mass[body]
    builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])


def _tail_joint_blend_weights(
    joint_count: int,
    segment_length: float,
    free_end_at_start: bool,
) -> np.ndarray:
    """Return smooth weights for the gravity-resistant distal tail material."""
    distances = segment_length * np.arange(1, joint_count + 1, dtype=np.float64)
    if constants.TAIL_STIFF_TRANSITION_LENGTH > 0.0:
        weights = np.clip(
            (constants.TAIL_STIFF_CORE_LENGTH + constants.TAIL_STIFF_TRANSITION_LENGTH - distances)
            / constants.TAIL_STIFF_TRANSITION_LENGTH,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)
    else:
        weights = (distances <= constants.TAIL_STIFF_CORE_LENGTH).astype(np.float64)
    return weights if free_end_at_start else weights[::-1].copy()
