# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton geometry and physics helpers for the dual-Franka shoelace task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

if TYPE_CHECKING:
    from newton import ModelBuilder


ASSET_DIR = Path(__file__).resolve().parents[5] / "scripts/demos/shoelace/assets"
CURVE_ASSET = ASSET_DIR / "curve.usd"
MODEL_ASSET = ASSET_DIR / "model.usd"
COLLIDER_ASSET = ASSET_DIR / "collider_simplified.usd"

TCP_OFFSET = (0.0, 0.0, 0.1034)

# Shared Franka-validated cable topology.
AUTHORED_SEGMENT_COUNT = 450
SHOELACE_SEGMENT_COUNT = 360
PINNED_FIRST = 78
PINNED_LAST = 281
PINNED_TUBE_SIDES = 6

# Shared Franka-validated cable material and contact defaults.
CABLE_DENSITY = 1150.0
CABLE_INERTIA_REGULARIZATION = 1.0e-6  # Additive isotropic proxy inertia [kg*m^2].
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 5.0
BEND_DAMPING = 1.0
DAHL_MAX_STRAIN = 0.20
DAHL_DECAY = 0.35
CONTACT_GAP = 1.0e-4
CONTACT_DISTANCE_CAP = 2.0e-3
CONTACT_KE = 1.0e6
CONTACT_KD = 0.0
GROUND_CONTACT_KD = 30.0
LACE_MU = 0.1
SHOE_MU = 0.02
GROUND_MU = 0.8
COLLISION_GROUP = 1
VBD_CONTACT_BUFFER = 256
CONTACTS_PER_ENV = 512
TRIANGLE_PAIRS_PER_ENV = 8192
MIN_TRIANGLE_PAIRS = 1_000_000

# Shared shoe geometry and display defaults.
TONGUE_UPPER_CENTER = (-0.008, 0.02, 0.10)
TONGUE_UPPER_SIZE = (0.05, 0.055, 0.006)
TONGUE_UPPER_PITCH = math.radians(28.0)
TONGUE_UPPER_Y_ROTATION = math.radians(5.0)
CABLE_COLOR = (112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0)


@dataclass
class ShoelaceBuild:
    """Ordered indices produced by the shared Newton cable builder.

    Attributes:
        body_chains: Left and right cable body indices for every Newton world.
        joint_chains: Left and right cable joint indices for every Newton world.
        mean_segment_length: Mean cable segment length [m].
    """

    body_chains: list[tuple[list[int], list[int]]]
    joint_chains: list[tuple[list[int], list[int]]]
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
    authored_centerline, cable_radius = load_usd_curve(CURVE_ASSET, "/World/Curve")
    authored_lengths = np.linalg.norm(np.diff(authored_centerline, axis=0), axis=1)
    if len(authored_lengths) != AUTHORED_SEGMENT_COUNT:
        raise ValueError(f"Expected {AUTHORED_SEGMENT_COUNT} authored segments, got {len(authored_lengths)}")
    centerline = resample_centerline(authored_centerline, SHOELACE_SEGMENT_COUNT)
    return centerline, cable_radius, float(authored_lengths.mean())


def parallel_transport_normals(centerline: np.ndarray) -> list[tuple[float, float, float]]:
    """Build one normal per centerline point using Newton parallel-transport frames."""
    quaternions = newton.utils.create_parallel_transport_cable_quaternions([wp.vec3(*p) for p in centerline])
    normals = [wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0)) for quaternion in quaternions]
    normals.append(normals[-1])
    return [(float(normal[0]), float(normal[1]), float(normal[2])) for normal in normals]


def tube_mesh(centerline: np.ndarray, radius: float, sides: int = PINNED_TUBE_SIDES) -> newton.Mesh:
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
    pinned_centerline = centerline[PINNED_FIRST + 1 : PINNED_LAST + 1]

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
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*CABLE_COLOR)])
        collision_cfg = [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]
        if not sim_utils.apply_collision_properties(str(mesh.GetPath()), collision_cfg, create_if_missing=True):
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
    collision_cfg = [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]
    if not sim_utils.apply_collision_properties(prim_path, collision_cfg, create_if_missing=True):
        raise RuntimeError(f"Failed to enable the shoe collider at {prim_path}")
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    sim_utils.set_prim_visibility(prim, False)
    return prim


def rigid_material(friction: float, damping: float) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Build standard-friction and Newton-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0),
        NewtonMaterialCfg(contact_stiffness=CONTACT_KE, contact_damping=damping),
    ]


def shoe_asset_cfg(visible: bool = True) -> RigidObjectCfg:
    """Build the environment-local kinematic shoe with a resettable root pose."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shoe",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(COLLIDER_ASSET),
            visible=visible,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
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
            usd_path=str(MODEL_ASSET),
            visual_material_bindings={"Model": material_path},
        ),
    )


def tongue_upper_asset_cfg() -> AssetBaseCfg:
    """Build the hidden upper-tongue collision panel."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/TongueUpper",
        spawn=sim_utils.CuboidCfg(
            size=TONGUE_UPPER_SIZE,
            visible=False,
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            physics_material=rigid_material(SHOE_MU, CONTACT_KD),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TONGUE_UPPER_CENTER,
            rot=(
                math.cos(0.5 * TONGUE_UPPER_Y_ROTATION) * math.sin(0.5 * TONGUE_UPPER_PITCH),
                math.sin(0.5 * TONGUE_UPPER_Y_ROTATION) * math.cos(0.5 * TONGUE_UPPER_PITCH),
                -math.sin(0.5 * TONGUE_UPPER_Y_ROTATION) * math.sin(0.5 * TONGUE_UPPER_PITCH),
                math.cos(0.5 * TONGUE_UPPER_Y_ROTATION) * math.cos(0.5 * TONGUE_UPPER_PITCH),
            ),
        ),
    )


def ground_asset_cfg(size: float) -> AssetBaseCfg:
    """Build the shared square ground plane with side length ``size`` [m]."""
    return AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.08, 0.08, 0.08),
            size=(size, size),
            physics_material=rigid_material(GROUND_MU, GROUND_CONTACT_KD),
        ),
        collision_group=-1,
    )


def light_asset_cfg() -> AssetBaseCfg:
    """Build the shared dome light."""
    return AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0)),
    )


def cable_spawn_cfg(
    centerline: np.ndarray | None = None,
    normals: list[tuple[float, float, float]] | None = None,
    cable_radius: float = 0.0015,
    reference_segment_length: float = 0.01,
) -> sim_utils.CableCfg:
    """Build a cable spawner, optionally configured with final asset geometry [m]."""
    spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.01)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CABLE_COLOR, roughness=0.8),
        physics_material=sim_utils.CableMaterialCfg(
            thickness=2.0 * cable_radius,
            density=CABLE_DENSITY,
            stretch_stiffness=1.0e9,
            bend_stiffness=1.0e8,
        ),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )
    if centerline is not None:
        configure_cable_spawn(spawn, centerline, normals, cable_radius, reference_segment_length)
    return spawn


def configure_cable_spawn(
    spawn: sim_utils.CableCfg,
    centerline: np.ndarray,
    normals: list[tuple[float, float, float]] | None,
    cable_radius: float,
    reference_segment_length: float,
) -> None:
    """Fill one cable spawner from centerline geometry and shared material defaults [m]."""
    spawn.positions = [tuple(point) for point in centerline]
    spawn.normals = parallel_transport_normals(centerline) if normals is None else normals
    cross_section_area = math.pi * cable_radius**2
    second_moment = 0.25 * math.pi * cable_radius**4
    spawn.physics_material.thickness = 2.0 * cable_radius
    spawn.physics_material.density = CABLE_DENSITY
    spawn.physics_material.stretch_stiffness = STRETCH_STIFFNESS * reference_segment_length / cross_section_area
    spawn.physics_material.bend_stiffness = BEND_STIFFNESS * reference_segment_length / second_moment


def ground_size(num_envs: int, env_spacing: float) -> float:
    """Return a ground-plane side length [m] covering the centered environment grid."""
    return max(2.0, env_spacing * (math.ceil(math.sqrt(num_envs)) + 1))


def install_shoe_material_reference() -> None:
    """Install the shared visual-material library at its expected stage path."""
    material_scope = UsdGeom.Scope.Define(sim_utils.get_current_stage(), "/World/ShoeMaterials").GetPrim()
    material_scope.GetReferences().AddReference(str(MODEL_ASSET), "/mat")


def configure_shoelace_builder(
    builder: ModelBuilder,
    centerline: np.ndarray,
    cable_radius: float,
    num_envs: int,
    cable_radii: np.ndarray | None = None,
    cable_colors: np.ndarray | None = None,
    lace_mu: float = LACE_MU,
    shoe_mu: float = SHOE_MU,
    write_shape_colors: bool = False,
    cable_inertia_regularization: float = CABLE_INERTIA_REGULARIZATION,
) -> ShoelaceBuild:
    """Configure shared cable frames, contacts, anchors, and collision filters.

    Args:
        builder: Newton model builder containing the replicated scene.
        centerline: Resampled cable centerline [m], shape [N, 3].
        cable_radius: Default cable radius [m].
        num_envs: Number of isolated Newton worlds.
        cable_radii: Optional per-environment cable radii [m].
        cable_colors: Optional per-environment linear RGB colors.
        lace_mu: Cable and pinned-span friction coefficient.
        shoe_mu: Shoe and tongue friction coefficient.
        write_shape_colors: Whether to override imported Newton shape colors.
        cable_inertia_regularization: Isotropic inertia added to each dynamic cable segment [kg*m^2].

    Returns:
        Ordered body and joint chains plus the mean segment length.
    """
    if len(centerline) - 1 != SHOELACE_SEGMENT_COUNT:
        raise ValueError(f"Expected {SHOELACE_SEGMENT_COUNT} cable segments, got {len(centerline) - 1}")
    if cable_radii is not None and cable_radii.shape != (num_envs,):
        raise ValueError(f"Expected {num_envs} cable radii, got shape {cable_radii.shape}")
    if cable_colors is not None and cable_colors.shape != (num_envs, 3):
        raise ValueError(f"Expected cable colors with shape {(num_envs, 3)}, got {cable_colors.shape}")

    body_chains = _find_chains(builder.body_label, builder.body_world, num_envs, "edge_body")
    joint_chains = _find_chains(builder.joint_label, builder.joint_world, num_envs, "cable")
    expected_body_counts = (PINNED_FIRST + 1, len(centerline) - 1 - PINNED_LAST)
    expected = tuple((count, count - 1) for count in expected_body_counts)
    for world, (bodies, joints) in enumerate(zip(body_chains, joint_chains, strict=True)):
        observed = tuple(
            (len(chain_bodies), len(chain_joints)) for chain_bodies, chain_joints in zip(bodies, joints, strict=True)
        )
        if observed != expected:
            raise RuntimeError(f"Unexpected cable topology in world {world}: {observed}, expected {expected}")

    body_by_label = _index_unique_labels(builder.body_label)
    shape_by_label = _index_unique_labels(builder.shape_label)
    pinned_centerline = centerline[PINNED_FIRST + 1 : PINNED_LAST + 1]
    pinned_meshes = (
        {float(radius): tube_mesh(pinned_centerline, float(radius)) for radius in np.unique(cable_radii)}
        if cable_radii is not None
        else {}
    )
    linear_colors = np.tile(CABLE_COLOR, (num_envs, 1)) if cable_colors is None else cable_colors
    display_colors = [tuple(Gf.ConvertLinearToDisplay(Gf.Vec3f(*map(float, color)))) for color in linear_colors]
    shoe_parts: list[tuple[int, int, int]] = []
    for world in range(num_envs):
        root = f"/World/envs/env_{world}/Shoe"
        shoe_body = _label_index(body_by_label, root)
        shoe_collider = _label_index(shape_by_label, f"{root}/Collider")
        tongue_collider = _label_index(shape_by_label, f"{root}/TongueUpper/geometry/mesh")
        pinned_shape = _label_index(shape_by_label, f"{root}/ShoelacePinned/geometry/mesh")
        if builder.shape_world[pinned_shape] != world:
            raise RuntimeError(f"Pinned mesh is not local to Newton world {world}")
        shoe_parts.append((shoe_collider, tongue_collider, pinned_shape))
        _make_body_static(builder, shoe_body)
        for shape in (shoe_collider, tongue_collider):
            builder.shape_flags[shape] = newton.ShapeFlags.COLLIDE_SHAPES
            builder.shape_collision_group[shape] = COLLISION_GROUP
            _set_shape_material(builder, shape, shoe_mu)
        builder.shape_flags[pinned_shape] = newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES
        builder.shape_collision_group[pinned_shape] = COLLISION_GROUP
        _set_shape_material(builder, pinned_shape, lace_mu)
        if write_shape_colors:
            builder.shape_color[pinned_shape] = display_colors[world]
        if cable_radii is not None:
            tube = pinned_meshes[float(cable_radii[world])]
            builder.shape_source[pinned_shape] = tube
            extent = np.ptp(np.asarray(tube.vertices, dtype=np.float64), axis=0)
            builder.shape_collision_radius[pinned_shape] = 0.5 * float(np.linalg.norm(extent))

    for shape, label in enumerate(builder.shape_label):
        if label.startswith("/World/Ground/"):
            _set_shape_material(builder, shape, GROUND_MU, GROUND_CONTACT_KD)

    exact_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
        [wp.vec3(*point) for point in centerline]
    )
    mean_segment_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
    segment_ranges = (range(0, PINNED_FIRST + 1), range(PINNED_LAST, len(centerline) - 1))
    for world, (world_bodies, world_joints, shoe) in enumerate(zip(body_chains, joint_chains, shoe_parts, strict=True)):
        shoe_collider, tongue_collider, pinned_shape = shoe
        randomized_radius = None if cable_radii is None else float(cable_radii[world])
        neighbor_window = _neighbor_filter_window(
            centerline, cable_radius if randomized_radius is None else randomized_radius
        )
        chain_shapes: list[list[int]] = []
        for bodies, segment_indices in zip(world_bodies, segment_ranges, strict=True):
            shapes: list[int] = []
            for body, segment in zip(bodies, segment_indices, strict=True):
                transform = builder.body_q[body]
                builder.body_q[body] = wp.transform(
                    wp.vec3(transform[0], transform[1], transform[2]), exact_quaternions[segment]
                )
                shape = int(builder.body_shapes[body][0])
                shapes.append(shape)
                builder.shape_collision_group[shape] = COLLISION_GROUP
                _set_shape_material(builder, shape, lace_mu)
                if write_shape_colors:
                    builder.shape_color[shape] = display_colors[world]
                if randomized_radius is None:
                    _correct_capsule_mass(builder, body, shape)
                else:
                    _resize_capsule(builder, body, shape, randomized_radius)
            _filter_rod_neighbors(builder, shapes, neighbor_window)
            chain_shapes.append(shapes)

        for body, shape in (
            (world_bodies[0][-1], chain_shapes[0][-1]),
            (world_bodies[1][0], chain_shapes[1][0]),
        ):
            _make_body_static(builder, body)
            for shoe_shape in (shoe_collider, tongue_collider):
                builder.add_shape_collision_filter_pair(shape, shoe_shape)
        for static_shape in (shoe_collider, tongue_collider):
            builder.add_shape_collision_filter_pair(pinned_shape, static_shape)
        for shape in (*chain_shapes[0][-neighbor_window:], *chain_shapes[1][:neighbor_window]):
            builder.add_shape_collision_filter_pair(shape, pinned_shape)

        configure_cable_inertia(builder, world_bodies[0] + world_bodies[1], cable_inertia_regularization)

    return ShoelaceBuild(body_chains, joint_chains, mean_segment_length)


def configure_cable_inertia(builder: ModelBuilder, cable_bodies: list[int], regularization: float) -> None:
    """Add explicit proxy inertia to dynamic cable segments before model finalization.

    Applies ``I_effective = I_geometry + regularization * identity`` once, after
    capsule mass/radius corrections and anchor pinning. Masses are unchanged;
    zero-mass anchors and bodies outside ``cable_bodies`` are untouched.
    This changes rotational dynamics, not material stiffness or damping.

    Args:
        builder: Newton builder with geometric cable mass properties.
        cable_bodies: Cable body indices, optionally including zero-mass anchors.
        regularization: Finite, nonnegative isotropic addition [kg*m^2]. Zero skips
            this addition but does not disable Newton's own inertia validation.

    Raises:
        ValueError: If regularization is negative or nonfinite.
    """
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Cable inertia regularization must be finite and nonnegative")
    if regularization == 0.0:
        return
    addition = wp.mat33(np.eye(3, dtype=np.float32) * regularization)
    for body in cable_bodies:
        if builder.body_mass[body] > 0.0:
            builder.body_inertia[body] += addition
            builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])


def cable_joint_dof_slice(builder: ModelBuilder, joints: list[int]) -> slice:
    """Return the contiguous four-DOF range occupied by one cable joint chain."""
    dof_start = int(builder.joint_qd_start[joints[0]])
    dof_end = int(builder.joint_qd_start[joints[-1]]) + 4
    if dof_end - dof_start != 4 * len(joints):
        raise RuntimeError("Cable joint DOFs must be contiguous")
    return slice(dof_start, dof_end)


def configure_dahl(builder: ModelBuilder, cable_joints: list[list[int]]) -> None:
    """Configure Dahl hysteresis for only the dynamic cable joints."""
    if "vbd:dahl_eps_max" not in builder.custom_attributes:
        newton.solvers.SolverVBD.register_custom_attributes(builder)
    dahl_eps = builder.custom_attributes["vbd:dahl_eps_max"].values
    dahl_tau = builder.custom_attributes["vbd:dahl_tau"].values
    for joint in (joint for joints in cable_joints for joint in joints):
        dahl_eps[joint] = DAHL_MAX_STRAIN
        dahl_tau[joint] = DAHL_DECAY


def _find_chains(labels: list[str], worlds: list[int], num_envs: int, suffix: str) -> list[tuple[list[int], list[int]]]:
    sides = [
        _sorted_world_label_indices(labels, worlds, num_envs, f"/Shoelace{side}/geometry/mesh_{suffix}_")
        for side in ("Left", "Right")
    ]
    return list(zip(*sides, strict=True))


def _sorted_world_label_indices(
    labels: list[str], worlds: list[int], num_envs: int, local_prefix: str
) -> list[list[int]]:
    matches: list[list[tuple[int, int]]] = [[] for _ in range(num_envs)]
    for index, (label, world) in enumerate(zip(labels, worlds, strict=True)):
        if 0 <= world < num_envs:
            prefix = f"/World/envs/env_{world}{local_prefix}"
            if label.startswith(prefix):
                matches[world].append((int(label.removeprefix(prefix)), index))
    return [[index for _, index in sorted(world_matches)] for world_matches in matches]


def _index_unique_labels(labels: list[str]) -> dict[str, int]:
    result = {label: index for index, label in enumerate(labels)}
    if len(result) != len(labels):
        raise RuntimeError("Newton labels must be unique after world replication")
    return result


def _label_index(label_indices: dict[str, int], expected: str) -> int:
    try:
        return label_indices[expected]
    except KeyError as exc:
        raise RuntimeError(f"Missing Newton label {expected!r}") from exc


def _neighbor_filter_window(points: np.ndarray, radius: float) -> int:
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    return max(1, int(np.floor((2.0 * radius + CONTACT_GAP) / minimum_length + 1.0e-9)) + 1)


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


def _resize_capsule(builder: ModelBuilder, body: int, shape: int, radius: float) -> None:
    half_height = float(builder.shape_scale[shape][1])
    scale = wp.vec3(radius, half_height, 0.0)
    mass, _, inertia = newton.geometry.compute_inertia_shape(newton.GeoType.CAPSULE, scale, None, CABLE_DENSITY)
    builder.shape_scale[shape] = (radius, half_height, 0.0)
    builder.shape_collision_radius[shape] = radius + half_height
    builder.body_mass[body] = mass
    builder.body_inertia[body] = inertia
    builder.body_inv_mass[body] = 1.0 / mass
    builder.body_inv_inertia[body] = wp.inverse(inertia)


def _set_shape_material(builder: ModelBuilder, shape: int, friction: float, damping: float = CONTACT_KD) -> None:
    builder.shape_material_ke[shape] = CONTACT_KE
    builder.shape_material_kd[shape] = damping
    builder.shape_material_mu[shape] = friction
    builder.shape_gap[shape] = CONTACT_GAP
