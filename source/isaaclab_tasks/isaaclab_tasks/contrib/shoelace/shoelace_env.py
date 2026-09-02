# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import newton.examples
import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager, VBDSolverCfg

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import spawn_from_usd

from isaaclab_contrib.coupling import CouplerProxyCfg

from .mdp.constants import (
    PINNED_FIRST,
    PINNED_LAST,
    SHOELACE_SEGMENT_COUNT,
    TCP_OFFSET,
)

if TYPE_CHECKING:
    import torch
    from newton import ModelBuilder

    from .shoelace_env_cfg import ShoelaceEnvCfg


# Cable material values use Newton's per-joint convention.
CABLE_DENSITY = 1150.0
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 0.50
BEND_DAMPING = 0.45
DAHL_MAX_STRAIN = 0.20
DAHL_DECAY = 0.35

# Contact values match the standalone demo, except robot proxies retain their
# lower-stiffness, high-friction defaults from the coupled solver configuration.
CONTACT_GAP = 1.0e-4
CONTACT_KE = 1.0e6
CONTACT_KD = 0.0
GROUND_CONTACT_KD = 30.0
LACE_MU = 0.7
SHOE_MU = 0.2
GROUND_MU = 0.8
COLLISION_GROUP = 1
PINNED_TUBE_SIDES = 6
SHOELACE_COLLIDER_ASSET = Path(__file__).resolve().parents[5] / "scripts/demos/assets/shoelace/collider_simplified.usd"


def _resolve_asset_dir() -> Path:
    """Return a shoelace asset directory containing curve, collision, and visual USDs."""
    repository_assets = Path(__file__).resolve().parents[5] / ".newton/newton/examples/assets/shoelace"
    package_assets = Path(newton.examples.__file__).resolve().parent / "assets/shoelace"
    required = ("curve.usd", "collider.usd", "model.usd")
    for directory in (repository_assets, package_assets):
        if all((directory / name).is_file() for name in required):
            return directory
    raise FileNotFoundError(
        "Shoelace assets were not found in the source checkout or installed Newton package. "
        "Initialize the repository's .newton checkout before running this task."
    )


def _load_usd_curve(path: Path, prim_path: str) -> tuple[np.ndarray, float]:
    """Load one world-space cable centerline and its radius [m]."""
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


def _spawn_collision_mesh_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the shoe shell and author its mesh as collision geometry."""
    root_prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    collider_prim = root_prim.GetStage().GetPrimAtPath(root_prim.GetPath().AppendChild("Collider"))
    if not collider_prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Expected a collision mesh at {collider_prim.GetPath()}")
    collision_api = UsdPhysics.CollisionAPI.Apply(collider_prim)
    collision_api.CreateCollisionEnabledAttr(True)
    return root_prim


def _parallel_transport_normals(centerline: np.ndarray) -> list[tuple[float, float, float]]:
    """Build normals that preserve Newton's parallel-transport cable frames."""
    positions = [wp.vec3(*point) for point in centerline]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(positions)
    normals = [wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0)) for quaternion in quaternions]
    normals.append(normals[-1])
    return [(float(normal[0]), float(normal[1]), float(normal[2])) for normal in normals]


def _resample_centerline(centerline: np.ndarray, num_segments: int) -> np.ndarray:
    """Resample a polyline to uniformly spaced arc-length segments."""
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("Shoelace centerline must not contain coincident adjacent points")
    arc_length = np.r_[0.0, np.cumsum(segment_lengths)]
    sample_arc_length = np.linspace(0.0, arc_length[-1], num_segments + 1)
    return np.column_stack(
        [np.interp(sample_arc_length, arc_length, centerline[:, axis]) for axis in range(centerline.shape[1])]
    )


def _tube_mesh(centerline: np.ndarray, radius: float, sides: int = 8) -> newton.Mesh:
    """Build a tube mesh around a fixed cable centerline."""
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    normals = np.asarray(_parallel_transport_normals(centerline))
    normals -= np.sum(normals * tangents, axis=1, keepdims=True) * tangents
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)

    angles = np.arange(sides) * (2.0 * math.pi / sides)
    radial_directions = (
        np.cos(angles)[None, :, None] * normals[:, None, :] + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    vertices = (centerline[:, None, :] + radius * radial_directions).reshape(-1, 3)

    first = np.arange((len(centerline) - 1) * sides).reshape(-1, sides)
    next_side = np.roll(first, -1, axis=1)
    next_ring = first + sides
    next_ring_side = next_side + sides
    indices = np.stack((first, next_ring_side, next_ring, first, next_side, next_ring_side), axis=-1).ravel()

    return newton.Mesh(
        vertices=vertices,
        indices=indices,
        normals=radial_directions.reshape(-1, 3),
        compute_inertia=False,
        is_solid=False,
    )


def _pinned_shoelace_spawner(centerline: np.ndarray, cable_radius: float) -> sim_utils.SpawnerCfg:
    """Build a cloned spawner for the fixed pinned span as one visible collision mesh."""
    pinned_centerline = centerline[PINNED_FIRST + 1 : PINNED_LAST + 1]

    @sim_utils.clone
    def spawn_pinned_shoelace(
        prim_path: str,
        _: sim_utils.SpawnerCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
        tube = _tube_mesh(pinned_centerline, cable_radius, sides=PINNED_TUBE_SIDES)
        mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), f"{prim_path}/geometry/mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in tube.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * (len(tube.indices) // 3))
        mesh.CreateFaceVertexIndicesAttr(tube.indices.tolist())
        mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, normal)) for normal in tube.normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0)])
        collision_prim = mesh.GetPrim()
        if not sim_utils.apply_collision_properties(
            str(collision_prim.GetPath()), [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]
        ):
            raise RuntimeError(f"Failed to enable pinned collision mesh at {collision_prim.GetPath()}")
        return root

    return sim_utils.SpawnerCfg(func=spawn_pinned_shoelace)


def _neighbor_filter_window(points: np.ndarray, radius: float) -> int:
    """Return the number of adjacent rod capsules that overlap by construction."""
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    contact_reach = 2.0 * radius + CONTACT_GAP
    return max(1, int(np.floor(contact_reach / minimum_length + 1.0e-9)) + 1)


def _filter_rod_neighbors(builder: ModelBuilder, shapes: list[int], window: int) -> None:
    """Disable collisions between geometrically overlapping neighbor segments."""
    for first, first_shape in enumerate(shapes):
        for second_shape in shapes[first + 1 : first + 1 + window]:
            builder.add_shape_collision_filter_pair(first_shape, second_shape)


def _resolve_grasp_assist_bodies(
    builder: ModelBuilder,
    cable_bodies: list[list[int]],
    num_envs: int,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve hand, finger, and three-segment tail body indices for each grasp."""
    hand_ids: list[int] = []
    finger_ids: list[int] = []
    tail_ids: list[int] = []
    for world, world_cable_bodies in enumerate(cable_bodies):
        if len(world_cable_bodies) % 2 != 0:
            raise RuntimeError(f"Expected two equal cable chains in Newton world {world}")
        chain_length = len(world_cable_bodies) // 2
        left_cable = world_cable_bodies[:chain_length]
        right_cable = world_cable_bodies[chain_length:]
        grasps = (("Left", right_cable[-3:]), ("Right", left_cable[:3]))
        for robot_side, grasp_tail_ids in grasps:
            root = f"/World/envs/env_{world}/Robot{robot_side}/"
            robot_bodies = [
                (body, str(label)) for body, label in enumerate(builder.body_label) if str(label).startswith(root)
            ]
            hand_matches = [body for body, label in robot_bodies if label.endswith("/panda_hand")]
            finger_matches = [
                [body for body, label in robot_bodies if label.endswith(f"/panda_{finger_side}finger")]
                for finger_side in ("left", "right")
            ]
            if len(hand_matches) != 1 or any(len(matches) != 1 for matches in finger_matches):
                raise RuntimeError(f"Unexpected grasp-assist robot topology below {root}")
            hand_ids.append(hand_matches[0])
            finger_ids.extend(matches[0] for matches in finger_matches)
            if len(grasp_tail_ids) != 3:
                raise RuntimeError(f"Expected three grasp-tail bodies in Newton world {world}")
            tail_ids.extend(grasp_tail_ids)
    if len(cable_bodies) != num_envs:
        raise RuntimeError(f"Expected {num_envs} Newton cable worlds, got {len(cable_bodies)}")
    return hand_ids, finger_ids, tail_ids


@wp.kernel(enable_backward=False)
def _apply_grasp_assist_kernel(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    hand_ids: wp.array(dtype=wp.int32),
    finger_ids: wp.array(dtype=wp.int32),
    tail_ids: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    local_anchors: wp.array(dtype=wp.vec3f),
    tcp_offset: wp.vec3f,
    acquisition_distance: float,
    release_distance: float,
    acquisition_closed_separation: float,
    release_open_separation: float,
    stiffness: float,
    damping: float,
    maximum_force: float,
):
    """Apply a breakable cable-side spring after a geometric grasp is acquired."""
    grasp = wp.tid()
    hand_id = hand_ids[grasp]
    finger_base = 2 * grasp
    tail_base = 3 * grasp
    hand_pose = body_q[hand_id]
    hand_position = wp.transform_get_translation(hand_pose)
    tcp_position = wp.transform_point(hand_pose, tcp_offset)
    tail_position = wp.vec3f(0.0)
    tail_velocity = wp.vec3f(0.0)
    for index in range(3):
        tail_id = tail_ids[tail_base + index]
        tail_position += wp.transform_get_translation(body_q[tail_id]) / 3.0
        tail_velocity += wp.spatial_top(body_qd[tail_id]) / 3.0

    finger_separation = wp.length(
        wp.transform_get_translation(body_q[finger_ids[finger_base]])
        - wp.transform_get_translation(body_q[finger_ids[finger_base + 1]])
    )
    distance = wp.length(tail_position - tcp_position)
    if active[grasp] == 0:
        if finger_separation <= acquisition_closed_separation and distance <= acquisition_distance:
            active[grasp] = 1
            local_anchors[grasp] = wp.transform_point(wp.transform_inverse(hand_pose), tail_position)
        else:
            return
    elif finger_separation >= release_open_separation or distance >= release_distance:
        active[grasp] = 0
        return

    target_position = wp.transform_point(hand_pose, local_anchors[grasp])
    hand_velocity = wp.spatial_top(body_qd[hand_id])
    hand_angular_velocity = wp.spatial_bottom(body_qd[hand_id])
    target_velocity = hand_velocity + wp.cross(hand_angular_velocity, target_position - hand_position)
    force = stiffness * (target_position - tail_position) + damping * (target_velocity - tail_velocity)
    force_length = wp.length(force)
    if force_length > maximum_force:
        force *= maximum_force / force_length
    closure_scale = wp.clamp(
        (acquisition_closed_separation - finger_separation) / (0.25 * acquisition_closed_separation),
        0.0,
        1.0,
    )
    force *= closure_scale
    distributed_force = force / 3.0
    for index in range(3):
        wp.atomic_add(
            body_f,
            tail_ids[tail_base + index],
            wp.spatial_vectorf(distributed_force[0], distributed_force[1], distributed_force[2], 0.0, 0.0, 0.0),
        )


class _ShoelacePhysics:
    """Configure shoelace parity before coupled MJWarp/VBD solver construction."""

    def __init__(
        self,
        centerline: np.ndarray,
        cable_radius: float,
        num_envs: int,
        joint_stiffness_scale: float,
        grasp_assist_enabled: bool,
        grasp_assist_acquisition_distance: float,
        grasp_assist_release_distance: float,
        grasp_assist_acquisition_closed_separation: float,
        grasp_assist_release_open_separation: float,
        grasp_assist_stiffness: float,
        grasp_assist_damping: float,
        grasp_assist_maximum_force: float,
    ):
        if not 0.0 < grasp_assist_acquisition_distance < grasp_assist_release_distance:
            raise ValueError("Grasp-assist distances must satisfy 0 < acquisition < release")
        if not 0.0 < grasp_assist_acquisition_closed_separation < grasp_assist_release_open_separation:
            raise ValueError("Grasp-assist finger separations must satisfy 0 < acquisition < release")
        if grasp_assist_stiffness < 0.0 or grasp_assist_damping < 0.0 or grasp_assist_maximum_force < 0.0:
            raise ValueError("Grasp-assist stiffness, damping, and maximum force must be non-negative")
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.joint_stiffness_scale = joint_stiffness_scale
        self.grasp_assist_enabled = grasp_assist_enabled and grasp_assist_maximum_force > 0.0
        self.grasp_assist_acquisition_distance = grasp_assist_acquisition_distance
        self.grasp_assist_release_distance = grasp_assist_release_distance
        self.grasp_assist_acquisition_closed_separation = grasp_assist_acquisition_closed_separation
        self.grasp_assist_release_open_separation = grasp_assist_release_open_separation
        self.grasp_assist_stiffness = grasp_assist_stiffness
        self.grasp_assist_damping = grasp_assist_damping
        self.grasp_assist_maximum_force = grasp_assist_maximum_force
        self.cable_bodies: list[list[int]] = []
        self.cable_joints: list[list[int]] = []
        self._grasp_assist_hand_ids: list[int] = []
        self._grasp_assist_finger_ids: list[int] = []
        self._grasp_assist_tail_ids: list[int] = []
        self._grasp_assist_active_view: torch.Tensor | None = None

    @property
    def grasp_assist_active(self) -> torch.Tensor | None:
        """Return per-environment left and right compliant grasp latch state."""
        return self._grasp_assist_active_view

    def reset_grasp_assist(self, env_ids: torch.Tensor) -> None:
        """Clear compliant grasp state for selected Newton worlds."""
        if self._grasp_assist_active_view is not None:
            self._grasp_assist_active_view[env_ids] = 0

    def register(self) -> None:
        """Register model lifecycle callbacks before the first simulation reset."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_task_builder")
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY, name="shoelace_task_model")

    def set_straight_rest_state(self) -> None:
        """Use a straight untwisted bend reference while retaining the authored state."""
        model = NewtonManager.get_model()
        rest = model.body_q.numpy()
        for bodies in self.cable_bodies:
            for chain in (bodies[: PINNED_FIRST + 1], bodies[PINNED_FIRST + 1 :]):
                rest[np.asarray(chain), 3:7] = rest[chain[0], 3:7]
        model.body_q.assign(rest)

    def _configure_builder(self, _: object) -> None:
        """Apply cable properties that must exist before model finalization."""
        builder = NewtonManager.get_builder()
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body, label in enumerate(builder.body_label):
            if "/RobotLeft/" in label or "/RobotRight/" in label:
                gravcomp.values[body] = 1.0

        def find_chains(labels: list[str], worlds: list[int], suffix: str) -> list[tuple[list[int], list[int]]]:
            sides = [
                self._sorted_world_label_indices(labels, worlds, f"/Shoelace{side}/geometry/mesh_{suffix}_")
                for side in ("Left", "Right")
            ]
            return list(zip(*sides, strict=True))

        body_chains = find_chains(builder.body_label, builder.body_world, "edge_body")
        joint_chains = find_chains(builder.joint_label, builder.joint_world, "cable")
        expected_body_counts = (PINNED_FIRST + 1, len(self.centerline) - 1 - PINNED_LAST)
        for world, (bodies, joints) in enumerate(zip(body_chains, joint_chains, strict=True)):
            observed = tuple(
                (len(chain_bodies), len(chain_joints))
                for chain_bodies, chain_joints in zip(bodies, joints, strict=True)
            )
            expected = tuple((count, count - 1) for count in expected_body_counts)
            if observed != expected:
                raise RuntimeError(f"Unexpected cable topology in world {world}: {observed}, expected {expected}")

        self.cable_bodies = [left + right for left, right in body_chains]
        self.cable_joints = [left + right for left, right in joint_chains]
        if self.grasp_assist_enabled:
            (
                self._grasp_assist_hand_ids,
                self._grasp_assist_finger_ids,
                self._grasp_assist_tail_ids,
            ) = _resolve_grasp_assist_bodies(builder, self.cable_bodies, self.num_envs)
        body_by_label = self._index_unique_labels(builder.body_label)
        shape_by_label = self._index_unique_labels(builder.shape_label)
        shoe_parts: list[tuple[int, int, int]] = []
        for world in range(self.num_envs):
            root = f"/World/envs/env_{world}/Shoe"
            shoe_body = self._label_index(body_by_label, root)
            shoe_collider = self._label_index(shape_by_label, f"{root}/Collider")
            tongue_collider = self._label_index(shape_by_label, f"{root}/TongueUpper/geometry/mesh")
            pinned_shape = self._label_index(shape_by_label, f"/World/envs/env_{world}/ShoelacePinned/geometry/mesh")
            if builder.shape_world[pinned_shape] != world:
                raise RuntimeError(f"Pinned mesh is not local to Newton world {world}")
            shoe_parts.append((shoe_collider, tongue_collider, pinned_shape))
            builder.body_mass[shoe_body] = 0.0
            builder.body_inv_mass[shoe_body] = 0.0
            builder.body_inertia[shoe_body] = wp.mat33()
            builder.body_inv_inertia[shoe_body] = wp.mat33()
            for shape in (shoe_collider, tongue_collider):
                builder.shape_flags[shape] = newton.ShapeFlags.COLLIDE_SHAPES
                builder.shape_collision_group[shape] = COLLISION_GROUP
                builder.shape_material_ke[shape] = CONTACT_KE
                builder.shape_material_kd[shape] = CONTACT_KD
                builder.shape_material_mu[shape] = SHOE_MU
                builder.shape_gap[shape] = CONTACT_GAP
            builder.shape_flags[pinned_shape] = newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES
            builder.shape_collision_group[pinned_shape] = COLLISION_GROUP
            builder.shape_material_ke[pinned_shape] = CONTACT_KE
            builder.shape_material_kd[pinned_shape] = CONTACT_KD
            builder.shape_material_mu[pinned_shape] = LACE_MU
            builder.shape_gap[pinned_shape] = CONTACT_GAP

        ground_shapes = [index for index, label in enumerate(builder.shape_label) if label.startswith("/World/Ground/")]
        for shape in ground_shapes:
            builder.shape_material_ke[shape] = CONTACT_KE
            builder.shape_material_kd[shape] = GROUND_CONTACT_KD
            builder.shape_material_mu[shape] = GROUND_MU
            builder.shape_gap[shape] = CONTACT_GAP

        exact_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
            [wp.vec3(*point) for point in self.centerline]
        )
        neighbor_window = _neighbor_filter_window(self.centerline, self.cable_radius)
        segment_ranges = (range(0, PINNED_FIRST + 1), range(PINNED_LAST, len(self.centerline) - 1))
        for world, (world_bodies, world_joints, shoe) in enumerate(
            zip(body_chains, joint_chains, shoe_parts, strict=True)
        ):
            shoe_collider, tongue_collider, pinned_shape = shoe
            chain_shapes: list[list[int]] = []
            for bodies, joints, segment_indices in zip(world_bodies, world_joints, segment_ranges, strict=True):
                shapes: list[int] = []
                for body, segment in zip(bodies, segment_indices, strict=True):
                    quaternion = exact_quaternions[segment]
                    transform = builder.body_q[body]
                    builder.body_q[body] = wp.transform(wp.vec3(transform[0], transform[1], transform[2]), quaternion)
                    shape = int(builder.body_shapes[body][0])
                    shapes.append(shape)
                    builder.shape_collision_group[shape] = COLLISION_GROUP
                    builder.shape_material_ke[shape] = CONTACT_KE
                    builder.shape_material_kd[shape] = CONTACT_KD
                    builder.shape_material_mu[shape] = LACE_MU
                    builder.shape_gap[shape] = CONTACT_GAP
                    radius = float(builder.shape_scale[shape][0])
                    segment_length = 2.0 * float(builder.shape_scale[shape][1])
                    capsule_mass_scale = 1.0 + 4.0 * radius / (3.0 * segment_length)
                    builder.body_mass[body] *= capsule_mass_scale
                    builder.body_inertia[body] *= capsule_mass_scale
                    builder.body_inv_mass[body] = 1.0 / builder.body_mass[body]
                    builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])

                _filter_rod_neighbors(builder, shapes, neighbor_window)
                chain_shapes.append(shapes)

                dof_start = int(builder.joint_qd_start[joints[0]])
                dof_end = int(builder.joint_qd_start[joints[-1]]) + 4
                if dof_end - dof_start != 4 * len(joints):
                    raise RuntimeError("Cable joint DOFs must be contiguous")
                builder.joint_target_ke[dof_start:dof_end] = (
                    STRETCH_STIFFNESS * self.joint_stiffness_scale,
                    STRETCH_STIFFNESS * self.joint_stiffness_scale,
                    BEND_STIFFNESS * self.joint_stiffness_scale,
                    BEND_STIFFNESS * self.joint_stiffness_scale,
                ) * len(joints)
                builder.joint_target_kd[dof_start:dof_end] = (
                    STRETCH_DAMPING * self.joint_stiffness_scale,
                    STRETCH_DAMPING * self.joint_stiffness_scale,
                    BEND_DAMPING * self.joint_stiffness_scale,
                    BEND_DAMPING * self.joint_stiffness_scale,
                ) * len(joints)

            for body, shape in ((world_bodies[0][-1], chain_shapes[0][-1]), (world_bodies[1][0], chain_shapes[1][0])):
                builder.body_mass[body] = 0.0
                builder.body_inv_mass[body] = 0.0
                builder.body_inertia[body] = wp.mat33()
                builder.body_inv_inertia[body] = wp.mat33()
                for shoe_shape in (shoe_collider, tongue_collider):
                    builder.add_shape_collision_filter_pair(shape, shoe_shape)

            for static_shape in (shoe_collider, tongue_collider):
                builder.add_shape_collision_filter_pair(pinned_shape, static_shape)

            seam_shapes = (*chain_shapes[0][-neighbor_window:], *chain_shapes[1][:neighbor_window])
            for shape in seam_shapes:
                builder.add_shape_collision_filter_pair(shape, pinned_shape)

    def _configure_model(self, _: object) -> None:
        """Configure cable hysteresis and compliant grasp transport before solver construction."""
        model = NewtonManager.get_model()
        dahl_eps = np.zeros(model.joint_count, dtype=np.float32)
        dahl_tau = np.zeros(model.joint_count, dtype=np.float32)
        cable_joints = np.asarray([joint for joints in self.cable_joints for joint in joints])
        dahl_eps[cable_joints] = DAHL_MAX_STRAIN
        dahl_tau[cable_joints] = DAHL_DECAY
        model.vbd.dahl_eps_max.assign(dahl_eps)
        model.vbd.dahl_tau.assign(dahl_tau)

        if not self.grasp_assist_enabled:
            return

        device = model.device
        grasp_count = 2 * self.num_envs
        self._grasp_assist_hand_ids_wp = wp.array(self._grasp_assist_hand_ids, dtype=wp.int32, device=device)
        self._grasp_assist_finger_ids_wp = wp.array(self._grasp_assist_finger_ids, dtype=wp.int32, device=device)
        self._grasp_assist_tail_ids_wp = wp.array(self._grasp_assist_tail_ids, dtype=wp.int32, device=device)
        self._grasp_assist_active = wp.zeros(grasp_count, dtype=wp.int32, device=device)
        self._grasp_assist_local_anchors = wp.zeros(grasp_count, dtype=wp.vec3f, device=device)
        self._grasp_assist_active_view = wp.to_torch(self._grasp_assist_active).view(self.num_envs, 2)
        NewtonManager.register_state_force_callback(self._apply_grasp_assist)

    def _apply_grasp_assist(self, state: newton.State) -> None:
        """Apply the graph-safe compliant grasp force before each solver substep."""
        # Proxy contact is one-way, so transport the acquired cable tail on the cable side.
        wp.launch(
            _apply_grasp_assist_kernel,
            dim=2 * self.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_f,
                self._grasp_assist_hand_ids_wp,
                self._grasp_assist_finger_ids_wp,
                self._grasp_assist_tail_ids_wp,
                self._grasp_assist_active,
                self._grasp_assist_local_anchors,
                wp.vec3f(*TCP_OFFSET),
                self.grasp_assist_acquisition_distance,
                self.grasp_assist_release_distance,
                self.grasp_assist_acquisition_closed_separation,
                self.grasp_assist_release_open_separation,
                self.grasp_assist_stiffness,
                self.grasp_assist_damping,
                self.grasp_assist_maximum_force,
            ],
            device=state.body_q.device,
        )

    def _sorted_world_label_indices(
        self,
        labels: list[str],
        worlds: list[int],
        local_prefix: str,
    ) -> list[list[int]]:
        """Group numerically suffixed labels by Newton world in one pass."""
        matches: list[list[tuple[int, int]]] = [[] for _ in range(self.num_envs)]
        for index, (label, world) in enumerate(zip(labels, worlds, strict=True)):
            if world < 0 or world >= self.num_envs:
                continue
            prefix = f"/World/envs/env_{world}{local_prefix}"
            if label.startswith(prefix):
                matches[world].append((int(label.removeprefix(prefix)), index))
        return [[index for _, index in sorted(world_matches)] for world_matches in matches]

    @staticmethod
    def _index_unique_labels(labels: list[str]) -> dict[str, int]:
        result = {label: index for index, label in enumerate(labels)}
        if len(result) != len(labels):
            raise RuntimeError("Newton labels must be unique after world replication")
        return result

    @staticmethod
    def _label_index(label_indices: dict[str, int], expected: str) -> int:
        try:
            return label_indices[expected]
        except KeyError as exc:
            raise RuntimeError(f"Missing Newton label {expected!r}") from exc


class ShoelaceEnv(ManagerBasedRLEnv):
    """Learn to approach, grasp, and untie a fixed knot using two Franka robots."""

    cfg: ShoelaceEnvCfg

    def __init__(self, cfg: ShoelaceEnvCfg, render_mode: str | None = None, **kwargs):
        asset_dir = _resolve_asset_dir()
        curve_asset = asset_dir / "curve.usd"
        self._model_asset = asset_dir / "model.usd"
        authored_centerline, self._cable_radius = _load_usd_curve(curve_asset, "/World/Curve")
        authored_segment_lengths = np.linalg.norm(np.diff(authored_centerline, axis=0), axis=1)
        if len(authored_segment_lengths) != 450:
            raise ValueError(f"Expected 450 authored shoelace segments, got {len(authored_segment_lengths)}")
        self._authored_mean_segment_length = float(authored_segment_lengths.mean())
        self._centerline = _resample_centerline(authored_centerline, SHOELACE_SEGMENT_COUNT)
        mean_segment_length = float(np.linalg.norm(np.diff(self._centerline, axis=0), axis=1).mean())
        self._configure_runtime_cfg(cfg, SHOELACE_COLLIDER_ASSET)
        self._physics = _ShoelacePhysics(
            self._centerline,
            self._cable_radius,
            cfg.scene.num_envs,
            self._authored_mean_segment_length / mean_segment_length,
            grasp_assist_enabled=cfg.grasp_assist_enabled,
            grasp_assist_acquisition_distance=cfg.grasp_assist_acquisition_distance,
            grasp_assist_release_distance=cfg.grasp_assist_release_distance,
            grasp_assist_acquisition_closed_separation=cfg.grasp_assist_acquisition_closed_separation,
            grasp_assist_release_open_separation=cfg.grasp_assist_release_open_separation,
            grasp_assist_stiffness=cfg.grasp_assist_stiffness,
            grasp_assist_damping=cfg.grasp_assist_damping,
            grasp_assist_maximum_force=cfg.grasp_assist_maximum_force,
        )

        super().__init__(cfg, render_mode, **kwargs)
        self._physics.set_straight_rest_state()

    def _configure_runtime_cfg(self, cfg: ShoelaceEnvCfg, collider_asset: Path) -> None:
        """Fill asset- and world-count-dependent configuration before validation."""
        expected_segments = len(self._centerline) - 1
        if expected_segments != SHOELACE_SEGMENT_COUNT:
            raise ValueError(f"Expected {SHOELACE_SEGMENT_COUNT} shoelace segments, got {expected_segments}")

        cfg.scene.shoe.spawn.usd_path = str(collider_asset)
        cfg.scene.shoe.spawn.func = _spawn_collision_mesh_usd
        cfg.scene.shoe.spawn.collision_props = None
        cfg.scene.shoe_visual.spawn.usd_path = str(self._model_asset)
        cfg.scene.shoelace_pinned_visual.spawn = _pinned_shoelace_spawner(self._centerline, self._cable_radius)
        cable_normals = _parallel_transport_normals(self._centerline)
        cable_centerlines = (
            self._centerline[: PINNED_FIRST + 2],
            self._centerline[PINNED_LAST:],
        )
        cable_normal_chains = (
            cable_normals[: PINNED_FIRST + 2],
            cable_normals[PINNED_LAST:],
        )
        cross_section_area = math.pi * self._cable_radius**2
        second_moment = 0.25 * math.pi * self._cable_radius**4
        cable_cfgs = (cfg.scene.shoelace_left.spawn, cfg.scene.shoelace_right.spawn)
        for cable_cfg, centerline, normals in zip(cable_cfgs, cable_centerlines, cable_normal_chains, strict=True):
            cable_cfg.positions = [tuple(point) for point in centerline]
            cable_cfg.normals = normals
            cable_cfg.physics_material.thickness = 2.0 * self._cable_radius
            cable_cfg.physics_material.density = CABLE_DENSITY
            cable_cfg.physics_material.stretch_stiffness = (
                STRETCH_STIFFNESS * self._authored_mean_segment_length / cross_section_area
            )
            cable_cfg.physics_material.bend_stiffness = (
                BEND_STIFFNESS * self._authored_mean_segment_length / second_moment
            )

        grid_side = math.ceil(math.sqrt(cfg.scene.num_envs))
        ground_size = max(2.0, cfg.scene.env_spacing * (grid_side + 1))
        cfg.scene.ground.spawn.size = (ground_size, ground_size)

        physics_cfg = cfg.sim.physics
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, CouplerProxyCfg):
            raise TypeError("The dual-Franka shoelace task requires Newton proxy coupling")
        vbd_entries = [entry for entry in physics_cfg.solver_cfg.entries if entry.name == "shoelace"]
        if len(vbd_entries) != 1 or not isinstance(vbd_entries[0].solver_cfg, VBDSolverCfg):
            raise TypeError("The shoelace coupler entry requires one VBD solver")
        proxy_pipelines = [proxy.collision_pipeline for proxy in physics_cfg.solver_cfg.proxies]
        if len(proxy_pipelines) != 1 or not isinstance(proxy_pipelines[0], NewtonCollisionPipelineCfg):
            raise TypeError("The shoelace coupler proxy requires one configured Newton collision pipeline")
        proxy_collision_cfg = proxy_pipelines[0]
        triangle_pair_capacity = max(1_000_000, cfg.triangle_pairs_per_env * cfg.scene.num_envs)
        physics_cfg.collision_cfg.rigid_contact_max = cfg.contacts_per_env * cfg.scene.num_envs
        physics_cfg.collision_cfg.max_triangle_pairs = triangle_pair_capacity
        proxy_collision_cfg.max_triangle_pairs = triangle_pair_capacity

    def _init_sim(self) -> None:
        """Register Newton lifecycle callbacks before the scene's first reset."""
        stage = sim_utils.get_current_stage()
        material_scope = UsdGeom.Scope.Define(stage, "/World/ShoeMaterials").GetPrim()
        material_scope.GetReferences().AddReference(str(self._model_asset), "/mat")
        self._physics.register()
        super()._init_sim()
