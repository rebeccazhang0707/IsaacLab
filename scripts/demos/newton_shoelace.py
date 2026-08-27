# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tighten and untie an authored shoelace with force-controlled Newton VBD dynamics.

Only the virtual fingertip forces are scripted. The selected lace segments stay dynamic, and no
pose or velocity target is tracked. The eyelet-held lace span is pinned while the bow loops are
first tightened, the two tails are pulled in sequence, and the loosened bight is extracted.

.. code-block:: bash

    # Run the complete 12 second sequence with the Newton visualizer.
    uv run python scripts/demos/newton_shoelace.py

    # Use the Kit visualizer instead.
    uv run --extra isaacsim python scripts/demos/newton_shoelace.py --visualizer kit

    # Run headless and verify the final untied state.
    uv run python scripts/demos/newton_shoelace.py --visualizer none --verify

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-controlled Newton VBD shoelace demo.", conflict_handler="resolve")
parser.add_argument("--contact_buffer", type=int, default=256, help="Per-body VBD rigid-contact capacity.")
parser.add_argument("--max_steps", type=int, default=720, help="Number of 60 Hz simulation steps.")
parser.add_argument("--verify", action="store_true", help="Check force delivery, stability, and final untying.")
parser.add_argument("--physics", default="newton_vbd", choices=["newton_vbd"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

if args_cli.contact_buffer < 1:
    parser.error("--contact_buffer must be at least 1.")
if args_cli.max_steps < 1:
    parser.error("--max_steps must be at least 1.")

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsCfg, PhysicsEvent
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

if TYPE_CHECKING:
    from newton import ModelBuilder, State


ASSET_DIR = Path(__file__).resolve().parents[2] / ".newton" / "newton" / "examples" / "assets" / "shoelace"
COLLIDER_ASSET = ASSET_DIR / "collider.usd"
CURVE_ASSET = ASSET_DIR / "curve.usd"
MODEL_ASSET = ASSET_DIR / "model.usd"

# Simulation cadence.
FPS = 60
SIM_SUBSTEPS = 10
SIM_ITERATIONS = 24
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS

# Lace material. Stiffnesses and damping below are Newton per-joint values.
CABLE_DENSITY = 1150.0
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 0.50
BEND_DAMPING = 0.45
DAHL_MAX_STRAIN = 0.20
DAHL_DECAY = 0.35

# Contact model.
GRAVITY = -9.81
CONTACT_GAP = 1.0e-4
CONTACT_KE = 1.0e6
CONTACT_KD = 0.0
GROUND_CONTACT_KD = 30.0
LACE_MU = 0.7
SHOE_MU = 0.2
GROUND_MU = 0.8
COLLISION_GROUP = 1

# Tightening and untying force schedule [s].
SETTLE_END = 0.6
TIGHTEN_END = 1.6
RELEASE_TIME = 2.1
LOOP_GRAB_LEFT = (53, 58)
LOOP_GRAB_RIGHT = (392, 397)
UNTIE_FIRST_START = 3.0
UNTIE_FIRST_END = 4.2
UNTIE_SECOND_START = 5.2
UNTIE_SECOND_END = 6.4
UNTIE_HOLD = 0.3
UNTIE_RELAX = 0.6
# The imported Lab contact trajectory can leave the loosened bight at the throat. A final force-only
# extraction gives the standalone demo a reproducible untied terminal state without adding a pose target.
EXTRACT_START = 7.4
EXTRACT_END = 9.6
EXTRACT_FORCE = 0.1
EXTRACT_DOWNWARD_BIAS = 1.0
TAIL_GRAB_FIRST = (446, 450)
TAIL_GRAB_SECOND = (0, 4)
TIGHTEN_FORCE_RAMP = 1.0
UNTIE_FORCE_RAMP = 1.0
TIGHTEN_FORCE = 0.1
UNTIE_FORCE_FIRST = 0.1
UNTIE_FORCE_SECOND = 0.1
GUIDED_DURATION = EXTRACT_END + UNTIE_HOLD + UNTIE_RELAX

# Shoe and display parameters.
FOOT_CENTER = (-0.01, 0.060, 0.075)
FOOT_RADIUS = 0.025
FOOT_HALF_LENGTH = 0.04
CABLE_COLOR = (112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0)

# Headless verification thresholds.
# The contact-enabled reference run reaches 9.5 mm on the constrained right loop.
MIN_TIGHTEN_OUTWARD_DISPLACEMENT = 0.009
MAX_FREE_SPEED = 3.0
MAX_JOINT_GAP = 0.0015
# Sampled extraction runs peak at 68--73 mm, while the force-free jammed control peaks at 47--49 mm.
MIN_BIGHT_REACH = 0.052
# The authored tied state has 100 free segments near the throat; successful runs finish with 62--65.
MIN_NEAR_KNOT_REDUCTION = 15


def _world_points(prim: Usd.Prim) -> np.ndarray:
    """Return point-based geometry positions in world coordinates."""
    points = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).reshape(4, 4)
    return (np.c_[points, np.ones(len(points))] @ transform)[:, :3]


def _load_usd_mesh(path: Path, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangulated world-space USD mesh, preserving its winding."""
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Missing prim {prim_path} in {path}")
    vertices = _world_points(prim)
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    left_handed = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).reshape(4, 4)
    flip = left_handed != (np.linalg.det(transform[:3, :3]) < 0.0)
    triangles = []
    offset = 0
    for count in counts:
        for index in range(1, count - 1):
            a, b, c = indices[offset], indices[offset + index], indices[offset + index + 1]
            triangles.append((a, c, b) if flip else (a, b, c))
        offset += count
    return vertices.astype(np.float32), np.asarray(triangles, dtype=np.int32).reshape(-1)


def _load_usd_curve(path: Path, prim_path: str) -> tuple[np.ndarray, float]:
    """Load one world-space cable centerline and its radius."""
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Missing prim {prim_path} in {path}")
    centerline = _world_points(prim)
    curves = UsdGeom.BasisCurves(prim)
    counts = list(curves.GetCurveVertexCountsAttr().Get())
    if counts != [len(centerline)]:
        raise ValueError(f"Expected one curve in {path}, got counts {counts}")
    widths = curves.GetWidthsAttr().Get()
    if not widths:
        raise ValueError(f"Curve in {path} has no widths attribute")
    return centerline, 0.5 * float(min(widths))


def _parallel_transport_normals(centerline: np.ndarray) -> list[tuple[float, float, float]]:
    """Build per-point normals that preserve Newton's parallel-transport cable frames."""
    positions = [wp.vec3(*point) for point in centerline]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(positions)
    normals = [wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0)) for quaternion in quaternions]
    normals.append(normals[-1])
    return [(float(normal[0]), float(normal[1]), float(normal[2])) for normal in normals]


@wp.kernel
def _query_mesh_clearance(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    distances: wp.array[wp.float32],
):
    index = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh_id, points[index], 10.0)
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    distances[index] = wp.length(points[index] - closest)


def _captured_segments(
    centerline: np.ndarray, radius: float, vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Find the contiguous lace span held against the shoe by the eyelets."""
    midpoints = 0.5 * (centerline[:-1] + centerline[1:])
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(triangles, dtype=wp.int32),
        support_winding_number=True,
    )
    points = wp.array(midpoints.astype(np.float32), dtype=wp.vec3)
    distances = wp.zeros(len(midpoints), dtype=wp.float32)
    wp.launch(_query_mesh_clearance, dim=len(midpoints), inputs=[mesh.id, points, distances])
    near = np.where(distances.numpy() < 2.0 * radius)[0]
    near = near[(near > LOOP_GRAB_LEFT[1]) & (near < LOOP_GRAB_RIGHT[0])]
    if len(near) < 2:
        raise ValueError("The authored lace does not appear to pass through the shoe")
    captured = np.zeros(len(midpoints), dtype=bool)
    captured[near.min() : near.max() + 1] = True
    return captured


def _neighbor_filter_window(points: np.ndarray, radius: float) -> int:
    """Return the number of adjacent rod capsules that overlap by construction."""
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    contact_reach = 2.0 * radius + CONTACT_GAP
    return max(1, int(np.floor(contact_reach / minimum_length + 1.0e-9)) + 1)


def _filter_rod_neighbors(builder: ModelBuilder, bodies: list[int], window: int) -> None:
    """Disable collisions between geometrically overlapping neighbor segments."""
    for first, first_body in enumerate(bodies):
        for second_body in bodies[first + 1 : min(first + 1 + window, len(bodies))]:
            for first_shape in builder.body_shapes.get(first_body, []):
                for second_shape in builder.body_shapes.get(second_body, []):
                    builder.add_shape_collision_filter_pair(int(first_shape), int(second_shape))


def _segment_axes(quaternions: np.ndarray) -> np.ndarray:
    """Return world-space local-Z axes for xyzw quaternions."""
    z_axis = np.zeros((len(quaternions), 3), dtype=quaternions.dtype)
    z_axis[:, 2] = 1.0
    twice_cross = 2.0 * np.cross(quaternions[:, :3], z_axis)
    return z_axis + quaternions[:, 3, None] * twice_cross + np.cross(quaternions[:, :3], twice_cross)


@wp.kernel
def _apply_pull_forces(
    grab_bodies: wp.array[wp.int32],
    grab_offsets: wp.array[wp.int32],
    grab_counts: wp.array[wp.int32],
    move_t0: wp.array[wp.float32],
    move_t1: wp.array[wp.float32],
    force_directions: wp.array[wp.vec3],
    force_magnitudes: wp.array[wp.float32],
    ramp_up: wp.array[wp.float32],
    ramp_down: wp.array[wp.float32],
    elapsed: wp.array[wp.float32],
    sim_dt: float,
    body_f: wp.array[wp.spatial_vector],
    peak_forces: wp.array[wp.float32],
):
    """Apply smooth open-loop force profiles to fully dynamic cable segments."""
    move = wp.tid()
    time = elapsed[0] + sim_dt
    t0 = move_t0[move]
    t1 = move_t1[move]
    if time < t0 or time > t1:
        return

    up_u = wp.clamp((time - t0) / wp.max(ramp_up[move], 1.0e-6), 0.0, 1.0)
    down_u = wp.clamp((t1 - time) / wp.max(ramp_down[move], 1.0e-6), 0.0, 1.0)
    up_scale = up_u * up_u * (3.0 - 2.0 * up_u)
    down_scale = down_u * down_u * (3.0 - 2.0 * down_u)
    magnitude = force_magnitudes[move] * wp.min(up_scale, down_scale)
    force = magnitude * force_directions[move] / float(grab_counts[move])
    offset = grab_offsets[move]
    for index in range(grab_counts[move]):
        body_f[grab_bodies[offset + index]] += wp.spatial_vector(force, wp.vec3())
    peak_forces[move] = wp.max(peak_forces[move], magnitude)


@wp.kernel
def _advance_elapsed(elapsed: wp.array[wp.float32], sim_dt: float, limit: float):
    elapsed[0] = wp.min(elapsed[0] + sim_dt, limit)


class ShoelaceController:
    """Configure the imported Newton model and apply the virtual fingertip forces."""

    def __init__(self, centerline: np.ndarray, cable_radius: float, collider_vertices: np.ndarray, collider_tris):
        self.centerline = centerline
        self.cable_radius = cable_radius
        captured = _captured_segments(centerline, cable_radius, collider_vertices, collider_tris)
        held = np.flatnonzero(captured)
        self.first_pin = int(held.min())
        self.last_pin = int(held.max())
        self.moves = self._build_tighten_moves() + self._build_untie_moves()
        self.cable_bodies: list[int] = []
        self.cable_joints: list[int] = []
        self.shoe_collider: int | None = None
        self._cable_shapes = np.empty(0, dtype=np.int32)
        self._initial_body_q = None
        self._initial_pulled_inv_mass = None
        self._tighten_initial_centers = None
        self.tighten_peak_outward_displacement = np.zeros(2, dtype=np.float64)
        self.max_free_speed = 0.0
        self.max_joint_gap = 0.0
        self.max_bight_reach = 0.0
        self.max_lace_shoe_contacts = 0
        self.initial_near_knot: int | None = None

    def register(self) -> None:
        """Register model lifecycle and per-substep force callbacks."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_builder")
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY, name="shoelace_model")
        NewtonManager.register_state_force_callback(self._apply_forces)

    def _configure_builder(self, _: object) -> None:
        """Apply cable properties that must exist before model finalization."""
        builder = NewtonManager.get_builder()
        body_prefix = "/World/Shoelace/geometry/mesh_edge_body_"
        joint_prefix = "/World/Shoelace/geometry/mesh_cable_"
        self.cable_bodies = self._sorted_label_indices(builder.body_label, body_prefix)
        self.cable_joints = self._sorted_label_indices(builder.joint_label, joint_prefix)
        expected_segments = len(self.centerline) - 1
        if len(self.cable_bodies) != expected_segments or len(self.cable_joints) != expected_segments - 1:
            raise RuntimeError(
                f"Unexpected cable topology: {len(self.cable_bodies)} bodies and {len(self.cable_joints)} joints"
            )

        # USD normals preserve the authored frame semantically, but reconstructing a quaternion from a
        # normal introduces a small round trip. Restore the exact source-demo frames before finalization.
        exact_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
            [wp.vec3(*point) for point in self.centerline]
        )
        for body, quaternion in zip(self.cable_bodies, exact_quaternions, strict=True):
            transform = builder.body_q[body]
            builder.body_q[body] = wp.transform(wp.vec3(transform[0], transform[1], transform[2]), quaternion)

        shoe_body = self._unique_label_index(builder.body_label, "/World/Shoe")
        shoe_collider = self._unique_label_index(builder.shape_label, "/World/Shoe/Collider")
        builder.body_mass[shoe_body] = 0.0
        builder.body_inv_mass[shoe_body] = 0.0
        builder.body_inertia[shoe_body] = wp.mat33()
        builder.body_inv_inertia[shoe_body] = wp.mat33()

        foot_shape = self._unique_label_index(builder.shape_label, "/World/Shoe/Foot/geometry/mesh")

        cable_shapes = []
        for body in self.cable_bodies:
            cable_shapes.extend(int(shape) for shape in builder.body_shapes.get(body, []))
        self.shoe_collider = shoe_collider
        self._cable_shapes = np.asarray(cable_shapes, dtype=np.int32)
        builder.shape_flags[shoe_collider] = newton.ShapeFlags.COLLIDE_SHAPES
        for shape in [shoe_collider, foot_shape, *cable_shapes]:
            builder.shape_collision_group[shape] = COLLISION_GROUP
        ground_shapes = [index for index, label in enumerate(builder.shape_label) if label.startswith("/World/Ground/")]
        for shape in (shoe_collider, foot_shape):
            builder.shape_material_ke[shape] = CONTACT_KE
            builder.shape_material_kd[shape] = CONTACT_KD
            builder.shape_material_mu[shape] = SHOE_MU
            builder.shape_gap[shape] = CONTACT_GAP
        for shape in ground_shapes:
            builder.shape_material_ke[shape] = CONTACT_KE
            builder.shape_material_kd[shape] = GROUND_CONTACT_KD
            builder.shape_material_mu[shape] = GROUND_MU
            builder.shape_gap[shape] = CONTACT_GAP

        # AOUSD cable import intentionally removes capsule-cap mass so material density describes
        # a cylindrical curve element. The source demo was tuned against ModelBuilder.add_rod's
        # capsule mass, so restore that local mass/inertia convention for migration parity.
        for body in self.cable_bodies:
            shape = int(builder.body_shapes[body][0])
            radius = float(builder.shape_scale[shape][0])
            segment_length = 2.0 * float(builder.shape_scale[shape][1])
            capsule_mass_scale = 1.0 + 4.0 * radius / (3.0 * segment_length)
            builder.body_mass[body] *= capsule_mass_scale
            builder.body_inertia[body] *= capsule_mass_scale
            builder.body_inv_mass[body] = 1.0 / builder.body_mass[body]
            builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])

        for segment in range(self.first_pin, self.last_pin + 1):
            body = self.cable_bodies[segment]
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33()
            builder.body_inv_inertia[body] = wp.mat33()
            for shape in builder.body_shapes.get(body, []):
                builder.add_shape_collision_filter_pair(int(shape), shoe_collider)

        _filter_rod_neighbors(
            builder,
            self.cable_bodies,
            _neighbor_filter_window(self.centerline, self.cable_radius),
        )

        for joint in self.cable_joints:
            dof = int(builder.joint_qd_start[joint])
            builder.joint_target_ke[dof : dof + 4] = [
                STRETCH_STIFFNESS,
                STRETCH_STIFFNESS,
                BEND_STIFFNESS,
                BEND_STIFFNESS,
            ]
            builder.joint_target_kd[dof : dof + 4] = [
                STRETCH_DAMPING,
                STRETCH_DAMPING,
                BEND_DAMPING,
                BEND_DAMPING,
            ]

    def _configure_model(self, _: object) -> None:
        """Configure Dahl hysteresis and force buffers before solver construction."""
        model = NewtonManager.get_model()
        dahl_eps = np.zeros(model.joint_count, dtype=np.float32)
        dahl_tau = np.zeros(model.joint_count, dtype=np.float32)
        dahl_eps[self.cable_joints] = DAHL_MAX_STRAIN
        dahl_tau[self.cable_joints] = DAHL_DECAY
        model.vbd.dahl_eps_max.assign(dahl_eps)
        model.vbd.dahl_tau.assign(dahl_tau)

        state = NewtonManager.get_state_0()
        self._initial_body_q = wp.clone(state.body_q)

        grab_bodies = []
        grab_offsets = []
        for move in self.moves:
            grab_offsets.append(len(grab_bodies))
            grab_bodies.extend(self.cable_bodies[segment] for segment in move["segments"])
        device = model.device
        self.grab_bodies_wp = wp.array(grab_bodies, dtype=wp.int32, device=device)
        self.grab_offsets_wp = wp.array(grab_offsets, dtype=wp.int32, device=device)
        self.grab_counts_wp = wp.array([len(move["segments"]) for move in self.moves], dtype=wp.int32, device=device)
        self.move_t0_wp = wp.array([move["t0"] for move in self.moves], dtype=wp.float32, device=device)
        self.move_t1_wp = wp.array([move["t1"] for move in self.moves], dtype=wp.float32, device=device)
        self.force_directions_wp = wp.array([move["direction"] for move in self.moves], dtype=wp.vec3, device=device)
        self.force_magnitudes_wp = wp.array([move["force"] for move in self.moves], dtype=wp.float32, device=device)
        self.force_ramp_up_wp = wp.array([move["ramp_up"] for move in self.moves], dtype=wp.float32, device=device)
        self.force_ramp_down_wp = wp.array([move["ramp_down"] for move in self.moves], dtype=wp.float32, device=device)
        self.peak_pull_forces_wp = wp.zeros(len(self.moves), dtype=wp.float32, device=device)
        self.elapsed_wp = wp.zeros(1, dtype=wp.float32, device=device)

        pulled_bodies = np.asarray(
            sorted({self.cable_bodies[segment] for move in self.moves for segment in move["segments"]}),
            dtype=np.int32,
        )
        self._pulled_bodies = pulled_bodies
        self._initial_pulled_inv_mass = model.body_inv_mass.numpy()[pulled_bodies].copy()
        self._free_mask = np.ones(len(self.cable_bodies), dtype=bool)
        self._free_mask[self.first_pin : self.last_pin + 1] = False
        self._segment_half_length = 0.5 * np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)

    def restore_authored_state(self) -> None:
        """Restore the knotted initial state after solver initialization evaluates FK."""
        if self._initial_body_q is None:
            raise RuntimeError("The shoelace model was not initialized")

        model = NewtonManager.get_model()
        rest = model.body_q.numpy()
        rest[np.asarray(self.cable_bodies), 3:7] = rest[self.cable_bodies[0], 3:7]
        model.body_q.assign(rest)
        for state in self._iter_states():
            state.body_q.assign(self._initial_body_q)
            state.body_qd.zero_()
            state.clear_forces()
        self.elapsed_wp.zero_()
        self.peak_pull_forces_wp.zero_()
        NewtonManager.invalidate_body_state()
        NewtonManager.refresh_contacts()

    def update_verification(self, sim_time: float) -> None:
        """Accumulate inexpensive stability metrics for ``--verify``."""
        state = NewtonManager.get_state_0()
        body_q = state.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError(f"Non-finite body transform at t={sim_time:.3f} s")
        lace_q = body_q[self.cable_bodies]
        contacts = NewtonManager.get_contacts()
        if contacts is not None:
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            shape_0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
            shape_1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
            shoe_mask = (shape_0 == self.shoe_collider) | (shape_1 == self.shoe_collider)
            other_shapes = np.where(shape_0 == self.shoe_collider, shape_1, shape_0)
            lace_shoe_contacts = int(np.count_nonzero(shoe_mask & np.isin(other_shapes, self._cable_shapes)))
            self.max_lace_shoe_contacts = max(self.max_lace_shoe_contacts, lace_shoe_contacts)
        if self.initial_near_knot is None:
            knot = 0.5 * (lace_q[self.first_pin, :3] + lace_q[self.last_pin, :3])
            free_positions = np.r_[lace_q[: self.first_pin, :3], lace_q[self.last_pin + 1 :, :3]]
            self.initial_near_knot = int((np.linalg.norm(free_positions - knot, axis=1) < 0.025).sum())
        if SETTLE_END <= sim_time <= TIGHTEN_END + FRAME_DT:
            centers = np.stack(
                (
                    lace_q[LOOP_GRAB_LEFT[0] : LOOP_GRAB_LEFT[1], :3].mean(axis=0),
                    lace_q[LOOP_GRAB_RIGHT[0] : LOOP_GRAB_RIGHT[1], :3].mean(axis=0),
                )
            )
            if self._tighten_initial_centers is None:
                self._tighten_initial_centers = centers.copy()
            directions = np.asarray([move["direction"] for move in self.moves[:2]])
            displacement = np.sum((centers - self._tighten_initial_centers) * directions, axis=1)
            self.tighten_peak_outward_displacement = np.maximum(self.tighten_peak_outward_displacement, displacement)

        if sim_time >= EXTRACT_START:
            knot = 0.5 * (lace_q[self.first_pin, :3] + lace_q[self.last_pin, :3])
            apex = lace_q[(LOOP_GRAB_LEFT[0] + LOOP_GRAB_LEFT[1]) // 2, :3]
            self.max_bight_reach = max(self.max_bight_reach, float(np.linalg.norm(apex - knot)))

        speed = np.linalg.norm(state.body_qd.numpy()[self.cable_bodies, :3], axis=1)
        self.max_free_speed = max(self.max_free_speed, float(speed[self._free_mask].max()))
        axes = _segment_axes(lace_q[:, 3:7])
        left = lace_q[:-1, :3] + axes[:-1] * self._segment_half_length[:-1, None]
        right = lace_q[1:, :3] - axes[1:] * self._segment_half_length[1:, None]
        self.max_joint_gap = max(self.max_joint_gap, float(np.linalg.norm(left - right, axis=1).max()))

    def verify(self) -> None:
        """Verify the force-driven sequence stayed dynamic, stable, and ended untied."""
        peak_forces = self.peak_pull_forces_wp.numpy()
        scheduled = np.asarray([move["force"] for move in self.moves], dtype=np.float32)
        if not np.allclose(peak_forces, scheduled, rtol=1.0e-3, atol=1.0e-5):
            raise AssertionError(f"Expected peak forces {scheduled.tolist()} N, got {peak_forces.tolist()} N")

        model = NewtonManager.get_model()
        pulled_inv_mass = model.body_inv_mass.numpy()[self._pulled_bodies]
        if not np.allclose(pulled_inv_mass, self._initial_pulled_inv_mass):
            raise AssertionError("A force-controlled grab segment became kinematic")
        if float(self.tighten_peak_outward_displacement.min()) < MIN_TIGHTEN_OUTWARD_DISPLACEMENT:
            displacement_mm = (self.tighten_peak_outward_displacement * 1000.0).round(1).tolist()
            raise AssertionError(f"Tightening did not move both loops outward: {displacement_mm} mm")
        if self.max_free_speed > MAX_FREE_SPEED:
            raise AssertionError(f"Peak free-segment speed was {self.max_free_speed:.2f} m/s")
        if self.max_joint_gap > MAX_JOINT_GAP:
            raise AssertionError(f"A cable joint opened by {self.max_joint_gap * 1000.0:.2f} mm")
        if self.max_lace_shoe_contacts == 0:
            raise AssertionError("The free lace never contacted the shoe collider")

        positions = NewtonManager.get_state_0().body_q.numpy()[self.cable_bodies, :3]
        if float(positions[:, 2].min()) < -2.0 * self.cable_radius:
            raise AssertionError(f"The lace fell through the floor: min z={positions[:, 2].min():.4f} m")
        spread = float(np.linalg.norm(positions - positions.mean(axis=0), axis=1).max())
        if spread > 0.6:
            raise AssertionError(f"The lace left the scene: spread={spread:.3f} m")
        knot = 0.5 * (positions[self.first_pin] + positions[self.last_pin])
        apex = positions[(LOOP_GRAB_LEFT[0] + LOOP_GRAB_LEFT[1]) // 2]
        reach = float(np.linalg.norm(apex - knot))
        free_positions = np.r_[positions[: self.first_pin], positions[self.last_pin + 1 :]]
        near_knot = int((np.linalg.norm(free_positions - knot, axis=1) < 0.025).sum())
        if self.initial_near_knot is None:
            raise AssertionError("The initial knot occupancy was not measured")
        print(
            "[INFO]: Verification metrics: "
            f"tighten={(self.tighten_peak_outward_displacement * 1000.0).round(1).tolist()} mm, "
            f"max_speed={self.max_free_speed:.3f} m/s, max_gap={self.max_joint_gap * 1000.0:.3f} mm, "
            f"peak_shoe_contacts={self.max_lace_shoe_contacts}, "
            f"peak_apex_reach={self.max_bight_reach * 1000.0:.1f} mm, final_apex_reach={reach * 1000.0:.1f} mm, "
            f"apex_vector={((apex - knot) * 1000.0).round(1).tolist()} mm, "
            f"near_knot={self.initial_near_knot}->{near_knot}."
        )
        if self.max_bight_reach < MIN_BIGHT_REACH:
            raise AssertionError(
                f"The second bight remained in the wrap: peak reach={self.max_bight_reach * 1000.0:.0f} mm"
            )
        near_knot_reduction = self.initial_near_knot - near_knot
        if near_knot_reduction < MIN_NEAR_KNOT_REDUCTION:
            raise AssertionError(
                f"The knot remained tied: throat occupancy decreased by only {near_knot_reduction} segments"
            )

        print(
            "[INFO]: Verification passed: "
            f"peak_forces={peak_forces.round(3).tolist()} N, "
            f"peak_apex_reach={self.max_bight_reach * 1000.0:.1f} mm, near_knot={near_knot}."
        )

    def _apply_forces(self, state: State) -> None:
        """Apply scheduled forces before one Newton solver substep."""
        wp.launch(
            _apply_pull_forces,
            dim=len(self.moves),
            inputs=[
                self.grab_bodies_wp,
                self.grab_offsets_wp,
                self.grab_counts_wp,
                self.move_t0_wp,
                self.move_t1_wp,
                self.force_directions_wp,
                self.force_magnitudes_wp,
                self.force_ramp_up_wp,
                self.force_ramp_down_wp,
                self.elapsed_wp,
                SIM_DT,
                state.body_f,
                self.peak_pull_forces_wp,
            ],
            device=state.body_f.device,
        )
        wp.launch(
            _advance_elapsed,
            dim=1,
            inputs=[self.elapsed_wp, SIM_DT, GUIDED_DURATION],
            device=state.body_f.device,
        )

    def _build_tighten_moves(self) -> list[dict]:
        """Build opposing force profiles that cinch the two bow loops."""
        midpoints = 0.5 * (self.centerline[:-1] + self.centerline[1:])
        knot = 0.5 * (midpoints[self.first_pin] + midpoints[self.last_pin])
        moves = []
        for lower, upper in (LOOP_GRAB_LEFT, LOOP_GRAB_RIGHT):
            segments = list(range(lower, min(upper, len(midpoints))))
            outward = midpoints[segments].mean(axis=0) - knot
            outward[2] = 0.0
            outward /= np.linalg.norm(outward)
            moves.append(
                {
                    "segments": segments,
                    "t0": SETTLE_END,
                    "t1": RELEASE_TIME,
                    "direction": outward,
                    "force": TIGHTEN_FORCE,
                    "ramp_up": TIGHTEN_FORCE_RAMP,
                    "ramp_down": RELEASE_TIME - TIGHTEN_END,
                }
            )
        return moves

    def _build_untie_moves(self) -> list[dict]:
        """Build sequential force profiles that pull both tails and extract the loosened bight."""
        midpoints = 0.5 * (self.centerline[:-1] + self.centerline[1:])
        knot = 0.5 * (midpoints[self.first_pin] + midpoints[self.last_pin])
        moves = []
        for (lower, upper), (start, pull_end), force in (
            (TAIL_GRAB_FIRST, (UNTIE_FIRST_START, UNTIE_FIRST_END), UNTIE_FORCE_FIRST),
            (TAIL_GRAB_SECOND, (UNTIE_SECOND_START, UNTIE_SECOND_END), UNTIE_FORCE_SECOND),
        ):
            segments = list(range(lower, min(upper, len(midpoints))))
            side = 1.0 if midpoints[segments].mean(axis=0)[0] >= knot[0] else -1.0
            outward = np.asarray((side, -0.35, 0.15), dtype=np.float64)
            outward /= np.linalg.norm(outward)
            moves.append(
                {
                    "segments": segments,
                    "t0": float(start),
                    "t1": float(pull_end + UNTIE_HOLD + UNTIE_RELAX),
                    "direction": outward,
                    "force": force,
                    "ramp_up": UNTIE_FORCE_RAMP,
                    "ramp_down": UNTIE_RELAX,
                }
            )
        segments = list(range(*LOOP_GRAB_LEFT))
        outward = midpoints[segments].mean(axis=0) - knot
        horizontal_norm = np.linalg.norm(outward[:2])
        outward[2] = -EXTRACT_DOWNWARD_BIAS * horizontal_norm
        outward /= np.linalg.norm(outward)
        moves.append(
            {
                "segments": segments,
                "t0": EXTRACT_START,
                "t1": EXTRACT_END + UNTIE_HOLD + UNTIE_RELAX,
                "direction": outward,
                "force": EXTRACT_FORCE,
                "ramp_up": UNTIE_FORCE_RAMP,
                "ramp_down": UNTIE_RELAX,
            }
        )

        return moves

    @staticmethod
    def _sorted_label_indices(labels: list[str], prefix: str) -> list[int]:
        matches = [
            (int(label.removeprefix(prefix)), index) for index, label in enumerate(labels) if label.startswith(prefix)
        ]
        return [index for _, index in sorted(matches)]

    @staticmethod
    def _unique_label_index(labels: list[str], expected: str) -> int:
        matches = [index for index, label in enumerate(labels) if label == expected]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Newton label {expected!r}, got {len(matches)}")
        return matches[0]

    @staticmethod
    def _iter_states():
        state_0 = NewtonManager.get_state_0()
        state_1 = NewtonManager.get_state_1()
        yield state_0
        if state_1 is not None and state_1 is not state_0:
            yield state_1


def _rigid_material(friction: float, damping: float) -> list:
    """Build composed standard-friction and Newton-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        NewtonMaterialCfg(contact_stiffness=CONTACT_KE, contact_damping=damping),
    ]


def design_scene(centerline: np.ndarray, cable_radius: float) -> None:
    """Author the shoe visual, collider, cable, ground, and lighting into the USD stage."""
    shoe_cfg = sim_utils.UsdFileCfg(
        usd_path=str(COLLIDER_ASSET),
        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
    )
    shoe_cfg.func("/World/Shoe", shoe_cfg)
    sim_utils.apply_collision_properties(
        "/World/Shoe/Collider", [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]
    )

    stage = sim_utils.get_current_stage()
    sim_utils.set_prim_visibility(stage.GetPrimAtPath("/World/Shoe/Collider"), False)
    material_scope = UsdGeom.Scope.Define(stage, "/World/Shoe/Materials").GetPrim()
    material_scope.GetReferences().AddReference(str(MODEL_ASSET), "/mat")
    shoe_visual_cfg = sim_utils.UsdFileCfg(
        usd_path=str(MODEL_ASSET),
        visual_material_bindings={"Model": "/World/Shoe/Materials/shoes"},
    )
    shoe_visual_cfg.func("/World/Shoe/Visual", shoe_visual_cfg)

    foot_cfg = sim_utils.CapsuleCfg(
        radius=FOOT_RADIUS,
        height=2.0 * FOOT_HALF_LENGTH,
        visible=False,
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        physics_material=_rigid_material(SHOE_MU, CONTACT_KD),
    )
    half_angle = -0.25 * math.pi
    foot_cfg.func(
        "/World/Shoe/Foot",
        foot_cfg,
        translation=FOOT_CENTER,
        orientation=(math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)),
    )

    ground_cfg = sim_utils.GroundPlaneCfg(
        color=(0.08, 0.08, 0.08),
        size=(2.0, 2.0),
        physics_material=_rigid_material(GROUND_MU, GROUND_CONTACT_KD),
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    mean_segment_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
    cross_section_area = math.pi * cable_radius**2
    second_moment = 0.25 * math.pi * cable_radius**4
    cable_cfg = sim_utils.CableCfg(
        positions=[tuple(point) for point in centerline],
        normals=_parallel_transport_normals(centerline),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CABLE_COLOR, roughness=0.8),
        physics_material=sim_utils.CableMaterialCfg(
            thickness=2.0 * cable_radius,
            density=CABLE_DENSITY,
            stretch_stiffness=STRETCH_STIFFNESS * mean_segment_length / cross_section_area,
            bend_stiffness=BEND_STIFFNESS * mean_segment_length / second_moment,
        ),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )
    cable_cfg.func("/World/Shoelace", cable_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0))
    light_cfg.func("/World/Light", light_cfg)


def run_simulator(sim: sim_utils.SimulationContext, controller: ShoelaceController) -> None:
    """Run the requested force-controlled sequence."""
    controller.restore_authored_state()
    for step in range(args_cli.max_steps):
        if not sim.is_headless_or_exist_active_visualizer():
            break
        sim.step(render=False)
        if args_cli.verify:
            controller.update_verification((step + 1) * FRAME_DT)
        if sim.is_rendering:
            sim.render()
    if args_cli.verify:
        if args_cli.max_steps < 720:
            raise ValueError("--verify requires --max_steps >= 720 to cover the full sequence")
        controller.verify()


def main() -> None:
    """Launch Isaac Lab with Newton VBD and run the shoelace demo."""
    centerline, cable_radius = _load_usd_curve(CURVE_ASSET, "/World/Curve")
    collider_vertices, collider_triangles = _load_usd_mesh(COLLIDER_ASSET, "/World/Collider")
    controller = ShoelaceController(centerline, cable_radius, collider_vertices, collider_triangles)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, VBDSolverCfg):
            raise TypeError("This demo requires the newton_vbd physics preset")
        physics_cfg.num_substeps = SIM_SUBSTEPS
        physics_cfg.collision_decimation = 1
        physics_cfg.default_shape_cfg = NewtonShapeCfg(
            gap=CONTACT_GAP,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=LACE_MU,
        )
        physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(contact_matching="latest")
        physics_cfg.solver_cfg.iterations = SIM_ITERATIONS
        physics_cfg.solver_cfg.rigid_contact_hard = True
        physics_cfg.solver_cfg.rigid_avbd_alpha = 0.0
        physics_cfg.solver_cfg.rigid_body_contact_buffer_size = args_cli.contact_buffer
        physics_cfg.solver_cfg.rigid_contact_history = True

        sim_cfg = sim_utils.SimulationCfg(
            dt=FRAME_DT,
            device=args_cli.device,
            gravity=(0.0, 0.0, GRAVITY),
            physics=physics_cfg,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        design_scene(centerline, cable_radius)
        controller.register()

        bounds_min = collider_vertices.min(axis=0)
        bounds_max = collider_vertices.max(axis=0)
        target = 0.5 * (bounds_min + bounds_max) + np.asarray((0.0, 0.0, 0.01))
        diagonal = float(np.linalg.norm(bounds_max - bounds_min))
        pitch = math.radians(-32.0)
        yaw = math.radians(135.0)
        front = np.asarray((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
        eye = target - 1.05 * diagonal * front
        sim.set_camera_view(eye=tuple(eye), target=tuple(target))

        sim.reset()
        print(
            f"[INFO]: Shoelace ready: {len(centerline) - 1} segments, "
            f"pinned span [{controller.first_pin}, {controller.last_pin}]."
        )
        run_simulator(sim, controller)


if __name__ == "__main__":
    main()
