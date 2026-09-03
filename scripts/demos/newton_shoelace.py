# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tighten and untie an authored shoelace with force-controlled Newton VBD dynamics.

Only the virtual fingertip forces are scripted. The bow loops and tails stay dynamic, and no pose
or velocity target is tracked. Two fixed endpoint segments anchor the eyelet-held span, whose
interior uses one visible static collision mesh.

.. code-block:: bash

    # Run the complete 12 second sequence with the Newton visualizer.
    uv run python scripts/demos/newton_shoelace.py

    # Run 1024 isolated Newton worlds in parallel (a large-memory workload).
    uv run python scripts/demos/newton_shoelace.py --num_envs 1024 --visualizer none

    # Randomize shoe color, cable color, and cable radius once per environment.
    uv run python scripts/demos/newton_shoelace.py --num_envs 16 --randomize --randomization_seed 7

    # Keep the authored shoelace pose by disabling gravity.
    uv run python scripts/demos/newton_shoelace.py --disable_gravity

"""

from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Force-controlled Newton VBD shoelace demo.", conflict_handler="resolve")
parser.add_argument("--max_steps", type=int, default=720, help="Number of 60 Hz simulation steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of isolated Newton worlds.")
parser.add_argument(
    "--randomize",
    action="store_true",
    help="Randomize shoe color, cable color, and cable radius once per environment at startup.",
)
parser.add_argument("--randomization_seed", type=int, default=0, help="Seed for per-environment randomization.")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity for this demo.")
add_launcher_args(parser)
parser.set_defaults(physics="newton_vbd", visualizer=["newton_gl"])
args_cli = parser.parse_args()

import newton
import numpy as np
import torch
import warp as wp
from _newton_force_schedule import BodyForceMove, OpenLoopBodyForceSchedule
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, VisualMaterial, VisualMaterialCfg
from isaaclab.physics import PhysicsCfg, PhysicsEvent
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from newton import ModelBuilder, State


ASSET_DIR = Path(__file__).resolve().parents[2] / ".newton" / "newton" / "examples" / "assets" / "shoelace"
COLLIDER_ASSET = Path(__file__).resolve().parent / "assets" / "shoelace" / "collider_simplified.usd"
CURVE_ASSET = ASSET_DIR / "curve.usd"
MODEL_ASSET = ASSET_DIR / "model.usd"

# Simulation cadence.
FPS = 60
SIM_SUBSTEPS = 10
SIM_ITERATIONS = 20  # VBD solve iterations per substep after contacts are generated.
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS
ENV_SPACING = 0.5

# Lace material. Stiffnesses and damping below are Newton per-joint values.
CABLE_DENSITY = 1150.0
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 10.0
BEND_DAMPING = 2.0
DAHL_MAX_STRAIN = 0.20
DAHL_DECAY = 0.35
CABLE_JOINT_STIFFNESS = (STRETCH_STIFFNESS, STRETCH_STIFFNESS, BEND_STIFFNESS, BEND_STIFFNESS)
CABLE_JOINT_DAMPING = (STRETCH_DAMPING, STRETCH_DAMPING, BEND_DAMPING, BEND_DAMPING)

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
CONTACT_BUFFER = 128  # Per-body VBD contact-list capacity after narrow phase.
CONTACTS_PER_ENV = 512  # Per-world narrow-phase rigid-contact output capacity.
TRIANGLE_PAIRS_PER_ENV = 8192  # Per-world mid-phase mesh triangle-pair capacity.

# Tightening and untying force schedule [s].
SETTLE_END = 0.6
TIGHTEN_END = 1.6
RELEASE_TIME = 2.1
TIGHTEN_LIFT_END = 1.1
# Asset-specific inclusive span held against the shoe by the eyelets.
PINNED_FIRST = 98
PINNED_LAST = 352
PINNED_TUBE_SIDES = 6
CABLE_RADIUS_BUCKETS = 5
CABLE_RADIUS_SCALE_RANGE = (0.85, 1.15)
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
TIGHTEN_LIFT_RAMP = 0.2
UNTIE_FORCE_RAMP = 1.0
TIGHTEN_FORCE = 0.1
TIGHTEN_LIFT_FORCE = 0.01
UNTIE_FORCE_FIRST = 0.1
UNTIE_FORCE_SECOND = 0.1

# Shoe and display parameters. A hidden panel follows the visual tongue below the moving lacing.
TONGUE_UPPER_CENTER = (-0.008, 0.02, 0.10)
TONGUE_UPPER_SIZE = (0.05, 0.055, 0.006)
TONGUE_UPPER_PITCH = math.radians(28.0)
TONGUE_UPPER_Y_ROTATION = math.radians(5.0)
CABLE_COLOR = (112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0)
CAMERA_EYE = (0.2159324, -0.2158787, 0.2631368)
CAMERA_TARGET = (-0.0000817, 0.0001354, 0.0722455)


def _load_usd_curve(path: Path, prim_path: str) -> tuple[np.ndarray, float]:
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


def _parallel_transport_normals(centerline: np.ndarray) -> list[tuple[float, float, float]]:
    """Build one normal per centerline point using Newton's parallel-transport frames."""
    positions = [wp.vec3(*point) for point in centerline]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(positions)
    normals = [wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0)) for quaternion in quaternions]
    normals.append(normals[-1])
    return [(float(normal[0]), float(normal[1]), float(normal[2])) for normal in normals]


def _tube_mesh(centerline: np.ndarray, radius: float, sides: int = 8) -> newton.Mesh:
    """Build an open triangulated tube around a fixed centerline [m]."""
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


def _neighbor_filter_window(points: np.ndarray, radius: float) -> int:
    """Return the number of adjacent rod capsules that overlap by construction."""
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    contact_reach = 2.0 * radius + CONTACT_GAP
    return max(1, int(np.floor(contact_reach / minimum_length + 1.0e-9)) + 1)


def _sample_randomization(num_envs: int, cable_radius: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-environment linear RGB colors and bucketed cable radii [m]."""
    rng = np.random.default_rng(seed)
    shoe_colors = np.asarray(
        [colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.45, 0.80), rng.uniform(0.45, 0.85)) for _ in range(num_envs)],
        dtype=np.float32,
    )
    radius_scales = np.linspace(*CABLE_RADIUS_SCALE_RANGE, CABLE_RADIUS_BUCKETS)
    cable_radii = cable_radius * rng.choice(radius_scales, size=num_envs)
    cable_colors = np.asarray(
        [colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.35, 0.75), rng.uniform(0.25, 0.70)) for _ in range(num_envs)],
        dtype=np.float32,
    )
    return shoe_colors, cable_colors, cable_radii


def _filter_rod_neighbors(builder: ModelBuilder, shapes: list[int], window: int) -> None:
    """Disable collisions between geometrically overlapping neighbor segments."""
    for first, first_shape in enumerate(shapes):
        for second_shape in shapes[first + 1 : first + 1 + window]:
            builder.add_shape_collision_filter_pair(first_shape, second_shape)


def _make_body_static(builder: ModelBuilder, body: int) -> None:
    """Remove all mass and inertia from one Newton body."""
    builder.body_mass[body] = builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33()
    builder.body_inv_inertia[body] = wp.mat33()


def _resize_capsule(builder: ModelBuilder, body: int, shape: int, radius: float) -> None:
    """Resize one cable capsule to ``radius`` [m] and update all mass properties."""
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
    """Set contact stiffness, damping, friction, and gap on one Newton shape."""
    builder.shape_material_ke[shape] = CONTACT_KE
    builder.shape_material_kd[shape] = damping
    builder.shape_material_mu[shape] = friction
    builder.shape_gap[shape] = CONTACT_GAP


class ShoelaceController:
    """Configure replicated Newton shoelaces and apply virtual fingertip forces.

    Args:
        centerline: Authored cable centerline [m], shape [N, 3].
        cable_radius: Authored cable radius [m].
        num_envs: Number of isolated Newton worlds.
        cable_radii: Optional randomized cable radii [m], shape [num_envs].
        cable_colors: Optional randomized linear RGB colors, shape [num_envs, 3].
    """

    def __init__(
        self,
        centerline: np.ndarray,
        cable_radius: float,
        num_envs: int,
        cable_radii: np.ndarray | None = None,
        cable_colors: np.ndarray | None = None,
    ) -> None:
        """Initialize authored geometry and optional per-environment variations."""
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.cable_radii = None if cable_radii is None else np.asarray(cable_radii, dtype=np.float64)
        if self.cable_radii is not None and self.cable_radii.shape != (num_envs,):
            raise ValueError(f"Expected {num_envs} cable radii, got shape {self.cable_radii.shape}")
        self.cable_colors = None if cable_colors is None else np.asarray(cable_colors, dtype=np.float32)
        if self.cable_colors is not None and self.cable_colors.shape != (num_envs, 3):
            raise ValueError(f"Expected cable colors with shape {(num_envs, 3)}, got {self.cable_colors.shape}")
        self.cable_bodies: list[list[int]] = []
        self.cable_joints: list[list[int]] = []
        self._initial_body_q = None
        self._force_schedule: OpenLoopBodyForceSchedule | None = None

    def register(self) -> None:
        """Register model lifecycle and per-substep force callbacks."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_builder")
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY, name="shoelace_model")
        NewtonManager.register_state_force_callback(self._apply_forces)

    def _configure_builder(self, _: object) -> None:
        """Finalize replicated cable geometry, materials, topology, and collision filters."""
        builder = NewtonManager.get_builder()

        def find_chains(labels: list[str], worlds: list[int], suffix: str) -> list[tuple[list[int], list[int]]]:
            """Find ordered left and right cable indices in every Newton world."""
            sides = [
                self._sorted_world_label_indices(labels, worlds, f"/Shoelace{side}/geometry/mesh_{suffix}_")
                for side in ("Left", "Right")
            ]
            return list(zip(*sides, strict=True))

        body_chains = find_chains(builder.body_label, builder.body_world, "edge_body")
        joint_chains = find_chains(builder.joint_label, builder.joint_world, "cable")
        expected_body_counts = (PINNED_FIRST + 1, len(self.centerline) - 1 - PINNED_LAST)
        expected = tuple((count, count - 1) for count in expected_body_counts)
        for world, (bodies, joints) in enumerate(zip(body_chains, joint_chains, strict=True)):
            observed = tuple(
                (len(chain_bodies), len(chain_joints))
                for chain_bodies, chain_joints in zip(bodies, joints, strict=True)
            )
            if observed != expected:
                raise RuntimeError(f"Unexpected cable topology in world {world}: {observed}, expected {expected}")

        self.cable_bodies = [left + right for left, right in body_chains]
        self.cable_joints = [left + right for left, right in joint_chains]

        body_by_label = self._index_unique_labels(builder.body_label)
        shape_by_label = self._index_unique_labels(builder.shape_label)
        pinned_centerline = self.centerline[PINNED_FIRST + 1 : PINNED_LAST + 1]
        pinned_meshes = (
            {radius: _tube_mesh(pinned_centerline, radius, PINNED_TUBE_SIDES) for radius in np.unique(self.cable_radii)}
            if self.cable_radii is not None
            else {}
        )
        linear_cable_colors = (
            np.tile(CABLE_COLOR, (self.num_envs, 1)) if self.cable_colors is None else self.cable_colors
        )
        display_cable_colors = [
            tuple(Gf.ConvertLinearToDisplay(Gf.Vec3f(*map(float, color)))) for color in linear_cable_colors
        ]
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
            _make_body_static(builder, shoe_body)
            for shape in (shoe_collider, tongue_collider):
                builder.shape_flags[shape] = newton.ShapeFlags.COLLIDE_SHAPES
                builder.shape_collision_group[shape] = COLLISION_GROUP
                _set_shape_material(builder, shape, SHOE_MU)
            builder.shape_flags[pinned_shape] = newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES
            builder.shape_color[pinned_shape] = display_cable_colors[world]
            builder.shape_collision_group[pinned_shape] = COLLISION_GROUP
            _set_shape_material(builder, pinned_shape, LACE_MU)
            if self.cable_radii is not None:
                tube = pinned_meshes[self.cable_radii[world]]
                builder.shape_source[pinned_shape] = tube
                extent = np.ptp(np.asarray(tube.vertices, dtype=np.float64), axis=0)
                builder.shape_collision_radius[pinned_shape] = 0.5 * float(np.linalg.norm(extent))

        ground_shapes = [index for index, label in enumerate(builder.shape_label) if label.startswith("/World/Ground/")]
        for shape in ground_shapes:
            _set_shape_material(builder, shape, GROUND_MU, GROUND_CONTACT_KD)

        # Restore exact source-demo frames and capsule mass/inertia while visiting each cable body once.
        exact_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
            [wp.vec3(*point) for point in self.centerline]
        )
        segment_ranges = (range(0, PINNED_FIRST + 1), range(PINNED_LAST, len(self.centerline) - 1))
        for world, (world_bodies, world_joints, shoe) in enumerate(
            zip(body_chains, joint_chains, shoe_parts, strict=True)
        ):
            shoe_collider, tongue_collider, pinned_shape = shoe
            cable_color = display_cable_colors[world]
            randomized_radius = None if self.cable_radii is None else float(self.cable_radii[world])
            neighbor_window = _neighbor_filter_window(
                self.centerline, self.cable_radius if randomized_radius is None else randomized_radius
            )
            chain_shapes: list[list[int]] = []
            for bodies, joints, segment_indices in zip(world_bodies, world_joints, segment_ranges, strict=True):
                shapes: list[int] = []
                for body, segment in zip(bodies, segment_indices, strict=True):
                    quaternion = exact_quaternions[segment]
                    transform = builder.body_q[body]
                    builder.body_q[body] = wp.transform(wp.vec3(transform[0], transform[1], transform[2]), quaternion)
                    shape = int(builder.body_shapes[body][0])
                    shapes.append(shape)
                    builder.shape_color[shape] = cable_color
                    builder.shape_collision_group[shape] = COLLISION_GROUP
                    if randomized_radius is None:
                        radius = float(builder.shape_scale[shape][0])
                        segment_length = 2.0 * float(builder.shape_scale[shape][1])
                        capsule_mass_scale = 1.0 + 4.0 * radius / (3.0 * segment_length)
                        builder.body_mass[body] *= capsule_mass_scale
                        builder.body_inertia[body] *= capsule_mass_scale
                        builder.body_inv_mass[body] = 1.0 / builder.body_mass[body]
                        builder.body_inv_inertia[body] = wp.inverse(builder.body_inertia[body])
                    else:
                        _resize_capsule(builder, body, shape, randomized_radius)

                _filter_rod_neighbors(builder, shapes, neighbor_window)
                chain_shapes.append(shapes)

                dof_start = int(builder.joint_qd_start[joints[0]])
                dof_end = int(builder.joint_qd_start[joints[-1]]) + 4
                if dof_end - dof_start != 4 * len(joints):
                    raise RuntimeError("Cable joint DOFs must be contiguous")
                builder.joint_target_ke[dof_start:dof_end] = CABLE_JOINT_STIFFNESS * len(joints)
                builder.joint_target_kd[dof_start:dof_end] = CABLE_JOINT_DAMPING * len(joints)

            for body, shape in ((world_bodies[0][-1], chain_shapes[0][-1]), (world_bodies[1][0], chain_shapes[1][0])):
                _make_body_static(builder, body)
                for shoe_shape in (shoe_collider, tongue_collider):
                    builder.add_shape_collision_filter_pair(shape, shoe_shape)

            for static_shape in (shoe_collider, tongue_collider):
                builder.add_shape_collision_filter_pair(pinned_shape, static_shape)

            seam_shapes = (*chain_shapes[0][-neighbor_window:], *chain_shapes[1][:neighbor_window])
            for shape in seam_shapes:
                builder.add_shape_collision_filter_pair(shape, pinned_shape)

    def _configure_model(self, _: object) -> None:
        """Configure Dahl hysteresis and the force schedule before solver construction."""
        model = NewtonManager.get_model()
        dahl_eps = np.zeros(model.joint_count, dtype=np.float32)
        dahl_tau = np.zeros(model.joint_count, dtype=np.float32)
        cable_joints = np.asarray([joint for joints in self.cable_joints for joint in joints])
        dahl_eps[cable_joints] = DAHL_MAX_STRAIN
        dahl_tau[cable_joints] = DAHL_DECAY
        model.vbd.dahl_eps_max.assign(dahl_eps)
        model.vbd.dahl_tau.assign(dahl_tau)

        state = NewtonManager.get_state_0()
        self._initial_body_q = wp.clone(state.body_q)
        moves = self._build_force_moves()
        self._force_schedule = OpenLoopBodyForceSchedule(
            moves,
            device=str(model.device),
            sim_dt=SIM_DT,
        )

    def restore_authored_state(self) -> None:
        """Restore the authored knot and clear velocities, forces, contacts, and schedule state."""
        if self._initial_body_q is None or self._force_schedule is None:
            raise RuntimeError("The shoelace model was not initialized")

        model = NewtonManager.get_model()
        rest = model.body_q.numpy()
        for bodies in self.cable_bodies:
            for chain in (bodies[: PINNED_FIRST + 1], bodies[PINNED_FIRST + 1 :]):
                rest[np.asarray(chain), 3:7] = rest[chain[0], 3:7]
        model.body_q.assign(rest)
        state_0 = NewtonManager.get_state_0()
        state_1 = NewtonManager.get_state_1()
        states = (state_0,) if state_1 is None or state_1 is state_0 else (state_0, state_1)
        for state in states:
            state.body_q.assign(self._initial_body_q)
            state.body_qd.zero_()
            state.clear_forces()
        self._force_schedule.reset()
        NewtonManager.invalidate_body_state()
        NewtonManager.refresh_contacts()

    def _apply_forces(self, state: State) -> None:
        """Apply scheduled forces before one Newton solver substep."""
        if self._force_schedule is None:
            raise RuntimeError("The shoelace force schedule was not initialized")
        self._force_schedule.apply(state)

    def _build_force_moves(self) -> list[BodyForceMove]:
        """Build the complete tightening and untying force sequence for every world."""
        midpoints = 0.5 * (self.centerline[:-1] + self.centerline[1:])
        knot = 0.5 * (midpoints[PINNED_FIRST] + midpoints[PINNED_LAST])
        moves: list[BodyForceMove] = []
        for world in range(self.num_envs):
            for lower, upper in (LOOP_GRAB_LEFT, LOOP_GRAB_RIGHT):
                segments = range(lower, min(upper, len(midpoints)))
                outward = midpoints[segments].mean(axis=0) - knot
                outward[2] = 0.0
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=SETTLE_END,
                        end_time=TIGHTEN_LIFT_END,
                        direction=(0.0, 0.0, 1.0),
                        magnitude=TIGHTEN_LIFT_FORCE,
                        ramp_up=TIGHTEN_LIFT_RAMP,
                        ramp_down=TIGHTEN_LIFT_RAMP,
                    )
                )
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=SETTLE_END,
                        end_time=RELEASE_TIME,
                        direction=tuple(outward),
                        magnitude=TIGHTEN_FORCE,
                        ramp_up=TIGHTEN_FORCE_RAMP,
                        ramp_down=RELEASE_TIME - TIGHTEN_END,
                    )
                )
        for world in range(self.num_envs):
            for (lower, upper), (start, pull_end), force in (
                (TAIL_GRAB_FIRST, (UNTIE_FIRST_START, UNTIE_FIRST_END), UNTIE_FORCE_FIRST),
                (TAIL_GRAB_SECOND, (UNTIE_SECOND_START, UNTIE_SECOND_END), UNTIE_FORCE_SECOND),
            ):
                segments = range(lower, min(upper, len(midpoints)))
                side = 1.0 if midpoints[segments].mean(axis=0)[0] >= knot[0] else -1.0
                moves.append(
                    BodyForceMove(
                        body_indices=self._bodies_for_segments(world, segments),
                        start_time=float(start),
                        end_time=float(pull_end + UNTIE_HOLD + UNTIE_RELAX),
                        direction=(side, -0.35, 0.15),
                        magnitude=force,
                        ramp_up=UNTIE_FORCE_RAMP,
                        ramp_down=UNTIE_RELAX,
                    )
                )
            segments = range(*LOOP_GRAB_LEFT)
            outward = midpoints[segments].mean(axis=0) - knot
            horizontal_norm = np.linalg.norm(outward[:2])
            outward[2] = -EXTRACT_DOWNWARD_BIAS * horizontal_norm
            moves.append(
                BodyForceMove(
                    body_indices=self._bodies_for_segments(world, segments),
                    start_time=EXTRACT_START,
                    end_time=EXTRACT_END + UNTIE_HOLD + UNTIE_RELAX,
                    direction=tuple(outward),
                    magnitude=EXTRACT_FORCE,
                    ramp_up=UNTIE_FORCE_RAMP,
                    ramp_down=UNTIE_RELAX,
                )
            )

        return moves

    def _bodies_for_segments(self, world: int, segments: range) -> tuple[int, ...]:
        """Map full-centerline segment indices to simulated bodies in one world."""
        omitted_segments = PINNED_LAST - PINNED_FIRST - 1
        bodies = self.cable_bodies[world]
        return tuple(bodies[segment if segment <= PINNED_FIRST else segment - omitted_segments] for segment in segments)

    def _sorted_world_label_indices(self, labels: list[str], worlds: list[int], local_prefix: str) -> list[list[int]]:
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
        """Index unique Newton labels for constant-time replicated asset lookup."""
        result = {label: index for index, label in enumerate(labels)}
        if len(result) != len(labels):
            raise RuntimeError("Newton labels must be unique after world replication")
        return result

    @staticmethod
    def _label_index(label_indices: dict[str, int], expected: str) -> int:
        """Look up one required Newton label with a contextual error."""
        try:
            return label_indices[expected]
        except KeyError as exc:
            raise RuntimeError(f"Missing Newton label {expected!r}") from exc


def _rigid_material(friction: float, damping: float) -> list[UsdPhysicsRigidBodyMaterialCfg | NewtonMaterialCfg]:
    """Build composed standard-friction and Newton-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0),
        NewtonMaterialCfg(contact_stiffness=CONTACT_KE, contact_damping=damping),
    ]


def _configure_shoe_collider(
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


def create_scene_cfg(centerline: np.ndarray, cable_radius: float, randomize: bool = False) -> InteractiveSceneCfg:
    """Create the replicated three-part shoelace scene.

    Args:
        centerline: Authored cable centerline [m], shape [N, 3].
        cable_radius: Authored cable radius [m].
        randomize: Whether to create per-environment writable shoe materials.

    Returns:
        Scene configuration for isolated Newton worlds.
    """
    mean_segment_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
    cross_section_area = math.pi * cable_radius**2
    second_moment = 0.25 * math.pi * cable_radius**4
    grid_side = math.ceil(math.sqrt(args_cli.num_envs))
    ground_size = max(2.0, ENV_SPACING * (grid_side + 1))
    cable_normals = _parallel_transport_normals(centerline)
    left_centerline = centerline[: PINNED_FIRST + 2]
    right_centerline = centerline[PINNED_LAST:]
    pinned_centerline = centerline[PINNED_FIRST + 1 : PINNED_LAST + 1]

    @sim_utils.clone
    def spawn_pinned(
        prim_path: str,
        _: sim_utils.SpawnerCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: object,
    ) -> Usd.Prim:
        """Spawn the fixed pinned span as one visible collision mesh."""
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
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*CABLE_COLOR)])
        collision_cfg = [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]
        if not sim_utils.apply_collision_properties(str(mesh.GetPath()), collision_cfg, create_if_missing=True):
            raise RuntimeError(f"Failed to enable pinned collision mesh at {mesh.GetPath()}")
        return root

    pinned_cfg = sim_utils.SpawnerCfg(func=spawn_pinned)

    def cable_cfg(points: np.ndarray, normals: list[tuple[float, float, float]]) -> sim_utils.CableCfg:
        """Create one dynamic cable side from centerline points [m] and frame normals."""
        return sim_utils.CableCfg(
            positions=[tuple(point) for point in points],
            normals=normals,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CABLE_COLOR, roughness=0.8),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=2.0 * cable_radius,
                density=CABLE_DENSITY,
                stretch_stiffness=STRETCH_STIFFNESS * mean_segment_length / cross_section_area,
                bend_stiffness=BEND_STIFFNESS * mean_segment_length / second_moment,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        )

    def env_asset(prim_suffix: str, spawn: sim_utils.SpawnerCfg) -> AssetBaseCfg:
        """Create an environment-scoped static asset configuration."""
        return AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}", spawn=spawn)

    shoe_material_cfg = (
        VisualMaterialCfg(
            prim_path="{ENV_REGEX_NS}/Shoe/Visual/material",
            spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.75),
            channels=("color",),
        )
        if randomize
        else None
    )
    shoe_material_path = "./material" if randomize else "/World/ShoeMaterials/shoes"

    @configclass
    class ShoelaceSceneCfg(InteractiveSceneCfg):
        """Replicated shoe and cable assets with shared ground and lighting."""

        shoe_material = shoe_material_cfg
        shoe = env_asset(
            "Shoe",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(COLLIDER_ASSET),
                rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            ),
        )
        shoe_collider = env_asset("Shoe/Collider", sim_utils.SpawnerCfg(func=_configure_shoe_collider))
        shoe_visual = env_asset(
            "Shoe/Visual",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(MODEL_ASSET),
                visual_material_bindings={"Model": shoe_material_path},
            ),
        )
        tongue_upper = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Shoe/TongueUpper",
            spawn=sim_utils.CuboidCfg(
                size=TONGUE_UPPER_SIZE,
                visible=False,
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
                physics_material=_rigid_material(SHOE_MU, CONTACT_KD),
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
        shoelace_left = env_asset("ShoelaceLeft", cable_cfg(left_centerline, cable_normals[: PINNED_FIRST + 2]))
        shoelace_pinned = env_asset("ShoelacePinned", pinned_cfg)
        shoelace_right = env_asset("ShoelaceRight", cable_cfg(right_centerline, cable_normals[PINNED_LAST:]))
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                color=(0.08, 0.08, 0.08),
                size=(ground_size, ground_size),
                physics_material=_rigid_material(GROUND_MU, GROUND_CONTACT_KD),
            ),
            collision_group=-1,
        )
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0)),
        )

    return ShoelaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=ENV_SPACING, replicate_physics=True)


def run_simulator(sim: sim_utils.SimulationContext, controller: ShoelaceController) -> None:
    """Restore the authored knot and run the requested force-controlled sequence.

    Args:
        sim: Initialized simulation context.
        controller: Configured shoelace controller.
    """
    controller.restore_authored_state()
    for _ in range(args_cli.max_steps):
        if not sim.is_headless_or_exist_active_visualizer():
            break
        sim.step(render=False)
        if sim.is_rendering:
            sim.render()


def main() -> None:
    """Launch Isaac Lab with Newton VBD and run the shoelace demo."""
    centerline, cable_radius = _load_usd_curve(CURVE_ASSET, "/World/Curve")
    shoe_colors = cable_colors = cable_radii = None
    if args_cli.randomize:
        shoe_colors, cable_colors, cable_radii = _sample_randomization(
            args_cli.num_envs, cable_radius, args_cli.randomization_seed
        )
    controller = ShoelaceController(centerline, cable_radius, args_cli.num_envs, cable_radii, cable_colors)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, VBDSolverCfg):
            raise TypeError("This demo requires the newton_vbd physics preset")
        physics_cfg.num_substeps = SIM_SUBSTEPS
        physics_cfg.collision_decimation = 2
        physics_cfg.default_shape_cfg = NewtonShapeCfg(
            gap=CONTACT_GAP,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=LACE_MU,
        )
        # The default explicit broad phase consumes the precomputed model.shape_contact_pairs.
        physics_cfg.collision_cfg = NewtonCollisionPipelineCfg(
            rigid_contact_max=CONTACTS_PER_ENV * args_cli.num_envs,
            max_triangle_pairs=max(1_000_000, TRIANGLE_PAIRS_PER_ENV * args_cli.num_envs),
        )
        physics_cfg.solver_cfg.iterations = SIM_ITERATIONS
        physics_cfg.solver_cfg.rigid_contact_hard = True
        physics_cfg.solver_cfg.rigid_avbd_alpha = 0.0
        physics_cfg.solver_cfg.rigid_body_contact_buffer_size = CONTACT_BUFFER

        sim_cfg = sim_utils.SimulationCfg(
            dt=FRAME_DT,
            device=args_cli.device,
            gravity=(0.0, 0.0, 0.0 if args_cli.disable_gravity else GRAVITY),
            physics=physics_cfg,
            visualizer_cfgs=[
                NewtonGLVisualizerCfg(
                    update_frequency=8,
                    enable_shadows=False,
                )
            ],
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        controller.register()

        stage = sim_utils.get_current_stage()
        material_scope = UsdGeom.Scope.Define(stage, "/World/ShoeMaterials").GetPrim()
        material_scope.GetReferences().AddReference(str(MODEL_ASSET), "/mat")
        scene = InteractiveScene(create_scene_cfg(centerline, cable_radius, args_cli.randomize))
        camera_eye, camera_target = CAMERA_EYE, CAMERA_TARGET
        if args_cli.num_envs > 1:
            extent = ENV_SPACING * math.ceil(math.sqrt(args_cli.num_envs))
            camera_eye = (0.8 * extent, -0.8 * extent, 0.7 * extent)
            camera_target = (0.0, 0.0, CAMERA_TARGET[2])
        sim.set_camera_view(eye=camera_eye, target=camera_target)

        sim.reset()
        if shoe_colors is not None:
            material = scene["shoe_material"]
            if not isinstance(material, VisualMaterial):
                raise TypeError("Randomized shoe material did not initialize as a VisualMaterial")
            VisualMaterial.write_channels(
                [material],
                {"color": torch.from_numpy(shoe_colors).unsqueeze(0)},
            )
            print(
                f"[INFO]: Randomized shoe/cable colors in {args_cli.num_envs} environments with seed "
                f"{args_cli.randomization_seed}; cable radius "
                f"{1.0e3 * cable_radii.min():.3f}-{1.0e3 * cable_radii.max():.3f} mm."
            )
        print(
            f"[INFO]: Shoelace ready: {args_cli.num_envs} worlds, "
            f"{len(centerline) - 1} visual segments/world, "
            f"{len(controller.cable_bodies[0])} VBD bodies/world, "
            f"anchors [{PINNED_FIRST}, {PINNED_LAST}]."
        )
        run_simulator(sim, controller)


if __name__ == "__main__":
    main()
