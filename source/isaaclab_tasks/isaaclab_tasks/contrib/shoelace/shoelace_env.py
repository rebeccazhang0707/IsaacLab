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
from isaaclab_newton.physics import NewtonCfg, NewtonManager, VBDSolverCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import spawn_from_usd

from isaaclab_contrib.coupling import CouplerProxyCfg

from .mdp.constants import PINNED_FIRST, PINNED_LAST

if TYPE_CHECKING:
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


class _ShoelacePhysics:
    """Configure shoelace parity before coupled MJWarp/VBD solver construction."""

    def __init__(self, centerline: np.ndarray, cable_radius: float, num_envs: int):
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.cable_bodies: list[list[int]] = []
        self.cable_joints: list[list[int]] = []

    def register(self) -> None:
        """Register model lifecycle callbacks before the first simulation reset."""
        NewtonManager.register_callback(self._configure_builder, PhysicsEvent.MODEL_INIT, name="shoelace_task_builder")
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY, name="shoelace_task_model")

    def set_straight_rest_state(self) -> None:
        """Use a straight untwisted bend reference while retaining the authored state."""
        model = NewtonManager.get_model()
        rest = model.body_q.numpy()
        for bodies in self.cable_bodies:
            rest[np.asarray(bodies), 3:7] = rest[bodies[0], 3:7]
        model.body_q.assign(rest)

    def _configure_builder(self, _: object) -> None:
        """Apply cable properties that must exist before model finalization."""
        builder = NewtonManager.get_builder()
        self.cable_bodies = self._sorted_world_label_indices(
            builder.body_label,
            builder.body_world,
            "/Shoelace/geometry/mesh_edge_body_",
        )
        self.cable_joints = self._sorted_world_label_indices(
            builder.joint_label,
            builder.joint_world,
            "/Shoelace/geometry/mesh_cable_",
        )
        expected_segments = len(self.centerline) - 1
        for world, (bodies, joints) in enumerate(zip(self.cable_bodies, self.cable_joints, strict=True)):
            if len(bodies) != expected_segments or len(joints) != expected_segments - 1:
                raise RuntimeError(
                    f"Unexpected cable topology in world {world}: {len(bodies)} bodies and {len(joints)} joints"
                )

        body_by_label = self._index_unique_labels(builder.body_label)
        shape_by_label = self._index_unique_labels(builder.shape_label)
        shoe_colliders: list[int] = []
        tongue_colliders: list[int] = []
        for world in range(self.num_envs):
            root = f"/World/envs/env_{world}/Shoe"
            shoe_body = self._label_index(body_by_label, root)
            shoe_collider = self._label_index(shape_by_label, f"{root}/Collider")
            tongue_collider = self._label_index(shape_by_label, f"{root}/TongueUpper/geometry/mesh")
            shoe_colliders.append(shoe_collider)
            tongue_colliders.append(tongue_collider)
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
        for bodies, joints, shoe_collider, tongue_collider in zip(
            self.cable_bodies, self.cable_joints, shoe_colliders, tongue_colliders, strict=True
        ):
            shapes: list[int] = []
            for body, quaternion in zip(bodies, exact_quaternions, strict=True):
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

            for body, shape in zip(
                bodies[PINNED_FIRST : PINNED_LAST + 1],
                shapes[PINNED_FIRST : PINNED_LAST + 1],
                strict=True,
            ):
                builder.body_mass[body] = 0.0
                builder.body_inv_mass[body] = 0.0
                builder.body_inertia[body] = wp.mat33()
                builder.body_inv_inertia[body] = wp.mat33()
                for shoe_shape in (shoe_collider, tongue_collider):
                    builder.add_shape_collision_filter_pair(shape, shoe_shape)

            _filter_rod_neighbors(builder, shapes, neighbor_window)

            dof_start = int(builder.joint_qd_start[joints[0]])
            dof_end = int(builder.joint_qd_start[joints[-1]]) + 4
            if dof_end - dof_start != 4 * len(joints):
                raise RuntimeError("Cable joint DOFs must be contiguous")
            builder.joint_target_ke[dof_start:dof_end] = (
                STRETCH_STIFFNESS,
                STRETCH_STIFFNESS,
                BEND_STIFFNESS,
                BEND_STIFFNESS,
            ) * len(joints)
            builder.joint_target_kd[dof_start:dof_end] = (
                STRETCH_DAMPING,
                STRETCH_DAMPING,
                BEND_DAMPING,
                BEND_DAMPING,
            ) * len(joints)

    def _configure_model(self, _: object) -> None:
        """Configure cable Dahl hysteresis before coupled solver construction."""
        model = NewtonManager.get_model()
        dahl_eps = np.zeros(model.joint_count, dtype=np.float32)
        dahl_tau = np.zeros(model.joint_count, dtype=np.float32)
        cable_joints = np.asarray([joint for joints in self.cable_joints for joint in joints])
        dahl_eps[cable_joints] = DAHL_MAX_STRAIN
        dahl_tau[cable_joints] = DAHL_DECAY
        model.vbd.dahl_eps_max.assign(dahl_eps)
        model.vbd.dahl_tau.assign(dahl_tau)

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
        collider_asset = asset_dir / "collider.usd"
        curve_asset = asset_dir / "curve.usd"
        self._model_asset = asset_dir / "model.usd"
        self._centerline, self._cable_radius = _load_usd_curve(curve_asset, "/World/Curve")
        self._configure_runtime_cfg(cfg, collider_asset)
        self._physics = _ShoelacePhysics(self._centerline, self._cable_radius, cfg.scene.num_envs)

        super().__init__(cfg, render_mode, **kwargs)
        self._physics.set_straight_rest_state()

    def _configure_runtime_cfg(self, cfg: ShoelaceEnvCfg, collider_asset: Path) -> None:
        """Fill asset- and world-count-dependent configuration before validation."""
        expected_segments = len(self._centerline) - 1
        if expected_segments != 450:
            raise ValueError(f"Expected 450 shoelace segments, got {expected_segments}")

        cfg.scene.shoe.spawn.usd_path = str(collider_asset)
        cfg.scene.shoe.spawn.func = _spawn_collision_mesh_usd
        cfg.scene.shoe.spawn.collision_props = None
        cfg.scene.shoe_visual.spawn.usd_path = str(self._model_asset)
        cable_cfg = cfg.scene.shoelace.spawn
        cable_cfg.positions = [tuple(point) for point in self._centerline]
        cable_cfg.normals = _parallel_transport_normals(self._centerline)
        mean_segment_length = float(np.linalg.norm(np.diff(self._centerline, axis=0), axis=1).mean())
        cross_section_area = math.pi * self._cable_radius**2
        second_moment = 0.25 * math.pi * self._cable_radius**4
        cable_cfg.physics_material.thickness = 2.0 * self._cable_radius
        cable_cfg.physics_material.density = CABLE_DENSITY
        cable_cfg.physics_material.stretch_stiffness = STRETCH_STIFFNESS * mean_segment_length / cross_section_area
        cable_cfg.physics_material.bend_stiffness = BEND_STIFFNESS * mean_segment_length / second_moment

        grid_side = math.ceil(math.sqrt(cfg.scene.num_envs))
        ground_size = max(2.0, cfg.scene.env_spacing * (grid_side + 1))
        cfg.scene.ground.spawn.size = (ground_size, ground_size)

        physics_cfg = cfg.sim.physics
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, CouplerProxyCfg):
            raise TypeError("The dual-Franka shoelace task requires Newton proxy coupling")
        vbd_entries = [entry for entry in physics_cfg.solver_cfg.entries if entry.name == "shoelace"]
        if len(vbd_entries) != 1 or not isinstance(vbd_entries[0].solver_cfg, VBDSolverCfg):
            raise TypeError("The shoelace coupler entry requires one VBD solver")
        vbd_cfg = vbd_entries[0].solver_cfg
        triangle_pair_capacity = max(1_000_000, cfg.triangle_pairs_per_env * cfg.scene.num_envs)
        # VBD history allocates persistent warm-start storage on its first step, which the proxy coupler cannot do
        # safely inside CUDA graph capture.
        contact_history = (
            not physics_cfg.use_cuda_graph and triangle_pair_capacity <= cfg.deterministic_triangle_pair_limit
        )
        physics_cfg.collision_cfg.rigid_contact_max = cfg.contacts_per_env * cfg.scene.num_envs
        physics_cfg.collision_cfg.max_triangle_pairs = triangle_pair_capacity
        physics_cfg.collision_cfg.contact_matching = "latest" if contact_history else "disabled"
        vbd_cfg.rigid_contact_history = contact_history

    def _init_sim(self) -> None:
        """Register Newton lifecycle callbacks before the scene's first reset."""
        stage = sim_utils.get_current_stage()
        material_scope = UsdGeom.Scope.Define(stage, "/World/ShoeMaterials").GetPrim()
        material_scope.GetReferences().AddReference(str(self._model_asset), "/mat")
        self._physics.register()
        super()._init_sim()
