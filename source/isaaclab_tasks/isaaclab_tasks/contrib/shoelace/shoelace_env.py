# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime setup for the dual-Franka Newton shoelace RL environment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent

from . import shoelace_physics as physics

if TYPE_CHECKING:
    from newton import ModelBuilder

    from .shoelace_env_cfg import ShoelaceEnvCfg


TAIL_BEND_STIFFNESS = 100.0
TAIL_BEND_DAMPING = 2.0
TAIL_STIFF_CORE_LENGTH = 0.050
TAIL_STIFF_TRANSITION_LENGTH = 0.012


@wp.kernel
def _reset_finger_tail_signed_distance(
    distance_cap: float,
    signed_distance: wp.array2d(dtype=wp.float32),
):
    """Reset every finger-tail distance to the no-contact sentinel [m]."""
    env_id, finger_slot = wp.tid()
    signed_distance[env_id, finger_slot] = distance_cap


@wp.kernel
def _aggregate_finger_tail_signed_distance(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    finger_slot_by_shape: wp.array(dtype=wp.int32),
    finger_env_by_shape: wp.array(dtype=wp.int32),
    tail_arm_by_shape: wp.array(dtype=wp.int32),
    signed_distance: wp.array2d(dtype=wp.float32),
):
    """Aggregate minimum signed surface separation for the four finger-tail pairs [m]."""
    contact_id = wp.tid()
    if contact_id >= contact_count[0]:
        return

    shape0 = contact_shape0[contact_id]
    shape1 = contact_shape1[contact_id]
    if shape0 < 0 or shape1 < 0:
        return

    finger_shape = shape0
    finger_slot = finger_slot_by_shape[shape0]
    tail_arm = tail_arm_by_shape[shape1]
    if finger_slot < 0 or tail_arm != finger_slot // 2:
        finger_shape = shape1
        finger_slot = finger_slot_by_shape[shape1]
        tail_arm = tail_arm_by_shape[shape0]
    if finger_slot < 0 or tail_arm != finger_slot // 2:
        return

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0_w = wp.transform_point(transform0, contact_point0[contact_id])
    point1_w = wp.transform_point(transform1, contact_point1[contact_id])
    distance = wp.dot(contact_normal[contact_id], point1_w - point0_w)
    distance -= contact_margin0[contact_id] + contact_margin1[contact_id]
    wp.atomic_min(signed_distance, finger_env_by_shape[finger_shape], finger_slot, distance)


def _load_settled_segment_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load validated gravity-settled cable segment poses [m, quaternion x-y-z-w]."""
    names = ("shoelace_left", "shoelace_right")
    with np.load(path, allow_pickle=False) as state:
        if tuple(state.files) != names:
            raise ValueError(f"Expected settled shoelace arrays {names}, got {tuple(state.files)}")
        poses = tuple(np.asarray(state[name], dtype=np.float32) for name in names)
    expected_shapes = (
        (physics.PINNED_FIRST + 1, 7),
        (physics.SHOELACE_SEGMENT_COUNT - physics.PINNED_LAST, 7),
    )
    for name, pose, expected_shape in zip(names, poses, expected_shapes, strict=True):
        if pose.shape != expected_shape or not np.isfinite(pose).all():
            raise ValueError(f"Invalid settled {name} poses: expected finite {expected_shape}, got {pose.shape}")
        if not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1.0, atol=2.0e-5):
            raise ValueError(f"Settled {name} poses contain non-unit quaternions")
    return poses


def _tail_joint_blend_weights(
    joint_count: int,
    segment_length: float,
    free_end_at_start: bool,
) -> np.ndarray:
    """Return smooth weights for the gravity-resistant distal tail material."""
    distances = segment_length * np.arange(1, joint_count + 1, dtype=np.float64)
    if TAIL_STIFF_TRANSITION_LENGTH > 0.0:
        weights = np.clip(
            (TAIL_STIFF_CORE_LENGTH + TAIL_STIFF_TRANSITION_LENGTH - distances) / TAIL_STIFF_TRANSITION_LENGTH,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)
    else:
        weights = (distances <= TAIL_STIFF_CORE_LENGTH).astype(np.float64)
    return weights if free_end_at_start else weights[::-1].copy()


class _ShoelacePhysics:
    """Configure the Newton cable model before coupled solver construction."""

    def __init__(
        self,
        centerline: np.ndarray,
        cable_radius: float,
        num_envs: int,
        joint_stiffness_scale: float,
        finger_mu: float,
        lace_mu: float,
        shoe_mu: float,
        cable_inertia_regularization: float = physics.CABLE_INERTIA_REGULARIZATION,
    ) -> None:
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.joint_stiffness_scale = joint_stiffness_scale
        self.finger_mu = finger_mu
        self.lace_mu = lace_mu
        self.shoe_mu = shoe_mu
        self.cable_inertia_regularization = cable_inertia_regularization
        self.cable_joints: list[list[int]] = []
        self.finger_tail_signed_distance: torch.Tensor | None = None
        self._finger_slot_by_shape: wp.array | None = None
        self._finger_env_by_shape: wp.array | None = None
        self._tail_arm_by_shape: wp.array | None = None
        self._finger_slot_by_shape_host: np.ndarray | None = None
        self._finger_env_by_shape_host: np.ndarray | None = None
        self._tail_arm_by_shape_host: np.ndarray | None = None
        self._finger_tail_signed_distance: wp.array | None = None

    def register(self) -> None:
        """Register the Newton model lifecycle callbacks."""
        NewtonManager.register_callback(
            self._configure_builder,
            PhysicsEvent.MODEL_INIT,
            name="shoelace_task_builder",
        )
        NewtonManager.register_callback(
            self._configure_model,
            PhysicsEvent.PHYSICS_READY,
            name="shoelace_task_contact_observer",
        )

    def _configure_builder(self, _: object) -> None:
        """Finalize cable materials, topology, anchors, and collision filters."""
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton builder is unavailable during MODEL_INIT")
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp.values = {} if gravcomp.values is None else gravcomp.values
        for body, label in enumerate(builder.body_label):
            if "/RobotLeft/" in label or "/RobotRight/" in label:
                gravcomp.values[body] = 1.0
        build = physics.configure_shoelace_builder(
            builder,
            self.centerline,
            self.cable_radius,
            self.num_envs,
            lace_mu=self.lace_mu,
            shoe_mu=self.shoe_mu,
            cable_inertia_regularization=self.cable_inertia_regularization,
        )
        self.cable_joints = [left + right for left, right in build.joint_chains]
        self._configure_contact_shape_maps(builder, build)
        for shape, label in enumerate(builder.shape_label):
            if "panda_leftfinger" in label or "panda_rightfinger" in label:
                builder.shape_material_mu[shape] = self.finger_mu

        for world_joints in build.joint_chains:
            for joints, free_end_at_start in zip(world_joints, (True, False), strict=True):
                weights = _tail_joint_blend_weights(len(joints), build.mean_segment_length, free_end_at_start)
                stiffness: list[float] = []
                damping: list[float] = []
                for weight in weights:
                    bend_stiffness = physics.BEND_STIFFNESS + weight * (TAIL_BEND_STIFFNESS - physics.BEND_STIFFNESS)
                    bend_damping = physics.BEND_DAMPING + weight * (TAIL_BEND_DAMPING - physics.BEND_DAMPING)
                    stiffness.extend((physics.STRETCH_STIFFNESS,) * 2 + (bend_stiffness,) * 2)
                    damping.extend((physics.STRETCH_DAMPING,) * 2 + (bend_damping,) * 2)
                dof_slice = physics.cable_joint_dof_slice(builder, joints)
                builder.joint_target_ke[dof_slice] = tuple(value * self.joint_stiffness_scale for value in stiffness)
                builder.joint_target_kd[dof_slice] = tuple(value * self.joint_stiffness_scale for value in damping)
        physics.configure_dahl(builder, self.cable_joints)

    def reset_contact_observation(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset cached finger-tail signed distances [m] for selected environments."""
        signed_distance = self.finger_tail_signed_distance
        if signed_distance is None:
            return
        if env_ids is None:
            signed_distance.fill_(physics.CONTACT_DISTANCE_CAP)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, dtype=torch.long, device=signed_distance.device)
            signed_distance[env_ids] = physics.CONTACT_DISTANCE_CAP

    def _configure_contact_shape_maps(self, builder: ModelBuilder, build: physics.ShoelaceBuild) -> None:
        """Map finger and free-tail collision shapes to four observation slots."""
        shape_count = len(builder.shape_label)
        finger_slot_by_shape = np.full(shape_count, -1, dtype=np.int32)
        finger_env_by_shape = np.full(shape_count, -1, dtype=np.int32)
        tail_arm_by_shape = np.full(shape_count, -1, dtype=np.int32)
        robot_names = ("RobotLeft", "RobotRight")
        finger_names = ("panda_leftfinger", "panda_rightfinger")

        for env_id, cable_bodies in enumerate(build.body_chains):
            left_cable_bodies, right_cable_bodies = cable_bodies
            tail_bodies_by_arm = (right_cable_bodies[-3:], left_cable_bodies[:3])
            for arm, tail_bodies in enumerate(tail_bodies_by_arm):
                for body in tail_bodies:
                    for shape in builder.body_shapes[body]:
                        tail_arm_by_shape[int(shape)] = arm

            for arm, robot_name in enumerate(robot_names):
                for finger, finger_name in enumerate(finger_names):
                    bodies = [
                        body
                        for body, label in enumerate(builder.body_label)
                        if int(builder.body_world[body]) == env_id
                        and f"/{robot_name}/" in str(label)
                        and str(label).rsplit("/", 1)[-1] == finger_name
                    ]
                    if len(bodies) != 1:
                        raise RuntimeError(
                            f"Expected one {robot_name}/{finger_name} body in Newton world {env_id}, got {bodies}"
                        )
                    body = bodies[0]
                    shapes = builder.body_shapes[body]
                    if len(shapes) == 0:
                        raise RuntimeError(f"Newton finger body {builder.body_label[body]!r} has no collision shapes")
                    slot = 2 * arm + finger
                    for shape in shapes:
                        shape = int(shape)
                        finger_slot_by_shape[shape] = slot
                        finger_env_by_shape[shape] = env_id

        self._finger_slot_by_shape_host = finger_slot_by_shape
        self._finger_env_by_shape_host = finger_env_by_shape
        self._tail_arm_by_shape_host = tail_arm_by_shape

    def _configure_model(self, _: object) -> None:
        """Allocate contact-observation buffers after the Newton model is finalized."""
        finger_slot_by_shape = self._finger_slot_by_shape_host
        finger_env_by_shape = self._finger_env_by_shape_host
        tail_arm_by_shape = self._tail_arm_by_shape_host
        if finger_slot_by_shape is None or finger_env_by_shape is None or tail_arm_by_shape is None:
            raise RuntimeError("Shoelace contact shape maps were not configured during MODEL_INIT")
        device = NewtonManager.get_model().device
        self._finger_slot_by_shape = wp.array(finger_slot_by_shape, dtype=wp.int32, device=device)
        self._finger_env_by_shape = wp.array(finger_env_by_shape, dtype=wp.int32, device=device)
        self._tail_arm_by_shape = wp.array(tail_arm_by_shape, dtype=wp.int32, device=device)
        self._finger_tail_signed_distance = wp.full(
            (self.num_envs, 4),
            physics.CONTACT_DISTANCE_CAP,
            dtype=wp.float32,
            device=device,
        )
        self.finger_tail_signed_distance = wp.to_torch(self._finger_tail_signed_distance)
        NewtonManager.unregister_post_step_callback(self._update_contact_observation)
        NewtonManager.register_post_step_callback(self._update_contact_observation)

    def _update_contact_observation(self) -> None:
        """Update current finger-tail signed distances from Newton collision candidates."""
        contacts = NewtonManager.get_contacts()
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        if contacts is None or self._finger_tail_signed_distance is None:
            raise RuntimeError("Shoelace contact observation requires an initialized Newton collision buffer")
        if self._finger_slot_by_shape is None or self._finger_env_by_shape is None or self._tail_arm_by_shape is None:
            raise RuntimeError("Shoelace contact observation shape maps are unavailable")
        wp.launch(
            _reset_finger_tail_signed_distance,
            dim=(self.num_envs, 4),
            inputs=[physics.CONTACT_DISTANCE_CAP],
            outputs=[self._finger_tail_signed_distance],
            device=model.device,
        )
        wp.launch(
            _aggregate_finger_tail_signed_distance,
            dim=contacts.rigid_contact_max,
            inputs=[
                state.body_q,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self._finger_slot_by_shape,
                self._finger_env_by_shape,
                self._tail_arm_by_shape,
            ],
            outputs=[self._finger_tail_signed_distance],
            device=model.device,
        )


class ShoelaceEnv(ManagerBasedRLEnv):
    """Expose the dual-Franka shoelace scene through the manager-based RL interface."""

    cfg: ShoelaceEnvCfg

    def __init__(self, cfg: ShoelaceEnvCfg, render_mode: str | None = None, **kwargs: object) -> None:
        self._centerline, self._cable_radius, authored_mean_segment_length = physics.load_shoelace()
        settled_state_asset = physics.ASSET_DIR / "settled_tail_clear_segment_poses.npz"
        self._settled_segment_poses = _load_settled_segment_poses(settled_state_asset)
        mean_segment_length = float(np.linalg.norm(np.diff(self._centerline, axis=0), axis=1).mean())
        self._configure_runtime_cfg(cfg, authored_mean_segment_length)
        self._physics = _ShoelacePhysics(
            self._centerline,
            self._cable_radius,
            cfg.scene.num_envs,
            authored_mean_segment_length / mean_segment_length,
            finger_mu=cfg.finger_mu,
            lace_mu=cfg.lace_mu,
            shoe_mu=cfg.shoe_mu,
            cable_inertia_regularization=cfg.cable_inertia_regularization,
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._install_settled_default_state()

    def _configure_runtime_cfg(self, cfg: ShoelaceEnvCfg, authored_mean_segment_length: float) -> None:
        """Fill asset geometry and environment-count-dependent capacities."""
        cfg.scene.shoelace_pinned_visual.spawn = physics.pinned_shoelace_spawner(self._centerline, self._cable_radius)
        cable_normals = physics.parallel_transport_normals(self._centerline)
        cable_centerlines = (
            self._centerline[: physics.PINNED_FIRST + 2],
            self._centerline[physics.PINNED_LAST :],
        )
        cable_normal_chains = (
            cable_normals[: physics.PINNED_FIRST + 2],
            cable_normals[physics.PINNED_LAST :],
        )
        for cable_cfg, centerline, normals in zip(
            (cfg.scene.shoelace_left.spawn, cfg.scene.shoelace_right.spawn),
            cable_centerlines,
            cable_normal_chains,
            strict=True,
        ):
            physics.configure_cable_spawn(
                cable_cfg,
                centerline,
                normals,
                self._cable_radius,
                authored_mean_segment_length,
            )

        ground_size = physics.ground_size(cfg.scene.num_envs, cfg.scene.env_spacing)
        cfg.scene.ground.spawn.size = (ground_size, ground_size)
        physics_cfg = cfg.sim.physics
        if not isinstance(physics_cfg, NewtonCfg):
            raise TypeError("The dual-Franka shoelace task requires Newton physics")
        physics_cfg.collision_cfg.rigid_contact_max = physics.CONTACTS_PER_ENV * cfg.scene.num_envs
        physics_cfg.collision_cfg.max_triangle_pairs = max(
            physics.MIN_TRIANGLE_PAIRS,
            physics.TRIANGLE_PAIRS_PER_ENV * cfg.scene.num_envs,
        )

    def _install_settled_default_state(self) -> None:
        """Install the offline gravity-settled cable poses as reset defaults."""
        for cable_name, local_pose in zip(
            ("shoelace_left", "shoelace_right"),
            self._settled_segment_poses,
            strict=True,
        ):
            cable = self.scene[cable_name]
            default_pose = cable.data.default_segment_pose_w.torch
            segment_pose = default_pose.new_tensor(local_pose).unsqueeze(0).expand_as(default_pose).clone()
            segment_pose[..., :3] += self.scene.env_origins.unsqueeze(1)
            segment_velocity = torch.zeros_like(cable.data.default_segment_velocity_w.torch)
            default_pose.copy_(segment_pose)
            cable.data.default_segment_velocity_w.torch.copy_(segment_velocity)
            cable.write_segment_pose_to_sim_index(segment_pose=segment_pose)
            cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity)

    def _init_sim(self) -> None:
        """Register the shoe material and Newton callback before model creation."""
        physics.install_shoe_material_reference()
        self._physics.register()
        super()._init_sim()

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset task state and invalidate cached contact observations."""
        super()._reset_idx(env_ids)
        self._physics.reset_contact_observation(env_ids)
