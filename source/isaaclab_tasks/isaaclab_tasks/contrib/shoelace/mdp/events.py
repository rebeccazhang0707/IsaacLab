# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset randomization for the dual-Franka shoelace task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ModelFlags
from newton.geometry import compute_inertia_shape
from newton.utils import create_parallel_transport_cable_quaternions

from pxr import Usd, UsdGeom

from isaaclab.assets import Articulation, CableObject, RigidObject
from isaaclab.envs.mdp.events import reset_joints_by_offset
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, sample_uniform

from .. import shoelace_constants as physics
from ..asset_authoring import _tail_joint_blend_weights, load_shoelace

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def configure_shoelace_physics(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Configure non-Dahl cable physics once, before installing settled poses.

    Uses the existing coupled solver's model-change notification to refresh its
    compact views and VBD material caches. Episode resets only restore state.

    Args:
        env: Environment containing the imported shoe and both cables.
        env_ids: Unused; startup configures every environment.
    """
    regularization = env.cfg.cable_inertia_regularization
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Cable inertia regularization must be finite and nonnegative")
    centerline, _, reference_length = load_shoelace()
    segment_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
    stiffness_scale = reference_length / segment_length
    orientations = torch.as_tensor(
        np.asarray(create_parallel_transport_cable_quaternions([wp.vec3(*point) for point in centerline])),
        dtype=torch.float32,
        device=env.device,
    )
    model = NewtonManager.get_model()
    mass = wp.to_torch(model.body_mass)
    inverse_mass = wp.to_torch(model.body_inv_mass)
    inertia = wp.to_torch(model.body_inertia)
    inverse_inertia = wp.to_torch(model.body_inv_inertia)
    shape_bodies = model.shape_body.numpy()
    shape_scales = model.shape_scale.numpy()
    shape_types = model.shape_type.numpy()
    for name, free_end_at_start in (("shoelace_left", True), ("shoelace_right", False)):
        cable: CableObject = env.scene[name]
        view = cable.root_view
        parents = torch.as_tensor(view.get_attribute("joint_parent", model).numpy(), device=env.device).long()[:, 0]
        children = torch.as_tensor(view.get_attribute("joint_child", model).numpy(), device=env.device).long()[:, 0]
        bodies = torch.cat((parents[:, :1], children), dim=1)
        # Newton 1.6 omits curve contact material/gap import. Configure the task's
        # generated shapes here; SHAPE_PROPERTIES also refreshes compact solver views.
        shapes = torch.isin(wp.to_torch(model.shape_body).long(), bodies.flatten())
        for field, value in (
            (model.shape_material_mu, env.cfg.lace_mu),
            (model.shape_material_ke, physics.CONTACT_KE),
            (model.shape_material_kd, physics.CONTACT_KD),
            (model.shape_gap, physics.CONTACT_GAP),
        ):
            wp.to_torch(field)[shapes] = value
        # VBD also uses model.body_q to derive rest invariants. Preserve the original
        # parallel-transport frames there, not just in the subsequently settled state.
        side = "Left" if free_end_at_start else "Right"
        # Physics-only replication keeps just the source USD prim; environments differ by translation.
        path = env.scene.env_prim_paths[0]
        curve = env.sim.stage.GetPrimAtPath(f"{path}/ShoelaceScene/Shoelace{side}/geometry/mesh")
        transform = UsdGeom.Xformable(curve).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = transform.ExtractRotationQuat()
        world_rotation = torch.tensor(
            (*rotation.GetImaginary(), rotation.GetReal()), dtype=torch.float32, device=env.device
        ).expand(env.num_envs, -1)
        local_rotation = (
            orientations[: cable.num_segments] if free_end_at_start else orientations[-cable.num_segments :]
        )
        poses = wp.to_torch(model.body_q)[bodies].clone()
        poses[..., 3:] = quat_mul(
            world_rotation[:, None].expand(-1, cable.num_segments, -1),
            local_rotation[None].expand(env.num_envs, -1, -1),
        )
        wp.to_torch(model.body_q)[bodies] = poses
        fixed = [-1 if free_end_at_start else 0]
        dynamic = torch.ones(cable.num_segments, dtype=torch.bool, device=env.device)
        dynamic[fixed] = False
        dynamic_bodies = bodies[:, dynamic]
        # Finalize may inflate tiny native inertias. Recompute the unregularized capsule
        # inertia using Newton's geometry API, rather than adding to that corrected value.
        unit_inertias = []
        for body in bodies[0].tolist():
            shapes = np.flatnonzero(shape_bodies == body)
            if len(shapes) != 1 or int(shape_types[shapes[0]]) != int(GeoType.CAPSULE):
                raise ValueError("Shoelace startup requires one capsule shape per segment")
            unit_mass, _, unit_inertia = compute_inertia_shape(
                GeoType.CAPSULE, shape_scales[shapes[0]], None, density=1.0
            )
            unit_inertias.append(np.asarray(unit_inertia).reshape(3, 3) / unit_mass)
        geometric = torch.as_tensor(np.asarray(unit_inertias), dtype=torch.float32, device=env.device)
        updated = mass[dynamic_bodies, None, None] * geometric[dynamic]
        updated += torch.eye(3, device=env.device) * regularization
        inertia[dynamic_bodies] = updated
        inverse_inertia[dynamic_bodies] = torch.linalg.inv(updated)
        anchors = bodies[:, fixed]
        mass[anchors] = inverse_mass[anchors] = 0.0
        inertia[anchors] = inverse_inertia[anchors] = 0.0
        # Joint IDs come from the asset view rather than assuming global contiguous layouts.
        starts = torch.as_tensor(view.get_attribute("joint_qd_start", model).numpy(), device=env.device).long()[:, 0]
        slots = starts[..., None] + torch.arange(4, device=env.device)
        weights = _tail_joint_blend_weights(cable.num_segments - 1, segment_length, free_end_at_start)
        for target, stretch, bend, tail in (
            (model.joint_target_ke, physics.STRETCH_STIFFNESS, physics.BEND_STIFFNESS, physics.TAIL_BEND_STIFFNESS),
            (model.joint_target_kd, physics.STRETCH_DAMPING, physics.BEND_DAMPING, physics.TAIL_BEND_DAMPING),
        ):
            values = np.column_stack((np.full_like(weights, stretch), bend + weights * (tail - bend)))
            values = np.repeat(values * stiffness_scale, 2, axis=1)
            wp.to_torch(target)[slots] = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    # The shoe already has native kinematic semantics; retain its zero-mass contact model as well.
    shoe: RigidObject = env.scene["shoe"]
    shoe.set_masses_index(masses=torch.zeros((env.num_envs, 1), device=env.device))
    shoe.set_inertias_index(inertias=torch.zeros((env.num_envs, 1, 9), device=env.device))
    NewtonManager.add_model_change(
        ModelFlags.BODY_PROPERTIES
        | ModelFlags.BODY_INERTIAL_PROPERTIES
        | ModelFlags.JOINT_DOF_PROPERTIES
        | ModelFlags.SHAPE_PROPERTIES
    )


def install_settled_default_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Install the offline gravity-settled cable poses as defaults at startup.

    Args:
        env: Environment containing both shoelace cables.
        env_ids: Unused; startup installs defaults for every environment.
    """
    names = ("shoelace_left", "shoelace_right")
    with np.load(physics.ASSET_DIR / "settled_tail_clear_segment_poses.npz", allow_pickle=False) as state:
        if tuple(state.files) != names:
            raise ValueError(f"Expected settled shoelace arrays {names}, got {tuple(state.files)}")
        for name in names:
            cable: CableObject = env.scene[name]
            default_pose = cable.data.default_segment_pose_w.torch
            local_pose = np.asarray(state[name], dtype=np.float32)
            if local_pose.shape != default_pose.shape[1:] or not np.isfinite(local_pose).all():
                raise ValueError(f"Invalid settled {name} poses: expected finite {tuple(default_pose.shape[1:])}")
            if not np.allclose(np.linalg.norm(local_pose[:, 3:], axis=1), 1.0, atol=2.0e-5):
                raise ValueError(f"Settled {name} poses contain non-unit quaternions")
            default_pose.copy_(default_pose.new_tensor(local_pose))
            default_pose[..., :3] += env.scene.env_origins.unsqueeze(1)
            velocity = cable.data.default_segment_velocity_w.torch
            velocity.zero_()
            cable.write_segment_pose_to_sim_index(segment_pose=default_pose)
            cable.write_segment_velocity_to_sim_index(segment_velocity=velocity)


def reset_arm_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Perturb arm joints about their defaults and synchronize position targets.

    Args:
        env: Environment containing the arm.
        env_ids: Environments to reset.
        position_range: Uniform joint position offset bounds [rad].
        asset_cfg: Arm articulation and joints to randomize; exclude finger joints.
    """
    reset_joints_by_offset(env, env_ids, position_range, (0.0, 0.0), asset_cfg)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch[env_ids][:, asset_cfg.joint_ids]
    robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_shoe_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
) -> None:
    """Translate the shoe and both laces together about their default positions.

    The pinned lace mesh is a child of the shoe; cable segment poses include the
    fixed anchors. Sampling from defaults avoids accumulating offsets across resets.
    Run after ``reset_scene_to_default`` to restore default velocities as well.

    Args:
        env: Shoelace environment.
        env_ids: Environments to reset.
        position_range: Uniform translation offset bounds [m], keyed by ``x``, ``y``,
            or ``z``. Omitted axes have zero offset.
    """
    shoe: RigidObject = env.scene["shoe"]
    bounds = torch.tensor([position_range.get(axis, (0.0, 0.0)) for axis in "xyz"], device=env.device)
    offset = sample_uniform(bounds[:, 0], bounds[:, 1], (len(env_ids), 3), env.device)
    root_pose = shoe.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids] + offset
    shoe.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)

    for name in ("shoelace_left", "shoelace_right"):
        cable: CableObject = env.scene[name]
        segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
        segment_pose[..., :3] += offset.unsqueeze(1)
        cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
