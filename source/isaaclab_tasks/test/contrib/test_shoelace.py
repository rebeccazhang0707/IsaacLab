# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused reset and material tests for the shoelace task."""

from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardManager, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.contrib.shoelace import shoelace_constants as constants
from isaaclab_tasks.contrib.shoelace.g1_inspire_env_cfg import G1ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.g1_inspire_hand import STRAIGHT_JOINT_NAMES
from isaaclab_tasks.contrib.shoelace.mdp.collision_ik import project_bounded_step, sphere_box_distance
from isaaclab_tasks.contrib.shoelace.mdp.events import reset_shoe_position
from isaaclab_tasks.contrib.shoelace.mdp.grasp import shoelace_grasp_quality
from isaaclab_tasks.contrib.shoelace.mdp.observations import (
    finger_tail_relative_velocity,
    gripper_close_error,
    tail_tcp_relative_speed,
    tails_to_tcp,
)
from isaaclab_tasks.contrib.shoelace.mdp.rewards import (
    binary_gripper_switch_rate,
    dense_task_reward,
    pregrasp_progress_reward,
)
from isaaclab_tasks.contrib.shoelace.mdp.terminations import shoelace_bilateral_pull_success
from isaaclab_tasks.contrib.shoelace.shoelace_assets import ShoelaceUsdCfg
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import configure_shoelace_builder


def test_arm_torso_distance_reports_surface_clearance_and_outward_gradients() -> None:
    """Distinguish face, corner and interior distances, including mirrored normals."""
    points = torch.tensor([[1.5, 0.0, 0.0], [1.3, 1.4, 0.0], [0.0, -0.2, 0.0]], dtype=torch.float64)
    radii = torch.full((3,), 0.1, dtype=torch.float64)
    center, extent = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    distance, gradient = sphere_box_distance(points, radii, center, extent)
    torch.testing.assert_close(distance, torch.tensor([0.4, 0.4, -0.9], dtype=torch.float64))
    torch.testing.assert_close(
        gradient, torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    )
    for axis in range(3):
        perturbation = torch.zeros_like(points)
        perturbation[:, axis] = 1.0e-6
        plus, _ = sphere_box_distance(points + perturbation, radii, center, extent)
        minus, _ = sphere_box_distance(points - perturbation, radii, center, extent)
        torch.testing.assert_close((plus - minus) / 2.0e-6, gradient[:, axis])


def test_arm_step_projection_respects_coupled_clearance_and_joint_bounds() -> None:
    """A command into two surfaces is corrected; degenerate constraints remain finite."""
    delta = torch.tensor([[-0.5, -0.2], [0.1, -0.1]])
    gradient = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0]] * 3])
    required = torch.tensor([[0.1, -0.1, 0.1], [0.1, 0.1, 0.1]])
    lower, upper = torch.full_like(delta, -0.2), torch.full_like(delta, 0.2)
    result = project_bounded_step(delta, gradient, required, lower, upper, 12)
    assert torch.isfinite(result).all()
    assert ((result >= lower) & (result <= upper)).all()
    assert (gradient[0] @ result[0] >= required[0] - 1.0e-6).all()
    torch.testing.assert_close(result[1], delta[1])


@pytest.mark.parametrize("robot", ["franka", "g1"])
def test_robot_collision_selection_preserves_only_grasping_fingers(robot: str, tmp_path: Path) -> None:
    """Select fingers in nested Franka and instanced G1 geometry without deleting shapes."""
    if robot == "franka":
        spawn_cfg = ShoelaceEnvCfg().scene.robot_left.spawn
        enabled = [
            "Geometry/panda_link0/panda_hand/panda_leftfinger",
            "Geometry/panda_link0/panda_hand/panda_rightfinger",
        ]
        disabled = ["Geometry/panda_link0", "Geometry/panda_link0/panda_hand", "Geometry/panda_leftfinger_guard"]
    else:
        spawn_cfg = G1ShoelaceEnvCfg().scene.robot.spawn
        enabled = ["L_thumb_proximal", "L_thumb_distal", "R_index_proximal", "R_index_intermediate"]
        disabled = ["pelvis", "left_wrist_yaw_link", "L_middle_proximal", "R_ring_intermediate"]
    asset_path = tmp_path / "robot.usda"
    geometry_path = tmp_path / "geometry.usda"
    geometry = Usd.Stage.CreateNew(str(geometry_path))
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, "/Geometry").GetPrim())
    collider = UsdGeom.Cube.Define(geometry, "/Geometry/shape").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    geometry.GetRootLayer().Save()
    source = Usd.Stage.CreateNew(str(asset_path))
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Robot").GetPrim())
    for body in enabled + disabled:
        collisions = UsdGeom.Xform.Define(source, f"/Robot/{body}/collisions").GetPrim()
        collisions.GetReferences().AddReference(str(geometry_path))
        collisions.SetInstanceable(robot == "g1")
    source.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        cfg = ShoelaceUsdCfg(
            usd_path=str(asset_path),
            collision_props=spawn_cfg.collision_props,
            make_uninstanceable=spawn_cfg.make_uninstanceable,
        )
        cfg.func("/World/Robot", cfg)
    for body in enabled + disabled:
        collider = stage.GetPrimAtPath(f"/World/Robot/{body}/collisions/shape")
        assert collider.IsA(UsdGeom.Cube)
        assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() == (body in enabled)


def test_g1_straight_pad_spawner_clones_mechanics_and_invisible_material_bound_colliders() -> None:
    """The imported colliders preserve mirrored open/closed grips and clone the straight-finger mechanism."""
    env_cfg = G1ShoelaceEnvCfg()
    cfg = env_cfg.scene.robot.spawn
    # This USD authoring test needs no active backend to create the pelvis world joint.
    cfg.fix_root_link = None
    stage = Usd.Stage.CreateInMemory()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    with sim_utils.use_stage(stage):
        cfg.func("/World/env_.*/Robot", cfg)
    for index in range(2):
        root = stage.GetPrimAtPath(f"/World/env_{index}/Robot")
        joints = [prim for prim in Usd.PrimRange(root) if prim.GetName() in STRAIGHT_JOINT_NAMES]
        assert len(joints) == 6
        for prim in joints:
            assert not any("Mimic" in schema for schema in prim.GetAppliedSchemas())
            assert all(not rel.GetTargets() for rel in prim.GetRelationships() if "mimic" in rel.GetName().lower())
            joint = UsdPhysics.RevoluteJoint(prim)
            assert joint.GetLowerLimitAttr().Get() == pytest.approx(-0.001)
            assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.001)
        for name in ("L_thumb_distal", "L_index_intermediate", "R_thumb_distal", "R_index_intermediate"):
            body = stage.GetPrimAtPath(root.GetPath().AppendChild(name))
            active = [
                prim
                for prim in Usd.PrimRange(body)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            ]
            assert len(active) == 1
            pad = active[0]
            assert pad.IsA(UsdGeom.Cube) and pad.GetName() == "pinch_box"
            assert UsdGeom.Imageable(pad).ComputeVisibility() == UsdGeom.Tokens.invisible
            assert tuple(pad.GetAttribute("xformOp:scale").Get()) == pytest.approx((0.016, 0.010, 0.003))
            material, _ = UsdShade.MaterialBindingAPI(pad).ComputeBoundMaterial(materialPurpose="physics")
            assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == constants.FINGER_MU
    _assert_g1_mirrored_grips(stage.GetPrimAtPath("/World/env_0/Robot"), env_cfg)


def _assert_g1_mirrored_grips(root: Usd.Prim, cfg: G1ShoelaceEnvCfg) -> None:
    """Check actual USD joint chains and box transforms, independently of the calibration constants."""
    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parent or not child:
            continue
        frames = []
        for position, rotation in (
            (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        ):
            matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(rotation))
            matrix.SetTranslateOnly(Gf.Vec3d(position))
            frames.append(np.asarray(matrix).T)
        joints.append((prim.GetName(), parent[0].name, child[0].name, frames, joint.GetAxisAttr().Get()))

    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    for state in ("open", "close"):
        positions = cfg.scene.robot.init_state.joint_pos.copy()
        commands = []
        for side, prefix in (("left", "L"), ("right", "R")):
            command = getattr(getattr(cfg.actions, f"{side}_gripper"), f"{state}_command_expr")
            if state == "open":
                assert all(positions[joint] == target for joint, target in command.items())
            positions.update(command)
            commands.append({joint.removeprefix(f"{prefix}_"): target for joint, target in command.items()})
        assert commands[0] == commands[1]

        bodies = {"torso_link": np.eye(4)}
        pending = list(joints)
        while pending:
            reached = [joint for joint in pending if joint[1] in bodies]
            if not reached:
                break
            for name, parent, child, (parent_frame, child_frame), axis in reached:
                vector = Gf.Vec3d(*np.eye(3)["XYZ".index(axis)])
                rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(vector, np.degrees(positions.get(name, 0.0))))
                bodies[child] = bodies[parent] @ parent_frame @ np.asarray(rotation).T @ np.linalg.inv(child_frame)
            pending = [joint for joint in pending if joint[2] not in bodies]

        wrists = [bodies[f"{side}_wrist_yaw_link"] for side in ("left", "right")]
        # Authored shoulder translations differ by 10 micrometres after reflection.
        np.testing.assert_allclose(wrists[1], mirror @ wrists[0] @ mirror, atol=2.0e-5)
        faces = []
        boxes = []
        for hand, prefix in enumerate(("L", "R")):
            hand_faces = []
            hand_boxes = []
            for finger, suffix in enumerate(("thumb_distal", "index_intermediate")):
                body_name = f"{prefix}_{suffix}"
                body = bodies[body_name]
                prim = root.GetStage().GetPrimAtPath(root.GetPath().AppendPath(f"{body_name}/pinch_box"))
                box = body @ np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
                face = box[:3, 3] + (0.5 if finger == 0 else -0.5) * box[:3, 2]
                landmark = body @ np.r_[cfg.finger_body_offsets[2 * hand + finger], 1.0]
                np.testing.assert_allclose(landmark[:3], face, atol=1.0e-7)
                hand_faces.append(face)
                hand_boxes.append(box)
            faces.append(np.asarray(hand_faces))
            boxes.append(hand_boxes)
            aperture = np.linalg.norm(hand_faces[1] - hand_faces[0])
            assert aperture == pytest.approx(cfg.grasp_open_apertures[hand] if state == "open" else 0.0028, abs=1e-6)
            if state == "close":
                tcp = wrists[hand] @ np.r_[cfg.tcp_body_offsets[hand][0], 1.0]
                np.testing.assert_allclose(tcp[:3], faces[hand].mean(axis=0), atol=1.0e-7)
        for left, right in zip(boxes[0], boxes[1], strict=True):
            np.testing.assert_allclose(right, mirror @ left @ mirror, atol=2.0e-5)


def test_binary_gripper_switch_cost_counts_commands_and_ignores_resets() -> None:
    """Tiny sign changes cost one switch, while arm motion and same-state changes remain free."""
    # Deliberately order the terms differently from the task's default action layout.
    previous = torch.tensor([[0.0, 1.0, -1.0, -0.8]]).repeat(5, 1)
    current = torch.tensor(
        [
            [0.8, -1.0, 1.0, -0.01],
            [-0.0001, 1.0, -1.0, -0.8],
            [-0.0001, 1.0, -1.0, 0.0],
            [-0.0001, 1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0, -0.8],
        ]
    )
    env = SimpleNamespace(
        num_envs=5,
        device="cpu",
        step_dt=1.0 / 30.0,
        sim=SimpleNamespace(is_playing=lambda: True),
        episode_length_buf=torch.tensor([10, 10, 10, 1, 1]),
        action_manager=SimpleNamespace(
            active_terms=["right_gripper", "left_arm", "left_gripper"],
            action_term_dim=[1, 2, 1],
            action=current,
            prev_action=previous,
        ),
    )
    manager = RewardManager({"switch": RewardTermCfg(func=binary_gripper_switch_rate, weight=-0.002)}, env)
    expected = torch.tensor([0.0, -0.002, -0.004, 0.0, 0.0])
    torch.testing.assert_close(manager.compute(env.step_dt), expected)
    # Changing the control rate must not change the cost of the same command transitions.
    env.step_dt = 1.0 / 60.0
    torch.testing.assert_close(manager.compute(env.step_dt), expected)


@pytest.mark.parametrize(
    ("open_position", "closed_position", "direction"),
    [(0.04, 0.0, -1.0), (0.0, 1.2, 1.0), ((0.6, 0.65), (0.8, 0.85), 1.0)],
)
def test_finger_tracking_residual_supports_both_joint_directions(
    open_position: float | tuple[float, float], closed_position: float | tuple[float, float], direction: float
) -> None:
    """Tracking residuals must handle both joint directions and different left/right strokes."""
    opened = torch.tensor(open_position).expand(2)
    closed = torch.tensor(closed_position).expand(2)
    positions = torch.stack([opened, (opened + closed) / 2, closed])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=positions),
            joint_pos_target=SimpleNamespace(torch=closed.expand_as(positions)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(3, 2, 4)),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(3, 2, 6)),
        )
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=torch.zeros(3, 3, 7)),
            segment_velocity_w=SimpleNamespace(torch=torch.zeros(3, 3, 6)),
        )
    )
    env = SimpleNamespace(
        num_envs=3,
        cfg=SimpleNamespace(tcp_body_offsets=((constants.TCP_OFFSET,),) * 2),
        scene={
            "robot": robot,
            "left": cable,
            "right": cable,
            "finger_tail_contacts": SimpleNamespace(data=torch.zeros(3, 4)),
        },
    )
    robot_cfgs = tuple(SceneEntityCfg("robot", joint_ids=[index], body_ids=[index]) for index in range(2))
    residual = gripper_close_error(env, robot_cfgs, closing_direction=direction) / (closed - opened).abs()
    torch.testing.assert_close(residual, torch.tensor([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]))


def test_tcp_observations_use_each_hand_offset_for_position_and_angular_velocity() -> None:
    """Distinct hand TCPs must rotate their positions and include angular motion when measuring slip."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            body_pos_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 2**-0.5, 2**-0.5]]])),
            body_link_vel_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]])),
        )
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=torch.zeros(1, 3, 7)),
            segment_velocity_w=SimpleNamespace(torch=torch.zeros(1, 3, 6)),
        )
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(tcp_body_offsets=(((0.1, 0.0, 0.0),), ((0.0, 0.2, 0.0),))),
        scene={"robot": robot, "left": cable, "right": cable},
    )
    robot_cfgs = (SceneEntityCfg("robot", body_ids=[0]),) * 2
    cable_cfgs = (SceneEntityCfg("left"), SceneEntityCfg("right"))
    torch.testing.assert_close(
        tails_to_tcp(env, cable_cfgs, robot_cfgs), torch.tensor([[0.0, -0.1, 0.0, 0.2, 0.0, 0.0]])
    )
    torch.testing.assert_close(tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs), torch.tensor([[0.2, 0.4]]))


def _geometry_env(num_envs: int):
    cfg = ShoelaceEnvCfg()
    cfg.finger_body_offsets = ((0.0, 0.0, 0.0),) * 4
    cfg.grasp_open_apertures = (0.03, 0.03)
    points = torch.zeros(num_envs, 4, 3)
    points[:, [0, 2], 0] = -0.0015
    points[:, [1, 3], 0] = 0.0015
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=points),
            body_quat_w=SimpleNamespace(torch=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(num_envs, 4, 4)),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 4, 6)),
            root_quat_w=SimpleNamespace(torch=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(num_envs, 4)),
        )
    )
    pose = torch.zeros(num_envs, 3, 7)
    pose[:, :, 1] = torch.tensor([-0.003, 0.0, 0.003])
    cables = [
        SimpleNamespace(
            data=SimpleNamespace(
                segment_pose_w=SimpleNamespace(torch=pose.clone()),
                segment_velocity_w=SimpleNamespace(torch=torch.zeros(num_envs, 3, 6)),
            )
        )
        for _ in range(2)
    ]
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.1,
        cfg=cfg,
        extras={"log": {}},
        scene={
            "robot": robot,
            "left": cables[0],
            "right": cables[1],
            "finger_tail_contacts": SimpleNamespace(data=torch.zeros(num_envs, 4)),
        },
        termination_manager=SimpleNamespace(
            dones=torch.zeros(num_envs, dtype=torch.bool), get_term=lambda name: torch.zeros(num_envs, dtype=torch.bool)
        ),
    )
    fingers = tuple(SceneEntityCfg("robot", body_ids=[i]) for i in range(4))
    cable_cfgs = (SceneEntityCfg("left"), SceneEntityCfg("right"))
    return env, fingers, cable_cfgs


def test_grasp_requires_local_two_surface_capture_and_uncancelled_slip() -> None:
    """Grasp scoring must use geometry without joint data, and opposed sliding must fail the pull gate."""
    env, fingers, cables = _geometry_env(4)
    points = env.scene["robot"].data.body_pos_w.torch
    points[0, [0, 2], 0] = -0.015
    points[0, [1, 3], 0] = 0.015
    env.scene["finger_tail_contacts"].data[1, [1, 3]] = 0.002
    velocity = env.scene["robot"].data.body_link_vel_w.torch
    # Motion across the tail must also fail, without a tail-direction-dependent projection.
    velocity[3, [0, 2], 0] = 0.1
    velocity[3, [1, 3], 0] = -0.1
    quality, finite = shoelace_grasp_quality(env, 0.0005, 0.08, cables, fingers, 0.001)
    assert finite.all()
    assert (quality[:2] < 1e-5).all()
    assert (quality[2] > 0.99).all()
    assert (quality[3] < 0.2).all()
    # Direction remains available to the actor, rather than cancelling at the midpoint.
    relative = finger_tail_relative_velocity(env, fingers, cables).reshape(4, 4, 3)
    torch.testing.assert_close(relative[3, :, 0], torch.tensor([-0.1, 0.1, -0.1, 0.1]))


def test_capture_preserving_asymmetric_closure_and_first_contact_earn_progress() -> None:
    """A moving midpoint must not lose approach credit while the tail stays bracketed by the surfaces."""
    env, fingers, cables = _geometry_env(1)
    cfg = RewardTermCfg(func=pregrasp_progress_reward)
    pregrasp = pregrasp_progress_reward(cfg, env)
    dense_cfg = RewardTermCfg(func=dense_task_reward, params=env.cfg.rewards.dense_task.params.copy())
    dense = dense_task_reward(dense_cfg, env)
    dense_params = dense_cfg.params | {"cable_cfgs": cables, "finger_cfgs": fingers}
    points = env.scene["robot"].data.body_pos_w.torch
    points[:, [0, 2], 0] = -0.015
    points[:, [1, 3], 0] = 0.015
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() == 0
    assert dense(env, **dense_params).item() == 0
    points[:, [1, 3], 0] = 0.003
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() > 0
    assert abs(dense(env, **dense_params).item()) < 1e-4
    # Restore a small captured aperture, then acquire just the first finger contact.
    points[:, [0, 2], 0] = -0.0015
    points[:, [1, 3], 0] = 0.0015
    env.scene["finger_tail_contacts"].data.fill_(0.002)
    pregrasp(env, 0.015, 0.01, cables, fingers)
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() == 0
    env.scene["finger_tail_contacts"].data[0, 0] = 0
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() > 0
    env.scene["finger_tail_contacts"].data.fill_(0.002)
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() < 0
    # Moving the closed pair away removes capture credit; reset never pays for it.
    points[:, :, 2] += 0.02
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() < 0
    pregrasp.reset([0])
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() == 0


def test_coarse_approach_saturates_locally_without_changing_far_field_or_cycle_credit() -> None:
    """An optional plateau removes local costs while preserving the far field and signed progress."""
    env, fingers, cables = _geometry_env(1)
    points = env.scene["robot"].data.body_pos_w.torch
    points[:, [0, 2], 0] = -0.015
    points[:, [1, 3], 0] = 0.015
    env.scene["finger_tail_contacts"].data.fill_(0.002)
    entities = {"cable_cfgs": cables, "finger_cfgs": fingers}
    old_params = env.cfg.rewards.dense_task.params | entities
    params = old_params | {"approach_saturation_radius": 0.01}
    original = dense_task_reward(RewardTermCfg(func=dense_task_reward, params=old_params), env)
    saturated = dense_task_reward(RewardTermCfg(func=dense_task_reward, params=params), env)
    points[:, :, 2] = 0.003
    assert original(env, **old_params).item() == 0.0
    assert saturated(env, **params).item() == 0.0
    points[:, :, 2] = 0.009
    assert original(env, **old_params).item() < 0.0
    assert saturated(env, **params).item() == 0.0
    points[:, :, 2] = 0.01
    assert saturated(env, **params).item() == 0.0
    points[:, :, 2] = 0.003
    assert original(env, **old_params).item() > 0.0
    assert saturated(env, **params).item() == 0.0

    points[:, :, 2] = 0.02
    original(env, **old_params)
    leaving = saturated(env, **params)
    assert leaving.item() < 0.0
    assert saturated(env, **params).item() == 0.0
    points[:, :, 2] = 0.015
    approaching = saturated(env, **params)
    assert approaching.item() > 0.0
    torch.testing.assert_close(approaching, original(env, **old_params))
    points[:, :, 2] = 0.009
    reentering = saturated(env, **params)
    assert reentering.item() > 0.0
    torch.testing.assert_close(leaving + approaching + reentering, torch.zeros(1))
    saturated.reset([0])
    assert saturated(env, **params).item() == 0.0
    with pytest.raises(ValueError, match="approach_saturation_radius"):
        saturated(env, **(params | {"approach_saturation_radius": float("nan")}))


def test_g1_fixed_tcp_approach_matches_observations_without_replacing_physical_grasp() -> None:
    """G1 approach follows the wrist TCP while actual pads still determine capture and grasp."""
    env, _, _ = _geometry_env(1)
    env.cfg = G1ShoelaceEnvCfg()
    robot = env.scene["robot"]
    robot.body_names = [
        "L_thumb_distal",
        "L_index_intermediate",
        "R_thumb_distal",
        "R_index_intermediate",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    robot.joint_names = [
        "L_thumb_proximal_pitch_joint",
        "L_index_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_index_proximal_joint",
    ]
    robot.num_bodies = len(robot.body_names)
    robot.num_joints = len(robot.joint_names)
    robot.find_bodies = lambda names, **kwargs: resolve_matching_names(names, robot.body_names, **kwargs)
    robot.find_joints = lambda names, **kwargs: resolve_matching_names(names, robot.joint_names, **kwargs)
    points = torch.zeros(1, 6, 3)
    robot.data.body_pos_w.torch = points
    robot.data.body_quat_w.torch = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, 6, 4).clone()
    robot.data.body_quat_w.torch[:, 4:] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    robot.data.body_link_vel_w.torch = torch.zeros(1, 6, 6)
    env.scene["shoelace_left"] = env.scene["left"]
    env.scene["shoelace_right"] = env.scene["right"]
    params = env.cfg.rewards.dense_task.params
    observation = env.cfg.observations.policy.tails_to_tcp
    for entities in (params["finger_cfgs"], params["approach_robot_cfgs"], observation.params["robot_cfgs"]):
        for entity in entities:
            entity.resolve(env.scene)
    offsets = torch.tensor(env.cfg.tcp_body_offsets)[:, 0]
    # A 90-degree wrist yaw maps local (x, y, z) to world (-y, x, z).
    points[:, 4:, 0] = offsets[:, 1]
    points[:, 4:, 1] = -offsets[:, 0]
    points[:, 4:, 2] = torch.tensor([0.02, 0.04]) - offsets[:, 2]
    pads = torch.tensor([[[-0.015, 0.0, 0.0], [0.015, 0.0, 0.0]]] * 2).reshape(1, 4, 3)
    points[:, :4] = pads - torch.tensor(env.cfg.finger_body_offsets)
    env.scene["finger_tail_contacts"].data.fill_(env.cfg.finger_contact_distance_cap)
    expected = torch.tensor([[0.0, 0.0, -0.02, 0.0, 0.0, -0.04]])
    torch.testing.assert_close(observation.func(env, **observation.params), expected, atol=1e-7, rtol=0)
    term = dense_task_reward(env.cfg.rewards.dense_task, env)
    assert term(env, **params).item() == 0.0

    # Asymmetric closure shifts the live midpoint, but neither coarse distance nor its reward.
    for index_x in (0.003, 0.015):
        pads[:, [1, 3], 0] = index_x
        points[:, :4] = pads - torch.tensor(env.cfg.finger_body_offsets)
        torch.testing.assert_close(observation.func(env, **observation.params), expected, atol=1e-7, rtol=0)
        assert term(env, **params).item() == 0.0
    # Equal observed TCP distances must have equal potential, even with different wrist distances.
    points[:, 4:, 0] += torch.tensor([0.02, 0.04])
    points[:, 4:, 2] -= torch.tensor([0.02, 0.04])
    sideways = observation.func(env, **observation.params)
    torch.testing.assert_close(sideways, torch.tensor([[-0.02, 0.0, 0.0, -0.04, 0.0, 0.0]]), atol=1e-7, rtol=0)
    torch.testing.assert_close(term(env, **params), torch.zeros(1), atol=1e-6, rtol=0)
    points[:, 4:, 0] -= torch.tensor([0.02, 0.04])
    points[:, 4:, 2] += torch.tensor([0.02, 0.04])
    torch.testing.assert_close(term(env, **params), torch.zeros(1), atol=1e-6, rtol=0)
    points[:, 4:, 2] -= 0.01
    nearer = expected.clone()
    nearer[:, [2, 5]] += 0.01
    torch.testing.assert_close(observation.func(env, **observation.params), nearer, atol=1e-7, rtol=0)
    approaching = term(env, **params)
    assert approaching.item() > 0.0
    assert term(env, **params).item() == 0.0
    points[:, 4:, 2] += 0.01
    torch.testing.assert_close(term(env, **params), -approaching)

    # A fixed TCP near the tail cannot fake a grasp, and a distant TCP cannot veto actual capture.
    points[:, 4:, 2] -= torch.tensor([0.02, 0.04])
    env.scene["finger_tail_contacts"].data.zero_()
    fingers, cables = params["finger_cfgs"], params["cable_cfgs"]
    quality, finite = shoelace_grasp_quality(env, 0.0005, 0.08, cables, fingers, 0.001)
    assert finite.all()
    assert (quality < 1e-5).all()
    points[:, 4:, 2] += torch.tensor([0.02, 0.04])
    pads[:, [0, 2], 0] = -env.cfg.grasp_target_diameter / 2
    pads[:, [1, 3], 0] = env.cfg.grasp_target_diameter / 2
    points[:, :4] = pads - torch.tensor(env.cfg.finger_body_offsets)
    quality, finite = shoelace_grasp_quality(env, 0.0005, 0.08, cables, fingers, 0.001)
    assert finite.all()
    assert (quality > 0.99).all()
    torch.testing.assert_close(observation.func(env, **observation.params), expected, atol=1e-7, rtol=0)


def test_franka_asymmetric_closure_earns_net_progress_without_midpoint_cancellation() -> None:
    """Franka capture-based approach preserves closure credit; release and escape cannot farm it."""
    env, fingers, cables = _geometry_env(1)
    points = env.scene["robot"].data.body_pos_w.torch
    points[:, [0, 2], 0] = -0.015
    points[:, [1, 3], 0] = 0.015
    points[:, :, 2] = 0.001
    env.scene["finger_tail_contacts"].data.fill_(0.002)
    cfg = env.cfg
    params = cfg.rewards.dense_task.params | {"cable_cfgs": cables, "finger_cfgs": fingers}
    dense = dense_task_reward(RewardTermCfg(func=dense_task_reward, params=params), env)
    pregrasp = pregrasp_progress_reward(RewardTermCfg(func=pregrasp_progress_reward), env)
    assert dense(env, **params).item() == 0.0
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() == 0.0
    points[:, [0, 2], 0] = -0.003
    coarse = dense(env, **params)
    close = pregrasp(env, 0.015, 0.01, cables, fingers)
    torch.testing.assert_close(coarse, torch.zeros(1))
    assert (cfg.rewards.dense_task.weight * coarse + cfg.rewards.pregrasp.weight * close).item() > 0.0
    assert dense(env, **params).item() == 0.0
    assert pregrasp(env, 0.015, 0.01, cables, fingers).item() == 0.0
    points[:, [0, 2], 0] = -0.015
    torch.testing.assert_close(pregrasp(env, 0.015, 0.01, cables, fingers), -close)
    assert dense(env, **params).item() == 0.0
    points[:, :, 2] = 0.007
    assert dense(env, **params).item() < 0.0
    dense.reset([0])
    assert dense(env, **params).item() == 0.0


def test_loaded_pull_success_survives_only_unselected_resets() -> None:
    """Success must require loaded pull history and clear it only for selected environments."""
    env, fingers, cables = _geometry_env(2)
    term = shoelace_bilateral_pull_success(TerminationTermCfg(func=shoelace_bilateral_pull_success), env)
    params = dict(
        throat_radius=0.025,
        minimum_pull_distance=0.004,
        minimum_tail_outward_distance=0.004,
        grasp_threshold=0.2,
        contact_std=0.0005,
        relative_speed_std=0.08,
        grasp_filter_time_constant=0.1,
        cable_cfgs=cables,
        finger_cfgs=fingers,
        contact_penetration_tolerance=0.001,
    )
    assert not term(env, **params).any()
    env.scene["right"].data.segment_pose_w.torch[:, :, 0] -= 0.005
    env.scene["left"].data.segment_pose_w.torch[:, :, 0] += 0.005
    env.scene["robot"].data.body_pos_w.torch[:, :2, 0] -= 0.005
    env.scene["robot"].data.body_pos_w.torch[:, 2:, 0] += 0.005
    assert term(env, **params).all()
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term(env, **params), torch.tensor([True, False]))


def test_tcp_tracks_both_surface_points_and_their_linear_and_angular_velocities() -> None:
    """Finger motion must move the TCP and contribute to slip even when the wrist frame stays fixed."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])),
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 2**-0.5, 2**-0.5], [0.0, 0.0, 0.0, 1.0]]])),
            body_link_vel_w=SimpleNamespace(
                torch=torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0, 0.0, -2.0]]])
            ),
        )
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=SimpleNamespace(torch=torch.zeros(1, 3, 7)),
            segment_velocity_w=SimpleNamespace(torch=torch.zeros(1, 3, 6)),
        )
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(tcp_body_offsets=(((0.2, 0.0, 0.0), (0.0, 0.4, 0.0)), ((0.0, 0.0, 0.0),) * 2)),
        scene={"robot": robot, "left": cable, "right": cable},
    )
    robot_cfgs = (SceneEntityCfg("robot", body_ids=[0, 1]),) * 2
    cable_cfgs = (SceneEntityCfg("left"), SceneEntityCfg("right"))
    torch.testing.assert_close(
        tails_to_tcp(env, cable_cfgs, robot_cfgs), torch.tensor([[-0.3, 2.0, 0.0, 0.0, 2.0, 0.0]]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs), torch.tensor([[4.04**0.5, 2.0]]))
    robot.data.body_pos_w.torch[0, 1, 0] = 5.0
    torch.testing.assert_close(
        tails_to_tcp(env, cable_cfgs, robot_cfgs), torch.tensor([[-0.3, 3.0, 0.0, 0.0, 3.0, 0.0]]), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("env_ids", [slice(1, 4, 2), torch.tensor([3, 1])])
def test_shoe_reset_accepts_slices_and_preserves_unselected_poses(env_ids: torch.Tensor | slice) -> None:
    """Apply the same translation to selected shoe and cable poses without expanding slices."""
    defaults = torch.zeros(4, 7)
    defaults[:, 6] = 1.0
    origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shoe_pose = defaults.clone()
    cable_poses = [defaults[:, None].expand(-1, 3, -1).clone() for _ in range(2)]
    cable_poses[1][..., 1] = 0.5
    initial_cable_poses = [pose.clone() for pose in cable_poses]
    shoe = SimpleNamespace(
        data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=defaults)),
        write_root_pose_to_sim_index=lambda *, root_pose, env_ids: shoe_pose.__setitem__(env_ids, root_pose),
    )
    cables = [
        SimpleNamespace(
            data=SimpleNamespace(default_segment_pose_w=SimpleNamespace(torch=pose.clone())),
            write_segment_pose_to_sim_index=lambda *, segment_pose, env_ids, buffer=pose: buffer.__setitem__(
                env_ids, segment_pose
            ),
        )
        for pose in cable_poses
    ]

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(device="cpu", scene=Scene(shoe=shoe, shoelace_left=cables[0], shoelace_right=cables[1]))
    reset_shoe_position(env, env_ids, {"x": (0.1, 0.1), "y": (-0.2, -0.2)})
    expected_shoe = defaults.clone()
    offset = torch.tensor([0.1, -0.2, 0.0])
    expected_shoe[env_ids, :3] += origins[env_ids] + offset
    torch.testing.assert_close(shoe_pose, expected_shoe)
    for actual, expected in zip(cable_poses, initial_cable_poses, strict=True):
        expected[env_ids, :, :3] += offset
        torch.testing.assert_close(actual, expected)


def test_cable_friction_uses_each_bound_usd_material() -> None:
    """Preserve distinct authored cable friction through model construction without external assets."""
    stage = Usd.Stage.CreateInMemory()
    builder = newton.ModelBuilder()
    builder.begin_world()
    cfg = ShoelaceEnvCfg()
    cfg.scene.clone_cfg.clone_template = "/Lab/Cells/Cell{}"
    expected = {}
    for side, friction in (("Left", 0.23), ("Right", 0.37)):
        path = f"/Lab/Cells/Cell0/ShoelaceScene/Shoelace{side}/geometry/mesh"
        curve = UsdGeom.BasisCurves.Define(stage, path).GetPrim()
        points = np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0)])
        segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()
        for name in ("referenceSegmentLength", "segmentLength"):
            curve.CreateAttribute(f"shoelace:{name}", Sdf.ValueTypeNames.Double).Set(segment_length)
        material = UsdShade.Material.Define(stage, f"{path}/ContactMaterial")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(friction)
        UsdShade.MaterialBindingAPI.Apply(curve).Bind(material, materialPurpose="physics")
        bodies, _ = builder.add_rod(rod=newton.Rod(points, radius=0.001), label=path, body_frame_origin="start")
        for body in bodies:
            expected.update({shape: friction for shape in builder.body_shapes[body]})
    builder.end_world()

    configure_shoelace_builder(builder, stage, 0, np.array([0.0, 0.0, 0.0, 1.0]), cfg)
    model = builder.finalize(device="cpu")
    for shape, friction in expected.items():
        assert model.shape_material_mu.numpy()[shape] == pytest.approx(friction)
