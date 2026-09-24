# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shoelace proxy inertia and physically consistent randomized resets."""

from types import SimpleNamespace

import gymnasium as gym
import newton
import numpy as np
import pytest
import torch
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from isaaclab_newton.physics import NewtonManager

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.contrib.shoelace import shoelace_constants as constants
from isaaclab_tasks.contrib.shoelace.asset_authoring import load_shoelace
from isaaclab_tasks.contrib.shoelace.mdp.events import configure_shoelace_physics
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg


@pytest.mark.parametrize(("num_envs", "override"), [(4, None), (1024, None), (1024, 1_024_000)])
def test_runtime_contact_capacity_scales_and_preserves_overrides(num_envs: int, override: int | None) -> None:
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = num_envs
    outer_override = 16_777_216 if override is not None else 0
    if override is not None:
        cfg.sim.physics.collision_cfg.max_triangle_pairs = outer_override
        cfg.sim.physics.solver_cfg.contact_max_triangle_pairs = override
        cfg.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor = 4.0
    cfg.validate()

    required = max(constants.MIN_TRIANGLE_PAIRS, constants.TRIANGLE_PAIRS_PER_ENV * num_envs)
    expected = max(required, outer_override)
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == expected
    internal_capacity = max(constants.MIN_TRIANGLE_PAIRS, override or 0)
    assert cfg.sim.physics.solver_cfg.contact_max_triangle_pairs == internal_capacity
    assert cfg.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor == max(
        4.0 if override is not None else 0.25, required * 0.25 / internal_capacity
    )


@pytest.mark.parametrize("regularization", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_proxy_inertia_is_rejected(regularization: float) -> None:
    """Reject negative and nonfinite regularization before editing body properties."""
    cfg = ShoelaceEnvCfg()
    cfg.cable_inertia_regularization = regularization
    cfg.validate()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        configure_shoelace_physics(SimpleNamespace(cfg=cfg), None)


def test_baked_asset_defers_task_physics_to_events() -> None:
    """Import native geometry/materials without custom model overrides or Dahl attributes."""
    cfg = ShoelaceEnvCfg()
    cfg.lace_mu, cfg.shoe_mu = 0.25, 0.5
    cfg.validate()
    stage = sim_utils.create_new_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    spawn = cfg.scene.shoelace_asset.spawn
    spawn.func("/World/ShoelaceScene", spawn)
    for prim in stage.Traverse():
        assert not any("dahl" in attr.GetName().lower() for attr in prim.GetAuthoredAttributes())
    native = newton.ModelBuilder()
    native_result = native.add_usd(stage, root_path="/World/ShoelaceScene", return_deformable_results=True)
    builder = build_source_builders(stage, ["/World/ShoelaceScene"], newton.ModelBuilder, [])["/World/ShoelaceScene"]
    for side in ("Left", "Right"):
        curve_path = f"/World/ShoelaceScene/Shoelace{side}/geometry/mesh"
        curve = stage.GetPrimAtPath(curve_path)
        bodies, _ = native_result["path_cable_map"][curve_path]
        geometric = np.asarray(native.body_inertia, dtype=np.float32).reshape(-1, 3, 3)[bodies]
        for name in (
            "fixedSegments",
            "inertiaRegularization",
            "segmentOrientations",
            "jointStiffnesses",
            "jointDampings",
        ):
            assert curve.GetAttribute(f"isaaclab:cable:{name}").Get() is None
        for segment in range(len(geometric)):
            body = builder.body_label.index(f"{curve_path}_edge_body_{segment}")
            assert builder.body_mass[body] == native.body_mass[bodies[segment]] > 0.0
            np.testing.assert_allclose(
                np.asarray(builder.body_inertia[body]).reshape(3, 3), geometric[segment], rtol=1.0e-6
            )
            for shape in builder.body_shapes[body]:
                native_shape = native.body_shapes[bodies[segment]][0]
                assert builder.shape_material_mu[shape] == native.shape_material_mu[native_shape]
    shoe = builder.shape_label.index("/World/ShoelaceScene/Shoe/Collider")
    assert builder.shape_material_mu[shoe] == pytest.approx(cfg.shoe_mu)
    assert any(
        "ShoelacePinned" in label and builder.shape_flags[index] & newton.ShapeFlags.VISIBLE
        for index, label in enumerate(builder.shape_label)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The coupled MJWarp/VBD rollout requires CUDA")
@pytest.mark.parametrize("regularization", [0.0, 3.0e-7, constants.CABLE_INERTIA_REGULARIZATION])
@torch.inference_mode()
def test_randomized_resets_preserve_shoe_lace_alignment_and_other_environments(regularization: float) -> None:
    """Resample selected starts without drift, stale targets, or displaced cable anchors."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 4
    cfg.cable_inertia_regularization = regularization
    cfg.sim.device = "cuda:0"
    with launch_simulation(cfg, {"visualizer": None, "visualizer_explicit": True}):
        env = gym.make("IsaacContrib-Shoelace-DualFranka", cfg=cfg).unwrapped
        assert type(env) is ManagerBasedRLEnv
        try:
            with np.load(constants.ASSET_DIR / "settled_tail_clear_segment_poses.npz") as state:
                for name in state.files:
                    defaults = env.scene[name].data.default_segment_pose_w.torch
                    expected = defaults.new_tensor(state[name]).unsqueeze(0).expand_as(defaults).clone()
                    expected[..., :3] += env.scene.env_origins.unsqueeze(1)
                    torch.testing.assert_close(defaults, expected, rtol=0, atol=0)
            env.reset()
            shoe = env.scene["shoe"]
            all_ids = torch.arange(env.num_envs, device=env.device)
            selected = all_ids[::2]
            untouched = all_ids[1::2]
            robots = [env.scene[name] for name in ("robot_left", "robot_right")]
            cables = [env.scene[name] for name in ("shoelace_left", "shoelace_right")]
            arm_events = [cfg.events.reset_left_arm, cfg.events.reset_right_arm]
            model = NewtonManager.get_model()
            cable_solver = NewtonManager._solver.solver("shoelace")
            assert isinstance(cable_solver, newton.solvers.SolverVBD)
            assert cable_solver.enable_dahl_friction is False
            assert cable_solver.rigid_compliant_alm is True
            # Compare startup inertia to an independent native USD import before finalize's correction.
            native = newton.ModelBuilder()
            imported = native.add_usd(str(constants.ASSET_DIR / "shoelace.usda"), return_deformable_results=True)
            native_inertia = np.asarray(native.body_inertia).reshape(-1, 3, 3)
            centerline, _, reference_length = load_shoelace()
            mean_length = np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean()
            authored_frames = np.asarray(newton.utils.create_parallel_transport_cable_quaternions(centerline))
            for side, fixed in (("Left", -1), ("Right", 0)):
                path = f"/Shoelace/Shoelace{side}/geometry/mesh"
                bodies, _ = imported["path_cable_map"][path]
                expected = native_inertia[bodies] + np.eye(3) * regularization
                expected[fixed] = 0.0
                for world in range(env.num_envs):
                    prefix = f"/World/envs/env_{world}/ShoelaceScene/Shoelace{side}/geometry/mesh"
                    indices = [model.body_label.index(f"{prefix}_edge_body_{i}") for i in range(len(bodies))]
                    shapes = np.flatnonzero(np.isin(model.shape_body.numpy(), indices))
                    assert len(shapes) == len(indices)
                    for field, value in (
                        ("shape_material_mu", cfg.lace_mu),
                        ("shape_material_ke", constants.CONTACT_KE),
                        ("shape_material_kd", constants.CONTACT_KD),
                        ("shape_gap", constants.CONTACT_GAP),
                    ):
                        np.testing.assert_allclose(getattr(model, field).numpy()[shapes], value)
                    actual = model.body_inertia.numpy()[indices]
                    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=1.0e-14)
                    dynamic = np.delete(np.arange(len(bodies)), fixed)
                    np.testing.assert_allclose(
                        actual[dynamic] @ model.body_inv_inertia.numpy()[indices][dynamic],
                        np.broadcast_to(np.eye(3), actual[dynamic].shape),
                        atol=1.0e-6,
                    )
                    assert model.body_mass.numpy()[indices[fixed]] == 0.0
                    frames = authored_frames[: len(bodies)] if side == "Left" else authored_frames[-len(bodies) :]
                    np.testing.assert_allclose(model.body_q.numpy()[indices, 3:], frames, atol=1.0e-6)
                    # Independently sample the material profile's core and ordinary regions.
                    joint_indices = [model.joint_label.index(f"{prefix}_cable_{i}") for i in range(1, len(bodies))]
                    slots = model.joint_qd_start.numpy()[joint_indices, None] + np.arange(4)
                    for field, stretch, bend, tail in (
                        (
                            "joint_target_ke",
                            constants.STRETCH_STIFFNESS,
                            constants.BEND_STIFFNESS,
                            constants.TAIL_BEND_STIFFNESS,
                        ),
                        (
                            "joint_target_kd",
                            constants.STRETCH_DAMPING,
                            constants.BEND_DAMPING,
                            constants.TAIL_BEND_DAMPING,
                        ),
                    ):
                        gains = getattr(model, field).numpy()[slots]
                        np.testing.assert_allclose(gains[:, :2], stretch * reference_length / mean_length, rtol=1.0e-6)
                        tip, ordinary = (0, -1) if side == "Left" else (-1, 0)
                        np.testing.assert_allclose(gains[tip, 2:], tail * reference_length / mean_length, rtol=1.0e-6)
                        np.testing.assert_allclose(
                            gains[ordinary, 2:], bend * reference_length / mean_length, rtol=1.0e-6
                        )
            model_before = {
                name: getattr(model, name).numpy().copy()
                for name in (
                    "body_mass",
                    "body_inv_mass",
                    "body_inertia",
                    "body_inv_inertia",
                    "joint_target_ke",
                    "joint_target_kd",
                    "shape_material_mu",
                    "shape_material_ke",
                    "shape_material_kd",
                    "shape_gap",
                )
            }
            for world in range(env.num_envs):
                root = f"/World/envs/env_{world}/ShoelaceScene/Shoe"
                body = model.body_label.index(root)
                # All shoe geometry must follow the same body, including the pinned lace span.
                for suffix in ("/Collider", "/TongueUpper/geometry/mesh", "/ShoelacePinned/geometry/mesh"):
                    shape = model.shape_label.index(root + suffix)
                    assert model.shape_body.numpy()[shape] == body

            for _ in range(3):
                shoe_before = shoe.data.root_pose_w.torch.clone()
                robot_before = [robot.data.joint_pos.torch.clone() for robot in robots]
                cable_before = [cable.data.segment_pose_w.torch.clone() for cable in cables]
                distances = env.scene["finger_tail_contacts"].data
                distances.fill_(-constants.CONTACT_DISTANCE_CAP)
                contacts_before = distances.clone()
                env.reset(env_ids=selected)
                torch.testing.assert_close(
                    distances[selected], torch.full_like(distances[selected], constants.CONTACT_DISTANCE_CAP)
                )
                torch.testing.assert_close(distances[untouched], contacts_before[untouched], rtol=0, atol=0)
                shoe_pose = shoe.data.root_pose_w.torch
                offset = shoe_pose[:, :3] - shoe.data.default_root_pose.torch[:, :3] - env.scene.env_origins
                assert not torch.equal(shoe_pose[selected], shoe_before[selected])
                assert not torch.equal(offset[selected[0]], offset[selected[1]])
                torch.testing.assert_close(shoe_pose[untouched], shoe_before[untouched], rtol=0, atol=0)
                torch.testing.assert_close(offset[:, 2], torch.zeros_like(offset[:, 2]))
                torch.testing.assert_close(shoe_pose[:, 3:], shoe.data.default_root_pose.torch[:, 3:])
                for axis, key in enumerate(("x", "y")):
                    lower, upper = cfg.events.reset_shoe.params["position_range"][key]
                    assert torch.all((offset[:, axis] >= lower - 1.0e-6) & (offset[:, axis] <= upper + 1.0e-6))

                for robot, before, event in zip(robots, robot_before, arm_events, strict=True):
                    joints = event.params["asset_cfg"].joint_ids
                    positions = robot.data.joint_pos.torch
                    defaults = robot.data.default_joint_pos.torch
                    delta = positions[selected][:, joints] - defaults[selected][:, joints]
                    lower, upper = event.params["position_range"]
                    assert torch.all((delta >= lower - 1.0e-6) & (delta <= upper + 1.0e-6))
                    assert not torch.equal(positions[selected], before[selected])
                    torch.testing.assert_close(positions[untouched], before[untouched], rtol=0, atol=0)
                    torch.testing.assert_close(
                        robot.actuators.target_command.position.torch[selected], positions[selected]
                    )
                    fingers, _ = robot.find_joints("panda_finger_joint.*")
                    torch.testing.assert_close(positions[selected][:, fingers], defaults[selected][:, fingers])
                    torch.testing.assert_close(
                        robot.data.joint_vel.torch[selected], robot.data.default_joint_vel.torch[selected]
                    )

                for cable, before in zip(cables, cable_before, strict=True):
                    expected = cable.data.default_segment_pose_w.torch[selected].clone()
                    expected[..., :3] += offset[selected].unsqueeze(1)
                    torch.testing.assert_close(cable.data.segment_pose_w.torch[selected], expected)
                    torch.testing.assert_close(
                        cable.data.segment_pose_w.torch[untouched], before[untouched], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        cable.data.segment_velocity_w.torch[selected],
                        cable.data.default_segment_velocity_w.torch[selected],
                    )

                # Advance through the captured coupled solver and verify both zero-mass anchors stay put.
                shoe_start = shoe.data.root_pose_w.torch.clone()
                anchors = [
                    cables[0].data.segment_pose_w.torch[:, -1].clone(),
                    cables[1].data.segment_pose_w.torch[:, 0].clone(),
                ]
                actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                actions[:, [6, 13]] = 1.0
                for _ in range(10):
                    observations, *_ = env.step(actions)
                    assert torch.isfinite(observations["policy"]).all()
                torch.testing.assert_close(shoe.data.root_pose_w.torch, shoe_start)
                for cable, segment, anchor in zip(cables, (-1, 0), anchors, strict=True):
                    torch.testing.assert_close(cable.data.segment_pose_w.torch[:, segment], anchor)

            for name, expected in model_before.items():
                np.testing.assert_array_equal(getattr(model, name).numpy(), expected)

            # Zero-width ranges recover the settled defaults after previously randomized episodes.
            for name in ("reset_left_arm", "reset_right_arm", "reset_shoe"):
                term = env.event_manager.get_term_cfg(name)
                term.params["position_range"] = {} if name == "reset_shoe" else (0.0, 0.0)
                env.event_manager.set_term_cfg(name, term)
            env.reset()
            torch.testing.assert_close(
                shoe.data.root_pos_w.torch, shoe.data.default_root_pose.torch[:, :3] + env.scene.env_origins
            )
            for robot in robots:
                torch.testing.assert_close(robot.data.joint_pos.torch, robot.data.default_joint_pos.torch)
            for cable in cables:
                torch.testing.assert_close(cable.data.segment_pose_w.torch, cable.data.default_segment_pose_w.torch)
        finally:
            env.close()
