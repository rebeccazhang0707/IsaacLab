# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shoelace proxy inertia and physically consistent randomized resets."""

import importlib
import warnings
from pathlib import Path
from types import ModuleType

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.shoelace import shoelace_physics
from isaaclab_tasks.contrib.shoelace.shoelace_env import ShoelaceEnv
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
    env = object.__new__(ShoelaceEnv)
    env._centerline, env._cable_radius, segment_length = shoelace_physics.load_shoelace()

    env._configure_runtime_cfg(cfg, segment_length)

    required = max(shoelace_physics.MIN_TRIANGLE_PAIRS, shoelace_physics.TRIANGLE_PAIRS_PER_ENV * num_envs)
    expected = max(required, outer_override)
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == expected
    internal_capacity = max(shoelace_physics.MIN_TRIANGLE_PAIRS, override or 0)
    assert cfg.sim.physics.solver_cfg.contact_max_triangle_pairs == internal_capacity
    assert cfg.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor == max(
        4.0 if override is not None else 0.25, required * 0.25 / internal_capacity
    )


@pytest.fixture(params=["task", "demo"])
def physics(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Exercise the self-contained task and demo physics implementations."""
    if request.param == "task":
        return shoelace_physics
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/demos/shoelace"))
    return importlib.import_module("_newton_shoelace")


@pytest.mark.parametrize("regularization", [0.0, 3.0e-7, 1.0e-6])
def test_cable_inertia_preserves_mass_and_unselected_bodies(physics: ModuleType, regularization: float) -> None:
    """Add isotropic inertia only to selected dynamic bodies and update their inverses."""
    builder = newton.ModelBuilder()
    geometric = 4.0e-11 * np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    cable = builder.add_body(mass=4.0e-5, inertia=wp.mat33(geometric))
    anchor = builder.add_body(mass=0.0, inertia=wp.mat33())
    robot = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
    masses = list(builder.body_mass)
    inverse_masses = list(builder.body_inv_mass)
    before = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3).copy()

    physics.configure_cable_inertia(builder, [cable, anchor], regularization)

    expected = before.copy()
    expected[cable] += np.eye(3, dtype=np.float32) * regularization
    actual = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3)
    np.testing.assert_array_equal(actual, expected)
    assert builder.body_mass == masses
    assert builder.body_inv_mass == inverse_masses
    for index in (cable, robot):
        inverse = np.asarray(builder.body_inv_inertia[index], dtype=np.float32).reshape(3, 3)
        np.testing.assert_allclose(actual[index] @ inverse, np.eye(3), atol=1.0e-6)
    np.testing.assert_array_equal(np.asarray(builder.body_inv_inertia[anchor]), 0.0)


@pytest.mark.parametrize("detailed", [False, True])
def test_proxy_inertia_survives_newton_validation(
    physics: ModuleType, detailed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve explicitly authored proxy inertia without a second Newton correction."""
    monkeypatch.setattr(newton, "use_coord_layout_targets", True)
    regularization = ShoelaceEnvCfg().cable_inertia_regularization
    assert regularization == physics.CABLE_INERTIA_REGULARIZATION == 1.0e-6
    builder = newton.ModelBuilder()
    builder.validate_inertia_detailed = detailed
    cable = builder.add_body(mass=4.0e-5, inertia=wp.mat33(np.diag([4.0e-11, 8.0e-11, 8.0e-11])))
    physics.configure_cable_inertia(builder, [cable], regularization)
    expected = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = builder.finalize(device="cpu")

    assert not caught, [str(warning.message) for warning in caught]
    np.testing.assert_array_equal(model.body_inertia.numpy(), expected)
    np.testing.assert_allclose(expected[cable] @ model.body_inv_inertia.numpy()[cable], np.eye(3), atol=1.0e-6)


@pytest.mark.parametrize("regularization", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_proxy_inertia_is_rejected(physics: ModuleType, regularization: float) -> None:
    """Reject negative and nonfinite regularization before editing body properties."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        physics.configure_cable_inertia(newton.ModelBuilder(), [], regularization)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The coupled MJWarp/VBD rollout requires CUDA")
@torch.inference_mode()
def test_randomized_resets_preserve_shoe_lace_alignment_and_other_environments() -> None:
    """Resample selected starts without drift, stale targets, or displaced cable anchors."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cuda:0"
    with launch_simulation(cfg, {"visualizer": None, "visualizer_explicit": True}):
        env = ShoelaceEnv(cfg)
        try:
            env.reset()
            shoe = env.scene["shoe"]
            all_ids = torch.arange(env.num_envs, device=env.device)
            selected = all_ids[::2]
            untouched = all_ids[1::2]
            robots = [env.scene[name] for name in ("robot_left", "robot_right")]
            cables = [env.scene[name] for name in ("shoelace_left", "shoelace_right")]
            arm_events = [cfg.events.reset_left_arm, cfg.events.reset_right_arm]
            model = NewtonManager.get_model()
            for world in range(env.num_envs):
                root = f"/World/envs/env_{world}/Shoe"
                body = model.body_label.index(root)
                # All shoe geometry must follow the same body, including the pinned lace span.
                for suffix in ("/Collider", "/TongueUpper/geometry/mesh", "/ShoelacePinned/geometry/mesh"):
                    shape = model.shape_label.index(root + suffix)
                    assert model.shape_body.numpy()[shape] == body

            for _ in range(3):
                shoe_before = shoe.data.root_pose_w.torch.clone()
                robot_before = [robot.data.joint_pos.torch.clone() for robot in robots]
                cable_before = [cable.data.segment_pose_w.torch.clone() for cable in cables]
                env.reset(env_ids=selected)
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
