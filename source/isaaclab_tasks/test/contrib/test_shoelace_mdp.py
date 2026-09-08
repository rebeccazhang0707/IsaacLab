# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the dual-Franka Newton shoelace task."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from tensordict import TensorDict

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import CurriculumTermCfg, SceneEntityCfg

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerProxyCfg

import isaaclab_tasks.contrib.shoelace.mdp.actions as shoelace_actions
import isaaclab_tasks.contrib.shoelace.mdp.actions_cfg as shoelace_actions_cfg
import isaaclab_tasks.contrib.shoelace.mdp.curriculums as shoelace_curriculums
import isaaclab_tasks.contrib.shoelace.mdp.events as shoelace_events
import isaaclab_tasks.contrib.shoelace.mdp.observations as shoelace_observations
import isaaclab_tasks.contrib.shoelace.mdp.rewards as shoelace_rewards
import isaaclab_tasks.contrib.shoelace.mdp.terminations as shoelace_terminations
import isaaclab_tasks.contrib.shoelace.mdp.utils as shoelace_utils
import isaaclab_tasks.contrib.shoelace.shoelace_env as shoelace_env_module
from isaaclab_tasks.contrib.shoelace.agents.models import (
    FixedObservationStatisticsMLPModel,
    FrozenBackboneFixedObservationStatisticsMLPModel,
    FrozenGripperOutputHeadFixedObservationStatisticsMLPModel,
    FrozenGripperSplitBackboneFixedObservationStatisticsMLPModel,
    FrozenPolicyFixedObservationStatisticsMLPModel,
    ShoelaceHybridActionDistribution,
    SplitBackboneFixedObservationStatisticsMLPModel,
)
from isaaclab_tasks.contrib.shoelace.agents.rsl_rl_ppo_cfg import ShoelacePPORunnerCfg
from isaaclab_tasks.contrib.shoelace.mdp.constants import (
    DYNAMIC_SEGMENT_COUNT,
    LEFT_CABLE_SEGMENT_COUNT,
    PINNED_FIRST,
    PINNED_LAST,
    REFERENCE_PULL_DIRECTIONS,
    RIGHT_CABLE_SEGMENT_COUNT,
    SHOELACE_SEGMENT_COUNT,
    TAIL_REGIONS,
)
from isaaclab_tasks.contrib.shoelace.mdp.utils import potential, pull_directions, task_state
from isaaclab_tasks.contrib.shoelace.shoelace_env import ShoelaceEnv
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg


def test_shoelace_task_uses_simplified_shoe_collider():
    """The task must share the demo's reduced collision-only shoe mesh."""
    collider_asset = shoelace_env_module.SHOELACE_COLLIDER_ASSET
    stage = Usd.Stage.Open(str(collider_asset))
    collider_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Collider"))

    assert collider_asset.name == "collider_simplified.usd"
    assert len(collider_mesh.GetFaceVertexCountsAttr().Get()) == 8000


def test_shoelace_task_uses_finite_gravity_settled_reset_poses():
    """The cable reset asset must contain one valid pose for every dynamic segment."""
    left_pose, right_pose = shoelace_env_module._load_settled_segment_poses(
        shoelace_env_module.SHOELACE_SETTLED_STATE_ASSET
    )

    assert left_pose.shape == (LEFT_CABLE_SEGMENT_COUNT, 7)
    assert right_pose.shape == (RIGHT_CABLE_SEGMENT_COUNT, 7)
    assert np.isfinite(left_pose).all()
    assert np.isfinite(right_pose).all()
    np.testing.assert_allclose(np.linalg.norm(left_pose[:, 3:], axis=1), 1.0, atol=2.0e-5)
    np.testing.assert_allclose(np.linalg.norm(right_pose[:, 3:], axis=1), 1.0, atol=2.0e-5)


def test_tail_bend_weights_leave_the_bow_at_baseline_stiffness():
    """Only the physical free ends should receive the tail material blend."""
    segment_length = 0.003
    joint_count = 30
    start_weights = shoelace_env_module._tail_joint_blend_weights(joint_count, segment_length, True)
    end_weights = shoelace_env_module._tail_joint_blend_weights(joint_count, segment_length, False)

    np.testing.assert_allclose(end_weights, start_weights[::-1])
    assert np.all(np.diff(start_weights) <= 0.0)
    assert np.all(start_weights[:16] == 1.0)
    assert np.all(start_weights[21:] == 0.0)
    assert np.any((start_weights > 0.0) & (start_weights < 1.0))


def test_shoe_collider_spawner_authors_collision_on_mesh(monkeypatch):
    """The collider mesh must remain physics geometry without visual-shape import."""
    stage = Usd.Stage.CreateInMemory()
    root_prim = stage.DefinePrim("/World/Shoe", "Xform")
    collider_prim = UsdGeom.Mesh.Define(stage, "/World/Shoe/Collider").GetPrim()
    monkeypatch.setattr(shoelace_env_module, "spawn_from_usd", lambda *args, **kwargs: root_prim)

    result = shoelace_env_module._spawn_collision_mesh_usd("/World/Shoe", object())

    assert result == root_prim
    assert UsdPhysics.CollisionAPI(collider_prim)
    assert UsdPhysics.CollisionAPI(collider_prim).GetCollisionEnabledAttr().Get()


def test_pinned_shoelace_spawner_authors_one_visible_collision_mesh(monkeypatch):
    """The fixed shoelace span must use the same mesh for rendering and collision."""
    stage = Usd.Stage.CreateInMemory()
    collision_calls: list[tuple[str, bool]] = []
    centerline = np.column_stack((np.zeros(361), np.zeros(361), np.linspace(0.0, 0.45, 361)))
    monkeypatch.setattr(
        shoelace_env_module.sim_utils,
        "create_prim",
        lambda prim_path, *_args, **_kwargs: stage.DefinePrim(prim_path, "Xform"),
    )
    monkeypatch.setattr(shoelace_env_module.sim_utils, "get_current_stage", lambda: stage)
    monkeypatch.setattr(
        shoelace_env_module.sim_utils,
        "apply_collision_properties",
        lambda prim_path, _fragments, create_if_missing=False: (
            collision_calls.append((prim_path, create_if_missing)) or True
        ),
    )

    cfg = shoelace_env_module._pinned_shoelace_spawner(centerline, 0.0015)
    root = cfg.func.__wrapped__("/World/ShoelacePinned", cfg)
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]

    assert [str(prim.GetPath()) for prim in meshes] == ["/World/ShoelacePinned/geometry/mesh"]
    assert collision_calls == [("/World/ShoelacePinned/geometry/mesh", True)]
    assert (
        len(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get())
        == (PINNED_LAST - PINNED_FIRST) * shoelace_env_module.PINNED_TUBE_SIDES
    )


def test_shoelace_task_uses_dual_franka_manager_contract():
    """The task exposes Cartesian arms, binary grippers, and no direct cable-force action."""
    cfg = ShoelaceEnvCfg()
    cfg.validate()

    assert isinstance(cfg, ManagerBasedRLEnvCfg)
    assert cfg.coupling_mode == "proxy"
    assert cfg.admm_iterations == 5
    assert cfg.admm_rho == pytest.approx(400.0)
    assert cfg.admm_gamma == pytest.approx(7.5e-5)
    assert cfg.admm_baumgarte == pytest.approx(0.75)
    assert cfg.admm_contact_matching == "latest"
    assert isinstance(cfg.sim.physics.solver_cfg, CouplerProxyCfg)
    assert isinstance(cfg.actions.left_arm, DifferentialInverseKinematicsActionCfg)
    assert isinstance(cfg.actions.right_arm, DifferentialInverseKinematicsActionCfg)
    assert isinstance(cfg.actions.left_arm, shoelace_actions_cfg.EMADifferentialInverseKinematicsActionCfg)
    assert isinstance(cfg.actions.right_arm, shoelace_actions_cfg.EMADifferentialInverseKinematicsActionCfg)
    assert cfg.actions.left_arm.alpha == pytest.approx(0.75)
    assert cfg.actions.right_arm.alpha == pytest.approx(0.75)
    assert cfg.actions.left_arm.warmup_steps == 3
    assert cfg.actions.right_arm.warmup_steps == 3
    assert cfg.actions.left_arm.warmup_steps_by_curriculum_level == (3,) * 81 + (0,) * 16
    assert cfg.actions.right_arm.warmup_steps_by_curriculum_level == (3,) * 81 + (0,) * 16
    assert isinstance(cfg.actions.left_gripper, BinaryJointPositionActionCfg)
    assert isinstance(cfg.actions.right_gripper, BinaryJointPositionActionCfg)
    assert not hasattr(cfg.actions, "force")
    assert cfg.observations.policy.last_action.func is shoelace_observations.filtered_last_action
    for robot_cfg in (cfg.scene.robot_left, cfg.scene.robot_right):
        hand_drive = robot_cfg.actuators["panda_hand"]
        assert hand_drive.stiffness == pytest.approx(6000.0)
        assert hand_drive.damping == pytest.approx(60.0)
    assert cfg.events.reset_shoelace.func is shoelace_events.ResetShoelaceCurriculum
    assert cfg.scene.env_spacing == pytest.approx(0.25)
    assert cfg.curriculum.pull_to_grasp.func is shoelace_curriculums.PullToGraspCurriculum
    assert cfg.curriculum.pull_to_grasp.params["level_count"] == 97
    assert cfg.curriculum.pull_to_grasp.params["promotion_success_rate"] == pytest.approx(0.5)
    assert "approach_level_count" not in cfg.curriculum.pull_to_grasp.params
    assert "grasp_assist_strengths" not in cfg.curriculum.pull_to_grasp.params
    assert cfg.curriculum.pull_to_grasp.params["current_level_fraction"] == pytest.approx(0.5)
    assert cfg.curriculum.pull_to_grasp.params["current_level_fraction_schedule"] == pytest.approx((0.2, 0.35, 0.5))
    assert cfg.curriculum.pull_to_grasp.params["terminal_level_fraction"] == pytest.approx(1.0)
    assert cfg.curriculum.pull_to_grasp.params["terminal_level_fraction_schedule"] == pytest.approx(
        (0.2, 0.35, 0.5, 0.75, 1.0)
    )
    assert cfg.curriculum.pull_to_grasp.params["fraction_increase_success_rate"] == pytest.approx(0.5)
    assert cfg.curriculum.pull_to_grasp.params["fraction_backoff_success_rate"] == pytest.approx(0.1)
    assert cfg.curriculum.pull_to_grasp.params["promotion_window_count"] == 1
    assert cfg.curriculum.pull_to_grasp.params["replay_level_weights"] == pytest.approx((0.5, 0.3, 0.2))
    assert cfg.curriculum.pull_to_grasp.params["initial_level"] == 0
    assert cfg.events.reset_shoelace.params["gripper_open_phase_fraction"] == pytest.approx(0.4)
    assert cfg.events.reset_shoelace.params["approach_phase_exponent"] == pytest.approx(2.0)
    assert cfg.events.reset_shoelace.params["grasp_joint_positions"][0] == pytest.approx(
        (0.389244, 0.011088, -0.521930, -2.569779, 1.149809, 2.555253, -0.150486)
    )
    assert cfg.events.reset_shoelace.params["grasp_joint_positions"][1] == pytest.approx(
        (-0.509291, -0.021903, 0.501640, -2.588014, -1.197501, 2.624648, 1.603688)
    )
    arm_states = cfg.events.reset_shoelace.params["arm_joint_positions_by_level"]
    assert tuple(len(states) for states in arm_states) == (97, 97)
    assert arm_states[1][67] == pytest.approx(
        (-0.509291, -0.021903, 0.501640, -2.588014, -1.197501, 2.624648, 1.603688)
    )
    assert arm_states[1][68] == pytest.approx(
        (-0.508908238594, 0.001041418021, 0.502844791584, -2.566197815865, -1.205718847, 2.614505814911, 1.608019054752)
    )
    assert arm_states[1][77] == pytest.approx(arm_states[1][68])
    assert arm_states[1][78] == pytest.approx(
        (
            -0.508039408424,
            0.004605980527,
            0.503710989481,
            -2.561184597482,
            -1.203628728833,
            2.611603975973,
            1.606203600583,
        )
    )
    assert arm_states[1][79] == pytest.approx(
        (
            -0.507449049679,
            0.019704037585,
            0.504276442333,
            -2.546659900388,
            -1.208664909196,
            2.604852518599,
            1.608572404884,
        )
    )
    assert arm_states[1][80] == pytest.approx(arm_states[1][79])
    assert arm_states[1][81] == pytest.approx(
        (
            -0.507592004576,
            0.017412899251,
            0.504158130358,
            -2.549777760137,
            -1.210331670910,
            2.606632777565,
            1.610694055455,
        )
    )
    assert arm_states[1][82] == pytest.approx(
        (
            -0.507656984074,
            0.016371472736,
            0.504104352187,
            -2.551194969114,
            -1.211089289871,
            2.607441986187,
            1.611658442079,
        )
    )
    assert arm_states[1][83] == pytest.approx(
        (
            -0.507669979974,
            0.016163187433,
            0.504093596553,
            -2.551478410909,
            -1.211240813663,
            2.607603827911,
            1.611851319403,
        )
    )
    assert arm_states[1][84] == pytest.approx(
        (
            -0.507681936202,
            0.015971564954,
            0.504083701370,
            -2.551739177361,
            -1.211380215552,
            2.607752722297,
            1.612028766542,
        )
    )
    assert arm_states[1][85] == pytest.approx(
        (
            -0.507690253578,
            0.015838262360,
            0.504076817764,
            -2.551920580110,
            -1.211477190779,
            2.607856301001,
            1.612152208029,
        )
    )
    assert arm_states[1][86] == pytest.approx(
        (
            -0.507698570953,
            0.015704959766,
            0.504069934158,
            -2.552101982859,
            -1.211574166006,
            2.607959879704,
            1.612275649517,
        )
    )
    approach_pose_1 = (
        (0.383982, -0.016711, -0.526566, -2.592917, 1.140252, 2.562327, -0.165460),
        (-0.510776, -0.033617, 0.501523, -2.619221, -1.247455, 2.646284, 1.657949),
    )
    bridge_fractions = ((0.125, 0.0), (0.25, 0.0), (0.25, 0.01), (0.25, 0.0125), (0.5, 0.0125))
    for level, fractions in enumerate(bridge_fractions, start=87):
        for arm_index, fraction in enumerate(fractions):
            expected = tuple(
                start + fraction * (end - start)
                for start, end in zip(arm_states[arm_index][86], approach_pose_1[arm_index], strict=True)
            )
            assert arm_states[arm_index][level] == pytest.approx(expected)
    assert arm_states[0][92] == pytest.approx(approach_pose_1[0])
    assert arm_states[1][92] == pytest.approx(approach_pose_1[1])
    assert arm_states[0][93] == pytest.approx(
        (0.374299, -0.059333, -0.531376, -2.610756, 1.098843, 2.563040, -0.152894)
    )
    assert arm_states[1][95] == pytest.approx(
        (-0.439454, -0.223472, 0.486224, -2.694287, -1.023344, 2.657568, 1.607431)
    )
    assert cfg.events.reset_shoelace.params["gripper_joint_positions_by_level"] == pytest.approx(
        (
            0.002,
            0.0025,
            0.0029375,
            0.003,
            0.0035,
            0.00353125,
            0.00353515625,
            0.0035390625,
            0.00354296875,
            0.003546875,
            0.00355078125,
            0.0035546875,
            0.00355859375,
            0.0035625,
            0.00356640625,
            0.0035703125,
            0.00357421875,
            0.003578125,
            0.00358203125,
            0.0035859375,
            0.00358984375,
            0.00359375,
            0.00359765625,
            0.0036015625,
            0.00360546875,
            0.003609375,
            0.00361328125,
            0.0036171875,
            0.00362109375,
            0.003625,
            0.00371875,
            0.00375,
            0.0038125,
            0.003875,
            0.0039375,
            0.0039765625,
            0.004,
            0.00400390625,
            0.0040078125,
            0.00401171875,
            0.004015625,
            0.004017578125,
            0.0040185546875,
            0.00401953125,
            0.0040205078125,
            0.004021484375,
            0.0040234375,
            0.0040244140625,
            0.004025390625,
            0.0040263671875,
            0.00402734375,
            0.00403125,
            0.004046875,
            0.0040580357142857145,
            0.004069196428571429,
            0.004080357142857143,
            0.004091517857142857,
            0.004102678571428571,
            0.004113839285714286,
            0.004125,
            0.004142857142857143,
            0.004160714285714286,
            0.004178571428571428,
            0.004196428571428571,
            0.004214285714285714,
            0.004232142857142857,
            0.00425,
            0.00428125,
            0.0043125,
            0.0043203125,
            0.004328125,
            0.005,
            0.0050625,
            0.005125,
            0.00515625,
            0.00516015625,
            0.0051640625,
            0.005166015625,
            0.006,
            0.008,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        )
    )

    assert cfg.events.reset_shoelace.params["closed_position"] == pytest.approx(0.002)
    assert cfg.actions.left_arm.scale == pytest.approx((0.005, 0.005, 0.005, 0.01, 0.01, 0.01))
    assert cfg.actions.left_arm.body_offset.pos == pytest.approx((0.0, 0.0, 0.1034))
    assert cfg.actions.left_gripper.open_command_expr["panda_finger_joint1"] == pytest.approx(0.01)
    assert cfg.actions.left_gripper.close_command_expr["panda_finger_joint1"] == pytest.approx(0.0015)
    assert cfg.actions.right_gripper.open_command_expr["panda_finger_joint1"] == pytest.approx(0.01)
    assert cfg.actions.right_gripper.close_command_expr["panda_finger_joint1"] == pytest.approx(0.0015)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.01)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.01)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_joint4"] == pytest.approx(-2.681384)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_joint5"] == pytest.approx(0.871062)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_joint6"] == pytest.approx(2.543250)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_joint4"] == pytest.approx(-2.726323)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_joint5"] == pytest.approx(-0.930591)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_joint6"] == pytest.approx(2.645580)
    assert not hasattr(cfg.terminations, "left_joint_velocity")
    assert not hasattr(cfg.terminations, "right_joint_velocity")
    assert cfg.rewards.failure.params["term_keys"] == [
        "unsafe",
        "lost_grasp",
        "missed_grasp",
        "insufficient_separation",
        "time_out",
    ]
    assert cfg.rewards.failure.params["exclude_term_keys"] == ["success"]
    assert cfg.rewards.failure.params["include_time_outs"] is True
    assert cfg.is_finite_horizon is True
    assert cfg.decimation == 4
    assert cfg.episode_length_s == pytest.approx(20.0)
    assert cfg.sim.physics.num_substeps == 5
    assert cfg.sim.physics.collision_decimation == 2
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.iterations == 20
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.rigid_body_contact_buffer_size == 256
    assert cfg.sim.physics.solver_cfg.proxies[0].collide_interval == 2
    assert cfg.triangle_pairs_per_env == 8192
    assert cfg.grasp_assist_release_distance == pytest.approx(
        cfg.terminations.lost_grasp.params["maximum_grasp_distance"]
    )
    assert cfg.grasp_assist_acquisition_distance == pytest.approx(0.012)
    assert cfg.grasp_assist_release_distance == pytest.approx(0.02)
    assert cfg.grasp_assist_acquisition_closed_separation == pytest.approx(0.0805)
    assert cfg.grasp_assist_release_open_separation == pytest.approx(0.081)
    assert cfg.grasp_assist_enabled is False
    assert cfg.grasp_assist_maximum_force == pytest.approx(2.0)
    assert SHOELACE_SEGMENT_COUNT == 360
    assert (PINNED_FIRST, PINNED_LAST) == (78, 281)
    assert (LEFT_CABLE_SEGMENT_COUNT, RIGHT_CABLE_SEGMENT_COUNT) == (79, 79)
    assert DYNAMIC_SEGMENT_COUNT == 158
    assert TAIL_REGIONS == ((357, 360), (0, 3))
    assert cfg.scene.tongue_upper.spawn.size == pytest.approx((0.05, 0.055, 0.006))
    assert cfg.scene.tongue_upper.init_state.pos == pytest.approx((-0.008, 0.02, 0.10))
    assert hasattr(cfg.scene, "shoelace_left")
    assert hasattr(cfg.scene, "shoelace_pinned_visual")
    assert cfg.scene.shoelace_pinned_visual.prim_path == "{ENV_REGEX_NS}/ShoelacePinned"
    assert hasattr(cfg.scene, "shoelace_right")
    assert not hasattr(cfg.scene, "shoelace")
    assert not hasattr(cfg.observations.policy, "reference_pull_directions")
    assert not hasattr(cfg.observations.policy, "tail_separation")
    assert not hasattr(cfg.observations.policy, "episode_phase")
    assert not hasattr(cfg.observations.policy, "inferred_grasp_state")
    assert not hasattr(cfg.observations.policy, "throat_density")
    assert hasattr(cfg.observations.privileged, "throat_density")
    assert cfg.rewards.dense_task.func is shoelace_rewards.reset_relative_dense_reward
    assert cfg.rewards.dense_task.weight == pytest.approx(10.0)
    assert cfg.rewards.dense_task.params["minimum_finger_position"] == pytest.approx(0.0003)
    assert cfg.rewards.dense_task.params["maximum_finger_position"] == pytest.approx(0.0025)
    assert cfg.rewards.dense_task.params["maximum_progress_rate"] == pytest.approx(3.0)
    assert cfg.rewards.dense_task.params["soft_min_temperature"] == pytest.approx(0.05)
    assert cfg.rewards.dense_task.params["soft_min_weight"] == pytest.approx(0.75)
    assert cfg.rewards.dense_task.params["confirmation_steps"] == 1
    assert cfg.rewards.dense_task.params["pull_weight"] == pytest.approx(0.25)
    assert cfg.rewards.grasp_acquisition.func is shoelace_rewards.bilateral_grasp_acquisition_event
    assert cfg.rewards.grasp_acquisition.weight == pytest.approx(10.0)
    assert cfg.rewards.grasp_acquisition.params["maximum_grasp_distance"] == pytest.approx(0.012)
    assert cfg.rewards.grasp_acquisition.params["minimum_finger_position"] == pytest.approx(0.0003)
    assert cfg.rewards.grasp_acquisition.params["maximum_finger_position"] == pytest.approx(0.0025)
    assert cfg.rewards.grasp_acquisition.params["confirmation_steps"] == 1
    assert cfg.terminations.success.params["maximum_finger_position"] == pytest.approx(0.0035)
    assert cfg.terminations.lost_grasp.params["maximum_finger_position"] == pytest.approx(0.0025)
    assert cfg.terminations.lost_grasp.params["release_confirmation_steps"] == 6
    assert cfg.terminations.missed_grasp.func is shoelace_terminations.missed_grasp_acquisition
    assert cfg.terminations.missed_grasp.params["deadline_steps"] == 8
    assert cfg.terminations.insufficient_separation.func is shoelace_terminations.insufficient_separation_progress
    assert cfg.terminations.insufficient_separation.params["minimum_progress_fraction"] == pytest.approx(0.5)
    assert cfg.terminations.insufficient_separation.params["deadline_steps"] == 64
    assert not hasattr(cfg.rewards, "eef_tracking")
    assert not hasattr(cfg.rewards, "approach_progress")
    assert not hasattr(cfg.rewards, "coordination")
    assert not hasattr(cfg.rewards, "remaining_grasp")
    assert not hasattr(cfg.rewards, "progress")
    assert not hasattr(cfg.rewards, "grasp_retention")
    assert not hasattr(cfg.rewards, "premature_close")
    assert cfg.rewards.success.func is shoelace_rewards.termination_event_reward
    assert cfg.rewards.success.weight == pytest.approx(60.0)
    assert cfg.rewards.failure.func is shoelace_rewards.termination_event_reward
    assert cfg.rewards.failure.weight == pytest.approx(-12.0)
    assert cfg.rewards.action_rate.weight == pytest.approx(-0.01)
    assert cfg.rewards.left_joint_velocity.func is shoelace_rewards.finite_joint_vel_l2
    assert cfg.rewards.left_joint_velocity.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.left_joint_velocity.params["maximum_penalty"] == pytest.approx(100.0)
    assert cfg.rewards.right_joint_velocity.func is shoelace_rewards.finite_joint_vel_l2
    assert cfg.rewards.right_joint_velocity.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.right_joint_velocity.params["maximum_penalty"] == pytest.approx(100.0)


def test_shoelace_arm_action_filters_after_first_post_reset_command(monkeypatch):
    """With zero warm-up, the EMA must leave each reset's first command unchanged."""
    processed_calls: list[torch.Tensor] = []

    def process_base(term, actions):
        term._raw_actions.copy_(actions)
        term._processed_actions.copy_(2.0 * actions)
        processed_calls.append(actions.clone())

    def reset_base(term, env_ids):
        term._raw_actions[env_ids] = 0.0

    monkeypatch.setattr(DifferentialInverseKinematicsAction, "process_actions", process_base)
    monkeypatch.setattr(DifferentialInverseKinematicsAction, "reset", reset_base)
    action = object.__new__(shoelace_actions.EMADifferentialInverseKinematicsAction)
    action._alpha = 0.75
    action._warmup_steps = 0
    action._warmup_steps_by_level = None
    action._raw_actions = torch.zeros(3, 2)
    action._processed_actions = torch.zeros(3, 2)
    action._filtered_actions = torch.zeros(3, 2)
    action._history_valid = torch.zeros(3, dtype=torch.bool)
    action._steps_since_reset = torch.zeros(3, dtype=torch.int64)
    action._warmup_steps_by_env = torch.zeros(3, dtype=torch.int64)

    first = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    second = torch.tensor([[2.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    action.process_actions(first)
    torch.testing.assert_close(processed_calls[-1], first)
    torch.testing.assert_close(action.raw_actions, first)
    action.process_actions(second)
    expected_second = 0.25 * first + 0.75 * second
    torch.testing.assert_close(processed_calls[-1], expected_second)
    torch.testing.assert_close(action.raw_actions, second)

    action.reset(torch.tensor([1]))
    third = torch.tensor([[4.0, 2.0], [9.0, 8.0], [2.0, 0.0]])
    expected_third = 0.25 * expected_second + 0.75 * third
    expected_third[1] = third[1]
    action.process_actions(third)
    torch.testing.assert_close(processed_calls[-1], expected_third)
    torch.testing.assert_close(action.filtered_actions, expected_third)
    torch.testing.assert_close(action.raw_actions, third)


def test_shoelace_arm_action_holds_commands_during_post_reset_warmup(monkeypatch):
    """The configured warm-up must preserve raw actions while holding executed arm commands."""
    processed_calls = []

    def process_base(term, actions):
        term._raw_actions.copy_(actions)
        term._processed_actions.copy_(2.0 * actions)
        processed_calls.append(actions.clone())

    monkeypatch.setattr(DifferentialInverseKinematicsAction, "process_actions", process_base)

    action = object.__new__(shoelace_actions.EMADifferentialInverseKinematicsAction)
    action._alpha = 0.75
    action._warmup_steps = 3
    action._raw_actions = torch.zeros(2, 2)
    action._processed_actions = torch.zeros(2, 2)
    action._filtered_actions = torch.zeros(2, 2)
    action._history_valid = torch.zeros(2, dtype=torch.bool)
    action._steps_since_reset = torch.zeros(2, dtype=torch.int64)
    action._warmup_steps_by_env = torch.full((2,), 3, dtype=torch.int64)

    command = torch.ones(2, 2)
    for _ in range(3):
        action.process_actions(command)
        torch.testing.assert_close(action.filtered_actions, torch.zeros_like(command))
        torch.testing.assert_close(action.raw_actions, command)

    action.process_actions(command)

    torch.testing.assert_close(action.filtered_actions, torch.full_like(command, 0.75))
    torch.testing.assert_close(processed_calls[-1], torch.full_like(command, 0.75))


def test_shoelace_arm_action_selects_warmup_from_curriculum_level(monkeypatch):
    """Each reset environment must use the warm-up duration for its sampled level."""

    def reset_base(term, env_ids):
        term._raw_actions[env_ids] = 0.0

    monkeypatch.setattr(DifferentialInverseKinematicsAction, "reset", reset_base)
    levels = torch.tensor((80, 81, 96), dtype=torch.long)
    curriculum_term = SimpleNamespace(levels=levels)
    curriculum_cfg = SimpleNamespace(func=curriculum_term)
    action = object.__new__(shoelace_actions.EMADifferentialInverseKinematicsAction)
    action.cfg = SimpleNamespace(warmup_curriculum_term_name="pull_to_grasp")
    action._env = SimpleNamespace(curriculum_manager=SimpleNamespace(cfg=SimpleNamespace(pull_to_grasp=curriculum_cfg)))
    action._warmup_steps = 3
    action._warmup_steps_by_level = torch.tensor((3,) * 81 + (0,) * 16, dtype=torch.int64)
    action._warmup_steps_by_env = torch.full((3,), 3, dtype=torch.int64)
    action._raw_actions = torch.ones(3, 2)
    action._processed_actions = torch.ones(3, 2)
    action._filtered_actions = torch.ones(3, 2)
    action._history_valid = torch.ones(3, dtype=torch.bool)
    action._steps_since_reset = torch.ones(3, dtype=torch.int64)

    action.reset(torch.tensor((0, 1, 2)))

    torch.testing.assert_close(action._warmup_steps_by_env, torch.tensor((3, 0, 0), dtype=torch.int64))
    torch.testing.assert_close(action._steps_since_reset, torch.zeros(3, dtype=torch.int64))
    torch.testing.assert_close(action.filtered_actions, torch.zeros(3, 2))


def test_filtered_last_action_matches_executed_arm_command_order():
    """Action history must expose filtered arms and raw binary grippers in policy order."""
    terms = {
        "left_arm": SimpleNamespace(filtered_actions=torch.full((2, 6), 1.0), raw_actions=torch.full((2, 6), -1.0)),
        "left_gripper": SimpleNamespace(raw_actions=torch.full((2, 1), 2.0)),
        "right_arm": SimpleNamespace(filtered_actions=torch.full((2, 6), 3.0), raw_actions=torch.full((2, 6), -3.0)),
        "right_gripper": SimpleNamespace(raw_actions=torch.full((2, 1), 4.0)),
    }
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=terms.__getitem__))

    observed = shoelace_observations.filtered_last_action(env)

    expected = torch.cat(
        (
            terms["left_arm"].filtered_actions,
            terms["left_gripper"].raw_actions,
            terms["right_arm"].filtered_actions,
            terms["right_gripper"].raw_actions,
        ),
        dim=-1,
    )
    assert observed.shape == (2, 14)
    torch.testing.assert_close(observed, expected)


def test_shoelace_play_mode_uses_complete_authored_reset():
    """Evaluation must use the full approach-and-grasp task rather than an easy curriculum level."""
    cfg = ShoelaceEnvCfg()

    cfg.play_mode()

    assert cfg.curriculum.pull_to_grasp.params["initial_level"] == (
        cfg.curriculum.pull_to_grasp.params["level_count"] - 1
    )
    assert cfg.curriculum.pull_to_grasp.params["current_level_fraction"] == pytest.approx(1.0)
    assert cfg.curriculum.pull_to_grasp.params["current_level_fraction_schedule"] == pytest.approx((1.0,))
    assert cfg.curriculum.pull_to_grasp.params["terminal_level_fraction"] == pytest.approx(1.0)
    assert cfg.curriculum.pull_to_grasp.params["terminal_level_fraction_schedule"] == pytest.approx((1.0,))


def test_grasp_assist_requires_close_geometry_and_releases_at_bounded_distance():
    """The assistant must not acquire remotely and must release at its bounded retention distance."""

    def body_poses(tail_distance: float, finger_separation: float = 0.01) -> wp.array:
        positions = (
            (0.0, 0.0, 0.0),
            (0.0, -0.5 * finger_separation, 0.0),
            (0.0, 0.5 * finger_separation, 0.0),
            (tail_distance, 0.0, 0.0),
            (tail_distance, 0.0, 0.0),
            (tail_distance, 0.0, 0.0),
        )
        return wp.array(
            [wp.transformf(position, (0.0, 0.0, 0.0, 1.0)) for position in positions],
            dtype=wp.transformf,
            device="cpu",
        )

    body_qd = wp.zeros(6, dtype=wp.spatial_vectorf, device="cpu")
    hand_ids = wp.array([0], dtype=wp.int32, device="cpu")
    finger_ids = wp.array([1, 2], dtype=wp.int32, device="cpu")
    tail_ids = wp.array([3, 4, 5], dtype=wp.int32, device="cpu")
    active = wp.zeros(1, dtype=wp.int32, device="cpu")
    local_anchors = wp.zeros(1, dtype=wp.vec3f, device="cpu")

    def apply(tail_distance: float, finger_separation: float = 0.01) -> np.ndarray:
        body_f = wp.zeros(6, dtype=wp.spatial_vectorf, device="cpu")
        wp.launch(
            shoelace_env_module._apply_grasp_assist_kernel,
            dim=1,
            inputs=[
                body_poses(tail_distance, finger_separation),
                body_qd,
                body_f,
                hand_ids,
                finger_ids,
                tail_ids,
                active,
                local_anchors,
                wp.vec3f(0.0),
                0.018,
                0.035,
                0.0805,
                0.081,
                20.0,
                0.0,
                2.0,
            ],
            device="cpu",
        )
        return body_f.numpy()

    apply(0.0181)
    assert active.numpy().tolist() == [0]
    apply(0.0, finger_separation=0.0806)
    assert active.numpy().tolist() == [0]

    apply(0.0, finger_separation=0.08)
    opening_forces = apply(0.0115, finger_separation=0.08)
    opening_scale = (0.0805 - 0.08) / (0.25 * 0.0805)
    expected_opening_force = 20.0 * -0.0115 * opening_scale / 3.0
    np.testing.assert_allclose(opening_forces[3:, 0], expected_opening_force, atol=1.0e-6)
    apply(0.035, finger_separation=0.08)

    apply(0.0)
    assert active.numpy().tolist() == [1]
    forces = apply(0.0115)
    expected_tail_force = 20.0 * -0.0115 / 3.0
    np.testing.assert_allclose(forces[3:, 0], expected_tail_force, atol=1.0e-6)
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1.0e-7)

    released_forces = apply(0.035)
    assert active.numpy().tolist() == [0]
    np.testing.assert_allclose(released_forces, 0.0, atol=1.0e-7)


@pytest.mark.parametrize(
    ("enabled", "maximum_force", "expected_assist_scale"),
    [
        (True, 2.0, 1.0),
        (False, 2.0, 0.0),
        (True, 0.0, 0.0),
    ],
)
def test_pull_to_grasp_curriculum_uses_same_levels_with_or_without_assistance(
    enabled: bool, maximum_force: float, expected_assist_scale: float
):
    """Grasp assistance must not change reset difficulty or the number of levels."""

    class Physics:
        grasp_assist_enabled = enabled and maximum_force > 0.0
        grasp_assist_maximum_force = maximum_force

        def set_grasp_assist_scale(self, env_ids: torch.Tensor, scale: torch.Tensor) -> None:
            pytest.fail("The reset curriculum must not control grasp-assist strength.")

    env = SimpleNamespace(num_envs=4, device="cpu", common_step_counter=0, _physics=Physics())
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.7,
        "minimum_episodes": 4,
        "current_level_fraction": 1.0,
        "initial_level": 10,
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    state = term(env, slice(None), **params)

    torch.testing.assert_close(term.levels, torch.full((4,), 10, dtype=torch.long))
    torch.testing.assert_close(term.difficulty, torch.ones(4))
    torch.testing.assert_close(term.grasp_assist_scale, torch.full((4,), expected_assist_scale))
    assert state["full_task_fraction"] == pytest.approx(1.0)
    assert state["mean_grasp_assist_scale"] == pytest.approx(expected_assist_scale)
    assert state["unassisted_fraction"] == pytest.approx(float(expected_assist_scale == 0.0))


@pytest.mark.parametrize(
    ("enabled", "maximum_force", "expected_enabled"),
    [
        (True, 2.0, True),
        (False, 2.0, False),
        (True, 0.0, False),
    ],
)
def test_grasp_assist_flag_controls_spring_latch_lifecycle(
    monkeypatch, enabled: bool, maximum_force: float, expected_enabled: bool
):
    """Disabled assistance must not allocate latch state or register its force callback."""

    class Buffer:
        def assign(self, value) -> None:
            self.value = np.asarray(value)

    model = SimpleNamespace(
        joint_count=1,
        device="cpu",
        vbd=SimpleNamespace(dahl_eps_max=Buffer(), dahl_tau=Buffer()),
    )
    callbacks = []
    monkeypatch.setattr(shoelace_env_module.NewtonManager, "get_model", staticmethod(lambda: model))
    monkeypatch.setattr(
        shoelace_env_module.NewtonManager,
        "register_state_force_callback",
        staticmethod(lambda callback: callbacks.append(callback)),
    )
    physics = shoelace_env_module._ShoelacePhysics(
        centerline=np.zeros((2, 3)),
        cable_radius=0.001,
        num_envs=1,
        joint_stiffness_scale=1.0,
        grasp_assist_enabled=enabled,
        grasp_assist_acquisition_distance=0.018,
        grasp_assist_release_distance=0.035,
        grasp_assist_acquisition_closed_separation=0.0805,
        grasp_assist_release_open_separation=0.081,
        grasp_assist_stiffness=20.0,
        grasp_assist_damping=0.04,
        grasp_assist_maximum_force=maximum_force,
    )
    physics.cable_joints = [[0]]
    physics._grasp_assist_hand_ids = [0, 1]
    physics._grasp_assist_finger_ids = [0, 1, 2, 3]
    physics._grasp_assist_tail_ids = [0, 1, 2, 3, 4, 5]

    physics._configure_model(None)

    assert physics.grasp_assist_enabled is expected_enabled
    assert len(callbacks) == int(expected_enabled)
    if expected_enabled:
        assert physics.grasp_assist_active is not None
        assert physics.grasp_assist_active.shape == (1, 2)
    else:
        assert physics.grasp_assist_active is None


def test_pull_to_grasp_curriculum_accepts_legacy_annealing_config_as_eleven_levels():
    """Historical annealing configs must collapse to their eleven reset-difficulty levels."""
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=0,
        _physics=SimpleNamespace(grasp_assist_enabled=True, grasp_assist_maximum_force=2.0),
    )
    params = {
        "level_count": 15,
        "approach_level_count": 11,
        "grasp_assist_strengths": (1.0, 0.75, 0.5, 0.25, 0.0),
        "success_term_name": "success",
        "promotion_success_rate": 0.7,
        "minimum_episodes": 4,
        "current_level_fraction": 1.0,
        "initial_level": 14,
    }

    with pytest.warns(FutureWarning, match="grasp_assist_enabled"):
        term = shoelace_curriculums.PullToGraspCurriculum(
            CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
        )

    state = term(env, slice(None), **params)

    torch.testing.assert_close(term.levels, torch.full((4,), 10, dtype=torch.long))
    torch.testing.assert_close(term.difficulty, torch.ones(4))
    assert state["current_level"] == pytest.approx(10.0)
    assert state["full_task_fraction"] == pytest.approx(1.0)


def test_pull_to_grasp_curriculum_expands_final_level_exposure():
    """The final reset level must grow beyond the replay ceiling only after successful windows."""

    class TerminationManager:
        successes = torch.ones(10, dtype=torch.bool)

        def get_term(self, name: str) -> torch.Tensor:
            assert name == "success"
            return self.successes

    env = SimpleNamespace(
        num_envs=10,
        device="cpu",
        common_step_counter=0,
        termination_manager=TerminationManager(),
    )
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.7,
        "minimum_episodes": 2,
        "current_level_fraction": 0.5,
        "current_level_fraction_schedule": (0.5,),
        "terminal_level_fraction": 1.0,
        "terminal_level_fraction_schedule": (0.2, 0.5, 1.0),
        "fraction_increase_success_rate": 0.5,
        "promotion_window_count": 1,
        "initial_level": 9,
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    term(env, slice(None), **params)
    env.common_step_counter = 1
    first_terminal_window = term(env, slice(None), **params)
    second_terminal_window = term(env, slice(None), **params)
    third_terminal_window = term(env, slice(None), **params)

    assert first_terminal_window["current_level"] == pytest.approx(10.0)
    assert first_terminal_window["current_level_fraction"] == pytest.approx(0.2)
    assert second_terminal_window["current_level_fraction"] == pytest.approx(0.5)
    assert third_terminal_window["current_level_fraction"] == pytest.approx(1.0)


def test_pull_to_grasp_curriculum_promotes_only_after_successful_window():
    """The reset level must advance from measured success rather than elapsed iterations."""

    class TerminationManager:
        successes = torch.ones(4, dtype=torch.bool)

        def get_term(self, name: str) -> torch.Tensor:
            assert name == "success"
            return self.successes

    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=0,
        termination_manager=TerminationManager(),
    )
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.75,
        "minimum_episodes": 4,
        "current_level_fraction": 1.0,
        "initial_level": 0,
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    term(env, slice(None), **params)
    env.common_step_counter = 1
    state = term(env, slice(None), **params)

    assert state["current_level"] == pytest.approx(1.0)
    torch.testing.assert_close(term.levels, torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(term.difficulty, torch.full((4,), 0.1))

    env.termination_manager.successes[:] = False
    state = term(env, slice(None), **params)
    assert state["current_level"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("initial_level", "expected_fractions"),
    [
        (5, (0.0, 0.0, 0.1, 0.15, 0.25, 0.5)),
        (2, (0.1875, 0.3125, 0.5)),
    ],
)
def test_pull_to_grasp_curriculum_replays_multiple_preceding_levels(
    initial_level: int, expected_fractions: tuple[float, ...]
):
    """Replay sampling must retain configured ratios and normalize unavailable levels."""
    env = SimpleNamespace(num_envs=40_000, device="cpu", common_step_counter=0)
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.75,
        "minimum_episodes": 128,
        "current_level_fraction": 0.5,
        "initial_level": initial_level,
        "replay_level_weights": (0.5, 0.3, 0.2),
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    term(env, slice(None), **params)
    sampled_levels = term.levels.clone()
    term(env, torch.arange(0, env.num_envs, 2), **params)

    sampled_fractions = torch.bincount(term.levels, minlength=len(expected_fractions)).float() / env.num_envs
    torch.testing.assert_close(
        sampled_fractions[: len(expected_fractions)], torch.tensor(expected_fractions), rtol=0.0, atol=0.01
    )
    torch.testing.assert_close(term.levels, sampled_levels)


def test_pull_to_grasp_curriculum_increases_exposure_before_promotion():
    """A new level must start at low exposure and require repeated mastery before promotion."""

    class TerminationManager:
        successes = torch.ones(10, dtype=torch.bool)

        def get_term(self, name: str) -> torch.Tensor:
            assert name == "success"
            return self.successes

    env = SimpleNamespace(
        num_envs=10,
        device="cpu",
        common_step_counter=0,
        termination_manager=TerminationManager(),
    )
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.7,
        "minimum_episodes": 5,
        "current_level_fraction": 0.5,
        "current_level_fraction_schedule": (0.2, 0.35, 0.5),
        "fraction_increase_success_rate": 0.5,
        "fraction_backoff_success_rate": 0.1,
        "promotion_window_count": 2,
        "initial_level": 5,
        "replay_level_weights": (0.5, 0.3, 0.2),
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    term(env, slice(None), **params)
    env.common_step_counter = 1
    first_window = term(env, slice(None), **params)
    second_window = term(env, slice(None), **params)

    assert first_window["current_level"] == pytest.approx(5.0)
    assert first_window["successful_window_count"] == pytest.approx(1.0)
    assert second_window["current_level"] == pytest.approx(6.0)
    assert second_window["current_level_fraction"] == pytest.approx(0.2)
    assert (term.levels == 6).sum().item() == 2


def test_pull_to_grasp_curriculum_backs_off_current_level_exposure():
    """Repeated low-success windows must reduce frontier exposure without demoting the level."""

    class TerminationManager:
        successes = torch.zeros(10, dtype=torch.bool)

        def get_term(self, name: str) -> torch.Tensor:
            assert name == "success"
            return self.successes

    env = SimpleNamespace(
        num_envs=10,
        device="cpu",
        common_step_counter=0,
        termination_manager=TerminationManager(),
    )
    params = {
        "level_count": 11,
        "success_term_name": "success",
        "promotion_success_rate": 0.7,
        "minimum_episodes": 3,
        "current_level_fraction": 0.5,
        "current_level_fraction_schedule": (0.2, 0.35, 0.5),
        "fraction_increase_success_rate": 0.5,
        "fraction_backoff_success_rate": 0.1,
        "promotion_window_count": 2,
        "initial_level": 6,
        "replay_level_weights": (0.5, 0.3, 0.2),
    }
    term = shoelace_curriculums.PullToGraspCurriculum(
        CurriculumTermCfg(func=shoelace_curriculums.PullToGraspCurriculum, params=params), env
    )

    term(env, slice(None), **params)
    env.common_step_counter = 1
    first_window = term(env, slice(None), **params)
    second_window = term(env, slice(None), **params)

    assert first_window["current_level"] == pytest.approx(6.0)
    assert first_window["current_level_fraction"] == pytest.approx(0.35)
    assert second_window["current_level"] == pytest.approx(6.0)
    assert second_window["current_level_fraction"] == pytest.approx(0.2)


def test_reset_curriculum_opens_gripper_before_increasing_approach_distance():
    """Early levels must stage closure at the tail before moving the arms toward pregrasp."""
    difficulty = torch.tensor(((0.0,), (0.2,), (0.4,), (0.5,), (0.6,), (0.7,), (1.0,)))

    arm_difficulty, gripper_difficulty = shoelace_events.ResetShoelaceCurriculum._phase_difficulties(
        difficulty, gripper_open_phase_fraction=0.4, approach_phase_exponent=2.0
    )

    torch.testing.assert_close(
        arm_difficulty,
        torch.tensor(((0.0,), (0.0,), (0.0,), (1.0 / 36.0,), (1.0 / 9.0,), (0.25,), (1.0,))),
    )
    torch.testing.assert_close(
        gripper_difficulty, torch.tensor(((0.0,), (0.5,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)))
    )


def test_reset_curriculum_selects_calibrated_states_by_discrete_level():
    """Each environment must receive the measured state for its sampled curriculum level."""
    cfg = ShoelaceEnvCfg()
    left_states = cfg.events.reset_shoelace.params["arm_joint_positions_by_level"][0]
    levels = torch.tensor((13, 10, 0), dtype=torch.long)

    selected = shoelace_events.ResetShoelaceCurriculum._select_joint_positions(torch.zeros((3, 7)), left_states, levels)

    torch.testing.assert_close(selected[0], torch.tensor(left_states[13]))
    torch.testing.assert_close(selected[1], torch.tensor(left_states[10]))
    torch.testing.assert_close(selected[2], torch.tensor(left_states[0]))


def test_shoelace_coupler_solves_robot_shoe_contact_in_mjwarp():
    """The robot solver must own the shoe while VBD receives it as a cable-contact proxy."""
    solver_cfg = ShoelaceEnvCfg().sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver_cfg.entries}
    proxy = solver_cfg.proxies[0]

    assert entries["robots"].bodies == [r"/World/envs/env_[^/]+/(Robot(Left|Right)|Shoe)"]
    assert entries["shoelace"].bodies == [r"/World/envs/env_[^/]+/Shoelace(Left|Right)"]
    assert r"/World/envs/env_[^/]+/Shoe" in proxy.bodies


def _runtime_config_env() -> ShoelaceEnv:
    """Build the minimal environment state needed by runtime configuration tests."""
    env = ShoelaceEnv.__new__(ShoelaceEnv)
    env._is_closed = True
    x_coordinates = np.linspace(0.0, 0.36, SHOELACE_SEGMENT_COUNT + 1)
    env._centerline = np.column_stack((x_coordinates, np.zeros_like(x_coordinates), np.zeros_like(x_coordinates)))
    env._cable_radius = 0.001
    env._authored_mean_segment_length = 0.001
    env._model_asset = Path("model.usd")
    return env


def test_shoelace_runtime_scales_outer_and_proxy_triangle_pair_capacities():
    """Both collision pipelines must scale triangle-pair capacity with the environment count."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 4096
    env = _runtime_config_env()

    env._configure_runtime_cfg(cfg, Path("collider.usd"))

    expected_capacity = cfg.triangle_pairs_per_env * cfg.scene.num_envs
    proxy_pipeline = cfg.sim.physics.solver_cfg.proxies[0].collision_pipeline
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == expected_capacity
    assert proxy_pipeline.max_triangle_pairs == expected_capacity


def test_shoelace_runtime_selects_symmetric_admm_coupling():
    """The ADMM mode must retain ownership entries and couple their contacts symmetrically."""
    cfg = ShoelaceEnvCfg()
    cfg.coupling_mode = "admm"
    cfg.admm_iterations = 7
    cfg.admm_rho = 2.5
    cfg.admm_gamma = 0.25
    cfg.admm_baumgarte = 0.75
    cfg.admm_contact_matching = "sticky"
    cfg.scene.num_envs = 2
    env = _runtime_config_env()

    env._configure_runtime_cfg(cfg, Path("collider.usd"))

    solver_cfg = cfg.sim.physics.solver_cfg
    assert isinstance(solver_cfg, CouplerAdmmCfg)
    assert [entry.name for entry in solver_cfg.entries] == ["robots", "shoelace"]
    assert solver_cfg.contact_pairs == [("robots", "shoelace")]
    assert solver_cfg.iterations == 7
    assert solver_cfg.rho == pytest.approx(2.5)
    assert solver_cfg.gamma == pytest.approx(0.25)
    assert solver_cfg.baumgarte == pytest.approx(0.75)
    assert solver_cfg.rigid_contact_matching == "sticky"
    assert cfg.sim.physics.collision_cfg.rigid_contact_max == cfg.contacts_per_env * cfg.scene.num_envs
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == 1_000_000


def test_shoelace_runtime_rejects_unknown_coupling_mode():
    """Unknown coupling modes must fail before the Newton model is constructed."""
    cfg = ShoelaceEnvCfg()
    cfg.coupling_mode = "unknown"
    env = _runtime_config_env()

    with pytest.raises(ValueError, match="Unsupported shoelace coupling mode"):
        env._configure_runtime_cfg(cfg, Path("collider.usd"))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("admm_iterations", 0, "at least one"),
        ("admm_rho", 0.0, "finite and positive"),
        ("admm_gamma", -1.0, "finite and nonnegative"),
        ("admm_baumgarte", float("nan"), "finite and nonnegative"),
        ("admm_contact_matching", "unknown", "Unsupported ADMM rigid contact matching mode"),
    ),
)
def test_shoelace_runtime_rejects_invalid_admm_parameters(field_name, value, message):
    """Invalid ADMM tuning parameters must fail before the Newton model is constructed."""
    cfg = ShoelaceEnvCfg()
    cfg.coupling_mode = "admm"
    setattr(cfg, field_name, value)
    env = _runtime_config_env()

    with pytest.raises(ValueError, match=message):
        env._configure_runtime_cfg(cfg, Path("collider.usd"))


def test_shoelace_agent_uses_asymmetric_observations():
    """The actor excludes simulator-only state while the critic receives it."""
    assert ShoelacePPORunnerCfg().obs_groups == {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }


def test_fixed_observation_statistics_model_preserves_loaded_statistics():
    """Checkpoint consolidation must not update the loaded observation statistics."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = FixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        2,
        hidden_dims=[8],
        obs_normalization=True,
    )
    statistics = {name: value.clone() for name, value in model.obs_normalizer.state_dict().items()}

    model.update_normalization(TensorDict({"policy": torch.full((8, 3), 7.0)}, batch_size=[8]))

    for name, expected in statistics.items():
        torch.testing.assert_close(model.obs_normalizer.state_dict()[name], expected)


def test_frozen_backbone_model_trains_only_output_layer_and_distribution():
    """Consolidation must isolate the action head from shared-backbone gradient interference."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = FrozenBackboneFixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution",
            "init_std": 0.2,
        },
    )

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable_names == {"distribution.std_param", "mlp.2.weight", "mlp.2.bias"}


def test_frozen_gripper_output_head_model_updates_only_arm_rows():
    """Output-head consolidation must not move either binary gripper decision boundary."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = FrozenGripperOutputHeadFixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution",
            "init_std": 0.2,
            "learn_std": False,
        },
    )
    output_layer = model.mlp[-1]
    weight_before = output_layer.weight.detach().clone()
    bias_before = output_layer.bias.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    model(observations).sum().backward()
    optimizer.step()

    gripper_rows = torch.tensor([6, 13])
    arm_rows = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    torch.testing.assert_close(output_layer.weight[gripper_rows], weight_before[gripper_rows])
    torch.testing.assert_close(output_layer.bias[gripper_rows], bias_before[gripper_rows])
    assert torch.count_nonzero(output_layer.weight[arm_rows] - weight_before[arm_rows]) > 0
    assert torch.count_nonzero(output_layer.bias[arm_rows] - bias_before[arm_rows]) > 0
    backbone_layers = list(model.mlp.children())[:-1]
    assert all(not parameter.requires_grad for layer in backbone_layers for parameter in layer.parameters())


def test_frozen_policy_model_preserves_every_actor_parameter():
    """Critic warm-up must prevent gradients from changing any policy parameter."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = FrozenPolicyFixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution",
            "init_std": 0.2,
        },
    )

    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_split_backbone_model_isolates_arm_and_gripper_gradients():
    """Arm and gripper losses must not update the other action family's backbone."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = SplitBackboneFixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution",
            "init_std": 0.2,
        },
    )
    output = model(observations)
    output[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].sum().backward()

    assert any(parameter.grad is not None for parameter in model.mlp.arm_mlp.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.mlp.gripper_mlp.parameters()
    )

    model.zero_grad(set_to_none=True)
    model(observations)[..., [6, 13]].sum().backward()

    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.mlp.arm_mlp.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.mlp.gripper_mlp.parameters())


def test_frozen_gripper_split_backbone_model_trains_only_arm_policy():
    """Arm consolidation must preserve every parameter that controls gripper decisions."""
    observations = TensorDict({"policy": torch.randn(8, 3)}, batch_size=[8])
    model = FrozenGripperSplitBackboneFixedObservationStatisticsMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[8],
        obs_normalization=True,
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.shoelace.agents.models:ShoelaceHybridActionDistribution",
            "init_std": 0.2,
        },
    )
    output = model(observations)
    output.sum().backward()

    assert all(parameter.requires_grad for parameter in model.mlp.arm_mlp.parameters())
    assert any(parameter.grad is not None for parameter in model.mlp.arm_mlp.parameters())
    assert all(not parameter.requires_grad for parameter in model.mlp.gripper_mlp.parameters())
    assert all(parameter.grad is None for parameter in model.mlp.gripper_mlp.parameters())


def test_shoelace_hybrid_distribution_matches_binary_gripper_contract():
    """Gripper actions must be discrete while arm actions retain Gaussian exploration."""
    distribution = ShoelaceHybridActionDistribution(14, init_std=0.2, learn_std=False, gripper_logit_scale=500.0)
    mlp_output = torch.zeros((4096, 14))
    mlp_output[:, 6] = -0.001
    mlp_output[:, 13] = 0.0014

    deterministic = distribution.deterministic_output(mlp_output)
    distribution.update(mlp_output)
    sampled = distribution.sample()

    torch.testing.assert_close(deterministic, mlp_output)
    assert set(torch.unique(torch.sign(sampled[:, 6])).tolist()) == {-1.0, 1.0}
    assert set(torch.unique(torch.sign(sampled[:, 13])).tolist()) == {-1.0, 1.0}
    torch.testing.assert_close(sampled[:, 6].abs(), torch.full((4096,), 0.001))
    torch.testing.assert_close(sampled[:, 13].abs(), torch.full((4096,), 0.0014))
    assert sampled[:, :6].std().item() == pytest.approx(0.2, abs=0.01)
    assert torch.isfinite(distribution.log_prob(sampled)).all()


def test_shoelace_hybrid_distribution_scales_only_continuous_arm_actions():
    """Arm calibration must scale means and noise without changing binary gripper outputs."""
    distribution = ShoelaceHybridActionDistribution(
        14,
        init_std=0.2,
        learn_std=False,
        arm_action_scale=0.7,
        gripper_logit_scale=500.0,
    )
    mlp_output = torch.ones((2, 14))
    mlp_output[:, (6, 13)] = -0.001
    expected = mlp_output.clone()
    expected[:, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)] *= 0.7

    distribution.update(mlp_output)

    torch.testing.assert_close(distribution.deterministic_output(mlp_output), expected)
    torch.testing.assert_close(distribution.mean[:, distribution._arm_indices], expected[:, distribution._arm_indices])
    torch.testing.assert_close(
        distribution.std[:, distribution._arm_indices],
        torch.full((2, 12), 0.14),
    )
    torch.testing.assert_close(distribution.as_deterministic_output_module()(mlp_output), expected)
    scripted_output = torch.jit.script(distribution.as_deterministic_output_module())(mlp_output)
    torch.testing.assert_close(scripted_output, expected)


def test_shoelace_hybrid_distribution_kl_matches_independent_factors():
    """The reported KL must include Gaussian arms and Bernoulli grippers exactly once."""
    distribution = ShoelaceHybridActionDistribution(14, init_std=0.2, learn_std=False, gripper_logit_scale=500.0)
    old_output = torch.zeros((3, 14))
    new_output = torch.full((3, 14), 0.1)
    old_std = torch.full((3, 14), 0.2)
    new_std = torch.full((3, 14), 0.25)
    old_params = (old_output, old_std)
    new_params = (new_output, new_std)

    actual = distribution.kl_divergence(old_params, new_params)
    arm_indices = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    gripper_indices = (6, 13)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_output[..., arm_indices], old_std[..., arm_indices]),
        torch.distributions.Normal(new_output[..., arm_indices], new_std[..., arm_indices]),
    ).sum(dim=-1)
    expected += torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(logits=500.0 * old_output[..., gripper_indices]),
        torch.distributions.Bernoulli(logits=500.0 * new_output[..., gripper_indices]),
    ).sum(dim=-1)

    torch.testing.assert_close(actual, expected)


def test_shoelace_agent_saves_intermediate_checkpoints():
    """Training must expose checkpoints frequently enough for phase-metric evaluation."""
    assert ShoelacePPORunnerCfg().save_interval == 50


def test_shoelace_agent_starts_with_complete_episodes():
    """Success curricula must not count artificially shortened startup episodes."""
    assert ShoelacePPORunnerCfg().init_at_random_ep_len is False


def test_shoelace_agent_uses_persistent_gripper_sampling():
    """The sampled gripper policy must suppress rare open actions during a sustained hold."""
    assert ShoelacePPORunnerCfg().actor.distribution_cfg.gripper_logit_scale == 2.0


def test_shoelace_agent_uses_contact_stable_arm_action_scale():
    """The arm action scale must preserve marginal cable contacts during pulling."""
    assert ShoelacePPORunnerCfg().actor.distribution_cfg.arm_action_scale == 0.7


def test_reference_pull_directions_match_normalized_demo_forces():
    """Each tail reference must preserve its scripted demo force direction."""
    reference = torch.empty(1)
    directions = pull_directions(reference)
    expected = torch.tensor(REFERENCE_PULL_DIRECTIONS)
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)

    torch.testing.assert_close(directions, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(directions, dim=-1), torch.ones(2))


def test_task_state_concatenates_dual_cables_and_preserves_tail_order():
    """The two cable assets must retain the authored right-tail, left-tail control order."""

    class Scene(dict):
        pass

    left_pose = torch.zeros((1, LEFT_CABLE_SEGMENT_COUNT, 7))
    right_pose = torch.zeros((1, RIGHT_CABLE_SEGMENT_COUNT, 7))
    left_pose[0, :, 0] = torch.arange(LEFT_CABLE_SEGMENT_COUNT)
    right_pose[0, :, 0] = 100.0 + torch.arange(RIGHT_CABLE_SEGMENT_COUNT)
    left_velocity = torch.zeros((1, LEFT_CABLE_SEGMENT_COUNT, 6))
    right_velocity = torch.zeros((1, RIGHT_CABLE_SEGMENT_COUNT, 6))
    scene = Scene(
        shoelace_left=SimpleNamespace(
            data=SimpleNamespace(
                segment_pose_w=SimpleNamespace(torch=left_pose),
                segment_velocity_w=SimpleNamespace(torch=left_velocity),
            )
        ),
        shoelace_right=SimpleNamespace(
            data=SimpleNamespace(
                segment_pose_w=SimpleNamespace(torch=right_pose),
                segment_velocity_w=SimpleNamespace(torch=right_velocity),
            )
        ),
    )
    scene.env_origins = torch.zeros((1, 3))
    env = SimpleNamespace(scene=scene)

    positions, _, knot, tail_positions, _ = task_state(
        env,
        (SceneEntityCfg("shoelace_left"), SceneEntityCfg("shoelace_right")),
    )

    assert positions.shape == (1, DYNAMIC_SEGMENT_COUNT, 3)
    torch.testing.assert_close(knot, torch.tensor([[89.0, 0.0, 0.0]]))
    torch.testing.assert_close(tail_positions[0, :, 0], torch.tensor([177.0, 1.0]))


def test_phase_rewards_gate_approach_and_directional_pull(monkeypatch):
    """Approach requires an open gripper and directional pull requires an inferred grasp."""
    distances = torch.tensor([[0.0, 0.05]])
    closure = torch.tensor([[0.0, 1.0]])
    grasped = torch.tensor([[True, False]])
    directions = pull_directions(torch.empty(1))
    tail_velocities = 0.04 * directions.unsqueeze(0)
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    monkeypatch.setattr(
        shoelace_rewards,
        "task_state",
        lambda *args: (None, None, None, None, tail_velocities),
    )

    reaching = shoelace_rewards.tail_reaching(None, 0.05, 0.04, 0.0, None, None, None)
    grasping = shoelace_rewards.tail_grasping(None, 0.02, 0.02, 0.02, 0.04, 0.0015, None, None, None)
    directional_pull = shoelace_rewards.directional_tail_pull(None, 0.04, 0.02, 0.02, None, None, None)

    torch.testing.assert_close(reaching, torch.tensor([0.5]))
    expected_grasping = 0.25 * (1.0 + torch.exp(torch.tensor(-6.25)))
    torch.testing.assert_close(grasping, expected_grasping.unsqueeze(0))
    torch.testing.assert_close(directional_pull, torch.tensor([0.5 * torch.tanh(torch.tensor(1.0))]))


def test_dense_reward_orders_approach_acquisition_and_untying(monkeypatch):
    """Bounded shaping must rank task progress above open-gripper hovering."""
    closure = torch.zeros((1, 2))
    positions = torch.zeros((1, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((1, 3))
    tail_positions = torch.zeros((1, 2, 3))
    tail_velocities = torch.zeros((1, 2, 3))

    monkeypatch.setattr(
        shoelace_rewards,
        "robot_tcp_position",
        lambda _, robot_cfg: tail_positions[:, 0 if robot_cfg == "left" else 1],
    )
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(
        shoelace_rewards,
        "task_state",
        lambda *args: (positions, None, knot, tail_positions, tail_velocities),
    )

    def compute_reward() -> torch.Tensor:
        return shoelace_rewards.shoelace_dense_reward(
            None,
            reach_std=0.05,
            grasp_std=0.02,
            throat_radius=0.025,
            maximum_throat_segments=52,
            tail_success_distance=0.09,
            tail_success_separation=0.18,
            target_speed=0.04,
            open_position=0.04,
            closed_position=0.0015,
            approach_weight=0.1,
            acquisition_weight=0.25,
            task_weight=1.0,
            pull_weight=0.25,
            asset_cfgs=None,
            left_robot_cfg="left",
            right_robot_cfg="right",
        )

    hovering = compute_reward()
    closure[0, 0] = 1.0
    one_sided = compute_reward()
    closure.fill_(1.0)
    acquired = compute_reward()
    positions.fill_(0.1)
    tail_positions.copy_(torch.tensor([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]]))
    tail_velocities[:, 0].copy_(0.04 * pull_directions(tail_velocities)[0])
    torch.testing.assert_close(compute_reward(), torch.tensor([1.35]), rtol=1.0e-5, atol=1.0e-5)
    tail_velocities.copy_(0.04 * pull_directions(tail_velocities).unsqueeze(0))
    untied = compute_reward()

    torch.testing.assert_close(hovering, torch.tensor([0.1]))
    torch.testing.assert_close(one_sided, torch.tensor([0.1]))
    torch.testing.assert_close(acquired, torch.tensor([0.35]), rtol=1.0e-5, atol=1.0e-5)
    pull = torch.tanh(torch.tensor(1.0))
    expected_untied = 1.35 + 0.25 * pull / (2.0 - pull)
    torch.testing.assert_close(untied, expected_untied.unsqueeze(0), rtol=1.0e-5, atol=1.0e-5)


def test_dense_reward_is_finite_for_non_finite_terminal_states(monkeypatch):
    """Invalid cable state must not propagate through the terminal-step reward."""
    closure = torch.ones((3, 2))
    positions = torch.zeros((3, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((3, 3))
    tail_positions = torch.zeros((3, 2, 3))
    tail_velocities = torch.zeros((3, 2, 3))
    tail_positions[0, 0, 0] = torch.nan
    tail_velocities[1, 0, 0] = torch.inf

    monkeypatch.setattr(
        shoelace_rewards,
        "robot_tcp_position",
        lambda _, robot_cfg: torch.zeros((3, 3)),
    )
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(
        shoelace_rewards,
        "task_state",
        lambda *args: (positions, None, knot, tail_positions, tail_velocities),
    )

    reward = shoelace_rewards.shoelace_dense_reward(
        None,
        reach_std=0.05,
        grasp_std=0.02,
        throat_radius=0.025,
        maximum_throat_segments=52,
        tail_success_distance=0.09,
        tail_success_separation=0.18,
        target_speed=0.04,
        open_position=0.04,
        closed_position=0.0015,
        approach_weight=0.1,
        acquisition_weight=0.25,
        task_weight=1.0,
        pull_weight=0.25,
        asset_cfgs=None,
        left_robot_cfg="left",
        right_robot_cfg="right",
    )

    torch.testing.assert_close(reward, torch.tensor([0.0, 0.0, 0.35]))
    assert torch.isfinite(reward).all()


def test_reset_relative_dense_reward_requires_confirmed_grasp_for_task_progress(monkeypatch):
    """Reset geometry must score zero and post-grasp weakest-margin progress must be signed."""
    positions = torch.zeros((1, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((1, 3))
    tail_positions = torch.tensor([[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]])
    tail_velocities = torch.zeros((1, 2, 3))
    tcp_positions = tail_positions.clone()
    tcp_positions[..., 2] += 0.03
    finger_positions = torch.zeros((1, 2))
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    monkeypatch.setattr(
        shoelace_rewards,
        "robot_tcp_position",
        lambda _, robot_cfg: tcp_positions[:, 0 if robot_cfg == "left" else 1],
    )
    monkeypatch.setattr(shoelace_rewards, "gripper_positions", lambda *args: finger_positions)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    monkeypatch.setattr(
        shoelace_rewards,
        "task_state",
        lambda *args: (positions, None, knot, tail_positions, tail_velocities),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.1)
    reset_params = {
        "maximum_grasp_distance": 0.015,
        "minimum_finger_position": 0.0003,
        "maximum_finger_position": 0.0025,
        "throat_radius": 0.025,
        "asset_cfgs": None,
        "left_robot_cfg": "left",
        "right_robot_cfg": "right",
    }
    term = shoelace_rewards.reset_relative_dense_reward(SimpleNamespace(params=reset_params), env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            reach_std=0.05,
            grasp_std=0.012,
            throat_radius=0.025,
            maximum_throat_segments=52,
            tail_success_distance=0.09,
            tail_success_separation=0.18,
            target_speed=0.04,
            open_position=0.01,
            minimum_finger_position=0.0003,
            maximum_finger_position=0.0025,
            maximum_grasp_distance=0.015,
            maximum_success_grasp_distance=0.012,
            maximum_progress_rate=3.0,
            soft_min_temperature=0.05,
            soft_min_weight=0.25,
            confirmation_steps=2,
            approach_weight=0.1,
            acquisition_weight=0.25,
            task_weight=1.0,
            pull_weight=0.25,
            asset_cfgs=None,
            left_robot_cfg="left",
            right_robot_cfg="right",
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    tcp_positions.copy_(tail_positions)
    assert compute_reward().item() > 0.0
    finger_positions.fill_(0.001)
    assert compute_reward().item() > 0.0

    grasped[:] = True
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))

    tcp_positions[:, 0, 0] += 0.01
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    tcp_positions.copy_(tail_positions)
    torch.testing.assert_close(compute_reward(), torch.zeros(1))

    tail_positions.copy_(torch.tensor([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]))
    tcp_positions.copy_(tail_positions)
    tail_velocities.copy_(0.04 * pull_directions(tail_velocities).unsqueeze(0))
    assert compute_reward().item() > 3.0

    grasped[:] = False
    tcp_positions[:, 1, 2] += 0.03
    assert compute_reward().item() < 0.0
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    tcp_positions.copy_(tail_positions)
    grasped[:] = True
    assert compute_reward().item() > 0.0
    tail_positions.copy_(torch.tensor([[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]]))
    tcp_positions.copy_(tail_positions)
    tail_velocities.zero_()
    assert compute_reward().item() < 0.0

    term.reset()
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[:] = False
    tcp_positions[..., 2] += 0.03
    torch.testing.assert_close(compute_reward(), torch.zeros(1))


def test_reset_relative_dense_reward_captures_first_action_progress(monkeypatch):
    """The reset state, rather than the post-action state, must define the task-progress baseline."""
    positions = torch.zeros((1, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((1, 3))
    tail_positions = torch.tensor([[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]])
    tail_velocities = torch.zeros((1, 2, 3))
    tcp_positions = tail_positions.clone()
    finger_positions = torch.full((1, 2), 0.001)
    grasped = torch.ones((1, 2), dtype=torch.bool)
    monkeypatch.setattr(
        shoelace_rewards,
        "robot_tcp_position",
        lambda _, robot_cfg: tcp_positions[:, 0 if robot_cfg == "left" else 1],
    )
    monkeypatch.setattr(shoelace_rewards, "gripper_positions", lambda *args: finger_positions)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    monkeypatch.setattr(
        shoelace_rewards,
        "task_state",
        lambda *args: (positions, None, knot, tail_positions, tail_velocities),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.1)
    reset_params = {
        "maximum_grasp_distance": 0.015,
        "minimum_finger_position": 0.0003,
        "maximum_finger_position": 0.0025,
        "throat_radius": 0.025,
        "asset_cfgs": None,
        "left_robot_cfg": "left",
        "right_robot_cfg": "right",
    }
    term = shoelace_rewards.reset_relative_dense_reward(SimpleNamespace(params=reset_params), env)
    term.reset()

    tail_positions.copy_(torch.tensor([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]))
    tcp_positions.copy_(tail_positions)
    reward = term(
        env,
        reach_std=0.05,
        grasp_std=0.012,
        throat_radius=0.025,
        maximum_throat_segments=52,
        tail_success_distance=0.09,
        tail_success_separation=0.18,
        target_speed=0.04,
        open_position=0.01,
        minimum_finger_position=0.0003,
        maximum_finger_position=0.0025,
        maximum_grasp_distance=0.015,
        maximum_progress_rate=3.0,
        soft_min_temperature=0.05,
        soft_min_weight=0.25,
        confirmation_steps=2,
        approach_weight=0.1,
        acquisition_weight=0.25,
        task_weight=1.0,
        pull_weight=0.25,
        asset_cfgs=None,
        left_robot_cfg="left",
        right_robot_cfg="right",
    )

    assert reward.item() > 0.0


def test_premature_close_penalizes_only_first_pre_acquisition_closure(monkeypatch):
    """Closing far from a tail is penalized once without blocking post-acquisition recovery."""
    distances = torch.tensor([[0.05, 0.01]])
    closure = torch.zeros((1, 2))
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_rewards.premature_close_event(None, env)

    def compute_penalty() -> torch.Tensor:
        return term(
            env,
            acquisition_distance=0.02,
            maximum_finger_position=0.02,
            open_position=0.04,
            closed_position=0.0015,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_penalty(), torch.zeros(1))
    closure[0, 0] = 1.0
    torch.testing.assert_close(compute_penalty(), torch.tensor([0.5]))
    torch.testing.assert_close(compute_penalty(), torch.zeros(1))

    grasped[0, 0] = True
    torch.testing.assert_close(compute_penalty(), torch.zeros(1))
    grasped[0, 0] = False
    closure[0, 0] = 0.5
    torch.testing.assert_close(compute_penalty(), torch.zeros(1))
    closure[0, 0] = 1.0
    torch.testing.assert_close(compute_penalty(), torch.zeros(1))


def test_grasp_state_requires_contact_sized_finger_aperture(monkeypatch):
    """A nearby tail must not count as grasped when the fingers are empty or too open."""
    distances = torch.tensor([[0.006, 0.006], [0.013, 0.006], [0.006, 0.006]])
    finger_positions = torch.tensor([[0.001, 0.0025], [0.001, 0.001], [0.0002, 0.0026]])
    monkeypatch.setattr(shoelace_utils, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_utils, "gripper_positions", lambda *args: finger_positions)

    grasped = shoelace_utils.grasp_state(
        SimpleNamespace(),
        maximum_distance=0.012,
        maximum_finger_position=0.0025,
        asset_cfgs=None,
        left_robot_cfg=None,
        right_robot_cfg=None,
        minimum_finger_position=0.0003,
    )

    torch.testing.assert_close(
        grasped,
        torch.tensor([[True, True], [False, True], [False, False]]),
    )


def test_grasp_acquisition_rewards_each_side_only_once(monkeypatch):
    """Strict acquisition emits one per-side event and does not reward regrasping."""
    grasped = torch.zeros((2, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.2)
    term = shoelace_rewards.grasp_acquisition_event(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            maximum_grasp_distance=0.01,
            maximum_finger_position=0.004,
            side_weights=(3.0, 1.0),
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(2))
    grasped[0, 0] = True
    grasped[1] = True
    torch.testing.assert_close(compute_reward(), torch.tensor([3.75, 5.0]))
    torch.testing.assert_close(compute_reward(), torch.zeros(2))

    grasped[0, 0] = False
    torch.testing.assert_close(compute_reward(), torch.zeros(2))
    grasped[0] = True
    torch.testing.assert_close(compute_reward(), torch.tensor([1.25, 0.0]))

    term.reset([0])
    torch.testing.assert_close(compute_reward(), torch.tensor([5.0, 0.0]))


def test_grasp_acquisition_requires_consecutive_confirmation_steps(monkeypatch):
    """Transient grasp geometry must not emit an acquisition event."""
    grasped = torch.tensor([[True, False]])
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args, **kwargs: grasped)
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.2)
    term = shoelace_rewards.grasp_acquisition_event(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            maximum_grasp_distance=0.012,
            maximum_finger_position=0.0025,
            side_weights=(1.0, 1.0),
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
            minimum_finger_position=0.0003,
            confirmation_steps=3,
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[:] = False
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[0, 0] = True
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.tensor([2.5]))


def test_bilateral_grasp_acquisition_requires_simultaneous_confirmation(monkeypatch):
    """Sequential per-tail contact must not collect the bilateral acquisition event."""
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args, **kwargs: grasped)
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.2)
    term = shoelace_rewards.bilateral_grasp_acquisition_event(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            maximum_grasp_distance=0.012,
            minimum_finger_position=0.0003,
            maximum_finger_position=0.0025,
            confirmation_steps=3,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    grasped[0, 0] = True
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[0] = torch.tensor([False, True])
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[:] = True
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    torch.testing.assert_close(compute_reward(), torch.tensor([5.0]))
    torch.testing.assert_close(compute_reward(), torch.zeros(1))

    term.reset()
    torch.testing.assert_close(compute_reward(), torch.zeros(1))


def test_bilateral_grasp_acquisition_does_not_reward_reset_grasp(monkeypatch):
    """A grasp already present at reset must not emit a farmable acquisition event."""
    grasped = torch.ones((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args, **kwargs: grasped)
    params = {
        "maximum_grasp_distance": 0.012,
        "minimum_finger_position": 0.0003,
        "maximum_finger_position": 0.0025,
        "asset_cfgs": None,
        "left_robot_cfg": None,
        "right_robot_cfg": None,
    }
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.2)
    term = shoelace_rewards.bilateral_grasp_acquisition_event(SimpleNamespace(params=params), env)

    term.reset()
    torch.testing.assert_close(term(env, confirmation_steps=1, **params), torch.zeros(1))

    grasped[:] = False
    term.reset()
    torch.testing.assert_close(term(env, confirmation_steps=1, **params), torch.zeros(1))
    grasped[:] = True
    torch.testing.assert_close(term(env, confirmation_steps=1, **params), torch.tensor([5.0]))


def test_tail_approach_progress_is_side_weighted_and_finite(monkeypatch):
    """Open-gripper approach must favor the lagging side without propagating invalid state."""
    distances = torch.full((2, 2), 0.05)
    closure = torch.zeros((2, 2))
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    env = SimpleNamespace(num_envs=2, device="cpu")
    term = shoelace_rewards.tail_approach_progress(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            std=0.05,
            open_position=0.04,
            closed_position=0.0,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
            side_weights=(3.0, 1.0),
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(2))
    distances[0, 0] = 0.04
    distances[1, 1] = 0.04
    torch.testing.assert_close(compute_reward(), torch.tensor([0.15, 0.05]))
    distances[0, 0] = torch.nan
    distances[1, 1] = 0.03
    torch.testing.assert_close(compute_reward(), torch.tensor([0.0, 0.05]))
    assert torch.isfinite(compute_reward()).all()


def test_second_tail_coordination_rewards_reaching_after_first_acquisition(monkeypatch):
    """Acquiring one tail must expose a continuous signal for reaching the other tail."""
    distances = torch.full((1, 2), 0.05)
    closure = torch.zeros((1, 2))
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)

    def compute_reward() -> torch.Tensor:
        return shoelace_rewards.second_tail_coordination(
            None,
            reach_std=0.05,
            grasp_std=0.01,
            open_position=0.04,
            closed_position=0.0,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    distances[0, 1] = 0.0
    closure[0, 1] = 1.0
    first_tail_only = compute_reward()
    distances[0, 0] = 0.02
    closer_to_second_tail = compute_reward()
    distances[0, 0] = 0.0
    closure[0, 0] = 1.0
    both_tails = compute_reward()

    assert 0.0 < first_tail_only.item() < closer_to_second_tail.item() < both_tails.item()
    torch.testing.assert_close(both_tails, torch.ones(1))


def test_second_tail_approach_progress_requires_strict_first_acquisition(monkeypatch):
    """Only progress toward the remaining open-gripper tail is rewarded after acquisition."""
    distances = torch.full((1, 2), 0.05)
    closure = torch.zeros((1, 2))
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_rewards.second_tail_approach_progress(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            std=0.05,
            maximum_grasp_distance=0.01,
            maximum_finger_position=0.004,
            open_position=0.04,
            closed_position=0.0,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    distances[0, 1] = 0.04
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[0, 0] = True
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[0, 0] = False
    distances[0, 1] = 0.03
    torch.testing.assert_close(compute_reward(), torch.tensor([0.2]))
    distances[0, 1] = 0.035
    torch.testing.assert_close(compute_reward(), torch.tensor([-0.1]))
    closure[0, 1] = 1.0
    distances[0, 1] = 0.025
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[0, 1] = True
    closure[0, 1] = 0.0
    distances[0, 1] = 0.015
    torch.testing.assert_close(compute_reward(), torch.zeros(1))


def test_remaining_tail_grasping_latches_simultaneous_acquisitions(monkeypatch):
    """Simultaneously acquired grippers must receive normalized close-command retention."""
    distances = torch.full((1, 2), 0.05)
    closure = torch.zeros((1, 2))
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    actions = {
        "left_gripper": SimpleNamespace(raw_actions=torch.zeros((1, 1))),
        "right_gripper": SimpleNamespace(raw_actions=torch.zeros((1, 1))),
    }
    monkeypatch.setattr(shoelace_rewards, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_rewards, "gripper_closed_fraction", lambda *args: closure)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        action_manager=SimpleNamespace(get_term=actions.__getitem__),
    )
    term = shoelace_rewards.remaining_tail_grasping(None, env)

    def compute_reward() -> torch.Tensor:
        return term(
            env,
            std=0.03,
            maximum_grasp_distance=0.01,
            maximum_finger_position=0.004,
            open_position=0.04,
            closed_position=0.0,
            command_temperature=0.5,
            command_weight=0.75,
            gripper_action_names=("left_gripper", "right_gripper"),
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    grasped[:] = True
    closure[0, 0] = 0.5
    closure[0, 1] = 1.0
    actions["right_gripper"].raw_actions[0, 0] = -1.0
    left_score = 0.25 * 0.5 + 0.75 * 0.5
    right_score = 0.25 + 0.75 * torch.sigmoid(torch.tensor(2.0))
    expected = 0.5 * (left_score + right_score)
    torch.testing.assert_close(compute_reward(), expected.unsqueeze(0))
    distances[:] = 0.1
    torch.testing.assert_close(compute_reward(), expected.unsqueeze(0))
    distances[0, 1] = torch.nan
    torch.testing.assert_close(compute_reward(), torch.zeros(1))
    term.reset()
    grasped[:] = False
    distances[:] = 0.0
    torch.testing.assert_close(compute_reward(), torch.zeros(1))


def test_untying_progress_reports_finite_potential_rate(monkeypatch):
    """Post-grasp shaping must retain regressions after acquisition without propagating NaN."""
    current = torch.tensor([1.0, 1.0])
    grasped = torch.zeros((2, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "task_state", lambda *args: (None, None, None, None, None))
    monkeypatch.setattr(shoelace_rewards, "potential", lambda *args: current)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.2)
    term = shoelace_rewards.untying_progress(None, env)

    def compute_progress() -> torch.Tensor:
        return term(
            env,
            throat_radius=0.025,
            tail_success_distance=0.09,
            tail_success_separation=0.18,
            maximum_grasp_distance=0.02,
            maximum_finger_position=0.02,
            maximum_progress_rate=3.0,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_progress(), torch.zeros(2))
    grasped[:] = True
    current.copy_(torch.tensor([1.1, 1.1]))
    torch.testing.assert_close(compute_progress(), torch.tensor([0.5, 0.5]))
    grasped[:] = False
    current.copy_(torch.tensor([1.0, 1.2]))
    torch.testing.assert_close(compute_progress(), torch.tensor([-0.5, 0.5]))
    current.copy_(torch.tensor([torch.nan, 1.3]))
    torch.testing.assert_close(compute_progress(), torch.tensor([0.0, 0.5]))
    current.copy_(torch.tensor([10.0, -10.0]))
    torch.testing.assert_close(compute_progress(), torch.tensor([3.0, -3.0]))
    term.reset()
    current.copy_(torch.tensor([1.2, 1.2]))
    torch.testing.assert_close(compute_progress(), torch.zeros(2))


def test_acquired_grasp_retention_starts_after_acquisition_and_resets(monkeypatch):
    """Retention shaping must begin after acquisition and track each side independently."""
    grasped = torch.zeros((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_rewards, "grasp_state", lambda *args: grasped)
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_rewards.acquired_grasp_retention(None, env)

    def compute_retention() -> torch.Tensor:
        return term(
            env,
            maximum_grasp_distance=0.01,
            maximum_finger_position=0.004,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
        )

    torch.testing.assert_close(compute_retention(), torch.zeros(1))
    grasped[0, 0] = True
    torch.testing.assert_close(compute_retention(), torch.zeros(1))
    torch.testing.assert_close(compute_retention(), torch.tensor([0.5]))
    grasped[0, 1] = True
    torch.testing.assert_close(compute_retention(), torch.tensor([0.5]))
    torch.testing.assert_close(compute_retention(), torch.ones(1))
    term.reset()
    torch.testing.assert_close(compute_retention(), torch.zeros(1))


def test_shoelace_unsafe_detects_non_finite_velocity(monkeypatch):
    """A non-finite cable velocity must terminate before it reaches observations."""
    positions = torch.zeros((2, DYNAMIC_SEGMENT_COUNT, 3))
    velocities = torch.zeros_like(positions)
    velocities[1, 0, 0] = torch.nan
    monkeypatch.setattr(
        shoelace_terminations,
        "task_state",
        lambda *args: (positions, velocities, None, None, None),
    )

    unsafe = shoelace_terminations.shoelace_unsafe(
        None,
        minimum_lace_height=-0.003,
        maximum_lace_spread=0.6,
        asset_cfgs=None,
    )

    torch.testing.assert_close(unsafe, torch.tensor([False, True]))


def test_shoelace_unsafe_detects_non_finite_robot_state(monkeypatch):
    """Non-finite robot state must reset with the coupled shoelace state."""
    positions = torch.zeros((2, DYNAMIC_SEGMENT_COUNT, 3))
    velocities = torch.zeros_like(positions)
    robot_state = {
        "joint_pos": torch.zeros((2, 9)),
        "joint_vel": torch.zeros((2, 9)),
        "body_pos_w": torch.zeros((2, 12, 3)),
        "body_quat_w": torch.zeros((2, 12, 4)),
    }
    robot_state["joint_vel"][1, 0] = torch.nan

    def make_robot(state):
        return SimpleNamespace(
            data=SimpleNamespace(**{name: SimpleNamespace(torch=value) for name, value in state.items()})
        )

    env = SimpleNamespace(scene={"robot_left": make_robot(robot_state), "robot_right": make_robot(robot_state)})
    monkeypatch.setattr(
        shoelace_terminations,
        "task_state",
        lambda *args: (positions, velocities, None, None, None),
    )

    unsafe = shoelace_terminations.shoelace_unsafe(
        env,
        minimum_lace_height=-0.003,
        maximum_lace_spread=0.6,
        asset_cfgs=None,
        left_robot_cfg=SceneEntityCfg("robot_left"),
        right_robot_cfg=SceneEntityCfg("robot_right"),
    )

    torch.testing.assert_close(unsafe, torch.tensor([False, True]))


def test_lost_grasp_uses_compliant_latch_when_available(monkeypatch):
    """A retained compliant grasp must not terminate until its latch releases."""
    distances = torch.zeros((1, 2))
    active = torch.ones((1, 2), dtype=torch.int32)
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(
        shoelace_terminations,
        "grasp_state",
        lambda *args: pytest.fail("The compliant latch must be the authoritative grasp state."),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        _physics=SimpleNamespace(grasp_assist_enabled=True, grasp_assist_active=active),
    )
    term = shoelace_terminations.lost_grasp(None, env)

    result = term(env, 0.01, 0.01, 0.025, None, None, None)
    torch.testing.assert_close(result, torch.tensor([False]))

    active[0, 1] = 0
    result = term(env, 0.01, 0.01, 0.025, None, None, None)
    torch.testing.assert_close(result, torch.tensor([True]))


def test_lost_grasp_uses_geometry_when_assistance_is_disabled(monkeypatch):
    """Disabled assistance must ignore stale latch state and use physical grasp geometry."""
    distances = torch.zeros((1, 2))
    active = torch.zeros((1, 2), dtype=torch.int32)
    acquired = torch.ones((1, 2), dtype=torch.bool)
    retained = torch.ones((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(
        shoelace_terminations,
        "grasp_state",
        lambda env, maximum_distance, *args: acquired if maximum_distance == 0.01 else retained,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        _physics=SimpleNamespace(grasp_assist_enabled=False, grasp_assist_active=active),
    )
    term = shoelace_terminations.lost_grasp(None, env)

    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([False]))
    retained[0, 1] = False
    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([True]))


def test_lost_grasp_requires_consecutive_unassisted_acquisition(monkeypatch):
    """A transient geometric match must not arm the unassisted loss termination."""
    distances = torch.zeros((1, 2))
    acquired = torch.ones((1, 2), dtype=torch.bool)
    retained = torch.ones((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(
        shoelace_terminations,
        "grasp_state",
        lambda _env, maximum_distance, *args, **kwargs: acquired if maximum_distance == 0.012 else retained,
    )
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_terminations.lost_grasp(None, env)

    def compute_termination() -> torch.Tensor:
        return term(
            env,
            acquisition_distance=0.012,
            maximum_finger_position=0.0025,
            maximum_grasp_distance=0.015,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
            minimum_finger_position=0.0003,
            confirmation_steps=3,
            release_confirmation_steps=2,
        )

    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    acquired[:] = False
    retained[:] = False
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    acquired[:] = True
    retained[:] = True
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    retained[0, 1] = False
    torch.testing.assert_close(compute_termination(), torch.tensor([False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([True]))


def test_lost_grasp_is_armed_by_reset_grasp(monkeypatch):
    """Opening immediately after a grasped reset must still trigger grasp loss."""
    distances = torch.zeros((1, 2))
    grasped = torch.ones((1, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_terminations, "grasp_state", lambda *args, **kwargs: grasped)
    params = {
        "acquisition_distance": 0.012,
        "minimum_finger_position": 0.0003,
        "maximum_finger_position": 0.0025,
        "maximum_grasp_distance": 0.02,
        "asset_cfgs": None,
        "left_robot_cfg": None,
        "right_robot_cfg": None,
    }
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_terminations.lost_grasp(SimpleNamespace(params=params), env)

    term.reset()
    grasped[:] = False
    torch.testing.assert_close(
        term(env, confirmation_steps=1, release_confirmation_steps=2, **params), torch.tensor([False])
    )
    torch.testing.assert_close(
        term(env, confirmation_steps=1, release_confirmation_steps=2, **params), torch.tensor([True])
    )


def test_lost_grasp_allows_recovery_before_bilateral_acquisition(monkeypatch):
    """Dropping the first tail during acquisition must not reset before the second is reached."""
    distances = torch.full((1, 2), 0.1)
    retained = torch.tensor([[True, False]])
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_terminations, "grasp_state", lambda *args: retained)
    env = SimpleNamespace(num_envs=1, device="cpu")
    term = shoelace_terminations.lost_grasp(None, env)

    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([False]))
    retained[:] = False
    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([False]))
    retained[:] = True
    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([False]))
    retained[0, 1] = False
    torch.testing.assert_close(term(env, 0.01, 0.01, 0.025, None, None, None), torch.tensor([True]))


def test_missed_grasp_acquisition_terminates_only_stalled_reachable_states(monkeypatch):
    """The short deadline must not penalize approach or a completed acquisition."""
    distances = torch.tensor([[0.005, 0.005], [0.005, 0.005], [0.050, 0.050]])
    grasped = torch.zeros((3, 2), dtype=torch.bool)
    monkeypatch.setattr(shoelace_terminations, "grasp_distances", lambda *args: distances)
    monkeypatch.setattr(shoelace_terminations, "grasp_state", lambda *args, **kwargs: grasped)
    env = SimpleNamespace(num_envs=3, device="cpu")
    term = shoelace_terminations.missed_grasp_acquisition(None, env)

    def compute_termination() -> torch.Tensor:
        return term(
            env,
            acquisition_distance=0.012,
            maximum_finger_position=0.0025,
            asset_cfgs=None,
            left_robot_cfg=None,
            right_robot_cfg=None,
            minimum_finger_position=0.0003,
            deadline_steps=3,
        )

    torch.testing.assert_close(compute_termination(), torch.tensor([False, False, False]))
    distances[0] = 0.050
    torch.testing.assert_close(compute_termination(), torch.tensor([False, False, False]))
    grasped[1] = True
    torch.testing.assert_close(compute_termination(), torch.tensor([True, False, False]))
    grasped[1] = False
    torch.testing.assert_close(compute_termination(), torch.tensor([True, False, False]))


def test_insufficient_separation_progress_uses_post_acquisition_deadline(monkeypatch):
    """Only acquired tails below the reset-relative separation target should terminate."""
    tail_positions = torch.zeros((3, 2, 3))
    tail_positions[:, 1, 0] = 0.12
    grasped = torch.tensor([[True, True], [True, True], [False, False]])
    monkeypatch.setattr(
        shoelace_terminations,
        "task_state",
        lambda *args: (torch.empty(0), torch.empty(0), torch.empty(0), tail_positions, torch.empty(0)),
    )
    monkeypatch.setattr(shoelace_terminations, "grasp_state", lambda *args, **kwargs: grasped)
    params = {
        "acquisition_distance": 0.012,
        "maximum_finger_position": 0.0025,
        "target_tail_separation": 0.18,
        "minimum_progress_fraction": 0.5,
        "deadline_steps": 3,
        "asset_cfgs": None,
        "left_robot_cfg": None,
        "right_robot_cfg": None,
    }
    env = SimpleNamespace(num_envs=3, device="cpu")
    term = shoelace_terminations.insufficient_separation_progress(SimpleNamespace(params=params), env)
    term.reset()
    tail_positions[:, 1, 0] = torch.tensor([0.151, 0.149, 0.149])

    def compute_termination() -> torch.Tensor:
        return term(env, **params)

    torch.testing.assert_close(compute_termination(), torch.tensor([False, False, False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False, False, False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False, True, False]))
    grasped[2] = True
    torch.testing.assert_close(compute_termination(), torch.tensor([False, True, False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False, True, False]))
    torch.testing.assert_close(compute_termination(), torch.tensor([False, True, True]))


def test_finite_joint_vel_l2_bounds_terminal_state_outliers():
    """A reset-bound robot must not emit invalid or destabilizing velocity penalties."""
    joint_velocities = torch.tensor([[1.0, 2.0], [torch.nan, 2.0], [torch.inf, 2.0], [1000.0, 2.0]])
    robot = SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=joint_velocities)))
    env = SimpleNamespace(scene={"robot": robot})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])

    penalty = shoelace_rewards.finite_joint_vel_l2(env, asset_cfg, maximum_penalty=100.0)

    torch.testing.assert_close(penalty, torch.tensor([5.0, 0.0, 0.0, 100.0]))


def test_termination_event_reward_cancels_reward_manager_time_scaling():
    """Configured event weights must equal their one-step reward contribution."""

    class TerminationManager:
        time_outs = torch.tensor([False, False, True])

        def get_term(self, name: str) -> torch.Tensor:
            terms = {
                "unsafe": torch.tensor([True, False, True]),
                "lost_grasp": torch.tensor([True, True, False]),
                "success": torch.tensor([True, False, False]),
                "time_out": torch.tensor([False, False, True]),
            }
            return terms[name]

    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.2, termination_manager=TerminationManager())
    term = shoelace_rewards.termination_event_reward.__new__(shoelace_rewards.termination_event_reward)
    term._term_names = ["unsafe", "lost_grasp"]

    event_rate = term(env, ["unsafe", "lost_grasp"])

    torch.testing.assert_close(event_rate * env.step_dt, torch.tensor([2.0, 1.0, 0.0]))

    term._excluded_term_names = ["success"]
    exclusive_event_rate = term(env, ["unsafe", "lost_grasp"], ["success"])

    torch.testing.assert_close(exclusive_event_rate * env.step_dt, torch.tensor([0.0, 1.0, 0.0]))

    term._term_names = ["unsafe", "lost_grasp", "time_out"]
    timeout_event_rate = term(
        env,
        ["unsafe", "lost_grasp", "time_out"],
        ["success"],
        include_time_outs=True,
    )

    torch.testing.assert_close(timeout_event_rate * env.step_dt, torch.tensor([0.0, 1.0, 2.0]))


def test_runtime_configuration_authors_split_cable_assets():
    """Runtime configuration must retain both dynamic cable ends and the pinned middle span."""
    env = ShoelaceEnv.__new__(ShoelaceEnv)
    env._is_closed = True
    env._centerline = np.column_stack((np.zeros(361), np.zeros(361), np.linspace(0.0, 0.45, 361)))
    env._authored_mean_segment_length = 0.45 / 450
    env._cable_radius = 0.0015
    env._model_asset = Path("model.usd")
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 2

    env._configure_runtime_cfg(cfg, Path("collider.usd"))

    assert len(cfg.scene.shoelace_left.spawn.positions) - 1 == LEFT_CABLE_SEGMENT_COUNT
    assert len(cfg.scene.shoelace_right.spawn.positions) - 1 == RIGHT_CABLE_SEGMENT_COUNT
    assert callable(cfg.scene.shoelace_pinned_visual.spawn.func)


def test_untying_potential_increases_for_separated_tails_and_a_clear_throat():
    """Separating both held tails and clearing free segments must increase dense progress."""
    positions = torch.zeros((1, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((1, 3))
    tail_positions = torch.tensor([[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]])
    baseline = potential(
        positions,
        knot,
        tail_positions,
        throat_radius=0.025,
        tail_success_distance=0.09,
        tail_success_separation=0.18,
    )

    positions[:, : LEFT_CABLE_SEGMENT_COUNT - 1, 0] = 0.1
    positions[:, LEFT_CABLE_SEGMENT_COUNT + 1 :, 0] = -0.1
    tail_positions = torch.tensor([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    untied = potential(
        positions,
        knot,
        tail_positions,
        throat_radius=0.025,
        tail_success_distance=0.09,
        tail_success_separation=0.18,
    )

    assert len(TAIL_REGIONS) == 2
    assert untied.item() > baseline.item()


def test_untying_potential_does_not_credit_the_already_farther_tail_distance():
    """The tail-distance term must track the lagging tail rather than their average."""
    positions = torch.zeros((1, DYNAMIC_SEGMENT_COUNT, 3))
    knot = torch.zeros((1, 3))
    baseline_tails = torch.tensor([[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]])
    one_sided_tails = torch.tensor([[[-0.10, 0.0, 0.0], [0.06, 0.0, 0.0]]])

    baseline = potential(positions, knot, baseline_tails, 0.025, 0.09, 0.18)
    one_sided = potential(positions, knot, one_sided_tails, 0.025, 0.09, 0.18)

    torch.testing.assert_close(one_sided - baseline, torch.tensor([0.04 / 0.18]))
