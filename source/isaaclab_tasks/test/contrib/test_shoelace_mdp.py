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

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.coupling import CouplerProxyCfg

import isaaclab_tasks.contrib.shoelace.mdp.rewards as shoelace_rewards
import isaaclab_tasks.contrib.shoelace.shoelace_env as shoelace_env_module
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
    collision_paths: list[str] = []
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
        lambda prim_path, _fragments: collision_paths.append(prim_path) or True,
    )

    cfg = shoelace_env_module._pinned_shoelace_spawner(centerline, 0.0015)
    root = cfg.func.__wrapped__("/World/ShoelacePinned", cfg)
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]

    assert [str(prim.GetPath()) for prim in meshes] == ["/World/ShoelacePinned/geometry/mesh"]
    assert collision_paths == ["/World/ShoelacePinned/geometry/mesh"]
    assert (
        len(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get())
        == (PINNED_LAST - PINNED_FIRST) * shoelace_env_module.PINNED_TUBE_SIDES
    )


def test_shoelace_task_uses_dual_franka_manager_contract():
    """The task exposes Cartesian arms, binary grippers, and no direct cable-force action."""
    cfg = ShoelaceEnvCfg()
    cfg.validate()

    assert isinstance(cfg, ManagerBasedRLEnvCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, CouplerProxyCfg)
    assert isinstance(cfg.actions.left_arm, DifferentialInverseKinematicsActionCfg)
    assert isinstance(cfg.actions.right_arm, DifferentialInverseKinematicsActionCfg)
    assert isinstance(cfg.actions.left_gripper, BinaryJointPositionActionCfg)
    assert isinstance(cfg.actions.right_gripper, BinaryJointPositionActionCfg)
    assert not hasattr(cfg.actions, "force")
    assert cfg.actions.left_arm.scale == pytest.approx((0.005, 0.005, 0.005, 0.01, 0.01, 0.01))
    assert cfg.actions.left_gripper.open_command_expr["panda_finger_joint1"] == pytest.approx(0.04)
    assert cfg.actions.left_gripper.close_command_expr["panda_finger_joint1"] == pytest.approx(0.0015)
    assert cfg.actions.right_gripper.open_command_expr["panda_finger_joint1"] == pytest.approx(0.04)
    assert cfg.actions.right_gripper.close_command_expr["panda_finger_joint1"] == pytest.approx(0.0015)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.04)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.04)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_joint4"] == pytest.approx(-2.5148)
    assert cfg.scene.robot_left.init_state.joint_pos["panda_joint6"] == pytest.approx(2.3775)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_joint4"] == pytest.approx(-2.5148)
    assert cfg.scene.robot_right.init_state.joint_pos["panda_joint6"] == pytest.approx(2.3775)
    assert not hasattr(cfg.terminations, "left_joint_velocity")
    assert not hasattr(cfg.terminations, "right_joint_velocity")
    assert cfg.rewards.failure.params["term_keys"] == ["unsafe", "lost_grasp"]
    assert cfg.decimation == 4
    assert cfg.episode_length_s == pytest.approx(10.0)
    assert cfg.sim.physics.num_substeps == 5
    assert cfg.sim.physics.collision_decimation == 2
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.iterations == 20
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.rigid_body_contact_buffer_size == 128
    assert cfg.sim.physics.solver_cfg.proxies[0].collide_interval == 2
    assert cfg.triangle_pairs_per_env == 8192
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
    assert cfg.rewards.dense_task.weight == pytest.approx(10.0)
    assert not hasattr(cfg.rewards, "progress")
    assert not hasattr(cfg.rewards, "reach_tails")
    assert not hasattr(cfg.rewards, "approach_progress")
    assert not hasattr(cfg.rewards, "grasp_tails")
    assert not hasattr(cfg.rewards, "directional_pull")
    assert not hasattr(cfg.rewards, "premature_close")
    assert cfg.rewards.success.func is shoelace_rewards.termination_event_reward
    assert cfg.rewards.success.weight == pytest.approx(60.0)
    assert cfg.rewards.failure.func is shoelace_rewards.termination_event_reward
    assert cfg.rewards.failure.weight == pytest.approx(-2.0)
    assert cfg.rewards.action_rate.weight == pytest.approx(-0.01)
    assert cfg.rewards.left_joint_velocity.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.right_joint_velocity.weight == pytest.approx(-1.0e-4)


def test_shoelace_coupler_solves_robot_shoe_contact_in_mjwarp():
    """The robot solver must own the shoe while VBD receives it as a cable-contact proxy."""
    solver_cfg = ShoelaceEnvCfg().sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver_cfg.entries}
    proxy = solver_cfg.proxies[0]

    assert entries["robots"].bodies == [r"/World/envs/env_[^/]+/(Robot(Left|Right)|Shoe)"]
    assert entries["shoelace"].bodies == [r"/World/envs/env_[^/]+/Shoelace(Left|Right)"]
    assert r"/World/envs/env_[^/]+/Shoe" in proxy.bodies


def test_shoelace_runtime_scales_outer_and_proxy_triangle_pair_capacities():
    """Both collision pipelines must scale triangle-pair capacity with the environment count."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 4096
    env = ShoelaceEnv.__new__(ShoelaceEnv)
    env._is_closed = True
    x_coordinates = np.linspace(0.0, 0.36, SHOELACE_SEGMENT_COUNT + 1)
    env._centerline = np.column_stack((x_coordinates, np.zeros_like(x_coordinates), np.zeros_like(x_coordinates)))
    env._cable_radius = 0.001
    env._authored_mean_segment_length = 0.001
    env._model_asset = Path("model.usd")

    env._configure_runtime_cfg(cfg, Path("collider.usd"))

    expected_capacity = cfg.triangle_pairs_per_env * cfg.scene.num_envs
    proxy_pipeline = cfg.sim.physics.solver_cfg.proxies[0].collision_pipeline
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == expected_capacity
    assert proxy_pipeline.max_triangle_pairs == expected_capacity


def test_shoelace_agent_uses_asymmetric_observations():
    """The actor excludes simulator-only state while the critic receives it."""
    assert ShoelacePPORunnerCfg().obs_groups == {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }


def test_shoelace_agent_saves_intermediate_checkpoints():
    """Training must expose checkpoints frequently enough for phase-metric evaluation."""
    assert ShoelacePPORunnerCfg().save_interval == 10


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
    closure.fill_(1.0)
    acquired = compute_reward()
    positions.fill_(0.1)
    tail_positions.copy_(torch.tensor([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]]))
    tail_velocities.copy_(0.04 * pull_directions(tail_velocities).unsqueeze(0))
    untied = compute_reward()

    torch.testing.assert_close(hovering, torch.tensor([0.1]))
    torch.testing.assert_close(acquired, torch.tensor([0.35]), rtol=1.0e-5, atol=1.0e-5)
    expected_untied = 1.35 + 0.25 * torch.tanh(torch.tensor(1.0))
    torch.testing.assert_close(untied, expected_untied.unsqueeze(0), rtol=1.0e-5, atol=1.0e-5)


def test_termination_event_reward_cancels_reward_manager_time_scaling():
    """Configured event weights must equal their one-step reward contribution."""

    class TerminationManager:
        time_outs = torch.tensor([False, False, True])

        def get_term(self, name: str) -> torch.Tensor:
            terms = {
                "unsafe": torch.tensor([True, False, True]),
                "lost_grasp": torch.tensor([True, True, False]),
            }
            return terms[name]

    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.2, termination_manager=TerminationManager())
    term = shoelace_rewards.termination_event_reward.__new__(shoelace_rewards.termination_event_reward)
    term._term_names = ["unsafe", "lost_grasp"]

    event_rate = term(env, ["unsafe", "lost_grasp"])

    torch.testing.assert_close(event_rate * env.step_dt, torch.tensor([2.0, 1.0, 0.0]))


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
