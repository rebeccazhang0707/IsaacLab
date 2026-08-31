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
    assert cfg.sim.physics.num_substeps == 4
    assert cfg.sim.physics.solver_cfg.entries[1].solver_cfg.iterations == 16
    assert SHOELACE_SEGMENT_COUNT == 360
    assert (PINNED_FIRST, PINNED_LAST) == (76, 284)
    assert (LEFT_CABLE_SEGMENT_COUNT, RIGHT_CABLE_SEGMENT_COUNT) == (77, 76)
    assert DYNAMIC_SEGMENT_COUNT == 153
    assert TAIL_REGIONS == ((357, 360), (0, 3))
    assert hasattr(cfg.scene, "shoelace_left")
    assert hasattr(cfg.scene, "shoelace_pinned_visual")
    assert hasattr(cfg.scene, "shoelace_right")
    assert not hasattr(cfg.scene, "shoelace")
    assert not hasattr(cfg.observations.policy, "reference_pull_directions")
    assert not hasattr(cfg.observations.policy, "tail_separation")
    assert not hasattr(cfg.observations.policy, "episode_phase")
    assert not hasattr(cfg.observations.policy, "inferred_grasp_state")
    assert not hasattr(cfg.observations.policy, "throat_density")
    assert hasattr(cfg.observations.privileged, "throat_density")


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
    torch.testing.assert_close(knot, torch.tensor([[88.0, 0.0, 0.0]]))
    torch.testing.assert_close(tail_positions[0, :, 0], torch.tensor([174.0, 1.0]))


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


def test_runtime_contact_history_is_disabled_during_cuda_graph_capture():
    """Runtime sizing must not re-enable VBD history inside CUDA graph capture."""
    env = ShoelaceEnv.__new__(ShoelaceEnv)
    env._is_closed = True
    env._centerline = np.column_stack((np.zeros(361), np.zeros(361), np.linspace(0.0, 0.45, 361)))
    env._authored_mean_segment_length = 0.45 / 450
    env._cable_radius = 0.0015
    env._model_asset = Path("model.usd")
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = 2

    env._configure_runtime_cfg(cfg, Path("collider.usd"))

    vbd_cfg = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    assert len(cfg.scene.shoelace_left.spawn.positions) - 1 == LEFT_CABLE_SEGMENT_COUNT
    assert len(cfg.scene.shoelace_right.spawn.positions) - 1 == RIGHT_CABLE_SEGMENT_COUNT
    assert callable(cfg.scene.shoelace_pinned_visual.spawn.func)
    assert cfg.sim.physics.use_cuda_graph
    assert not vbd_cfg.rigid_contact_history
    assert cfg.sim.physics.collision_cfg.contact_matching == "disabled"

    eager_cfg = ShoelaceEnvCfg()
    eager_cfg.scene.num_envs = 2
    eager_cfg.sim.physics.use_cuda_graph = False
    env._configure_runtime_cfg(eager_cfg, Path("collider.usd"))

    eager_vbd_cfg = eager_cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    assert eager_vbd_cfg.rigid_contact_history
    assert eager_cfg.sim.physics.collision_cfg.contact_matching == "latest"


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
