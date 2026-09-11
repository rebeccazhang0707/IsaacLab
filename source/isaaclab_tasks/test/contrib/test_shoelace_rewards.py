# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified shoelace task reward and success condition."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.contrib.shoelace.mdp.rewards as shoelace_rewards
import isaaclab_tasks.contrib.shoelace.mdp.terminations as shoelace_terminations
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import (
    CABLE_CFGS,
    TAIL_SUCCESS_X_SEPARATION,
    ActionsCfg,
    RewardsCfg,
    ShoelaceEnvCfg,
    TerminationsCfg,
)
from isaaclab_tasks.contrib.shoelace.shoelace_physics import CONTACT_DISTANCE_CAP


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


def _termination_manager(num_envs: int) -> SimpleNamespace:
    return SimpleNamespace(
        dones=torch.zeros(num_envs, dtype=torch.bool),
        get_term=lambda name: torch.zeros(num_envs, dtype=torch.bool),
    )


def _cables(tail_x: torch.Tensor) -> dict[str, SimpleNamespace]:
    """Build translated cable states in robot-arm order, including fixed seam anchors."""
    cables = {}
    for name, arm, tail_slice in (("shoelace_left", 1, slice(0, 3)), ("shoelace_right", 0, slice(-3, None))):
        poses = torch.zeros((len(tail_x), 79, 7))
        poses[:, :, 3] = 1.0
        poses[:, tail_slice, 0] = tail_x[:, arm, None]
        cables[name] = SimpleNamespace(
            data=SimpleNamespace(
                segment_pose_w=_proxy(poses), segment_velocity_w=_proxy(torch.zeros_like(poses[:, :, :6]))
            )
        )
    return cables


@pytest.mark.parametrize("tail_distance", [0.0, 0.02])
@pytest.mark.parametrize("first_arm", [0, 1])
def test_dense_reward_orders_approach_contact_grasp_and_pull(
    monkeypatch: pytest.MonkeyPatch, tail_distance: float, first_arm: int
) -> None:
    """Acquisition and pulling should pay for progress, with no net credit for a closed state cycle."""
    tail_vectors = torch.full((1, 6), 0.05)
    signed_distance = torch.full((1, 4), CONTACT_DISTANCE_CAP)
    relative_speed = torch.zeros((1, 2))
    tail_x = torch.tensor([[-0.05, 0.05]])
    translation = torch.zeros(3)
    other_arm = 1 - first_arm
    first_contact = slice(2 * first_arm, 2 * first_arm + 2)
    other_contact = slice(2 * other_arm, 2 * other_arm + 2)
    outward_sign = (-1.0, 1.0)
    robots = {
        "robot_left": SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.tensor([[0.01]])))),
        "robot_right": SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.tensor([[0.01]])))),
    }
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        scene={**robots, **_cables(tail_x)},
        extras={},
        termination_manager=_termination_manager(1),
    )
    robot_cfgs = (
        SimpleNamespace(name="robot_left", joint_ids=[0]),
        SimpleNamespace(name="robot_right", joint_ids=[0]),
    )
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: tail_vectors)
    monkeypatch.setattr(shoelace_rewards, "finger_tail_signed_distance", lambda *args: signed_distance)
    monkeypatch.setattr(shoelace_rewards, "tail_tcp_relative_speed", lambda *args: relative_speed)
    term = shoelace_rewards.dense_task_reward(None, env)
    reward_history = []

    def compute() -> torch.Tensor:
        env.scene.update(_cables(tail_x))
        for cable_cfg in CABLE_CFGS:
            env.scene[cable_cfg.name].data.segment_pose_w.torch[:, :, :3] += translation
        reward = term(
            env,
            reach_std=0.05,
            contact_std=5.0e-4,
            relative_speed_std=0.08,
            grasp_filter_time_constant=1.0e-6,
            open_position=0.01,
            closed_position=0.001,
            success_x_separation=0.18,
            cable_cfgs=CABLE_CFGS,
            robot_cfgs=robot_cfgs,
            acquisition_weight=0.3,
            approach_fraction=0.3,
        )
        reward_history.append(reward.clone())
        return reward

    initial_reward = compute()
    torch.testing.assert_close(initial_reward, torch.zeros(1))
    initial_log = env.extras["log"]
    assert initial_log["Metrics/shoelace/approach_distance_m"].item() > 0.0
    assert initial_log["Metrics/shoelace/grasp_both"].item() == 0.0
    tail_vectors.zero_()
    assert compute().item() > 0.0
    assert env.extras["log"]["Metrics/shoelace/approach_distance_m"].item() == 0.0
    assert initial_log["Metrics/shoelace/approach_distance_m"].item() > 0.0

    # A physical grasp can leave the tail center offset from the nominal TCP.
    tail_vectors[:, ::3] = tail_distance
    compute()
    translation[:] = torch.tensor([1.52, -0.02, 0.1])
    assert abs(compute().item()) < 1.0e-5
    robots["robot_left"].data.joint_pos.torch.fill_(0.001)
    robots["robot_right"].data.joint_pos.torch.fill_(0.001)
    assert abs(compute().item()) < 1.0e-5

    tail_x[:] = torch.tensor([[-0.09, 0.09]])
    assert abs(compute().item()) < 1.0e-5
    tail_x[:] = torch.tensor([[-0.05, 0.05]])
    assert abs(compute().item()) < 1.0e-5

    signed_distance[:, first_contact].zero_()
    assert compute().item() > 0.0
    assert env.extras["log"][f"Metrics/shoelace/grasp_{('left', 'right')[first_arm]}"].item() > 0.99
    assert env.extras["log"][f"Metrics/shoelace/grasp_{('left', 'right')[other_arm]}"].item() < 1.0e-5
    assert env.extras["log"]["Metrics/shoelace/grasp_both"].item() < 1.0e-5

    # The ungrasped arm's tail cannot pay the grasped arm's pull reward.
    tail_x[:, other_arm] += outward_sign[other_arm] * 0.04
    assert abs(compute().item()) < 1.0e-5
    tail_x[:, other_arm] -= outward_sign[other_arm] * 0.04
    assert abs(compute().item()) < 1.0e-5

    # A single grasp connects directly to pulling, including below the reset separation.
    tail_x[:, first_arm] -= outward_sign[first_arm] * 0.02
    assert compute().item() < 0.0
    tail_x[:, first_arm] += outward_sign[first_arm] * 0.01
    assert compute().item() > 0.0
    tail_x[:, first_arm] += outward_sign[first_arm] * 0.05
    assert compute().item() > 0.0
    first_pull_key = f"Metrics/shoelace/pull_{('left', 'right')[first_arm]}"
    other_pull_key = f"Metrics/shoelace/pull_{('left', 'right')[other_arm]}"
    single_pull = env.extras["log"][first_pull_key].clone()
    assert single_pull.item() > 0.5
    assert env.extras["log"][other_pull_key].item() < 1.0e-5

    signed_distance.zero_()
    assert compute().item() > 0.0
    tail_x[:, other_arm] += outward_sign[other_arm] * 0.04
    assert compute().item() > 0.0
    torch.testing.assert_close(env.extras["log"][first_pull_key], single_pull)
    assert env.extras["log"][other_pull_key].item() > 0.5

    # Either hand can be acquired first; releasing and reacquiring cannot accumulate credit.
    signed_distance[:, first_contact].fill_(CONTACT_DISTANCE_CAP)
    release_left = compute()
    assert release_left.item() < 0.0
    signed_distance[:, other_contact].fill_(CONTACT_DISTANCE_CAP)
    release_right = compute()
    assert release_right.item() < 0.0
    signed_distance[:, other_contact].zero_()
    torch.testing.assert_close(compute(), -release_right)
    signed_distance[:, first_contact].zero_()
    torch.testing.assert_close(compute(), -release_left)

    assert 0.5 < env.extras["log"][first_pull_key].item() < 1.0
    assert 0.5 < env.extras["log"][other_pull_key].item() < 1.0
    assert abs(compute().item()) < 1.0e-6

    signed_distance.fill_(-4.0e-3)
    assert compute().item() < 0.0

    tail_vectors.fill_(0.05)
    signed_distance.fill_(CONTACT_DISTANCE_CAP)
    tail_x[:] = torch.tensor([[-0.05, 0.05]])
    for robot in robots.values():
        robot.data.joint_pos.torch.fill_(0.01)
    compute()
    torch.testing.assert_close(torch.stack(reward_history).sum(dim=0) * env.step_dt, torch.zeros(1))


def test_success_is_only_finite_absolute_tail_x_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy distance-only helper remains available for custom configurations."""
    separation = torch.tensor([0.179, 0.18, 0.25, torch.nan])
    monkeypatch.setattr(shoelace_terminations, "tail_x_separation", lambda *args: separation)

    success = shoelace_terminations.tail_x_separation_success(None, threshold=0.18, cable_cfgs=None)

    torch.testing.assert_close(success, torch.tensor([False, True, True, False]))


def test_success_requires_cleared_throat_and_both_tails_away_from_anchors() -> None:
    """Reject distance-only hacks, uncleared throats and invalid internal cable states."""
    # Two anchors do not count; exactly 52 free segments in the throat is accepted.
    cables = _cables(torch.tensor([[-0.10, 0.10]]).repeat(8, 1))
    left = cables["shoelace_left"].data.segment_pose_w.torch
    right = cables["shoelace_right"].data.segment_pose_w.torch
    left[:, 3:-1, 0] = 0.04
    right[:, 1:-3, 0] = -0.04
    left[:, 3:55, 0] = 0.0
    left[1, 55, 0] = 0.0  # 53 free segments: knot still occupies the throat.
    right[2, -3:, 0] = -0.05
    left[2, :3, 0] = 0.15  # 20 cm separation, but one tail remains too close.
    left[3, 60, 0] = torch.nan  # Invalid internal segment, even with finite tails.
    right[4, 0, 0] = torch.inf  # Invalid anchor cannot look like an empty throat.
    cables["shoelace_left"].data.segment_velocity_w.torch[5, 60, 0] = torch.nan
    left[6, :3, :3] = torch.tensor([0.08, 0.06, 0.0])
    right[6, -3:, :3] = torch.tensor([-0.08, -0.06, 0.0])  # Long 3D distance, insufficient X separation.
    # A different environment origin and randomized shoe translation preserve the classification.
    for cable in cables.values():
        cable.data.segment_pose_w.torch[7, :, :3] += torch.tensor([1.52, -2.98, 0.2])
    cfg = TerminationsCfg().success
    success = cfg.func(SimpleNamespace(scene=cables), **cfg.params)
    torch.testing.assert_close(success, torch.tensor([True, False, False, False, False, False, False, True]))


def test_dense_reward_filters_contact_and_resets_only_selected_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contact filtering should survive invalid samples and unrelated environment resets, then penalize slip."""
    num_envs = 2
    signed_distance = torch.full((num_envs, 4), CONTACT_DISTANCE_CAP)
    relative_speed = torch.zeros((num_envs, 2))
    x_separation = torch.full((num_envs,), 0.10)
    robots = {
        name: SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.full((num_envs, 1), 0.001))))
        for name in ("robot_left", "robot_right")
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=1.0 / 30.0,
        scene=robots,
        extras={},
        termination_manager=_termination_manager(num_envs),
    )
    cfg = RewardsCfg().dense_task
    cfg.params["robot_cfgs"] = tuple(SimpleNamespace(name=name, joint_ids=[0]) for name in robots)
    term = shoelace_rewards.dense_task_reward(cfg, env)
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: torch.zeros((num_envs, 6)))
    monkeypatch.setattr(shoelace_rewards, "finger_tail_signed_distance", lambda *args: signed_distance)
    monkeypatch.setattr(shoelace_rewards, "tail_tcp_relative_speed", lambda *args: relative_speed)
    monkeypatch.setattr(shoelace_rewards, "tail_x_separation", lambda *args: x_separation)
    monkeypatch.setattr(
        shoelace_rewards,
        "tail_outward_x",
        lambda *args: x_separation[:, None].expand(-1, 2) / 2.0,
        raising=False,
    )

    def compute() -> torch.Tensor:
        return term(env, **cfg.params)

    torch.testing.assert_close(compute(), torch.zeros(num_envs))
    assert set(env.extras["log"]) == {
        f"Metrics/shoelace/{name}"
        for name in (
            "approach_distance_m",
            "grasp_left",
            "grasp_right",
            "grasp_both",
            "pull_left",
            "pull_right",
            "pull_x_separation_m",
            "success_rate",
            "valid_fraction",
        )
    }
    signed_distance.zero_()
    assert (compute() > 0.0).all()
    assert (compute() > 0.0).all()

    signed_distance[0] = torch.nan
    reward = compute()
    assert reward[0].item() == 0.0
    assert reward[1].item() > 0.0
    assert env.extras["log"]["Metrics/shoelace/valid_fraction"].item() == 0.5
    assert all(torch.isfinite(value) for value in env.extras["log"].values())
    signed_distance.zero_()
    assert (compute() > 0.0).all()

    term.reset([0])
    x_separation[0] = 0.08
    reward = compute()
    assert reward[0].item() == 0.0
    assert reward[1].item() > 0.0

    x_separation.fill_(0.14)
    assert (compute() > 0.0).all()
    grasp_before_slip = env.extras["log"]["Metrics/shoelace/grasp_both"].clone()
    relative_speed.fill_(0.8)
    assert (compute() < 0.0).all()
    assert env.extras["log"]["Metrics/shoelace/grasp_both"] < grasp_before_slip

    x_separation[:] = torch.tensor([0.18, 0.10])
    env.termination_manager.dones[0] = True
    env.termination_manager.get_term = lambda name: x_separation >= TAIL_SUCCESS_X_SEPARATION
    compute()
    assert env.extras["log"]["Metrics/shoelace/success_rate"].item() == 1.0
    env.termination_manager.dones.fill_(True)
    compute()
    terminal_log = env.extras["log"]
    assert terminal_log["Metrics/shoelace/success_rate"].item() == 0.5
    # The environment replaces the log dictionary before resetting reward terms.
    env.extras["log"] = {"unrelated_metric": 1.0}
    term.reset([0])
    for name, value in terminal_log.items():
        torch.testing.assert_close(env.extras["log"][name], value)
    assert env.extras["log"]["unrelated_metric"] == 1.0

    env.termination_manager.dones.zero_()
    x_separation.fill_(0.10)
    compute()
    assert env.extras["log"]["Metrics/shoelace/success_rate"].item() == 0.5

    signed_distance.fill_(torch.nan)
    compute()
    assert env.extras["log"]["Metrics/shoelace/valid_fraction"].item() == 0.0
    assert env.extras["log"]["Metrics/shoelace/grasp_both"].item() == 0.0
    assert all(torch.isfinite(value) for value in env.extras["log"].values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize(
    "term_names",
    [
        tuple(field.name for field in dataclasses.fields(ActionsCfg)),
        ("right_gripper", "right_arm", "left_gripper", "left_arm"),
    ],
)
def test_arm_action_penalties_exclude_grippers_and_distinguish_steady_motion(term_names: tuple[str, ...]) -> None:
    """Only arm commands should incur magnitude and change costs, regardless of action ordering."""
    left_arm = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 20.0
    right_arm = left_arm.flip(dims=(1,)) - 0.5
    left_arm[0] = right_arm[0] = 0.0
    previous_left_arm = left_arm.clone()
    previous_right_arm = right_arm.clone()
    previous_left_arm[2] *= -1.0
    previous_right_arm[2] *= -1.0
    actions = {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_gripper": torch.ones(3, 1),
        "right_gripper": -torch.ones(3, 1),
    }
    previous_actions = {
        "left_arm": previous_left_arm,
        "right_arm": previous_right_arm,
        "left_gripper": -actions["left_gripper"],
        "right_gripper": -actions["right_gripper"],
    }
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            active_terms=list(term_names),
            action_term_dim=[actions[name].shape[1] for name in term_names],
            action=torch.cat([actions[name] for name in term_names], dim=-1),
            prev_action=torch.cat([previous_actions[name] for name in term_names], dim=-1),
        ),
    )
    expected_magnitude = left_arm.square().sum(dim=-1) + right_arm.square().sum(dim=-1)
    expected_rate = (left_arm - previous_left_arm).square().sum(dim=-1) + (right_arm - previous_right_arm).square().sum(
        dim=-1
    )
    cfg = RewardsCfg()
    for term, expected in (
        (cfg.arm_action_magnitude, expected_magnitude),
        (cfg.arm_action_rate, expected_rate),
    ):
        penalty = term.func(env, **term.params)
        torch.testing.assert_close(penalty, expected)
        assert (penalty * term.weight <= 0.0).all()


def test_task_config_uses_arm_penalties_and_matching_success_threshold() -> None:
    """Enable independent pulling with a smaller bilateral bonus and geometric untying success."""
    cfg = ShoelaceEnvCfg()

    assert [field.name for field in dataclasses.fields(RewardsCfg)] == [
        "dense_task",
        "arm_action_rate",
        "arm_action_magnitude",
    ]
    assert cfg.rewards.arm_action_rate.func is shoelace_rewards.arm_action_rate_l2
    assert cfg.rewards.arm_action_rate.weight == pytest.approx(-0.001)
    assert cfg.rewards.arm_action_magnitude.func is shoelace_rewards.arm_action_l2
    assert cfg.rewards.arm_action_magnitude.weight == pytest.approx(-0.001)
    assert [field.name for field in dataclasses.fields(TerminationsCfg)] == ["success", "time_out"]
    assert cfg.rewards.dense_task.func is shoelace_rewards.dense_task_reward
    assert cfg.rewards.dense_task.params["acquisition_weight"] == pytest.approx(0.3)
    assert cfg.rewards.dense_task.params["approach_fraction"] == pytest.approx(0.3)
    assert cfg.rewards.dense_task.params["bilateral_pull_fraction"] == pytest.approx(0.2)
    assert cfg.rewards.dense_task.params["success_x_separation"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)
    assert cfg.terminations.success.params["threshold"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)
    assert cfg.terminations.success.func is shoelace_terminations.shoelace_success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
