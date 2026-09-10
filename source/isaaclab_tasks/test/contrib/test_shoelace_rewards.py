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


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("tail_distance", [0.0, 0.02])
def test_dense_reward_orders_approach_contact_grasp_and_pull(
    monkeypatch: pytest.MonkeyPatch, legacy: bool, tail_distance: float
) -> None:
    """Acquisition and pulling should pay for progress, with no net credit for a closed state cycle."""
    tail_vectors = torch.full((1, 6), 0.05)
    signed_distance = torch.full((1, 4), CONTACT_DISTANCE_CAP)
    relative_speed = torch.zeros((1, 2))
    x_separation = torch.tensor([0.10])
    robots = {
        "robot_left": SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.tensor([[0.01]])))),
        "robot_right": SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.tensor([[0.01]])))),
    }
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=0.1, scene=robots, extras={}, termination_manager=_termination_manager(1)
    )
    robot_cfgs = (
        SimpleNamespace(name="robot_left", joint_ids=[0]),
        SimpleNamespace(name="robot_right", joint_ids=[0]),
    )
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: tail_vectors)
    monkeypatch.setattr(shoelace_rewards, "finger_tail_signed_distance", lambda *args: signed_distance)
    monkeypatch.setattr(shoelace_rewards, "tail_tcp_relative_speed", lambda *args: relative_speed)
    monkeypatch.setattr(shoelace_rewards, "tail_x_separation", lambda *args: x_separation)
    term = shoelace_rewards.dense_task_reward(None, env)
    reward_history = []
    stage_params = {"acquisition_weight": 0.3, "approach_fraction": 0.3}
    if legacy:
        stage_params = {
            "maximum_progress_rate": 3.0,
            "approach_weight": 0.15,
            "grasp_weight": 0.35,
            "task_weight": 1.0,
        }

    def compute() -> torch.Tensor:
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
            **stage_params,
        )
        reward_history.append(reward.clone())
        return reward

    if legacy:
        with pytest.warns(DeprecationWarning, match="acquisition_weight and approach_fraction"):
            initial_reward = compute()
    else:
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
    robots["robot_left"].data.joint_pos.torch.fill_(0.001)
    robots["robot_right"].data.joint_pos.torch.fill_(0.001)
    assert abs(compute().item()) < 1.0e-5

    x_separation.fill_(0.18)
    assert abs(compute().item()) < 1.0e-5
    x_separation.fill_(0.10)
    assert abs(compute().item()) < 1.0e-5

    signed_distance[:, :2].zero_()
    # Each arm earns half the grasp budget independently of its TCP-to-tail distance.
    acquisition_gain = 0.175 if legacy else 0.105
    assert compute().item() * env.step_dt == pytest.approx(acquisition_gain, abs=1.0e-5)
    assert env.extras["log"]["Metrics/shoelace/grasp_left"].item() > 0.99
    assert env.extras["log"]["Metrics/shoelace/grasp_right"].item() < 1.0e-5
    assert env.extras["log"]["Metrics/shoelace/grasp_both"].item() < 1.0e-5

    x_separation.fill_(0.18)
    assert abs(compute().item()) < 1.0e-5
    x_separation.fill_(0.10)
    assert abs(compute().item()) < 1.0e-5

    signed_distance.zero_()
    assert compute().item() * env.step_dt == pytest.approx(acquisition_gain, abs=1.0e-5)

    # Either hand can be acquired first; releasing and reacquiring cannot accumulate credit.
    signed_distance[:, :2].fill_(CONTACT_DISTANCE_CAP)
    assert compute().item() * env.step_dt == pytest.approx(-acquisition_gain, abs=1.0e-5)
    signed_distance[:, 2:].fill_(CONTACT_DISTANCE_CAP)
    assert compute().item() * env.step_dt == pytest.approx(-acquisition_gain, abs=1.0e-5)
    signed_distance[:, 2:].zero_()
    assert compute().item() * env.step_dt == pytest.approx(acquisition_gain, abs=1.0e-5)
    signed_distance[:, :2].zero_()
    assert compute().item() * env.step_dt == pytest.approx(acquisition_gain, abs=1.0e-5)

    x_separation.fill_(0.18)
    assert compute().item() * env.step_dt == pytest.approx(1.0 if legacy else 0.7, abs=1.0e-5)
    assert env.extras["log"]["Metrics/shoelace/pull_progress"].item() == pytest.approx(1.0)
    assert env.extras["log"]["Metrics/shoelace/pull_score"].item() > 0.99
    assert abs(compute().item()) < 1.0e-6

    signed_distance.fill_(-4.0e-3)
    assert compute().item() < 0.0

    tail_vectors.fill_(0.05)
    signed_distance.fill_(CONTACT_DISTANCE_CAP)
    x_separation.fill_(0.10)
    for robot in robots.values():
        robot.data.joint_pos.torch.fill_(0.01)
    compute()
    torch.testing.assert_close(torch.stack(reward_history).sum(dim=0) * env.step_dt, torch.zeros(1))


def test_success_is_only_finite_absolute_tail_x_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The requested success contract should have one geometric threshold."""
    separation = torch.tensor([0.179, 0.18, 0.25, torch.nan])
    monkeypatch.setattr(shoelace_terminations, "tail_x_separation", lambda *args: separation)

    success = shoelace_terminations.tail_x_separation_success(None, threshold=0.18, cable_cfgs=None)

    torch.testing.assert_close(success, torch.tensor([False, True, True, False]))


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

    def compute() -> torch.Tensor:
        return term(env, **cfg.params)

    torch.testing.assert_close(compute(), torch.zeros(num_envs))
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
    reward = compute()
    assert reward[0].item() == 0.0
    assert reward[1].item() > 0.0

    x_separation.fill_(0.14)
    assert (compute() > 0.0).all()
    relative_speed.fill_(0.8)
    assert (compute() < 0.0).all()
    assert env.extras["log"]["Metrics/shoelace/grasp_slip_mps"].item() == pytest.approx(0.8)

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
    """Keep the geometric objective while enabling weaker magnitude than action-change costs."""
    cfg = ShoelaceEnvCfg()

    assert [field.name for field in dataclasses.fields(RewardsCfg)] == [
        "dense_task",
        "arm_action_rate",
        "arm_action_magnitude",
    ]
    assert cfg.rewards.arm_action_rate.func is shoelace_rewards.arm_action_rate_l2
    assert cfg.rewards.arm_action_rate.weight == pytest.approx(-0.01)
    assert cfg.rewards.arm_action_magnitude.func is shoelace_rewards.arm_action_l2
    assert cfg.rewards.arm_action_magnitude.weight == pytest.approx(-0.001)
    assert [field.name for field in dataclasses.fields(TerminationsCfg)] == ["success", "time_out"]
    assert cfg.rewards.dense_task.func is shoelace_rewards.dense_task_reward
    assert cfg.rewards.dense_task.params["acquisition_weight"] == pytest.approx(0.3)
    assert cfg.rewards.dense_task.params["approach_fraction"] == pytest.approx(0.3)
    assert "maximum_progress_rate" not in cfg.rewards.dense_task.params
    assert cfg.rewards.dense_task.params["success_x_separation"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)
    assert cfg.terminations.success.params["threshold"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
