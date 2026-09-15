# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified shoelace task reward and success condition."""

import dataclasses
import math
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
def test_legacy_dense_reward_orders_approach_contact_grasp_and_pull(
    monkeypatch: pytest.MonkeyPatch, tail_distance: float, first_arm: int
) -> None:
    """Acquisition and pulling should pay for progress, with no net credit for a closed state cycle."""
    tail_vectors = torch.full((1, 6), 1.0)
    signed_distance = torch.full((1, 4), CONTACT_DISTANCE_CAP)
    relative_speed = torch.zeros((1, 2))
    tail_x = torch.tensor([[-0.05, 0.05]])
    initial_tail_x = tail_x.clone()
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
    cfg = RewardsCfg().dense_task
    cfg.params["pull_use_high_water_mark"] = False
    cfg.params["robot_cfgs"] = robot_cfgs
    cfg.params["grasp_filter_time_constant"] = 1.0e-6
    term = shoelace_rewards.dense_task_reward(cfg, env)
    reward_history = []

    def compute() -> torch.Tensor:
        env.scene.update(_cables(tail_x))
        for cable_cfg in CABLE_CFGS:
            env.scene[cable_cfg.name].data.segment_pose_w.torch[:, :, :3] += translation
        reward = term(env, **cfg.params)
        # Report signed physical motion, even without a grasp or after translating the whole scene.
        expected_displacement = (tail_x - initial_tail_x) * torch.tensor(outward_sign)
        for arm, side in enumerate(("left", "right")):
            torch.testing.assert_close(
                env.extras["log"][f"Metrics/shoelace/pull_{side}_displacement_m"],
                expected_displacement[0, arm],
                atol=1.0e-6,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                env.extras["log"][f"Metrics/shoelace/pull_{side}"],
                env.extras["log"][f"Metrics/shoelace/pull_{side}_score"],
            )
        reward_history.append(reward.clone())
        return reward

    initial_reward = compute()
    torch.testing.assert_close(initial_reward, torch.zeros(1))
    initial_log = env.extras["log"]
    assert initial_log["Metrics/shoelace/approach_distance_m"].item() > 0.0
    assert initial_log["Metrics/shoelace/grasp_both"].item() == 0.0
    tail_vectors.reshape(1, 2, 3)[:, first_arm].zero_()
    first_approach = compute().item() * env.step_dt * cfg.weight
    tail_vectors.reshape(1, 2, 3)[:, other_arm].zero_()
    second_approach = compute().item() * env.step_dt * cfg.weight
    # The 3-point approach budget gives 0.75 to one arm and the remaining 2.25 to the other.
    assert first_approach == pytest.approx(0.75, abs=1.0e-5)
    assert second_approach == pytest.approx(2.25, abs=1.0e-5)
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
    first_grasp = compute().item() * env.step_dt * cfg.weight
    # A single grasp earns 0.45 acquisition credit plus 0.20 initial-position pull credit.
    assert first_grasp == pytest.approx(0.65, abs=1.0e-5)
    assert env.extras["log"][f"Metrics/shoelace/grasp_{('left', 'right')[first_arm]}"].item() > 0.99
    assert env.extras["log"][f"Metrics/shoelace/grasp_{('left', 'right')[other_arm]}"].item() < 1.0e-5
    assert env.extras["log"]["Metrics/shoelace/grasp_both"].item() < 1.0e-5
    # Stationary grasp acquisition raises the score to 0.5, not the displacement.
    first_score_key = f"Metrics/shoelace/pull_{('left', 'right')[first_arm]}_score"
    assert env.extras["log"][first_score_key].item() == pytest.approx(0.5, abs=1.0e-5)

    signed_distance[:, other_contact].zero_()
    second_grasp = compute().item() * env.step_dt * cfg.weight
    # At the initial positions, H(0.5, 0.5) = 1/3 and both grasps receive the full acquisition budget.
    both_grasp_credit = 3.0 + 4.0 * (0.2 * 0.5 + 0.8 / 3.0)
    assert first_grasp + second_grasp == pytest.approx(both_grasp_credit, abs=3.0e-5)
    signed_distance[:, other_contact].fill_(CONTACT_DISTANCE_CAP)
    assert compute().item() * env.step_dt * cfg.weight == pytest.approx(-second_grasp, abs=1.0e-5)

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

    tail_vectors.fill_(1.0)
    signed_distance.fill_(CONTACT_DISTANCE_CAP)
    tail_x[:] = torch.tensor([[-0.05, 0.05]])
    for robot in robots.values():
        robot.data.joint_pos.torch.fill_(0.01)
    compute()
    torch.testing.assert_close(torch.stack(reward_history).sum(dim=0) * env.step_dt, torch.zeros(1))


@pytest.mark.parametrize("first_arm", [0, 1])
def test_pull_pays_new_displacement_without_regrasp_or_passive_motion_credit(
    monkeypatch: pytest.MonkeyPatch, first_arm: int
) -> None:
    """Only new physical records with current grasps earn the bounded pull budget."""
    grasp = torch.zeros((2, 2))
    finite = torch.ones(2, dtype=torch.bool)
    outward = torch.full((2, 2), 0.05)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.1, extras={}, termination_manager=_termination_manager(2))
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: torch.zeros((2, 6)))
    monkeypatch.setattr(shoelace_rewards, "_grasp_quality", lambda *args: (grasp, finite))
    monkeypatch.setattr(shoelace_rewards, "tail_outward_x", lambda *args: outward)
    monkeypatch.setattr(shoelace_rewards, "tail_x_separation", lambda *args: outward.sum(dim=1))
    cfg = RewardsCfg().dense_task
    cfg.params["acquisition_weight"] = 0.0
    cfg.params["grasp_filter_time_constant"] = 1.0e-6
    term = cfg.func(cfg, env)
    budget = cfg.weight
    scale = (cfg.params["success_x_separation"] - outward[0].sum().item()) / 2
    other_arm = 1 - first_arm

    def compute() -> torch.Tensor:
        return term(env, **cfg.params) * cfg.weight * env.step_dt

    torch.testing.assert_close(compute(), torch.zeros(2))
    grasp.fill_(0.9)
    torch.testing.assert_close(compute(), torch.zeros(2))  # Acquisition alone is not pulling.
    grasp.fill_(0.5)
    outward += scale * 0.25
    first_pull = compute()
    assert (first_pull > 0).all()  # Grasp dropped, but remains valid and both tails moved outward.
    torch.testing.assert_close(compute(), torch.zeros(2))
    outward -= scale * 0.25
    torch.testing.assert_close(compute(), torch.zeros(2))
    outward += scale * 0.25
    torch.testing.assert_close(compute(), torch.zeros(2))  # No second payment for the same record.

    # No retrospective credit for passive records, even with residual filtered grasp on release.
    cfg.params["grasp_filter_time_constant"] = 0.1
    grasp.zero_()
    outward += scale * 0.25
    torch.testing.assert_close(compute(), torch.zeros(2))
    grasp.fill_(0.9)
    torch.testing.assert_close(compute(), torch.zeros(2))

    finite[0] = False
    outward[0] = 0.05 + scale * 2
    torch.testing.assert_close(compute(), torch.zeros(2))
    finite[0] = True
    outward[0] = outward[1]
    torch.testing.assert_close(compute(), torch.zeros(2))

    # Only the selected reset clears records; its first sample seeds without credit.
    term.reset([0])
    outward[0] = 0.05
    torch.testing.assert_close(compute(), torch.zeros(2))
    outward[0] += scale * 0.25
    reset_pull = compute()
    torch.testing.assert_close(reset_pull, torch.tensor([first_pull[0], 0.0]))

    # A new record for one arm earns only independent credit, without a current opposite grasp.
    grasp[:, other_arm] = 0
    outward[:, first_arm] += scale * 0.25
    independent = compute()
    expected_independent = budget * (1 - cfg.params["bilateral_pull_fraction"]) * 0.25 / 2
    torch.testing.assert_close(independent, torch.full((2,), expected_independent))
    grasp[:, other_arm] = 0.9
    torch.testing.assert_close(compute(), torch.zeros(2))  # No delayed cooperation bonus on regrasp.
    outward[:] = 0.05 + scale * 2
    final_pull = compute()
    assert (final_pull > 0).all()
    assert reset_pull[0] + independent[0] + final_pull[0] <= budget
    assert first_pull[1] + independent[1] + final_pull[1] <= budget
    outward += scale
    torch.testing.assert_close(compute(), torch.zeros(2))  # Progress saturates at the physical target.


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
    reward_cfg = RewardsCfg().success
    for step_dt in (1.0 / 30.0, 1.0 / 60.0):
        env = SimpleNamespace(
            step_dt=step_dt,
            termination_manager=SimpleNamespace(
                get_term=lambda name: success if name == "success" else torch.zeros_like(success)
            ),
        )
        reward = reward_cfg.func(env, **reward_cfg.params) * reward_cfg.weight * step_dt
        torch.testing.assert_close(reward, 5.0 * success.float())


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
            "pull_left_score",
            "pull_right_score",
            "pull_left_displacement_m",
            "pull_right_displacement_m",
            "pull_x_separation_m",
            "success_rate",
            "valid_fraction",
        )
    }
    signed_distance.zero_()
    assert (compute() > 0.0).all()
    assert (compute() > 0.0).all()

    signed_distance[0] = torch.nan
    x_separation[0] = 0.40
    reward = compute()
    assert reward[0].item() == 0.0
    assert reward[1].item() > 0.0
    assert env.extras["log"]["Metrics/shoelace/valid_fraction"].item() == 0.5
    assert all(torch.isfinite(value) for value in env.extras["log"].values())
    for side in ("left", "right"):
        assert env.extras["log"][f"Metrics/shoelace/pull_{side}_displacement_m"].item() == 0.0
    signed_distance.zero_()
    x_separation[0] = 0.10
    assert (compute() > 0.0).all()

    term.reset([0])
    x_separation[0] = 0.08
    reward = compute()
    assert reward[0].item() == 0.0
    assert reward[1].item() > 0.0
    for side in ("left", "right"):
        assert env.extras["log"][f"Metrics/shoelace/pull_{side}_displacement_m"].item() == 0.0

    x_separation.fill_(0.14)
    assert (compute() > 0.0).all()
    # Only the reset environment rebases: outward motion is 0.03 and 0.02 m per arm.
    expected_displacement = ((x_separation - torch.tensor([0.08, 0.10])) / 2.0).mean()
    for side in ("left", "right"):
        torch.testing.assert_close(
            env.extras["log"][f"Metrics/shoelace/pull_{side}_displacement_m"], expected_displacement
        )
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
    for side in ("left", "right"):
        assert env.extras["log"][f"Metrics/shoelace/pull_{side}_displacement_m"].item() == 0.0
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


@pytest.mark.parametrize("step_dt", [1.0 / 30.0, 1.0 / 60.0])
def test_loaded_grasp_tolerates_solver_penetration_but_rejects_gaps_and_slip(
    monkeypatch: pytest.MonkeyPatch, step_dt: float
) -> None:
    """Loaded two-finger contact must not look like a lost grasp merely due to solver penetration."""
    signed_distance = torch.zeros((7, 4))
    signed_distance[1] = torch.tensor([-0.00054, -0.00092, -0.00090, -0.00056])
    signed_distance[2] = 0.0009
    signed_distance[3] = -0.004
    signed_distance[4, [0, 2]] = CONTACT_DISTANCE_CAP
    relative_speed = torch.zeros((7, 2))
    relative_speed[6] = 0.8
    robots = {
        name: SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.full((7, 1), 0.001))))
        for name in ("robot_left", "robot_right")
    }
    for robot in robots.values():
        robot.data.joint_pos.torch[5] = 0.01
    env = SimpleNamespace(num_envs=7, device="cpu", step_dt=step_dt, scene=robots)
    cfg = RewardsCfg().grasp_hold
    cfg.params["robot_cfgs"] = tuple(SimpleNamespace(name=name, joint_ids=[0]) for name in robots)
    monkeypatch.setattr(shoelace_rewards, "finger_tail_signed_distance", lambda *args: signed_distance)
    monkeypatch.setattr(shoelace_rewards, "tail_tcp_relative_speed", lambda *args: relative_speed)
    term = cfg.func(cfg, env)
    scores = term(env, **cfg.params)

    torch.testing.assert_close(scores[1], scores[0])
    assert scores[0] > 0.99
    assert scores[2] < 0.02
    assert (scores[3:] < 1.0e-5).all()
    cfg.params["contact_penetration_tolerance"] = 0.0
    term.reset()
    assert term(env, **cfg.params)[1] < 0.05  # Explicit opt-out restores the old contact score.
    cfg.params["contact_penetration_tolerance"] = -0.001
    with pytest.raises(ValueError, match="contact_penetration_tolerance"):
        term(env, **cfg.params)


@pytest.mark.parametrize("step_dt", [1.0 / 30.0, 1.0 / 60.0])
def test_hold_budget_keeps_acquisition_credit_without_renewal_on_regrasp(
    monkeypatch: pytest.MonkeyPatch, step_dt: float
) -> None:
    """Full-rate quality-time is episode limited; later grasps retain a smaller maintenance reward."""
    grasp = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    finite = torch.ones(2, dtype=torch.bool)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=step_dt)
    monkeypatch.setattr(shoelace_rewards, "_grasp_quality", lambda *args: (grasp, finite))
    cfg = RewardsCfg().grasp_hold
    cfg.params["grasp_filter_time_constant"] = 1.0e-6
    term = cfg.func(cfg, env)

    def compute() -> torch.Tensor:
        return term(env, **cfg.params) * cfg.weight * step_dt

    total = torch.stack([compute() for _ in range(round(10 / step_dt))]).sum(dim=0)
    # Two full-rate quality-seconds, then 20%: bilateral 2 + .2 * 8, unilateral 2 + .2 * .5.
    torch.testing.assert_close(total, torch.tensor([3.6, 2.1]), atol=1.0e-5, rtol=1.0e-5)
    grasp.zero_()
    torch.testing.assert_close(compute(), torch.zeros(2))
    grasp[:] = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(compute() / step_dt, torch.tensor([0.2, 0.05]))

    term.reset([0])
    torch.testing.assert_close(compute() / step_dt, torch.tensor([1.0, 0.05]))
    finite[0] = False
    grasp[0] = torch.nan
    for _ in range(round(3 / step_dt)):
        assert compute()[0] == 0
    finite[0] = True
    grasp[0] = 1.0
    torch.testing.assert_close(compute() / step_dt, torch.tensor([1.0, 0.05]))

    # The budget is timestep independent, including a step crossing the full-rate boundary.
    cfg.params["full_reward_duration"] = step_dt * 1.5
    term.reset()
    partial_total = torch.stack([compute() for _ in range(3)]).sum(dim=0)
    torch.testing.assert_close(partial_total, step_dt * torch.tensor([1.5 + 0.2 * 1.5, 0.75]))


@pytest.mark.parametrize("step_dt", [1.0 / 30.0, 1.0 / 60.0])
def test_grasp_hold_rewards_duration_and_cooperation_without_dense_term(
    monkeypatch: pytest.MonkeyPatch, step_dt: float
) -> None:
    """Pay for retained physical grasps, preserving filters across invalid samples and selective resets."""
    signed_distance = torch.full((4, 4), CONTACT_DISTANCE_CAP)
    signed_distance[0].zero_()
    signed_distance[1, :2].zero_()
    signed_distance[2, 2:].zero_()
    relative_speed = torch.zeros((4, 2))
    robots = {
        name: SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.full((4, 1), 0.001))))
        for name in ("robot_left", "robot_right")
    }
    env = SimpleNamespace(num_envs=4, device="cpu", step_dt=step_dt, scene=robots)
    cfg = RewardsCfg().grasp_hold
    cfg.params["robot_cfgs"] = tuple(SimpleNamespace(name=name, joint_ids=[0]) for name in robots)
    monkeypatch.setattr(shoelace_rewards, "finger_tail_signed_distance", lambda *args: signed_distance)
    monkeypatch.setattr(shoelace_rewards, "tail_tcp_relative_speed", lambda *args: relative_speed)
    term = cfg.func(cfg, env)

    def compute() -> torch.Tensor:
        return term(env, **cfg.params) * cfg.weight * env.step_dt

    stable_reward = compute()
    duration_steps = round(1.0 / step_dt)
    stable_total = torch.stack([compute() for _ in range(duration_steps)]).sum(dim=0)
    # After the manager applies dt and weight, one second pays 1.0 for both and 0.25 for either alone.
    torch.testing.assert_close(stable_total, torch.tensor([1.0, 0.25, 0.25, 0.0]), atol=1.0e-6, rtol=1.0e-4)

    signed_distance[0] = torch.nan
    relative_speed[1] = torch.inf
    invalid_reward = compute()
    torch.testing.assert_close(invalid_reward[:2], torch.zeros(2))
    signed_distance[0].zero_()
    relative_speed.zero_()
    torch.testing.assert_close(compute(), stable_reward)

    # Opening empty or occupied grippers does not earn hold credit; contact chatter pays less than retention.
    chatter_rewards = []
    for step in range(duration_steps):
        for robot in robots.values():
            robot.data.joint_pos.torch[0] = 0.01 if step % 2 == 0 else 0.001
        chatter_rewards.append(compute()[0])
    assert torch.stack(chatter_rewards).sum() < stable_total[0]

    for robot in robots.values():
        robot.data.joint_pos.torch[0] = 0.01
    before_reset = compute()
    term.reset([0])
    after_reset = compute()
    assert after_reset[0].item() == 0.0
    torch.testing.assert_close(after_reset[1:], before_reset[1:])

    # Sustained slip and deep penetration remove the remaining physical grasp credit.
    relative_speed[1] = 0.8
    signed_distance[2] = -0.004
    for _ in range(duration_steps):
        after_loss = compute()
    assert (after_loss < 1.0e-5).all()


@pytest.mark.parametrize("step_dt", [1.0 / 30.0, 1.0 / 60.0])
@pytest.mark.parametrize("first_arm", [0, 1])
def test_pregrasp_rewards_fine_alignment_and_only_nearby_actual_closure(
    monkeypatch: pytest.MonkeyPatch, step_dt: float, first_arm: int
) -> None:
    """Either arm can earn bounded pregrasp progress without credit for far closure or repeated cycles."""
    vectors = torch.full((1, 2, 3), 1.0)
    robots = {
        name: SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.full((1, 1), 0.01))))
        for name in ("robot_left", "robot_right")
    }
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=step_dt, scene=robots)
    cfg = RewardsCfg().pregrasp
    cfg.params["robot_cfgs"] = tuple(SimpleNamespace(name=name, joint_ids=[0]) for name in robots)
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: vectors.flatten(start_dim=1))
    term = cfg.func(cfg, env)
    positions = [robot.data.joint_pos.torch for robot in robots.values()]
    history = []

    def compute() -> float:
        reward = (term(env, **cfg.params) * cfg.weight * step_dt).item()
        history.append(reward)
        return reward

    assert compute() == 0.0
    positions[first_arm].fill_(0.001)
    assert compute() == 0.0  # Closing away from the tail has no pregrasp value.
    positions[first_arm].fill_(0.01)
    assert compute() == 0.0

    vectors[0, first_arm] = torch.tensor([0.012, 0.0, 0.0])
    expected_alignment = 0.375 * math.exp(-((0.012 / cfg.params["alignment_std"]) ** 2))
    assert compute() == pytest.approx(expected_alignment)
    positions[first_arm].fill_(0.001)
    assert compute() == 0.0
    positions[first_arm].fill_(0.01)
    assert compute() == 0.0

    vectors[0, first_arm, 0] = cfg.params["closure_radius"]
    assert compute() > 0.0
    positions[first_arm].fill_(0.001)
    assert compute() == 0.0  # The compact closure gate is zero at its boundary.
    positions[first_arm].fill_(0.01)
    assert compute() == 0.0

    vectors[0, first_arm, 0] = cfg.params["closure_radius"] / 2
    assert compute() > 0.0
    positions[first_arm].fill_(0.001)
    # At half the radius, the gate is (1 - 1/4)^2 = 9/16.
    assert compute() == pytest.approx(0.125 * 9 / 16, abs=1.0e-7)
    assert compute() == 0.0  # Holding the same state does not earn an attempt reward repeatedly.
    positions[first_arm].fill_(0.01)
    assert compute() == pytest.approx(-0.125 * 9 / 16, abs=1.0e-7)

    vectors[0, first_arm].zero_()
    assert compute() > 0.0
    assert sum(history) == pytest.approx(0.375, abs=1.0e-6)
    positions[first_arm].fill_(0.001)
    assert compute() == pytest.approx(0.125)
    vectors[0, 1 - first_arm].zero_()
    assert compute() == pytest.approx(0.375)
    positions[1 - first_arm].fill_(0.001)
    assert compute() == pytest.approx(0.125)
    assert sum(history) == pytest.approx(1.0, abs=1.0e-6)

    vectors.fill_(1.0)
    for position in positions:
        position.fill_(0.01)
    assert compute() == pytest.approx(-1.0)
    assert sum(history) == pytest.approx(0.0, abs=1.0e-6)


def test_pregrasp_preserves_invalid_samples_and_resets_selected_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid geometry or joint state cannot advance progress, and reset gives no new acquisition credit."""
    vectors = torch.full((3, 6), 1.0)
    robots = {
        name: SimpleNamespace(data=SimpleNamespace(joint_pos=_proxy(torch.full((3, 1), 0.01))))
        for name in ("robot_left", "robot_right")
    }
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.1, scene=robots)
    cfg = RewardsCfg().pregrasp
    cfg.params["robot_cfgs"] = tuple(SimpleNamespace(name=name, joint_ids=[0]) for name in robots)
    monkeypatch.setattr(shoelace_rewards, "tails_to_tcp", lambda *args: vectors)
    term = cfg.func(cfg, env)

    def compute() -> torch.Tensor:
        return term(env, **cfg.params) * cfg.weight * env.step_dt

    torch.testing.assert_close(compute(), torch.zeros(3))
    vectors.zero_()
    torch.testing.assert_close(compute(), torch.full((3,), 0.75))
    vectors[0] = torch.nan
    for robot in robots.values():
        robot.data.joint_pos.torch[1] = torch.inf
        robot.data.joint_pos.torch[2] = 0.001
    torch.testing.assert_close(compute(), torch.tensor([0.0, 0.0, 0.25]))
    vectors[0].zero_()
    for robot in robots.values():
        robot.data.joint_pos.torch[1] = 0.01
    torch.testing.assert_close(compute(), torch.zeros(3))

    vectors[[0, 2]] = 1.0
    term.reset([0])
    torch.testing.assert_close(compute(), torch.tensor([0.0, 0.0, -1.0]))
    for robot in robots.values():
        robot.data.joint_pos.torch[1] = 0.001
    torch.testing.assert_close(compute(), torch.tensor([0.0, 0.25, 0.0]))
    term.reset()
    torch.testing.assert_close(compute(), torch.zeros(3))

    # Disabling closure leaves the alignment budget unchanged for a focused ablation.
    cfg.params["closure_weight"] = 0.0
    term.reset()
    compute()
    for robot in robots.values():
        robot.data.joint_pos.torch[1] = 0.01
    torch.testing.assert_close(compute(), torch.zeros(3))


def test_task_config_uses_arm_penalties_and_matching_success_threshold() -> None:
    """Budget cooperative progress, time-based retention, and geometric completion separately."""
    cfg = ShoelaceEnvCfg()

    assert [field.name for field in dataclasses.fields(RewardsCfg)] == [
        "dense_task",
        "pregrasp",
        "grasp_hold",
        "success",
        "arm_action_rate",
        "arm_action_magnitude",
    ]
    assert cfg.rewards.arm_action_rate.func is shoelace_rewards.arm_action_rate_l2
    assert cfg.rewards.arm_action_rate.weight == pytest.approx(-0.001)
    assert cfg.rewards.arm_action_magnitude.func is shoelace_rewards.arm_action_l2
    assert cfg.rewards.arm_action_magnitude.weight == pytest.approx(-0.001)
    assert [field.name for field in dataclasses.fields(TerminationsCfg)] == ["success", "time_out"]
    assert cfg.rewards.dense_task.func is shoelace_rewards.dense_task_reward
    assert cfg.rewards.pregrasp.func is shoelace_rewards.pregrasp_progress_reward
    assert cfg.rewards.pregrasp.weight == pytest.approx(1.0)
    assert cfg.rewards.pregrasp.params["alignment_std"] == pytest.approx(0.015)
    assert cfg.rewards.pregrasp.params["closure_radius"] == pytest.approx(0.01)
    assert cfg.rewards.pregrasp.params["alignment_weight"] == pytest.approx(0.75)
    assert cfg.rewards.pregrasp.params["closure_weight"] == pytest.approx(0.25)
    assert cfg.rewards.dense_task.params["reach_std"] == pytest.approx(0.08)
    assert cfg.rewards.dense_task.params["acquisition_weight"] == pytest.approx(0.6)
    assert cfg.rewards.dense_task.params["approach_fraction"] == pytest.approx(0.5)
    assert cfg.rewards.dense_task.params["bilateral_approach_fraction"] == pytest.approx(0.5)
    assert cfg.rewards.dense_task.params["bilateral_grasp_fraction"] == pytest.approx(0.7)
    assert cfg.rewards.dense_task.params["bilateral_pull_fraction"] == pytest.approx(0.8)
    assert cfg.rewards.dense_task.params["pull_use_high_water_mark"] is True
    assert cfg.rewards.dense_task.params["pull_grasp_threshold"] == pytest.approx(0.2)
    assert cfg.rewards.grasp_hold.weight == pytest.approx(1.0)
    assert cfg.rewards.grasp_hold.params["bilateral_grasp_fraction"] == pytest.approx(0.5)
    assert cfg.rewards.grasp_hold.params["full_reward_duration"] == pytest.approx(2.0)
    assert cfg.rewards.grasp_hold.params["sustained_reward_fraction"] == pytest.approx(0.2)
    for key in ("contact_std", "contact_penetration_tolerance", "relative_speed_std", "grasp_filter_time_constant"):
        assert cfg.rewards.grasp_hold.params[key] == cfg.rewards.dense_task.params[key]
    assert cfg.rewards.dense_task.params["contact_penetration_tolerance"] == pytest.approx(0.001)
    assert cfg.rewards.dense_task.params["success_x_separation"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)
    assert cfg.terminations.success.params["threshold"] == pytest.approx(TAIL_SUCCESS_X_SEPARATION)
    assert cfg.terminations.success.func is shoelace_terminations.shoelace_success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
