# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused reset and material tests for the shoelace task."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.managers import RewardManager, RewardTermCfg, SceneEntityCfg, TerminationTermCfg

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
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import configure_shoelace_builder


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
