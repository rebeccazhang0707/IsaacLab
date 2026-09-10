# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the task-local shoelace contact observations."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.shoelace.mdp.observations import gripper_close_error, tail_tcp_relative_speed
from isaaclab_tasks.contrib.shoelace.shoelace_env import _aggregate_finger_tail_signed_distance
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import (
    CONTACT_OBSERVATION_HISTORY_LENGTH,
    ObservationsCfg,
)
from isaaclab_tasks.contrib.shoelace.shoelace_physics import CONTACT_DISTANCE_CAP


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


def test_signed_distance_aggregates_only_matching_finger_tail_pairs() -> None:
    """The kernel should retain minimum signed separation and ignore mismatched arms."""
    body_q = wp.array([wp.transform() for _ in range(6)], dtype=wp.transform, device="cpu")
    shape_body = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32, device="cpu")
    contact_shape0 = wp.array([0, 1, 2, 4], dtype=wp.int32, device="cpu")
    contact_shape1 = wp.array([1, 0, 3, 1], dtype=wp.int32, device="cpu")
    contact_point0 = wp.array([wp.vec3() for _ in range(4)], dtype=wp.vec3, device="cpu")
    contact_point1 = wp.array(
        [wp.vec3(1.0e-3, 0.0, 0.0), wp.vec3(-1.0e-4, 0.0, 0.0), wp.vec3(1.2e-3, 0.0, 0.0), wp.vec3()],
        dtype=wp.vec3,
        device="cpu",
    )
    contact_normal = wp.array([wp.vec3(1.0, 0.0, 0.0) for _ in range(4)], dtype=wp.vec3, device="cpu")
    contact_margin0 = wp.array([4.0e-4, 5.0e-5, 1.0e-4, 0.0], dtype=wp.float32, device="cpu")
    contact_margin1 = wp.array([1.0e-4, 5.0e-5, 1.0e-4, 0.0], dtype=wp.float32, device="cpu")
    finger_slot_by_shape = wp.array([0, -1, 1, -1, 2, -1], dtype=wp.int32, device="cpu")
    finger_env_by_shape = wp.array([0, -1, 0, -1, 1, -1], dtype=wp.int32, device="cpu")
    tail_arm_by_shape = wp.array([-1, 0, -1, 0, -1, 1], dtype=wp.int32, device="cpu")
    signed_distance = wp.full((2, 4), CONTACT_DISTANCE_CAP, dtype=wp.float32, device="cpu")

    wp.launch(
        _aggregate_finger_tail_signed_distance,
        dim=4,
        inputs=[
            body_q,
            shape_body,
            wp.array([4], dtype=wp.int32, device="cpu"),
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_margin0,
            contact_margin1,
            finger_slot_by_shape,
            finger_env_by_shape,
            tail_arm_by_shape,
        ],
        outputs=[signed_distance],
        device="cpu",
    )

    expected = np.full((2, 4), CONTACT_DISTANCE_CAP, dtype=np.float32)
    expected[0, 0] = -2.0e-4
    expected[0, 1] = 1.0e-3
    np.testing.assert_allclose(signed_distance.numpy(), expected, atol=1.0e-8)


def test_policy_observation_contract_is_minimal_and_temporal() -> None:
    """Only signed contact distance should carry short policy-step history."""
    policy = ObservationsCfg.PolicyCfg()

    assert not hasattr(policy, "left_finger_pos")
    assert not hasattr(policy, "right_finger_pos")
    assert not hasattr(policy, "tail_velocities")
    assert policy.finger_tail_signed_distance.history_length == CONTACT_OBSERVATION_HISTORY_LENGTH
    assert policy.gripper_close_error.history_length == 0
    assert policy.tail_tcp_relative_speed.history_length == 0
    assert policy.finger_tail_signed_distance.clip == (-CONTACT_DISTANCE_CAP, CONTACT_DISTANCE_CAP)


def test_gripper_error_and_tail_tcp_speed_are_compact_scalars() -> None:
    """The two scalar pairs should encode blocked closure and TCP-relative motion."""
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    robot_left = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_proxy(torch.tensor([[5.0e-3]])),
            joint_pos_target=_proxy(torch.tensor([[1.0e-3]])),
            body_pos_w=_proxy(torch.zeros(1, 1, 3)),
            body_quat_w=_proxy(identity),
            body_link_vel_w=_proxy(torch.tensor([[[0.1, 0.0, 0.0, 0.0, 1.0, 0.0]]])),
        )
    )
    robot_right = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_proxy(torch.tensor([[1.0e-3]])),
            joint_pos_target=_proxy(torch.tensor([[5.0e-3]])),
            body_pos_w=_proxy(torch.zeros(1, 1, 3)),
            body_quat_w=_proxy(identity),
            body_link_vel_w=_proxy(torch.zeros(1, 1, 6)),
        )
    )
    for robot in (robot_left, robot_right):
        robot.data.root_quat_w = _proxy(identity[:, 0])

    left_pose = torch.zeros(1, 3, 7)
    right_pose = torch.zeros(1, 3, 7)
    left_velocity = torch.zeros(1, 3, 6)
    right_velocity = torch.zeros(1, 3, 6)
    left_velocity[..., 1] = 0.2
    right_velocity[..., 0] = 0.3034
    scene = {
        "robot_left": robot_left,
        "robot_right": robot_right,
        "shoelace_left": SimpleNamespace(
            data=SimpleNamespace(segment_pose_w=_proxy(left_pose), segment_velocity_w=_proxy(left_velocity))
        ),
        "shoelace_right": SimpleNamespace(
            data=SimpleNamespace(segment_pose_w=_proxy(right_pose), segment_velocity_w=_proxy(right_velocity))
        ),
    }
    env = SimpleNamespace(scene=scene)
    robot_cfgs = (
        SimpleNamespace(name="robot_left", joint_ids=[0], body_ids=[0]),
        SimpleNamespace(name="robot_right", joint_ids=[0], body_ids=[0]),
    )
    cable_cfgs = (SimpleNamespace(name="shoelace_left"), SimpleNamespace(name="shoelace_right"))

    torch.testing.assert_close(gripper_close_error(env, robot_cfgs), torch.tensor([[4.0e-3, 0.0]]))
    torch.testing.assert_close(tail_tcp_relative_speed(env, cable_cfgs, robot_cfgs), torch.tensor([[0.1, 0.2]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
