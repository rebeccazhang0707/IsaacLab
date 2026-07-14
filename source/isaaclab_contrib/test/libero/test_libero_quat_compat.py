# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for LIBERO demo/sim/policy quaternion order conversion."""

from __future__ import annotations

import math

import pytest
import torch

from isaaclab_contrib.tasks.manipulation.libero.mdp.quat import (
    convert_pose7,
    convert_quat,
    pose7_to_policy,
    pose7_to_sim,
)


def test_identity_quat_roundtrip():
    """Identity: WXYZ [1,0,0,0] ↔ XYZW [0,0,0,1]."""
    wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
    xyzw = convert_quat(wxyz, from_order="wxyz", to_order="xyzw")
    torch.testing.assert_close(xyzw, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(convert_quat(xyzw, from_order="xyzw", to_order="wxyz"), wxyz)


def test_90deg_about_z_quat_conversion():
    """+90° about Z: WXYZ [√2/2, 0, 0, √2/2] ↔ XYZW [0, 0, √2/2, √2/2]."""
    half = math.sqrt(0.5)
    wxyz = torch.tensor([half, 0.0, 0.0, half])
    xyzw = convert_quat(wxyz, from_order="wxyz", to_order="xyzw")
    torch.testing.assert_close(xyzw, torch.tensor([0.0, 0.0, half, half]))
    torch.testing.assert_close(convert_quat(xyzw, from_order="xyzw", to_order="wxyz"), wxyz)


def test_pose7_demo_to_sim_and_policy_boundaries():
    """Demo WXYZ pose → sim XYZW → policy WXYZ preserves position and orientation."""
    half = math.sqrt(0.5)
    demo_pose = torch.tensor([[0.1, -0.2, 0.3, half, 0.0, 0.0, half]])  # WXYZ +90° Z
    sim_pose = pose7_to_sim(demo_pose, from_order="wxyz")
    torch.testing.assert_close(sim_pose[0, :3], demo_pose[0, :3])
    torch.testing.assert_close(sim_pose[0, 3:7], torch.tensor([0.0, 0.0, half, half]))
    policy_pose = pose7_to_policy(sim_pose, to_order="wxyz")
    torch.testing.assert_close(policy_pose, demo_pose)


def test_pose7_identity_when_orders_match():
    """No conversion when from_order == to_order."""
    pose = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    out = convert_pose7(pose, from_order="xyzw", to_order="xyzw")
    torch.testing.assert_close(out, pose)
    # Same tensor when no conversion needed.
    assert out is pose


def test_pose7_batched_roundtrip():
    """Batched pose7 WXYZ ↔ XYZW roundtrip."""
    poses_wxyz = torch.tensor(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],  # identity
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 180° about X (wxyz)
        ]
    )
    poses_xyzw = convert_pose7(poses_wxyz, from_order="wxyz", to_order="xyzw")
    torch.testing.assert_close(poses_xyzw[:, :3], poses_wxyz[:, :3])
    torch.testing.assert_close(poses_xyzw[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(poses_xyzw[1, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(convert_pose7(poses_xyzw, from_order="xyzw", to_order="wxyz"), poses_wxyz)


def test_pose7_rejects_short_vectors():
    with pytest.raises(ValueError, match="pose\\[\\.\\.\\., 7\\]"):
        convert_pose7(torch.zeros(4), from_order="wxyz", to_order="xyzw")
