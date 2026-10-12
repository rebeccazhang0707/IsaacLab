# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for gradual finger targets and isolated action resets."""

from types import SimpleNamespace

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.contrib.shoelace.mdp.inspire_hand import InspireFingerAction
from isaaclab_tasks.contrib.shoelace.mdp.inspire_hand_cfg import InspireFingerActionCfg


def _finger_action() -> tuple[InspireFingerAction, SimpleNamespace]:
    names = ["yaw", "pitch", "intermediate", "distal", "index", "index_follower"]
    opened = torch.tensor([1.1, 0.0, 0.0, 0.0, 0.4, 0.4]).expand(3, 6).clone()
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=opened.clone()),
            joint_pos=SimpleNamespace(torch=opened.clone()),
        ),
        find_joints=lambda expressions, as_proxy: (ProxyArray(wp.array([0, 1, 2, 3, 4, 5], device="cpu")), names),
    )
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.1, scene={"robot": asset})
    cfg = InspireFingerActionCfg(
        asset_name="robot",
        joint_names=names,
        open_command_expr=dict(zip(names, [1.1, 0.0, 0.0, 0.0, 0.4, 0.4], strict=True)),
        close_command_expr=dict(zip(names, [1.1, 0.25, 0.4, 0.6, 0.6, 0.6], strict=True)),
        max_target_velocity=0.4,
    )
    return InspireFingerAction(cfg, env), asset


def test_finger_targets_preserve_mimic_ratios_and_rate_bound() -> None:
    """Closing must stay on the mimic trajectory without exceeding the configured target speed."""
    action, _ = _finger_action()
    previous = torch.tensor([1.1, 0.0, 0.0, 0.0, 0.4, 0.4]).expand(3, 6).clone()
    for _ in range(16):
        action.process_actions(torch.tensor([[-1.0], [-1.0], [1.0]]))
        targets = action.processed_actions
        assert (targets - previous).abs().max() <= 0.040001
        torch.testing.assert_close(targets[:, 2], 1.6 * targets[:, 1])
        torch.testing.assert_close(targets[:, 3], 2.4 * targets[:, 1])
        torch.testing.assert_close(targets[:, 4], targets[:, 5])
        torch.testing.assert_close((targets[:, 4] - 0.4) / 0.2, targets[:, 1] / 0.25)
        torch.testing.assert_close(targets[2], torch.tensor([1.1, 0.0, 0.0, 0.0, 0.4, 0.4]))
        previous = targets.clone()
    torch.testing.assert_close(targets[:2], torch.tensor([1.1, 0.25, 0.4, 0.6, 0.6, 0.6]).expand(2, 6))


def test_partial_reset_restarts_only_selected_finger_trajectories() -> None:
    """A reset finger must restart from its reset pose without altering neighboring environments."""
    action, asset = _finger_action()
    command = -torch.ones(3, 1)
    for _ in range(16):
        action.process_actions(command)
    before = action.processed_actions.clone()
    asset.data.joint_pos.torch[:] = before
    asset.data.joint_pos.torch[1] = torch.tensor([1.1, 0.0, 0.0, 0.0, 0.4, 0.4])
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.processed_actions[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(action.processed_actions[1], torch.tensor([1.1, 0.0, 0.0, 0.0, 0.4, 0.4]))
    action.process_actions(command)
    assert action.processed_actions[1, 3] <= 0.040001
    torch.testing.assert_close(action.processed_actions[[0, 2]], before[[0, 2]])
