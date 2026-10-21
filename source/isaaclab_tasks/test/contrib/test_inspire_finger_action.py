# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for gradual finger targets and isolated action resets."""

from types import SimpleNamespace

import torch
import warp as wp

from isaaclab.managers import ActionManager
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_euler_xyz, quat_mul
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.contrib.shoelace.g1_inspire_env_cfg import G1ShoelaceEnvCfg
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


def test_g1_binary_commands_move_indices_and_hold_thumb_targets() -> None:
    """The task's real action configuration must retain fixed thumb targets through open/close cycles."""
    cfg = G1ShoelaceEnvCfg()
    for action_cfg in (cfg.actions.left_gripper, cfg.actions.right_gripper):
        names = list(action_cfg.open_command_expr)
        opened = torch.tensor([action_cfg.open_command_expr[name] for name in names])[None]
        asset = SimpleNamespace(
            data=SimpleNamespace(
                default_joint_pos=SimpleNamespace(torch=opened.clone()),
                joint_pos=SimpleNamespace(torch=opened.clone()),
            ),
            find_joints=lambda expressions, as_proxy: (
                ProxyArray(wp.array(list(range(len(names))), device="cpu")),
                names,
            ),
        )
        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=1.0 / 30.0, scene={"robot": asset})
        action = InspireFingerAction(action_cfg, env)
        indices = [i for i, name in enumerate(names) if "thumb" in name]
        straight = [i for i, name in enumerate(names) if "intermediate" in name or "distal" in name]
        index_proximal = next(i for i, name in enumerate(names) if "index_proximal" in name)
        for command in (-1.0, 1.0):
            for _ in range(12):
                action.process_actions(torch.tensor([[command]]))
                torch.testing.assert_close(action.processed_actions[:, indices], opened[:, indices])
                torch.testing.assert_close(action.processed_actions[:, straight], torch.zeros(1, len(straight)))
            if command < 0:
                assert action.processed_actions[0, index_proximal] > opened[0, index_proximal] + 0.2
            else:
                torch.testing.assert_close(action.processed_actions, opened)


def test_g1_fixed_tcp_actions_ignore_finger_motion_and_rotate_about_the_control_point() -> None:
    """Real G1 arm actions must control the wrist-local TCP without compensating finger closure."""
    cfg = G1ShoelaceEnvCfg()
    for arm, side in enumerate(("left", "right")):
        action_cfg = getattr(cfg.actions, f"{side}_arm")
        env, q, targets, body_pos, body_quat, update_geometry = _linear_wrist_env(side)
        # Only the joint selection changes: retain the task's controller, body, offset and scales.
        action_cfg.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        finger_cfg = InspireFingerActionCfg(
            asset_name="robot",
            joint_names=["finger"],
            open_command_expr={"finger": 0.0},
            close_command_expr={"finger": -0.018},
            max_target_velocity=0.06,
        )
        manager = ActionManager(SimpleNamespace(arm=action_cfg, **{f"{side}_gripper": finger_cfg}), env)
        offset = q.new_tensor(cfg.tcp_body_offsets[arm][0]).expand(1, 3)
        assert offset.norm() > 0.01

        def tcp_position():
            return body_pos[:, 1] + quat_apply(body_quat[:, 1], offset)

        def step(command):
            manager.process_action(command)
            # Solve the same relative target, rather than accumulating another policy command.
            for _ in range(6):
                manager.apply_action()
                q.copy_(targets)
                update_geometry()

        initial_arm = q[:, :6].clone()
        initial_tcp = tcp_position().clone()
        initial_midpoint = body_pos[:, 2:].mean(dim=1).clone()
        command = torch.zeros(1, 7)
        command[:, -1] = -1.0
        for _ in range(6):
            step(command)
            torch.testing.assert_close(q[:, :6], initial_arm, atol=1e-6, rtol=0)
            torch.testing.assert_close(tcp_position(), initial_tcp, atol=1e-6, rtol=0)
        assert (body_pos[:, 2:].mean(dim=1) - initial_midpoint).norm() > 0.008
        command[:, -1] = 1.0
        for _ in range(6):
            step(command)
            torch.testing.assert_close(q[:, :6], initial_arm, atol=1e-6, rtol=0)
        torch.testing.assert_close(q[:, 6], torch.zeros(1), atol=1e-6, rtol=0)

        # Root axes are rotated 90 degrees in world, and the wrist has its own nonzero yaw.
        command[:, 0] = 1.0
        step(command)
        translated = initial_tcp + torch.tensor([[0.0, 0.003, 0.0]])
        torch.testing.assert_close(tcp_position(), translated, atol=1e-6, rtol=0)
        command[:, 0] = 0.0
        command[:, 5] = 1.0
        wrist_before = body_pos[:, 1].clone()
        for _ in range(5):
            step(command)
            torch.testing.assert_close(tcp_position(), translated, atol=1e-6, rtol=0)
        torch.testing.assert_close(q[:, 5], initial_arm[:, 5] + 0.05, atol=1e-6, rtol=0)
        assert (body_pos[:, 1] - wrist_before).norm() > 1e-4


def _linear_wrist_env(side: str):
    """Supply a rotated Cartesian wrist and one moving pad without emulating action behavior."""
    names = ["x", "y", "z", "rx", "ry", "rz", "finger"]
    q = torch.zeros(1, 7)
    q[:, 5] = 0.7
    targets = q.clone()
    root = torch.tensor([[1.0, 2.0, 0.0]])
    rotation = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])
    body_pos = torch.zeros(1, 4, 3)
    body_quat = rotation[:, None].expand(-1, 4, -1).clone()
    jacobian = torch.zeros(1, 3, 6, 7)
    basis = matrix_from_quat(rotation)
    jacobian[:, :, :3, :3] = basis[:, None]
    jacobian[:, :, 3:, 3:6] = basis[:, None]
    jacobian[:, 2, :3, 6] = basis[:, :, 0]

    def find_joints(expressions, as_proxy=False):
        indices = [names.index(name) for name in expressions]
        return (ProxyArray(wp.array(indices, device="cpu")) if as_proxy else indices), expressions

    def set_targets(target, joint_ids):
        indices = wp.to_torch(joint_ids).long() if isinstance(joint_ids, wp.array) else joint_ids
        targets[:, indices] = target

    def update_geometry():
        wrist = root + quat_apply(rotation, q[:, :3])
        wrist_rotation = quat_mul(rotation, quat_from_euler_xyz(q[:, 3], q[:, 4], q[:, 5]))
        body_pos[:, 0] = root
        body_pos[:, 1] = wrist
        body_quat[:, 1:] = wrist_rotation[:, None]
        left = q.new_tensor([[-0.01, 0.0, 0.0]])
        right = -left.clone()
        right[:, 0] += q[:, 6]
        body_pos[:, 2] = wrist + quat_apply(wrist_rotation, left)
        body_pos[:, 3] = wrist + quat_apply(wrist_rotation, right)

    asset = SimpleNamespace(
        num_joints=7,
        num_base_dofs=0,
        is_fixed_base=True,
        find_joints=find_joints,
        find_bodies=lambda expression: ([1], [expression]) if expression == f"{side}_wrist_yaw_link" else ([], []),
        set_joint_position_target_index=set_targets,
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=q.clone()),
            joint_pos=SimpleNamespace(torch=q),
            root_pos_w=SimpleNamespace(torch=root),
            root_quat_w=SimpleNamespace(torch=rotation),
            body_pos_w=SimpleNamespace(torch=body_pos),
            body_quat_w=SimpleNamespace(torch=body_quat),
            body_link_jacobian_w=SimpleNamespace(torch=jacobian),
        ),
    )
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=0.1, scene={"robot": asset}, sim=SimpleNamespace(is_playing=lambda: True)
    )
    update_geometry()
    return env, q, targets, body_pos, body_quat, update_geometry
