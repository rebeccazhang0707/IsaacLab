# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Approach, grasp, hold, and pull shoelace tails with two scripted Franka arms."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

wp.config.enable_backward = False

from isaaclab_newton.physics import NewtonManager  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerProxyCfg  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import isaaclab_tasks.contrib.shoelace.shoelace_env as shoelace_env_module  # noqa: E402
from isaaclab_tasks.contrib.shoelace.mdp.constants import REFERENCE_PULL_DIRECTIONS, TCP_OFFSET  # noqa: E402
from isaaclab_tasks.contrib.shoelace.mdp.utils import task_state, untying_metrics  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg, setup_preset_cli  # noqa: E402


def _tcp_positions(env) -> torch.Tensor:
    positions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        hand_position = robot.data.body_pos_w.torch[:, body_ids[0]] - env.scene.env_origins
        hand_quaternion = robot.data.body_quat_w.torch[:, body_ids[0]]
        offset = hand_position.new_tensor(TCP_OFFSET).expand_as(hand_position)
        positions.append(hand_position + math_utils.quat_apply(hand_quaternion, offset))
    return torch.stack(positions, dim=1)


def _hand_axes(env) -> torch.Tensor:
    axes = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        quaternion = robot.data.body_quat_w.torch[:, body_ids[0]]
        basis = torch.eye(3, device=env.device).expand(quaternion.shape[0], -1, -1)
        expanded_quaternion = quaternion.unsqueeze(1).expand(-1, 3, -1)
        axes.append(math_utils.quat_apply(expanded_quaternion.reshape(-1, 4), basis.reshape(-1, 3)).reshape(-1, 3, 3))
    return torch.stack(axes, dim=1)


def _hand_quaternions_w(env) -> torch.Tensor:
    quaternions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        quaternions.append(robot.data.body_quat_w.torch[:, body_ids[0]].clone())
    return torch.stack(quaternions, dim=1)


def _outward_tilt_quaternions_w(env, tilt_degrees: float, roll_degrees: float = 0.0) -> torch.Tensor:
    """Build mirrored hand orientations with local +Z tilted outward from world -Z."""
    angle = torch.deg2rad(torch.tensor(tilt_degrees, device=env.device))
    sine = torch.sin(angle)
    cosine = torch.cos(angle)
    zero = cosine.new_zeros(())
    left_x = torch.stack((cosine, zero, -sine))
    left_y = torch.tensor((0.0, -1.0, 0.0), device=env.device)
    left_z = torch.stack((-sine, zero, -cosine))
    right_x = torch.stack((-cosine, zero, -sine))
    right_y = torch.tensor((0.0, 1.0, 0.0), device=env.device)
    right_z = torch.stack((sine, zero, -cosine))
    matrices = torch.stack(
        (
            torch.stack((left_x, left_y, left_z), dim=1),
            torch.stack((right_x, right_y, right_z), dim=1),
        )
    )
    roll = torch.deg2rad(torch.tensor(roll_degrees, device=env.device))
    roll_cosine = torch.cos(roll)
    roll_sine = torch.sin(roll)
    one = roll_cosine.new_ones(())
    roll_matrix = torch.stack(
        (
            torch.stack((roll_cosine, -roll_sine, zero)),
            torch.stack((roll_sine, roll_cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    matrices = matrices @ roll_matrix
    return math_utils.quat_from_matrix(matrices)


def _mirrored_world_yaw_quaternions_w(env, yaw_degrees: float) -> torch.Tensor:
    """Apply opposite world-Z yaw angles to the left and right hands."""
    current_quaternions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        current_quaternions.append(robot.data.body_quat_w.torch[0, body_ids[0]])
    current_quaternions_w = torch.stack(current_quaternions)
    yaw_angles = torch.deg2rad(current_quaternions_w.new_tensor((yaw_degrees, -yaw_degrees)))
    yaw_axes = torch.zeros((2, 3), device=env.device)
    yaw_axes[:, 2] = 1.0
    yaw_quaternions_w = math_utils.quat_from_angle_axis(yaw_angles, yaw_axes)
    return math_utils.quat_mul(yaw_quaternions_w, current_quaternions_w)


def _hand_orientation_errors_b(env, target_quaternions_w: torch.Tensor) -> torch.Tensor:
    errors = []
    for arm, robot_name in enumerate(("robot_left", "robot_right")):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        current_quaternion_w = robot.data.body_quat_w.torch[:, body_ids[0]]
        inverse_root = math_utils.quat_inv(robot.data.root_quat_w.torch)
        current_quaternion_b = math_utils.quat_mul(inverse_root, current_quaternion_w)
        target_quaternion_w = target_quaternions_w[arm].expand_as(current_quaternion_w)
        target_quaternion_b = math_utils.quat_mul(inverse_root, target_quaternion_w)
        zero = torch.zeros((current_quaternion_w.shape[0], 3), device=env.device)
        _, error = math_utils.compute_pose_error(
            zero,
            current_quaternion_b,
            zero,
            target_quaternion_b,
            rot_error_type="axis_angle",
        )
        errors.append(error)
    return torch.stack(errors, dim=1)


def _tail_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    asset_cfgs = (type("AssetCfg", (), {"name": "shoelace_left"})(), type("AssetCfg", (), {"name": "shoelace_right"})())
    state = task_state(env, asset_cfgs)
    return state[3], state[4]


def _tail_directions_w(env) -> torch.Tensor:
    """Return unit vectors from each tail's inner sample toward its free end."""
    left_tail_chain = env.scene["shoelace_right"].data.segment_pose_w.torch[..., :3]
    right_tail_chain = env.scene["shoelace_left"].data.segment_pose_w.torch[..., :3]
    directions = torch.stack(
        (
            left_tail_chain[:, 0] - left_tail_chain[:, 2],
            right_tail_chain[:, -1] - right_tail_chain[:, -3],
        ),
        dim=1,
    )
    return directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _finger_positions(env) -> torch.Tensor:
    positions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        joint_ids, _ = robot.find_joints("panda_finger_joint1")
        positions.append(robot.data.joint_pos.torch[:, joint_ids[0]])
    return torch.stack(positions, dim=1)


def _arm_joint_positions(env) -> torch.Tensor:
    positions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        joint_ids, _ = robot.find_joints("panda_joint[1-7]")
        positions.append(robot.data.joint_pos.torch[:, joint_ids])
    return torch.stack(positions, dim=1)


def _cable_rms_linear_speed(env) -> torch.Tensor:
    velocities = torch.cat(
        [env.scene[name].data.segment_velocity_w.torch[..., :3] for name in ("shoelace_left", "shoelace_right")],
        dim=1,
    )
    return torch.sqrt(torch.mean(torch.square(velocities), dim=(1, 2)))


def _save_settled_state(env, path: Path) -> None:
    """Save the first environment's cable poses as a local zero-velocity reset state."""
    origin = env.scene.env_origins[0].cpu().numpy()
    poses = {}
    for name in ("shoelace_left", "shoelace_right"):
        pose = env.scene[name].data.segment_pose_w.torch[0].cpu().numpy().copy()
        pose[:, :3] -= origin
        poses[name] = pose.astype(np.float32)
    np.savez(path, **poses)


def _save_demonstration(
    path: Path,
    observations: list[torch.Tensor],
    actions: list[torch.Tensor],
    phases: list[str],
    selected_lengths: dict[int, int],
    metadata: dict,
) -> None:
    """Save selected environment trajectories for offline imitation."""
    observation_steps = torch.stack(observations)
    action_steps = torch.stack(actions)
    selected_observations = []
    selected_actions = []
    selected_phases = []
    trajectory_lengths = []
    environment_ids = []
    for env_id, length in sorted(selected_lengths.items()):
        selected_observations.append(observation_steps[:length, env_id])
        selected_actions.append(action_steps[:length, env_id])
        selected_phases.extend(phases[:length])
        trajectory_lengths.append(length)
        environment_ids.append(env_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observations": torch.cat(selected_observations),
            "actions": torch.cat(selected_actions),
            "phases": selected_phases,
            "trajectory_lengths": trajectory_lengths,
            "environment_ids": environment_ids,
            "metadata": metadata,
        },
        path,
    )


def _set_term(actions: torch.Tensor, env, term_name: str, values: torch.Tensor | float) -> None:
    offset = 0
    for active_name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        if active_name == term_name:
            actions[:, offset : offset + dimension] = values
            return
        offset += dimension
    raise KeyError(f"Missing action term {term_name!r}; active terms: {env.action_manager.active_terms}")


def _add_term(actions: torch.Tensor, env, term_name: str, values: torch.Tensor) -> None:
    offset = 0
    for active_name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        if active_name == term_name:
            actions[:, offset : offset + dimension] += values
            return
        offset += dimension
    raise KeyError(f"Missing action term {term_name!r}; active terms: {env.action_manager.active_terms}")


def _set_term_masked(
    actions: torch.Tensor,
    env,
    term_name: str,
    mask: torch.Tensor,
    values: torch.Tensor | float,
) -> None:
    """Set an action term only for selected environments."""
    offset = 0
    for active_name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        if active_name == term_name:
            current = actions[:, offset : offset + dimension]
            replacement = torch.as_tensor(values, dtype=actions.dtype, device=actions.device)
            if replacement.ndim == 0:
                replacement = replacement.expand_as(current)
            actions[:, offset : offset + dimension] = torch.where(mask.unsqueeze(-1), replacement, current)
            return
        offset += dimension
    raise KeyError(f"Missing action term {term_name!r}; active terms: {env.action_manager.active_terms}")


def _set_tcp_translation(
    actions: torch.Tensor,
    env,
    displacement_w: torch.Tensor,
    arm_action_scale: float,
) -> None:
    for arm, robot_name in enumerate(("robot_left", "robot_right")):
        root_quaternion = env.scene[robot_name].data.root_quat_w.torch
        displacement_b = math_utils.quat_apply_inverse(root_quaternion, displacement_w[:, arm])
        arm_action = torch.zeros((actions.shape[0], 6), device=env.device)
        arm_action[:, :3] = displacement_b / arm_action_scale
        _set_term(actions, env, f"{robot_name.removeprefix('robot_')}_arm", arm_action)


def _set_tcp_pose_delta(
    actions: torch.Tensor,
    env,
    displacement_w: torch.Tensor,
    rotation_b: torch.Tensor,
    position_action_scale: float,
    rotation_action_scale: float,
) -> None:
    for arm, robot_name in enumerate(("robot_left", "robot_right")):
        root_quaternion = env.scene[robot_name].data.root_quat_w.torch
        displacement_b = math_utils.quat_apply_inverse(root_quaternion, displacement_w[:, arm])
        arm_action = torch.zeros((actions.shape[0], 6), device=env.device)
        arm_action[:, :3] = displacement_b / position_action_scale
        arm_action[:, 3:] = rotation_b[:, arm] / rotation_action_scale
        _set_term(actions, env, f"{robot_name.removeprefix('robot_')}_arm", arm_action)


def _load_checkpoint_policy(
    path: Path,
    device: str,
    arm_action_scale: float,
    stochastic: bool,
    action_std: float | None,
):
    """Load a standard-MLP actor and reproduce its hybrid action distribution."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["actor_state_dict"]
    required_keys = (
        "obs_normalizer._mean",
        "obs_normalizer._std",
        "distribution.std_param",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
    )
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        raise ValueError(f"Checkpoint does not contain the standard shoelace actor keys: {missing_keys}")
    arm_indices = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    gripper_indices = (6, 13)
    std = state["distribution.std_param"] if action_std is None else torch.full((14,), action_std, device=device)

    def policy(observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1.0e-2)
        hidden = F.elu(F.linear(normalized, state["mlp.0.weight"], state["mlp.0.bias"]))
        hidden = F.elu(F.linear(hidden, state["mlp.2.weight"], state["mlp.2.bias"]))
        output = F.linear(hidden, state["mlp.4.weight"], state["mlp.4.bias"])
        output[:, arm_indices] *= arm_action_scale
        if stochastic:
            output[:, arm_indices] += (
                arm_action_scale * std[list(arm_indices)] * torch.randn_like(output[:, arm_indices])
            )
            gripper_logits = 2.0 * output[:, gripper_indices]
            gripper_positive = torch.bernoulli(torch.sigmoid(gripper_logits))
            gripper_magnitude = output[:, gripper_indices].abs().clamp_min(1.0e-6)
            output[:, gripper_indices] = (2.0 * gripper_positive - 1.0) * gripper_magnitude
        return output.clamp(-1.0, 1.0)

    return policy


def _state(env) -> dict[str, torch.Tensor]:
    tcp = _tcp_positions(env).clone()
    tail, tail_velocity = _tail_state(env)
    tail = tail.clone()
    return {
        "tcp": tcp,
        "tail": tail,
        "tail_velocity": tail_velocity.clone(),
        "distance": torch.linalg.vector_norm(tail - tcp, dim=-1),
        "finger": _finger_positions(env).clone(),
        "arm_joint": _arm_joint_positions(env).clone(),
        "hand_axes": _hand_axes(env).clone(),
    }


def _tail_to_tcp_hand(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Express each tail-to-TCP vector in its assigned hand frame."""
    tail_to_tcp_w = state["tail"] - state["tcp"]
    return torch.sum(state["hand_axes"] * tail_to_tcp_w.unsqueeze(-2), dim=-1)


def _success_metrics(env, success_params: dict) -> dict[str, torch.Tensor]:
    """Evaluate the task success gates without enabling automatic episode resets."""
    positions, velocities, knot, tail_positions, _ = task_state(env, success_params["asset_cfgs"])
    throat_count, tail_distances, tail_separation = untying_metrics(
        positions,
        knot,
        tail_positions,
        success_params["throat_radius"],
    )
    tcp_positions = _tcp_positions(env)
    grasp_distances = torch.linalg.vector_norm(tail_positions - tcp_positions, dim=-1)
    finger_positions = _finger_positions(env)
    grasped = (
        (grasp_distances <= success_params["maximum_success_grasp_distance"])
        & (finger_positions >= success_params["minimum_finger_position"])
        & (finger_positions <= success_params["maximum_finger_position"])
    )
    finite = torch.isfinite(positions).all(dim=(1, 2)) & torch.isfinite(velocities).all(dim=(1, 2))
    spread = torch.linalg.vector_norm(positions - positions.mean(dim=1, keepdim=True), dim=-1).amax(dim=1)
    safe = (
        finite
        & (positions[..., 2].amin(dim=1) >= success_params["minimum_lace_height"])
        & (spread <= success_params["maximum_lace_spread"])
    )
    success = (
        (throat_count <= success_params["maximum_throat_segments"])
        & (tail_distances.amin(dim=1) >= success_params["tail_success_distance"])
        & (tail_separation >= success_params["tail_success_separation"])
        & grasped.all(dim=1)
        & safe
    )
    return {
        "throat_count": throat_count,
        "tail_distances": tail_distances,
        "tail_separation": tail_separation,
        "grasped": grasped,
        "safe": safe,
        "success": success,
    }


def _acquisition_summary(
    mask: torch.Tensor,
    tail_to_tcp_hand: torch.Tensor,
    finger_positions: torch.Tensor,
) -> dict[str, int | list[list[float]] | list[float] | None]:
    """Summarize captured acquisition states for one rollout outcome."""
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "tail_to_tcp_hand_mean_mm": None, "finger_mean_mm": None}
    return {
        "count": count,
        "tail_to_tcp_hand_mean_mm": (tail_to_tcp_hand[mask].mean(dim=0) * 1000).tolist(),
        "finger_mean_mm": (finger_positions[mask].mean(dim=0) * 1000).tolist(),
    }


def _calibrated_curriculum_states(
    approach_trace: list[dict[str, torch.Tensor]],
    reference_tail: torch.Tensor,
    level_count: int,
    gripper_open_phase_fraction: float,
    approach_phase_exponent: float,
    open_position: float,
    closed_position: float,
) -> list[dict[str, object]]:
    distances = torch.stack(
        [torch.linalg.vector_norm(reference_tail[0] - sample["tcp"][0], dim=-1) for sample in approach_trace]
    )
    start_distance = distances[0].clamp_min(1.0e-8)
    normalized_distances = distances / start_distance
    joint_positions = torch.stack([sample["arm_joint"][0] for sample in approach_trace])
    states = []
    for level in range(level_count):
        difficulty = level / (level_count - 1)
        gripper_difficulty = min(difficulty / gripper_open_phase_fraction, 1.0)
        arm_difficulty = (
            max((difficulty - gripper_open_phase_fraction) / (1.0 - gripper_open_phase_fraction), 0.0)
            ** approach_phase_exponent
        )
        arm_positions = []
        measured_distance = []
        for arm in range(2):
            sample = int(torch.argmin(torch.abs(normalized_distances[:, arm] - arm_difficulty)).item())
            arm_positions.append([round(float(value), 6) for value in joint_positions[sample, arm]])
            measured_distance.append(round(float(distances[sample, arm]), 6))
        states.append(
            {
                "level": level,
                "target_distance_fraction": arm_difficulty,
                "measured_distance_m": measured_distance,
                "gripper_position_m": closed_position + gripper_difficulty * (open_position - closed_position),
                "arm_joint_positions_rad": arm_positions,
            }
        )
    return states


def _apply_curriculum_reset_level(env, level: int) -> None:
    """Apply one deterministic curriculum reset level to every environment."""
    env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    difficulty_term = env.curriculum_manager.cfg.pull_to_grasp.func
    difficulty_term.levels[env_ids] = level
    env.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=0)
    env.scene.write_data_to_sim()
    env.sim.forward()


def _sample_joint_interpolation_waypoints(
    env,
    arm_joint_positions_by_level: tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]],
    start_level: int,
    end_level: int,
    waypoint_count: int,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Sample reachable TCP poses along a linear arm-joint interpolation."""
    waypoints = []
    fractions = torch.linspace(0.0, 1.0, waypoint_count + 1, device=env.device)[1:]
    for waypoint_index, fraction in enumerate(fractions, start=1):
        for robot_name, joint_positions_by_level in zip(
            ("robot_left", "robot_right"), arm_joint_positions_by_level, strict=True
        ):
            robot = env.scene[robot_name]
            joint_ids, _ = robot.find_joints("panda_joint.*")
            start = torch.tensor(joint_positions_by_level[start_level], device=env.device)
            end = torch.tensor(joint_positions_by_level[end_level], device=env.device)
            joint_positions = torch.lerp(start, end, fraction).unsqueeze(0).expand(env.num_envs, -1)
            robot.write_joint_position_to_sim_index(position=joint_positions, joint_ids=joint_ids)
        env.sim.forward()
        waypoints.append(
            (
                f"joint_{start_level}_{end_level}_{waypoint_index}",
                _tcp_positions(env)[0].clone(),
                _hand_quaternions_w(env)[0].clone(),
            )
        )
    return waypoints


def _play_curriculum_levels(env, level_count: int, hold_seconds: float, cycle_count: int) -> None:
    """Display every calibrated reset level sequentially without advancing physics."""
    if cycle_count == 0 and not env.sim.has_active_visualizers():
        raise ValueError("Infinite curriculum playback requires an active visualizer.")

    cycle = 0
    while cycle_count == 0 or cycle < cycle_count:
        for level in range(level_count):
            if not env.sim.is_headless_or_exist_active_visualizer():
                return
            _apply_curriculum_reset_level(env, level)
            print(f"[CURRICULUM PLAYBACK] level {level}/{level_count - 1}", flush=True)

            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                if not env.sim.is_headless_or_exist_active_visualizer():
                    return
                env.sim.render()
                time.sleep(1.0 / 60.0)
        cycle += 1


def _shape_material_summary() -> dict[str, dict[str, float | int | list[float]]]:
    model = NewtonManager.get_model()
    friction = wp.to_torch(model.shape_material_mu).cpu()
    categories = {
        "finger": ("panda_leftfinger", "panda_rightfinger"),
        "dynamic_lace": ("/ShoelaceLeft/", "/ShoelaceRight/"),
    }
    summary = {}
    for category, patterns in categories.items():
        indices = [
            index for index, label in enumerate(model.shape_label) if any(pattern in label for pattern in patterns)
        ]
        values = friction[indices]
        summary[category] = {
            "shape_count": len(indices),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "unique": [round(float(value), 6) for value in torch.unique(values)],
        }
    return summary


def _install_finger_mu_override(friction: float) -> None:
    original_configure_builder = shoelace_env_module._ShoelacePhysics._configure_builder

    def _configure_builder(physics, event) -> None:
        original_configure_builder(physics, event)
        builder = NewtonManager.get_builder()
        for shape, label in enumerate(builder.shape_label):
            if "panda_leftfinger" in label or "panda_rightfinger" in label:
                builder.shape_material_mu[shape] = friction

    shoelace_env_module._ShoelacePhysics._configure_builder = _configure_builder


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-Shoelace-DualFranka")
    parser.add_argument(
        "--approach_steps",
        type=int,
        default=240,
        help="Number of approach steps; set to zero to close directly from the reset pose.",
    )
    parser.add_argument("--approach_speed", type=float, default=0.04, help="Maximum TCP approach speed [m/s].")
    parser.add_argument("--approach_tolerance", type=float, default=0.001, help="TCP-to-tail tolerance [m].")
    parser.add_argument(
        "--left_tcp_offset_w",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Move only the left TCP by this fixed world-frame offset from its reset pose [m].",
    )
    parser.add_argument(
        "--right_tcp_offset_w",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Move only the right TCP by this fixed world-frame offset from its reset pose [m].",
    )
    parser.add_argument(
        "--left_tail_to_tcp_target_hand",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target left tail-to-TCP vector in the left-hand frame [m].",
    )
    parser.add_argument(
        "--right_tail_to_tcp_target_hand",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target right tail-to-TCP vector in the right-hand frame [m].",
    )
    parser.add_argument(
        "--pregrasp_clearance",
        type=float,
        default=0.0,
        help="Offset the hand-frame target along local +Z for a collision-free pregrasp waypoint [m].",
    )
    parser.add_argument(
        "--pregrasp_world_z_clearance",
        type=float,
        default=None,
        help="Offset an absolute TCP target upward along world +Z for a collision-free pregrasp waypoint [m].",
    )
    parser.add_argument(
        "--pregrasp_steps",
        type=int,
        default=120,
        help="Maximum control steps for the optional pregrasp waypoint.",
    )
    parser.add_argument(
        "--freeze_approach_tail_targets",
        action="store_true",
        help="Snapshot tail positions at each approach stage instead of chasing displaced tails.",
    )
    parser.add_argument(
        "--grasp_approach_close_step",
        type=int,
        default=None,
        help="Close both grippers at this zero-based step of the final scripted grasp approach.",
    )
    parser.add_argument(
        "--grasp_approach_handoff_distance",
        type=float,
        default=None,
        help="Stop and close each arm independently once its TCP enters this tail-distance window [m].",
    )
    parser.add_argument(
        "--approach_reference_level",
        type=int,
        default=None,
        help="Track the hand orientations measured from this curriculum reset level during approach.",
    )
    parser.add_argument(
        "--approach_orientation_steps",
        type=int,
        default=0,
        help="Maximum steps for converging to the reference hand orientations before tail-relative approach.",
    )
    parser.add_argument(
        "--approach_waypoint_levels",
        type=int,
        nargs="+",
        default=None,
        help="Track TCP poses sampled from these curriculum reset levels before the tail-relative approach.",
    )
    parser.add_argument(
        "--approach_waypoint_steps",
        type=int,
        default=120,
        help="Maximum control steps for each curriculum-pose approach waypoint.",
    )
    parser.add_argument(
        "--joint_interpolation_levels",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Append dense FK waypoints sampled along a linear joint interpolation between two reset levels.",
    )
    parser.add_argument(
        "--joint_interpolation_waypoints",
        type=int,
        default=8,
        help="Number of FK waypoints sampled from the joint interpolation.",
    )
    parser.add_argument(
        "--joint_interpolation_waypoint_steps",
        type=int,
        default=30,
        help="Maximum control steps for each joint-interpolation FK waypoint.",
    )
    parser.add_argument(
        "--final_waypoint_handoff_distance",
        type=float,
        default=None,
        help="Stop each arm at the final waypoint after its TCP enters this tail-distance window [m].",
    )
    parser.add_argument(
        "--align_hand_roll_to_tail",
        action="store_true",
        help="Roll each reference hand so the tail tangent lies in its local X-Z plane.",
    )
    parser.add_argument(
        "--left_grasp_joint_positions",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override the left-arm grasp joint state for diagnostic runs [rad].",
    )
    parser.add_argument(
        "--right_grasp_joint_positions",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override the right-arm grasp joint state for diagnostic runs [rad].",
    )
    parser.add_argument(
        "--outward_tilt_degrees",
        type=float,
        default=None,
        help=(
            "Tilt each hand's local +Z from world -Z by this mirrored angle [deg]; positive angles tilt outward "
            "and negative angles tilt inward."
        ),
    )
    parser.add_argument("--tilt_steps", type=int, default=120)
    parser.add_argument("--tilt_speed_degrees_s", type=float, default=30.0)
    parser.add_argument(
        "--hand_roll_degrees",
        type=float,
        default=0.0,
        help="Rotate both target hand frames around their local +Z approach axes [deg].",
    )
    parser.add_argument(
        "--mirrored_world_yaw_degrees",
        type=float,
        default=None,
        help="Apply positive world-Z yaw to the left hand and negative world-Z yaw to the right hand [deg].",
    )
    parser.add_argument(
        "--orientation_hold_seconds",
        type=float,
        default=0.0,
        help="Hold the post-orientation pose for this wall-clock duration before approaching [s].",
    )
    parser.add_argument(
        "--orientation_advance_m",
        type=float,
        default=0.0,
        help="Advance each TCP toward its baked tail center while applying the orientation target [m].",
    )
    parser.add_argument("--close_steps", type=int, default=30)
    parser.add_argument("--hold_steps", type=int, default=30)
    parser.add_argument("--pull_steps", type=int, default=45)
    parser.add_argument("--pull_speed", type=float, default=0.04)
    parser.add_argument(
        "--pull_tracking_gain",
        type=float,
        default=0.0,
        help="Correct each pull command toward its tail-to-TCP vector at pull start; zero preserves open-loop pull.",
    )
    parser.add_argument(
        "--stop_on_success",
        action="store_true",
        help="Stop the scripted pull as soon as the complete task success gate is satisfied.",
    )
    parser.add_argument(
        "--policy_checkpoint",
        type=Path,
        default=None,
        help="Hand control to a deterministic RSL-RL actor after the scripted approach.",
    )
    parser.add_argument(
        "--policy_steps",
        type=int,
        default=600,
        help="Maximum policy-control steps after the scripted approach.",
    )
    parser.add_argument(
        "--policy_handoff_close_steps",
        type=int,
        default=0,
        help="Hold both arms and close both grippers for this many steps before checkpoint control.",
    )
    parser.add_argument(
        "--policy_arm_action_scale",
        type=float,
        default=0.7,
        help="Continuous arm-output scale used by the checkpoint's action distribution.",
    )
    parser.add_argument(
        "--policy_stochastic",
        action="store_true",
        help="Sample the checkpoint's Gaussian-arm and Bernoulli-gripper action distribution.",
    )
    parser.add_argument(
        "--policy_action_std",
        type=float,
        default=None,
        help="Override the checkpoint's pre-scale Gaussian arm standard deviation.",
    )
    parser.add_argument(
        "--policy_force_grippers_closed",
        action="store_true",
        help="Keep both grippers closed during checkpoint handoff for expert-data diagnostics.",
    )
    parser.add_argument(
        "--policy_gripper_close_distance",
        type=float,
        default=None,
        help="Latch each policy-controlled gripper closed after its TCP enters this tail distance [m].",
    )
    parser.add_argument(
        "--policy_gripper_close_socket_error",
        type=float,
        default=None,
        help="Additionally require this maximum hand-frame socket error before latching either gripper closed [m].",
    )
    parser.add_argument(
        "--policy_right_socket_feedback_gain",
        type=float,
        default=None,
        help="Right-hand pre-contact socket correction gain applied to policy translation actions [1/m].",
    )
    parser.add_argument(
        "--policy_right_socket_feedback_axis_scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Per-axis hand-frame scale applied to right socket feedback.",
    )
    parser.add_argument(
        "--policy_socket_feedback_activation_distance",
        type=float,
        default=0.03,
        help="Activate right-hand socket feedback inside this tail-to-TCP distance [m].",
    )
    parser.add_argument(
        "--policy_socket_feedback_max_action",
        type=float,
        default=0.2,
        help="Maximum norm of the socket-feedback translation residual in normalized action space.",
    )
    parser.add_argument(
        "--policy_right_arm_bias",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional right-arm residual added to checkpoint actions for handoff diagnosis.",
    )
    parser.add_argument(
        "--policy_right_arm_bias_uniform_scale",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Sample one fixed per-environment right-arm residual from symmetric uniform bounds.",
    )
    parser.add_argument(
        "--policy_right_arm_bias_uniform_center",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Center of the per-environment right-arm residual search bounds.",
    )
    parser.add_argument(
        "--policy_right_hand_translation_bias",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional right-arm translation residual expressed in the current hand frame.",
    )
    parser.add_argument(
        "--policy_right_arm_bias_steps",
        type=int,
        default=0,
        help="Number of initial policy steps receiving the right-arm residual; zero applies it to every step.",
    )
    parser.add_argument(
        "--policy_right_regrasp_offset_hand",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="After bilateral acquisition, regrasp at this right-hand-frame TCP offset [m].",
    )
    parser.add_argument("--policy_right_regrasp_open_steps", type=int, default=10)
    parser.add_argument(
        "--policy_right_regrasp_clearance",
        type=float,
        default=0.0,
        help="Right-hand-frame -Z clearance used before the lateral regrasp move [m].",
    )
    parser.add_argument("--policy_right_regrasp_retract_steps", type=int, default=15)
    parser.add_argument("--policy_right_regrasp_move_steps", type=int, default=30)
    parser.add_argument("--policy_right_regrasp_insert_steps", type=int, default=30)
    parser.add_argument("--policy_right_regrasp_close_steps", type=int, default=10)
    parser.add_argument(
        "--left_pull_direction_w",
        type=float,
        nargs=3,
        default=REFERENCE_PULL_DIRECTIONS[0],
        metavar=("X", "Y", "Z"),
        help="Left TCP pull direction in the world frame.",
    )
    parser.add_argument(
        "--right_pull_direction_w",
        type=float,
        nargs=3,
        default=REFERENCE_PULL_DIRECTIONS[1],
        metavar=("X", "Y", "Z"),
        help="Right TCP pull direction in the world frame.",
    )
    parser.add_argument("--run_label", default="")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument(
        "--grasp_assist",
        action="store_true",
        help="Enable the task's diagnostic grasp spring; disabled by default.",
    )
    parser.add_argument(
        "--play_curriculum_levels",
        action="store_true",
        help="Display reset levels 0 through 10 sequentially without advancing physics.",
    )
    parser.add_argument(
        "--level_hold_seconds",
        type=float,
        default=4.0,
        help="Wall-clock display time for each curriculum level [s].",
    )
    parser.add_argument(
        "--level_cycles",
        type=int,
        default=1,
        help="Number of curriculum playback cycles; zero repeats until the viewer closes.",
    )
    parser.add_argument("--lace_mu", type=float, default=None)
    parser.add_argument("--finger_mu", type=float, default=None)
    parser.add_argument("--gripper_stiffness", type=float, default=None)
    parser.add_argument("--gripper_damping", type=float, default=100.0)
    parser.add_argument(
        "--gripper_maximum_velocity",
        type=float,
        default=None,
        help="Override the binary gripper target velocity limit [m/s] for diagnostics.",
    )
    parser.add_argument(
        "--bend_stiffness_scale",
        type=float,
        default=1.0,
        help="Scale the task's shoelace bend stiffness for diagnostic runs.",
    )
    parser.add_argument(
        "--tail_bend_stiffness_scale",
        type=float,
        default=1.0,
        help="Scale only the task's stiff free-tail bend stiffness for diagnostic runs.",
    )
    parser.add_argument(
        "--tail_bend_damping_scale",
        type=float,
        default=1.0,
        help="Scale only the task's stiff free-tail bend damping for diagnostic runs.",
    )
    parser.add_argument(
        "--cable_damping_scale",
        type=float,
        default=1.0,
        help="Scale shoelace stretch and bend damping for diagnostic runs.",
    )
    parser.add_argument(
        "--settled_state_path",
        type=Path,
        default=None,
        help="Override the task's baked shoelace reset-state asset.",
    )
    parser.add_argument(
        "--save_settled_state",
        type=Path,
        default=None,
        help="Save cable poses after the approach phase to this NPZ path.",
    )
    parser.add_argument(
        "--save_demonstration",
        type=Path,
        default=None,
        help="Save observations and actions when the rollout reaches the exact success condition.",
    )
    parser.add_argument(
        "--save_acquisition_demonstration",
        type=Path,
        default=None,
        help="Save each trajectory prefix through its first confirmed bilateral acquisition.",
    )
    parser.add_argument(
        "--initial_level",
        type=int,
        default=None,
        help="Reset curriculum level. By default, start from the full open-pregrasp pose.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the environment and scripted policy.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument(
        "--open_position",
        type=float,
        default=None,
        help="Open position per finger [m]. By default, use the task configuration.",
    )
    parser.add_argument(
        "--closed_position",
        type=float,
        default=None,
        help="Closed position per finger [m]. By default, use the task configuration.",
    )
    parser.add_argument(
        "--num_substeps",
        type=int,
        default=None,
        help="Override physics substeps; by default, preserve the task configuration.",
    )
    parser.add_argument("--vbd_iterations", type=int, default=20)
    parser.add_argument("--coupling_mode", choices=("proxy", "admm"), default="proxy")
    parser.add_argument("--coupler_iterations", type=int, default=None)
    parser.add_argument("--admm_iterations", type=int, default=5)
    parser.add_argument("--admm_rho", type=float, default=400.0)
    parser.add_argument("--admm_gamma", type=float, default=0.0)
    parser.add_argument("--admm_baumgarte", type=float, default=0.5)
    parser.add_argument("--admm_contact_matching", choices=("disabled", "latest", "sticky"), default="latest")
    parser.add_argument("--proxy_mass_scale", type=float, default=None)
    parser.add_argument("--proxy_mode", choices=("lagged", "staggered"), default=None)
    parser.add_argument("--collide_interval", type=int, default=2)
    parser.add_argument("--contact_history", action="store_true")
    parser.add_argument("--disable_cuda_graph", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=None)
    args_cli, hydra_args = setup_preset_cli(parser)
    if hydra_args:
        raise ValueError(f"Unexpected Hydra overrides: {hydra_args}")
    if args_cli.hold_steps < 0:
        raise ValueError("Hold steps must be non-negative.")
    if args_cli.approach_steps < 0 or args_cli.close_steps <= 0 or args_cli.pull_steps <= 0:
        raise ValueError("Approach steps must be non-negative; close and pull steps must be positive.")
    if args_cli.approach_speed <= 0.0 or args_cli.approach_tolerance < 0.0:
        raise ValueError("Approach speed must be positive and tolerance must be non-negative.")
    if args_cli.pregrasp_clearance < 0.0 or args_cli.pregrasp_steps <= 0:
        raise ValueError("Pregrasp clearance must be non-negative and pregrasp steps must be positive.")
    if args_cli.pregrasp_world_z_clearance is not None and args_cli.pregrasp_world_z_clearance <= 0.0:
        raise ValueError("World-Z pregrasp clearance must be positive.")
    if args_cli.pregrasp_world_z_clearance is not None and args_cli.pregrasp_clearance > 0.0:
        raise ValueError("Specify either local-Z or world-Z pregrasp clearance, not both.")
    if args_cli.grasp_approach_close_step is not None and args_cli.grasp_approach_close_step < 0:
        raise ValueError("Grasp-approach close step must be non-negative.")
    if args_cli.grasp_approach_handoff_distance is not None and args_cli.grasp_approach_handoff_distance <= 0.0:
        raise ValueError("Grasp-approach handoff distance must be positive.")
    if args_cli.grasp_approach_close_step is not None and args_cli.grasp_approach_handoff_distance is not None:
        raise ValueError("Specify either a fixed grasp-approach close step or a distance-triggered handoff, not both.")
    if args_cli.approach_waypoint_steps <= 0:
        raise ValueError("Approach waypoint steps must be positive.")
    if args_cli.joint_interpolation_waypoints <= 0 or args_cli.joint_interpolation_waypoint_steps <= 0:
        raise ValueError("Joint-interpolation waypoint count and steps must be positive.")
    if args_cli.approach_orientation_steps < 0:
        raise ValueError("Approach orientation steps must be non-negative.")
    if args_cli.approach_orientation_steps > 0 and args_cli.approach_reference_level is None:
        raise ValueError("Approach orientation steps require --approach_reference_level.")
    if args_cli.final_waypoint_handoff_distance is not None and args_cli.final_waypoint_handoff_distance <= 0.0:
        raise ValueError("Final waypoint handoff distance must be positive.")
    if args_cli.policy_steps <= 0:
        raise ValueError("Policy steps must be positive.")
    if args_cli.policy_handoff_close_steps < 0:
        raise ValueError("Policy handoff close steps must be non-negative.")
    if args_cli.gripper_maximum_velocity is not None and (
        not np.isfinite(args_cli.gripper_maximum_velocity) or args_cli.gripper_maximum_velocity <= 0.0
    ):
        raise ValueError("Gripper maximum velocity must be finite and positive.")
    if not np.isfinite(args_cli.policy_arm_action_scale) or args_cli.policy_arm_action_scale <= 0.0:
        raise ValueError("Policy arm action scale must be finite and positive.")
    if args_cli.policy_action_std is not None and (
        not np.isfinite(args_cli.policy_action_std) or args_cli.policy_action_std < 0.0
    ):
        raise ValueError("Policy action standard deviation must be finite and non-negative.")
    if args_cli.policy_right_arm_bias is not None and not np.isfinite(args_cli.policy_right_arm_bias).all():
        raise ValueError("Policy right-arm residual must be finite.")
    if (
        args_cli.policy_right_hand_translation_bias is not None
        and not np.isfinite(args_cli.policy_right_hand_translation_bias).all()
    ):
        raise ValueError("Policy right-hand translation residual must be finite.")
    if args_cli.policy_right_arm_bias is not None and args_cli.policy_right_hand_translation_bias is not None:
        raise ValueError("Specify either a base-frame or hand-frame policy residual, not both.")
    if args_cli.policy_right_arm_bias_uniform_scale is not None:
        if not np.isfinite(args_cli.policy_right_arm_bias_uniform_scale).all() or np.any(
            np.asarray(args_cli.policy_right_arm_bias_uniform_scale) < 0.0
        ):
            raise ValueError("Policy right-arm residual search scales must be finite and non-negative.")
        if not np.any(np.asarray(args_cli.policy_right_arm_bias_uniform_scale) > 0.0):
            raise ValueError("At least one policy right-arm residual search scale must be positive.")
        if args_cli.policy_right_arm_bias is not None or args_cli.policy_right_hand_translation_bias is not None:
            raise ValueError("Residual search cannot be combined with a fixed policy residual.")
    if not np.isfinite(args_cli.policy_right_arm_bias_uniform_center).all():
        raise ValueError("Policy right-arm residual search center must be finite.")
    if args_cli.policy_gripper_close_distance is not None and args_cli.policy_gripper_close_distance <= 0.0:
        raise ValueError("Policy gripper close distance must be positive.")
    if args_cli.policy_gripper_close_socket_error is not None and args_cli.policy_gripper_close_socket_error <= 0.0:
        raise ValueError("Policy gripper close socket error must be positive.")
    if args_cli.policy_gripper_close_socket_error is not None and args_cli.policy_gripper_close_distance is None:
        raise ValueError("Policy gripper close socket error requires --policy_gripper_close_distance.")
    if args_cli.policy_right_socket_feedback_gain is not None and args_cli.policy_right_socket_feedback_gain <= 0.0:
        raise ValueError("Policy right socket feedback gain must be positive.")
    if not np.isfinite(args_cli.policy_right_socket_feedback_axis_scale).all():
        raise ValueError("Policy right socket feedback axis scale must be finite.")
    if args_cli.policy_socket_feedback_activation_distance <= 0.0:
        raise ValueError("Policy socket feedback activation distance must be positive.")
    if args_cli.policy_socket_feedback_max_action <= 0.0:
        raise ValueError("Policy socket feedback maximum action must be positive.")
    if args_cli.policy_right_socket_feedback_gain is not None and args_cli.policy_gripper_close_socket_error is None:
        raise ValueError("Policy right socket feedback requires --policy_gripper_close_socket_error.")
    if args_cli.policy_force_grippers_closed and args_cli.policy_gripper_close_distance is not None:
        raise ValueError("Specify either forced policy closure or distance-triggered policy closure, not both.")
    if args_cli.policy_right_arm_bias_steps < 0:
        raise ValueError("Policy right-arm residual steps must be non-negative.")
    if (
        args_cli.policy_right_regrasp_offset_hand is not None
        and not np.isfinite(args_cli.policy_right_regrasp_offset_hand).all()
    ):
        raise ValueError("Policy right regrasp offset must be finite.")
    if not np.isfinite(args_cli.policy_right_regrasp_clearance) or args_cli.policy_right_regrasp_clearance < 0.0:
        raise ValueError("Policy right regrasp clearance must be finite and non-negative.")
    if (
        min(
            args_cli.policy_right_regrasp_open_steps,
            args_cli.policy_right_regrasp_retract_steps,
            args_cli.policy_right_regrasp_move_steps,
            args_cli.policy_right_regrasp_insert_steps,
            args_cli.policy_right_regrasp_close_steps,
        )
        <= 0
    ):
        raise ValueError("Policy right regrasp stage lengths must be positive.")
    if args_cli.policy_checkpoint is not None and not args_cli.policy_checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {args_cli.policy_checkpoint}")
    if not np.isfinite(args_cli.left_pull_direction_w).all() or np.linalg.norm(args_cli.left_pull_direction_w) == 0.0:
        raise ValueError("Left pull direction must be finite and non-zero.")
    if not np.isfinite(args_cli.right_pull_direction_w).all() or np.linalg.norm(args_cli.right_pull_direction_w) == 0.0:
        raise ValueError("Right pull direction must be finite and non-zero.")
    if not 0.0 <= args_cli.pull_tracking_gain <= 1.0:
        raise ValueError("Pull tracking gain must lie in [0, 1].")
    if args_cli.left_tcp_offset_w is not None and not np.isfinite(args_cli.left_tcp_offset_w).all():
        raise ValueError("Left TCP offset must contain finite values.")
    if args_cli.right_tcp_offset_w is not None and not np.isfinite(args_cli.right_tcp_offset_w).all():
        raise ValueError("Right TCP offset must contain finite values.")
    if (
        args_cli.left_tail_to_tcp_target_hand is not None
        and not np.isfinite(args_cli.left_tail_to_tcp_target_hand).all()
    ):
        raise ValueError("Left hand-frame tail-to-TCP target must contain finite values.")
    if (
        args_cli.right_tail_to_tcp_target_hand is not None
        and not np.isfinite(args_cli.right_tail_to_tcp_target_hand).all()
    ):
        raise ValueError("Right hand-frame tail-to-TCP target must contain finite values.")
    if args_cli.left_tcp_offset_w is not None and args_cli.left_tail_to_tcp_target_hand is not None:
        raise ValueError("Specify either a left TCP offset or a left hand-frame tail target, not both.")
    if args_cli.right_tcp_offset_w is not None and args_cli.right_tail_to_tcp_target_hand is not None:
        raise ValueError("Specify either a right TCP offset or a right hand-frame tail target, not both.")
    if args_cli.pregrasp_clearance > 0.0 and (
        (args_cli.left_tcp_offset_w is None and args_cli.left_tail_to_tcp_target_hand is None)
        or (args_cli.right_tcp_offset_w is None and args_cli.right_tail_to_tcp_target_hand is None)
    ):
        raise ValueError("A pregrasp clearance requires a TCP target for both arms.")
    if args_cli.pregrasp_world_z_clearance is not None and (
        args_cli.left_tcp_offset_w is None or args_cli.right_tcp_offset_w is None
    ):
        raise ValueError("World-Z pregrasp clearance requires absolute TCP targets for both arms.")
    if args_cli.align_hand_roll_to_tail and args_cli.approach_reference_level is None:
        raise ValueError("Tail-aligned hand roll requires an approach reference level.")
    if args_cli.left_grasp_joint_positions is not None and not np.isfinite(args_cli.left_grasp_joint_positions).all():
        raise ValueError("Left grasp joint positions must contain finite values.")
    if args_cli.right_grasp_joint_positions is not None and not np.isfinite(args_cli.right_grasp_joint_positions).all():
        raise ValueError("Right grasp joint positions must contain finite values.")
    if args_cli.bend_stiffness_scale <= 0.0:
        raise ValueError("Bend stiffness scale must be positive.")
    if args_cli.tail_bend_stiffness_scale <= 0.0:
        raise ValueError("Tail bend stiffness scale must be positive.")
    if args_cli.tail_bend_damping_scale <= 0.0:
        raise ValueError("Tail bend damping scale must be positive.")
    if args_cli.cable_damping_scale <= 0.0:
        raise ValueError("Cable damping scale must be positive.")
    if args_cli.open_position is not None and args_cli.open_position <= 0.0:
        raise ValueError("Open position must be positive.")
    if args_cli.outward_tilt_degrees is not None and not -90.0 <= args_cli.outward_tilt_degrees <= 90.0:
        raise ValueError("Mirrored hand tilt must be in [-90, 90] degrees.")
    if not -180.0 <= args_cli.hand_roll_degrees <= 180.0:
        raise ValueError("Hand roll must be in [-180, 180] degrees.")
    if args_cli.mirrored_world_yaw_degrees is not None and not 0.0 <= args_cli.mirrored_world_yaw_degrees <= 45.0:
        raise ValueError("Mirrored world yaw must be in [0, 45] degrees.")
    if args_cli.outward_tilt_degrees is not None and args_cli.mirrored_world_yaw_degrees is not None:
        raise ValueError("Specify either mirrored hand tilt or mirrored world yaw, not both.")
    if args_cli.orientation_hold_seconds < 0.0:
        raise ValueError("Orientation hold time must be non-negative.")
    if args_cli.orientation_advance_m < 0.0:
        raise ValueError("Orientation advance distance must be non-negative.")
    if args_cli.tilt_steps <= 0 or args_cli.tilt_speed_degrees_s <= 0.0:
        raise ValueError("Tilt steps and tilt speed must be positive.")
    if args_cli.level_hold_seconds <= 0.0 or args_cli.level_cycles < 0:
        raise ValueError("Level hold time must be positive and level cycles must be non-negative.")
    if args_cli.num_envs <= 0:
        raise ValueError("Number of environments must be positive.")
    if args_cli.num_envs > 1 and args_cli.policy_checkpoint is None:
        raise ValueError("Parallel collection requires --policy_checkpoint.")
    if (
        args_cli.save_demonstration is not None
        and args_cli.save_acquisition_demonstration is not None
        and args_cli.save_demonstration.resolve() == args_cli.save_acquisition_demonstration.resolve()
    ):
        raise ValueError("Success and acquisition demonstrations require distinct output paths.")

    torch.manual_seed(args_cli.seed)
    shoelace_env_module.BEND_STIFFNESS *= args_cli.bend_stiffness_scale
    shoelace_env_module.TAIL_BEND_STIFFNESS *= args_cli.bend_stiffness_scale * args_cli.tail_bend_stiffness_scale
    shoelace_env_module.STRETCH_DAMPING *= args_cli.cable_damping_scale
    shoelace_env_module.BEND_DAMPING *= args_cli.cable_damping_scale
    shoelace_env_module.TAIL_BEND_DAMPING *= args_cli.cable_damping_scale * args_cli.tail_bend_damping_scale
    if args_cli.settled_state_path is not None:
        shoelace_env_module.SHOELACE_SETTLED_STATE_ASSET = args_cli.settled_state_path.resolve()
    if args_cli.lace_mu is not None:
        shoelace_env_module.LACE_MU = args_cli.lace_mu
    if args_cli.finger_mu is not None:
        _install_finger_mu_override(args_cli.finger_mu)
    env_cfg = parse_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    if args_cli.gripper_maximum_velocity is not None:
        env_cfg.actions.left_gripper.maximum_velocity = args_cli.gripper_maximum_velocity
        env_cfg.actions.right_gripper.maximum_velocity = args_cli.gripper_maximum_velocity
    has_orientation_target = (
        args_cli.outward_tilt_degrees is not None or args_cli.mirrored_world_yaw_degrees is not None
    )
    scripted_steps = (
        (args_cli.tilt_steps if has_orientation_target else 0)
        + len(args_cli.approach_waypoint_levels or ()) * args_cli.approach_waypoint_steps
        + (
            args_cli.joint_interpolation_waypoints * args_cli.joint_interpolation_waypoint_steps
            if args_cli.joint_interpolation_levels is not None
            else 0
        )
        + args_cli.approach_orientation_steps
        + (
            args_cli.pregrasp_steps
            if args_cli.pregrasp_clearance > 0.0 or args_cli.pregrasp_world_z_clearance is not None
            else 0
        )
        + args_cli.approach_steps
        + (
            args_cli.policy_handoff_close_steps + args_cli.policy_steps
            if args_cli.policy_checkpoint is not None
            else args_cli.close_steps + args_cli.hold_steps + args_cli.pull_steps
        )
    )
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        (scripted_steps + 1) * env_cfg.sim.dt * env_cfg.decimation,
    )
    env_cfg.grasp_assist_enabled = args_cli.grasp_assist
    env_cfg.coupling_mode = args_cli.coupling_mode
    env_cfg.admm_iterations = args_cli.admm_iterations
    env_cfg.admm_rho = args_cli.admm_rho
    if args_cli.left_grasp_joint_positions is not None or args_cli.right_grasp_joint_positions is not None:
        reset_params = env_cfg.events.reset_shoelace.params
        left_grasp, right_grasp = reset_params["grasp_joint_positions"]
        left_override = (
            left_grasp if args_cli.left_grasp_joint_positions is None else tuple(args_cli.left_grasp_joint_positions)
        )
        right_override = (
            right_grasp if args_cli.right_grasp_joint_positions is None else tuple(args_cli.right_grasp_joint_positions)
        )
        reset_params["grasp_joint_positions"] = (left_override, right_override)
        left_levels, right_levels = reset_params["arm_joint_positions_by_level"]
        left_levels = tuple(left_override if tuple(level) == tuple(left_grasp) else level for level in left_levels)
        right_levels = tuple(right_override if tuple(level) == tuple(right_grasp) else level for level in right_levels)
        reset_params["arm_joint_positions_by_level"] = (left_levels, right_levels)
    for robot_cfg in (env_cfg.scene.robot_left, env_cfg.scene.robot_right):
        if args_cli.gripper_stiffness is not None:
            robot_cfg.actuators["panda_hand"].stiffness = args_cli.gripper_stiffness
        robot_cfg.actuators["panda_hand"].damping = args_cli.gripper_damping
        if args_cli.open_position is not None:
            robot_cfg.init_state.joint_pos["panda_finger_joint.*"] = args_cli.open_position
    if args_cli.open_position is not None:
        env_cfg.actions.left_gripper.open_command_expr["panda_finger_joint1"] = args_cli.open_position
        env_cfg.actions.right_gripper.open_command_expr["panda_finger_joint1"] = args_cli.open_position
        env_cfg.events.reset_shoelace.params["open_position"] = args_cli.open_position
    if args_cli.closed_position is not None:
        env_cfg.actions.left_gripper.close_command_expr["panda_finger_joint1"] = args_cli.closed_position
        env_cfg.actions.right_gripper.close_command_expr["panda_finger_joint1"] = args_cli.closed_position
        env_cfg.events.reset_shoelace.params["closed_position"] = args_cli.closed_position
    closed_position = env_cfg.actions.left_gripper.close_command_expr["panda_finger_joint1"]
    open_position = env_cfg.actions.left_gripper.open_command_expr["panda_finger_joint1"]
    if open_position <= closed_position:
        raise ValueError("Open position must be greater than closed position.")
    grasp_params = env_cfg.rewards.grasp_acquisition.params
    acquisition_distance = grasp_params["maximum_grasp_distance"]
    minimum_finger_position = grasp_params["minimum_finger_position"]
    maximum_finger_position = grasp_params["maximum_finger_position"]
    lost_grasp_params = env_cfg.terminations.lost_grasp.params
    retention_distance = lost_grasp_params["maximum_grasp_distance"]
    acquisition_confirmation_steps = max(int(lost_grasp_params["confirmation_steps"]), 1)
    release_confirmation_steps = max(int(lost_grasp_params["release_confirmation_steps"]), 1)
    physics_cfg = env_cfg.sim.physics
    if args_cli.num_substeps is not None:
        physics_cfg.num_substeps = args_cli.num_substeps
    if args_cli.disable_cuda_graph:
        physics_cfg.use_cuda_graph = False
    coupler_cfg = physics_cfg.solver_cfg
    if not isinstance(coupler_cfg, CouplerProxyCfg):
        raise TypeError(f"Expected a CouplerProxyCfg task template, got {type(coupler_cfg).__name__}")
    if args_cli.coupling_mode == "admm":
        coupler_cfg = CouplerAdmmCfg(
            entries=coupler_cfg.entries,
            contact_pairs=[("robots", "shoelace")],
            iterations=args_cli.admm_iterations,
            rho=args_cli.admm_rho,
            gamma=args_cli.admm_gamma,
            baumgarte=args_cli.admm_baumgarte,
            rigid_contact_matching=args_cli.admm_contact_matching,
        )
        physics_cfg.solver_cfg = coupler_cfg
    elif args_cli.coupler_iterations is not None:
        coupler_cfg.iterations = args_cli.coupler_iterations
    shoelace_solver_cfg = next(entry for entry in coupler_cfg.entries if entry.name == "shoelace").solver_cfg
    shoelace_solver_cfg.iterations = args_cli.vbd_iterations
    proxy_cfg = coupler_cfg.proxies[0] if isinstance(coupler_cfg, CouplerProxyCfg) else None
    if proxy_cfg is not None:
        if args_cli.proxy_mass_scale is not None:
            proxy_cfg.mass_scale = args_cli.proxy_mass_scale
        if args_cli.proxy_mode is not None:
            proxy_cfg.mode = args_cli.proxy_mode
        proxy_cfg.collide_interval = args_cli.collide_interval
    elif args_cli.proxy_mass_scale is not None or args_cli.proxy_mode is not None:
        raise ValueError("Proxy mass scale and mode are only available with --coupling_mode proxy")
    if args_cli.contact_history:
        shoelace_solver_cfg.rigid_contact_history = True
        if proxy_cfg is not None:
            proxy_cfg.collision_pipeline.contact_matching = "latest"
        else:
            coupler_cfg.rigid_contact_matching = "latest"

    curriculum = env_cfg.curriculum.pull_to_grasp.params
    if args_cli.open_position is not None or args_cli.closed_position is not None:
        level_count = curriculum["level_count"]
        open_phase_fraction = env_cfg.events.reset_shoelace.params["gripper_open_phase_fraction"]
        env_cfg.events.reset_shoelace.params["gripper_joint_positions_by_level"] = tuple(
            closed_position
            + min((level / (level_count - 1)) / open_phase_fraction, 1.0) * (open_position - closed_position)
            for level in range(level_count)
        )
    initial_level = curriculum["level_count"] - 1 if args_cli.initial_level is None else args_cli.initial_level
    curriculum["initial_level"] = initial_level
    curriculum["current_level_fraction"] = 1.0
    curriculum["current_level_fraction_schedule"] = (1.0,)
    curriculum["terminal_level_fraction"] = 1.0
    curriculum["terminal_level_fraction_schedule"] = (1.0,)

    success_params = env_cfg.terminations.success.params
    socket_targets = torch.tensor(env_cfg.rewards.dense_task.params["socket_targets"], device=env_cfg.sim.device)
    maximum_socket_error = env_cfg.rewards.dense_task.params["maximum_socket_error"]
    for termination_name in vars(env_cfg.terminations):
        setattr(env_cfg.terminations, termination_name, None)
    for reward_name in list(vars(env_cfg.rewards)):
        setattr(env_cfg.rewards, reward_name, None)

    with launch_simulation(env_cfg, args_cli):
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        env = gym_env.unwrapped
        try:
            gym_env.reset(seed=args_cli.seed)
            reference_hand_quaternions_w = None
            approach_alignment_roll_degrees = [0.0, 0.0]
            if args_cli.approach_reference_level is not None:
                if not 0 <= args_cli.approach_reference_level < curriculum["level_count"]:
                    raise ValueError(f"Approach reference level must lie in [0, {curriculum['level_count'] - 1}].")
                _apply_curriculum_reset_level(env, args_cli.approach_reference_level)
                reference_hand_quaternions_w = _hand_quaternions_w(env)[0]
                if args_cli.align_hand_roll_to_tail:
                    tail_directions_hand = torch.sum(
                        _hand_axes(env)[0] * _tail_directions_w(env)[0].unsqueeze(1), dim=-1
                    )
                    raw_roll = torch.atan2(tail_directions_hand[:, 1], tail_directions_hand[:, 0])
                    half_turn = torch.pi * 0.5
                    alignment_roll = torch.remainder(raw_roll + half_turn, torch.pi) - half_turn
                    local_z = torch.zeros((2, 3), device=env.device)
                    local_z[:, 2] = 1.0
                    alignment_quaternion = math_utils.quat_from_angle_axis(alignment_roll, local_z)
                    reference_hand_quaternions_w = math_utils.quat_mul(
                        reference_hand_quaternions_w, alignment_quaternion
                    )
                    approach_alignment_roll_degrees = torch.rad2deg(alignment_roll).tolist()
                _apply_curriculum_reset_level(env, initial_level)
            approach_waypoints = []
            for waypoint_level in args_cli.approach_waypoint_levels or ():
                if not 0 <= waypoint_level < curriculum["level_count"]:
                    raise ValueError(f"Approach waypoint level must lie in [0, {curriculum['level_count'] - 1}].")
                _apply_curriculum_reset_level(env, waypoint_level)
                approach_waypoints.append(
                    (
                        str(waypoint_level),
                        _tcp_positions(env)[0].clone(),
                        _hand_quaternions_w(env)[0].clone(),
                        args_cli.approach_waypoint_steps,
                    )
                )
            if approach_waypoints:
                _apply_curriculum_reset_level(env, initial_level)
            if args_cli.joint_interpolation_levels is not None:
                start_level, end_level = args_cli.joint_interpolation_levels
                if not 0 <= start_level < curriculum["level_count"] or not 0 <= end_level < curriculum["level_count"]:
                    raise ValueError(f"Joint-interpolation levels must lie in [0, {curriculum['level_count'] - 1}].")
                interpolation_waypoints = _sample_joint_interpolation_waypoints(
                    env,
                    env_cfg.events.reset_shoelace.params["arm_joint_positions_by_level"],
                    start_level,
                    end_level,
                    args_cli.joint_interpolation_waypoints,
                )
                approach_waypoints.extend(
                    (*waypoint, args_cli.joint_interpolation_waypoint_steps) for waypoint in interpolation_waypoints
                )
                _apply_curriculum_reset_level(env, initial_level)
            if args_cli.play_curriculum_levels:
                _play_curriculum_levels(
                    env,
                    level_count=curriculum["level_count"],
                    hold_seconds=args_cli.level_hold_seconds,
                    cycle_count=args_cli.level_cycles,
                )
                return
            shape_materials = _shape_material_summary()
            actions = torch.zeros(gym_env.action_space.shape, device=env.device)
            _set_term(actions, env, "left_gripper", 1.0)
            _set_term(actions, env, "right_gripper", 1.0)
            demonstration_observations = []
            demonstration_actions = []
            demonstration_phases = []
            record_demonstration = (
                args_cli.save_demonstration is not None or args_cli.save_acquisition_demonstration is not None
            )

            def _step(step_actions: torch.Tensor, phase: str):
                if record_demonstration:
                    demonstration_observations.append(
                        env.observation_manager.compute()["policy"].detach().cpu().clone()
                    )
                    demonstration_actions.append(step_actions.detach().cpu().clone())
                    demonstration_phases.append(phase)
                return gym_env.step(step_actions)

            reset_state = _state(env)
            initial_success_metrics = _success_metrics(env, success_params)
            initial_tcp_to_tail_w = reset_state["tail"] - reset_state["tcp"]
            initial_tcp_to_tail_hand = _tail_to_tcp_hand(reset_state)
            acquired_at_reset = (
                (reset_state["distance"][0] <= acquisition_distance)
                & (reset_state["finger"][0] >= minimum_finger_position)
                & (reset_state["finger"][0] <= maximum_finger_position)
            )

            policy_dt = env_cfg.sim.dt * env_cfg.decimation
            arm_action_scale = float(env_cfg.actions.left_arm.scale[0])
            arm_rotation_scale = float(env_cfg.actions.left_arm.scale[3])
            tilt_step_count = 0
            target_hand_quaternions_w = None
            orientation_target_tcp = reset_state["tcp"]
            if args_cli.outward_tilt_degrees is not None:
                target_hand_quaternions_w = _outward_tilt_quaternions_w(
                    env, args_cli.outward_tilt_degrees, args_cli.hand_roll_degrees
                )
            elif args_cli.mirrored_world_yaw_degrees is not None:
                target_hand_quaternions_w = _mirrored_world_yaw_quaternions_w(env, args_cli.mirrored_world_yaw_degrees)
            if target_hand_quaternions_w is not None:
                target_direction_w = reset_state["tail"] - reset_state["tcp"]
                target_distance = torch.linalg.vector_norm(target_direction_w, dim=-1, keepdim=True)
                advance = torch.minimum(
                    target_distance,
                    target_distance.new_full(target_distance.shape, args_cli.orientation_advance_m),
                )
                orientation_target_tcp = reset_state["tcp"] + target_direction_w * (
                    advance / target_distance.clamp_min(1.0e-8)
                )
                maximum_tilt_step = torch.deg2rad(
                    torch.tensor(args_cli.tilt_speed_degrees_s * policy_dt, device=env.device)
                )
                maximum_position_step = args_cli.approach_speed * policy_dt
                tilt_tolerance = torch.deg2rad(torch.tensor(0.5, device=env.device))
                position_tolerance = 5.0e-4
                for tilt_step_count in range(1, args_cli.tilt_steps + 1):
                    current = _state(env)
                    orientation_error_b = _hand_orientation_errors_b(env, target_hand_quaternions_w)
                    error_angle = torch.linalg.vector_norm(orientation_error_b, dim=-1)
                    step_angle = torch.minimum(error_angle, maximum_tilt_step.expand_as(error_angle))
                    rotation_b = orientation_error_b * (step_angle / error_angle.clamp_min(1.0e-8)).unsqueeze(-1)
                    position_error_w = orientation_target_tcp - current["tcp"]
                    position_error_norm = torch.linalg.vector_norm(position_error_w, dim=-1)
                    position_step = torch.minimum(
                        position_error_norm,
                        position_error_norm.new_full(position_error_norm.shape, maximum_position_step),
                    )
                    displacement_w = position_error_w * (
                        position_step / position_error_norm.clamp_min(1.0e-8)
                    ).unsqueeze(-1)
                    _set_tcp_pose_delta(
                        actions,
                        env,
                        displacement_w,
                        rotation_b,
                        arm_action_scale,
                        arm_rotation_scale,
                    )
                    _step(actions, "orient")
                    if bool(torch.all((error_angle <= tilt_tolerance) & (position_error_norm <= position_tolerance))):
                        break
            after_tilt = _state(env)
            if target_hand_quaternions_w is not None and args_cli.orientation_hold_seconds > 0.0:
                deadline = time.monotonic() + args_cli.orientation_hold_seconds
                while time.monotonic() < deadline and env.sim.is_headless_or_exist_active_visualizer():
                    env.sim.render()
                    time.sleep(1.0 / 60.0)
            if target_hand_quaternions_w is None:
                tilt_error_degrees = [0.0, 0.0]
            else:
                tilt_error_degrees = torch.rad2deg(
                    torch.linalg.vector_norm(_hand_orientation_errors_b(env, target_hand_quaternions_w), dim=-1)[0]
                ).tolist()
            maximum_approach_step = args_cli.approach_speed * policy_dt
            approach_trace = [after_tilt]
            approach_samples = []
            total_approach_steps = 0
            maximum_waypoint_rotation_step = torch.deg2rad(
                torch.tensor(args_cli.tilt_speed_degrees_s * policy_dt, device=env.device)
            )
            for waypoint_index, (
                waypoint_label,
                target_tcp_w,
                target_quaternions_w,
                maximum_waypoint_steps,
            ) in enumerate(approach_waypoints):
                is_final_waypoint = waypoint_index == len(approach_waypoints) - 1
                waypoint_reached = torch.zeros((actions.shape[0], 2), dtype=torch.bool, device=env.device)
                for waypoint_step in range(maximum_waypoint_steps):
                    current = _state(env)
                    if is_final_waypoint and args_cli.final_waypoint_handoff_distance is not None:
                        waypoint_reached |= current["distance"] <= args_cli.final_waypoint_handoff_distance
                    position_error_w = target_tcp_w.unsqueeze(0) - current["tcp"]
                    position_error_norm = torch.linalg.vector_norm(position_error_w, dim=-1)
                    position_step = torch.minimum(
                        position_error_norm,
                        position_error_norm.new_full(position_error_norm.shape, maximum_approach_step),
                    )
                    displacement_w = position_error_w * (
                        position_step / position_error_norm.clamp_min(1.0e-8)
                    ).unsqueeze(-1)
                    orientation_error_b = _hand_orientation_errors_b(env, target_quaternions_w)
                    orientation_error_norm = torch.linalg.vector_norm(orientation_error_b, dim=-1)
                    rotation_step = torch.minimum(
                        orientation_error_norm, maximum_waypoint_rotation_step.expand_as(orientation_error_norm)
                    )
                    rotation_b = orientation_error_b * (
                        rotation_step / orientation_error_norm.clamp_min(1.0e-8)
                    ).unsqueeze(-1)
                    displacement_w = torch.where(waypoint_reached.unsqueeze(-1), 0.0, displacement_w)
                    rotation_b = torch.where(waypoint_reached.unsqueeze(-1), 0.0, rotation_b)
                    _set_tcp_pose_delta(
                        actions,
                        env,
                        displacement_w,
                        rotation_b,
                        arm_action_scale,
                        arm_rotation_scale,
                    )
                    _step(actions, f"waypoint_{waypoint_label}")
                    current = _state(env)
                    approach_trace.append(current)
                    total_approach_steps += 1
                    if waypoint_step in {0, 4, 9, 19, 29, 44, maximum_waypoint_steps - 1}:
                        approach_samples.append(
                            {
                                "stage": f"level_{waypoint_label}",
                                "step": total_approach_steps,
                                "distance_mm": (current["distance"][0] * 1000).tolist(),
                                "tail_to_tcp_hand_mm": (_tail_to_tcp_hand(current)[0] * 1000).tolist(),
                            }
                        )
                    if bool(
                        torch.all(
                            (position_error_norm <= args_cli.approach_tolerance)
                            & (orientation_error_norm <= torch.deg2rad(orientation_error_norm.new_tensor(0.5)))
                        )
                    ) or bool(torch.all(waypoint_reached)):
                        break
            approach_orientation_step_count = 0
            approach_orientation_stage_result = None
            if reference_hand_quaternions_w is not None and args_cli.approach_orientation_steps > 0:
                stage_start = _state(env)
                target_tcp_w = stage_start["tcp"].clone()
                orientation_tolerance = torch.deg2rad(stage_start["tcp"].new_tensor(0.5))
                for approach_orientation_step_count in range(1, args_cli.approach_orientation_steps + 1):
                    current = _state(env)
                    position_error_w = target_tcp_w - current["tcp"]
                    position_error_norm = torch.linalg.vector_norm(position_error_w, dim=-1)
                    position_step = torch.minimum(
                        position_error_norm,
                        position_error_norm.new_full(position_error_norm.shape, maximum_approach_step),
                    )
                    displacement_w = position_error_w * (
                        position_step / position_error_norm.clamp_min(1.0e-8)
                    ).unsqueeze(-1)
                    orientation_error_b = _hand_orientation_errors_b(env, reference_hand_quaternions_w)
                    orientation_error_norm = torch.linalg.vector_norm(orientation_error_b, dim=-1)
                    rotation_step = torch.minimum(
                        orientation_error_norm,
                        maximum_waypoint_rotation_step.expand_as(orientation_error_norm),
                    )
                    rotation_b = orientation_error_b * (
                        rotation_step / orientation_error_norm.clamp_min(1.0e-8)
                    ).unsqueeze(-1)
                    _set_tcp_pose_delta(
                        actions,
                        env,
                        displacement_w,
                        rotation_b,
                        arm_action_scale,
                        arm_rotation_scale,
                    )
                    _step(actions, "approach_orientation")
                    approach_trace.append(_state(env))
                    total_approach_steps += 1
                    if bool(
                        torch.all(
                            (position_error_norm <= args_cli.approach_tolerance)
                            & (orientation_error_norm <= orientation_tolerance)
                        )
                    ):
                        break
                stage_end = _state(env)
                approach_orientation_stage_result = {
                    "tcp_delta_mm": ((stage_end["tcp"] - stage_start["tcp"])[0] * 1000).tolist(),
                    "tail_delta_mm": ((stage_end["tail"] - stage_start["tail"])[0] * 1000).tolist(),
                    "final_distance_mm": (stage_end["distance"][0] * 1000).tolist(),
                    "orientation_error_degrees": torch.rad2deg(
                        torch.linalg.vector_norm(_hand_orientation_errors_b(env, reference_hand_quaternions_w), dim=-1)[
                            0
                        ]
                    ).tolist(),
                }
            left_tcp_target_w = None
            if args_cli.left_tcp_offset_w is not None:
                left_tcp_target_w = reset_state["tcp"][:, 0] + reset_state["tcp"].new_tensor(args_cli.left_tcp_offset_w)
            right_tcp_target_w = None
            if args_cli.right_tcp_offset_w is not None:
                right_tcp_target_w = reset_state["tcp"][:, 1] + reset_state["tcp"].new_tensor(
                    args_cli.right_tcp_offset_w
                )
            tail_to_tcp_targets_hand = (args_cli.left_tail_to_tcp_target_hand, args_cli.right_tail_to_tcp_target_hand)
            fixed_target_quaternions_w = None
            if reference_hand_quaternions_w is not None:
                fixed_target_quaternions_w = reference_hand_quaternions_w
            elif left_tcp_target_w is not None or right_tcp_target_w is not None:
                fixed_target_quaternions_w = _mirrored_world_yaw_quaternions_w(env, 0.0)
            approach_stages = []
            if args_cli.pregrasp_clearance > 0.0 or args_cli.pregrasp_world_z_clearance is not None:
                pregrasp_targets_hand = tuple(
                    None if target is None else (target[0], target[1], target[2] + args_cli.pregrasp_clearance)
                    for target in tail_to_tcp_targets_hand
                )
                approach_stages.append(("pregrasp", args_cli.pregrasp_steps, pregrasp_targets_hand))
            approach_stages.append(("grasp", args_cli.approach_steps, tail_to_tcp_targets_hand))
            has_explicit_approach_target = any(
                target is not None
                for target in (
                    left_tcp_target_w,
                    right_tcp_target_w,
                    *tail_to_tcp_targets_hand,
                )
            )
            approach_stage_results = []
            for stage_name, maximum_steps, stage_targets_hand in approach_stages:
                stage_start = _state(env)
                stage_reached = torch.zeros((actions.shape[0], 2), dtype=torch.bool, device=env.device)
                frozen_stage_tcp_targets_w = [None, None]
                if args_cli.freeze_approach_tail_targets:
                    stage_quaternions_w = (
                        _hand_quaternions_w(env)[0]
                        if fixed_target_quaternions_w is None
                        else fixed_target_quaternions_w
                    )
                    for arm, tail_to_tcp_target_hand in enumerate(stage_targets_hand):
                        if tail_to_tcp_target_hand is None:
                            continue
                        target_hand = stage_start["tcp"].new_tensor(tail_to_tcp_target_hand)
                        target_w = math_utils.quat_apply(stage_quaternions_w[arm], target_hand)
                        frozen_stage_tcp_targets_w[arm] = stage_start["tail"][:, arm] - target_w
                for stage_step in range(maximum_steps):
                    current = _state(env)
                    if stage_name == "grasp" and stage_step == args_cli.grasp_approach_close_step:
                        _set_term(actions, env, "left_gripper", -1.0)
                        _set_term(actions, env, "right_gripper", -1.0)
                    if stage_name == "grasp" and args_cli.grasp_approach_handoff_distance is not None:
                        stage_reached |= current["distance"] <= args_cli.grasp_approach_handoff_distance
                        _set_term(
                            actions,
                            env,
                            "left_gripper",
                            torch.where(stage_reached[:, 0], -1.0, 1.0).unsqueeze(-1),
                        )
                        _set_term(
                            actions,
                            env,
                            "right_gripper",
                            torch.where(stage_reached[:, 1], -1.0, 1.0).unsqueeze(-1),
                        )
                    error_w = (
                        torch.zeros_like(current["tcp"])
                        if has_explicit_approach_target
                        else current["tail"] - current["tcp"]
                    )
                    for arm, (tcp_target_w, tail_to_tcp_target_hand) in enumerate(
                        zip((left_tcp_target_w, right_tcp_target_w), stage_targets_hand, strict=True)
                    ):
                        if tcp_target_w is not None:
                            stage_tcp_target_w = tcp_target_w
                            if stage_name == "pregrasp":
                                if args_cli.pregrasp_world_z_clearance is not None:
                                    clearance_w = error_w.new_tensor((0.0, 0.0, args_cli.pregrasp_world_z_clearance))
                                    stage_tcp_target_w = stage_tcp_target_w + clearance_w
                                else:
                                    clearance_hand = error_w.new_tensor((0.0, 0.0, args_cli.pregrasp_clearance))
                                    clearance_w = math_utils.quat_apply(fixed_target_quaternions_w[arm], clearance_hand)
                                    stage_tcp_target_w = stage_tcp_target_w - clearance_w
                            error_w[:, arm] = stage_tcp_target_w - current["tcp"][:, arm]
                        elif tail_to_tcp_target_hand is not None:
                            frozen_tcp_target_w = frozen_stage_tcp_targets_w[arm]
                            if frozen_tcp_target_w is not None:
                                error_w[:, arm] = frozen_tcp_target_w - current["tcp"][:, arm]
                            else:
                                target_hand = error_w.new_tensor(tail_to_tcp_target_hand)
                                target_w = torch.sum(current["hand_axes"][:, arm] * target_hand[None, :, None], dim=1)
                                error_w[:, arm] -= target_w
                        if stage_name == "grasp" and args_cli.grasp_approach_handoff_distance is not None:
                            error_w[:, arm] = torch.where(stage_reached[:, arm].unsqueeze(-1), 0.0, error_w[:, arm])
                    distance = torch.linalg.vector_norm(error_w, dim=-1)
                    step_distance = torch.minimum(distance, distance.new_full(distance.shape, maximum_approach_step))
                    step_distance = torch.where(distance > args_cli.approach_tolerance, step_distance, 0.0)
                    displacement_w = error_w * (step_distance / distance.clamp_min(1.0e-8)).unsqueeze(-1)
                    if fixed_target_quaternions_w is None:
                        _set_tcp_translation(actions, env, displacement_w, arm_action_scale)
                    else:
                        rotation_b = _hand_orientation_errors_b(env, fixed_target_quaternions_w)
                        _set_tcp_pose_delta(
                            actions,
                            env,
                            displacement_w,
                            rotation_b,
                            arm_action_scale,
                            arm_rotation_scale,
                        )
                    _step(actions, stage_name)
                    current = _state(env)
                    approach_trace.append(current)
                    total_approach_steps += 1
                    if stage_step in {0, 4, 9, 19, 29, 44, maximum_steps - 1}:
                        approach_samples.append(
                            {
                                "stage": stage_name,
                                "step": total_approach_steps,
                                "distance_mm": (current["distance"][0] * 1000).tolist(),
                                "tail_to_tcp_hand_mm": (_tail_to_tcp_hand(current)[0] * 1000).tolist(),
                            }
                        )
                    if any(target is not None for target in stage_targets_hand):
                        active_distance = distance
                    else:
                        active_distance = (
                            current["distance"]
                            if right_tcp_target_w is None
                            else torch.linalg.vector_norm(right_tcp_target_w - current["tcp"][:, 1], dim=-1)
                        )
                    if bool(torch.all(active_distance <= args_cli.approach_tolerance)):
                        break
                    if stage_name == "grasp" and bool(torch.all(stage_reached)):
                        break
                stage_end = _state(env)
                frozen_target_error_mm = []
                for arm, frozen_tcp_target_w in enumerate(frozen_stage_tcp_targets_w):
                    frozen_target_error_mm.append(
                        None
                        if frozen_tcp_target_w is None
                        else float(
                            torch.linalg.vector_norm(frozen_tcp_target_w - stage_end["tcp"][:, arm], dim=-1)[0] * 1000
                        )
                    )
                approach_stage_results.append(
                    {
                        "stage": stage_name,
                        "tcp_delta_mm": ((stage_end["tcp"] - stage_start["tcp"])[0] * 1000).tolist(),
                        "tail_delta_mm": ((stage_end["tail"] - stage_start["tail"])[0] * 1000).tolist(),
                        "frozen_target_error_mm": frozen_target_error_mm,
                        "final_distance_mm": (stage_end["distance"][0] * 1000).tolist(),
                    }
                )
            after_approach = _state(env)
            if reference_hand_quaternions_w is None:
                approach_orientation_error_degrees = [0.0, 0.0]
            else:
                approach_orientation_error_degrees = torch.rad2deg(
                    torch.linalg.vector_norm(_hand_orientation_errors_b(env, reference_hand_quaternions_w), dim=-1)[0]
                ).tolist()
            after_approach_cable_rms_speed = _cable_rms_linear_speed(env)
            if args_cli.save_settled_state is not None:
                _save_settled_state(env, args_cli.save_settled_state)
            _set_tcp_translation(actions, env, torch.zeros_like(after_approach["tcp"]), arm_action_scale)

            if args_cli.policy_checkpoint is not None:
                if args_cli.policy_handoff_close_steps > 0:
                    env.action_manager.reset()
                    actions.zero_()
                    _set_term(actions, env, "left_gripper", -1.0)
                    _set_term(actions, env, "right_gripper", -1.0)
                    for _ in range(args_cli.policy_handoff_close_steps):
                        _step(actions, "handoff_close")
                after_handoff_close = _state(env)
                policy = _load_checkpoint_policy(
                    args_cli.policy_checkpoint.resolve(),
                    env.device,
                    args_cli.policy_arm_action_scale,
                    args_cli.policy_stochastic,
                    args_cli.policy_action_std,
                )
                policy_right_arm_bias_by_env = None
                if args_cli.policy_right_arm_bias_uniform_scale is not None:
                    search_scale = actions.new_tensor(args_cli.policy_right_arm_bias_uniform_scale)
                    search_center = actions.new_tensor(args_cli.policy_right_arm_bias_uniform_center)
                    policy_right_arm_bias_by_env = 2.0 * torch.rand((actions.shape[0], 6), device=env.device) - 1.0
                    policy_right_arm_bias_by_env *= search_scale
                    policy_right_arm_bias_by_env += search_center
                    policy_right_arm_bias_by_env[0] = search_center
                env.action_manager.reset()
                first_bilateral_acquisition_steps = torch.full(
                    (actions.shape[0],), -1, dtype=torch.int64, device=env.device
                )
                first_success_steps = torch.full_like(first_bilateral_acquisition_steps, -1)
                lost_grasp_policy_steps = torch.full_like(first_bilateral_acquisition_steps, -1)
                completed_policy_steps = 0
                active = torch.ones(actions.shape[0], dtype=torch.bool, device=env.device)
                acquired_latched = torch.zeros((actions.shape[0], 2), dtype=torch.bool, device=env.device)
                acquisition_candidate_steps = torch.zeros_like(acquired_latched, dtype=torch.int64)
                bilaterally_acquired = torch.zeros(actions.shape[0], dtype=torch.bool, device=env.device)
                strict_socket_candidate_steps = torch.zeros_like(acquired_latched, dtype=torch.int64)
                strict_socket_acquired = torch.zeros(actions.shape[0], dtype=torch.bool, device=env.device)
                acquisition_tail_to_tcp_hand = torch.full(
                    (actions.shape[0], 2, 3), torch.nan, dtype=actions.dtype, device=env.device
                )
                acquisition_finger_positions = torch.full(
                    (actions.shape[0], 2), torch.nan, dtype=actions.dtype, device=env.device
                )
                release_candidate_steps = torch.zeros(actions.shape[0], dtype=torch.int64, device=env.device)
                successful_demonstration_lengths = {}
                acquisition_demonstration_lengths = {}
                maximum_distance = after_handoff_close["distance"].clone()
                minimum_socket_errors = torch.linalg.vector_norm(
                    _tail_to_tcp_hand(after_handoff_close) - socket_targets,
                    dim=-1,
                )
                minimum_throat_count = initial_success_metrics["throat_count"].clone()
                maximum_tail_distances = initial_success_metrics["tail_distances"].clone()
                maximum_tail_separation = initial_success_metrics["tail_separation"].clone()
                policy_samples = []
                final_policy_actions = torch.zeros_like(actions)
                policy_gripper_close_latched = torch.zeros((actions.shape[0], 2), dtype=torch.bool, device=env.device)
                right_regrasp_has_clearance = args_cli.policy_right_regrasp_clearance > 0.0
                right_regrasp_retract_steps = (
                    args_cli.policy_right_regrasp_retract_steps if right_regrasp_has_clearance else 0
                )
                right_regrasp_insert_steps = (
                    args_cli.policy_right_regrasp_insert_steps if right_regrasp_has_clearance else 0
                )
                right_regrasp_retract_end = args_cli.policy_right_regrasp_open_steps + right_regrasp_retract_steps
                right_regrasp_move_end = right_regrasp_retract_end + args_cli.policy_right_regrasp_move_steps
                right_regrasp_insert_end = right_regrasp_move_end + right_regrasp_insert_steps
                right_regrasp_steps = right_regrasp_insert_end + args_cli.policy_right_regrasp_close_steps
                right_regrasp_counters = torch.full((actions.shape[0],), -1, dtype=torch.int64, device=env.device)
                right_regrasp_completed = torch.zeros(actions.shape[0], dtype=torch.bool, device=env.device)
                right_regrasp_tail_to_tcp_hand = torch.full(
                    (actions.shape[0], 2, 3), torch.nan, dtype=actions.dtype, device=env.device
                )
                right_regrasp_finger_positions = torch.full(
                    (actions.shape[0], 2), torch.nan, dtype=actions.dtype, device=env.device
                )
                right_regrasp_start_tcp = torch.full(
                    (actions.shape[0], 3), torch.nan, dtype=actions.dtype, device=env.device
                )
                right_regrasp_start_hand_axes = torch.full(
                    (actions.shape[0], 3, 3), torch.nan, dtype=actions.dtype, device=env.device
                )
                right_regrasp_end_tcp = torch.full_like(right_regrasp_start_tcp, torch.nan)
                policy_state = after_handoff_close
                sample_steps = {1, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, args_cli.policy_steps}
                for step in range(args_cli.policy_steps):
                    right_regrasp_was_started = right_regrasp_counters >= 0
                    observation = env.observation_manager.compute()["policy"]
                    with torch.inference_mode():
                        policy_actions = policy(observation)
                    policy_actions = policy_actions.clone()
                    if args_cli.policy_force_grippers_closed:
                        _set_term(policy_actions, env, "left_gripper", -1.0)
                        _set_term(policy_actions, env, "right_gripper", -1.0)
                    if args_cli.policy_gripper_close_distance is not None:
                        close_candidate = policy_state["distance"] <= args_cli.policy_gripper_close_distance
                        if args_cli.policy_gripper_close_socket_error is not None:
                            socket_error = torch.linalg.vector_norm(
                                _tail_to_tcp_hand(policy_state) - socket_targets,
                                dim=-1,
                            )
                            close_candidate &= socket_error <= args_cli.policy_gripper_close_socket_error
                        policy_gripper_close_latched |= close_candidate
                        _set_term(
                            policy_actions,
                            env,
                            "left_gripper",
                            torch.where(policy_gripper_close_latched[:, 0], -1.0, 1.0).unsqueeze(-1),
                        )
                        _set_term(
                            policy_actions,
                            env,
                            "right_gripper",
                            torch.where(policy_gripper_close_latched[:, 1], -1.0, 1.0).unsqueeze(-1),
                        )
                    if args_cli.policy_right_hand_translation_bias is not None and (
                        args_cli.policy_right_arm_bias_steps == 0 or step < args_cli.policy_right_arm_bias_steps
                    ):
                        bias_hand = policy_actions.new_tensor(args_cli.policy_right_hand_translation_bias)
                        right_hand_axes_w = _hand_axes(env)[:, 1]
                        bias_w = torch.sum(right_hand_axes_w * bias_hand[None, :, None], dim=1)
                        right_root_quaternion = env.scene["robot_right"].data.root_quat_w.torch
                        bias_b = math_utils.quat_apply_inverse(right_root_quaternion, bias_w)
                        bias_action = torch.zeros((policy_actions.shape[0], 6), device=env.device)
                        bias_action[:, :3] = bias_b
                        _add_term(policy_actions, env, "right_arm", bias_action)
                        policy_actions.clamp_(-1.0, 1.0)
                    if args_cli.policy_right_socket_feedback_gain is not None:
                        feedback_active = (
                            active
                            & (~policy_gripper_close_latched[:, 1])
                            & (policy_state["distance"][:, 1] <= args_cli.policy_socket_feedback_activation_distance)
                        )
                        socket_error_hand = _tail_to_tcp_hand(policy_state)[:, 1] - socket_targets[1]
                        axis_scale = socket_error_hand.new_tensor(args_cli.policy_right_socket_feedback_axis_scale)
                        feedback_hand = args_cli.policy_right_socket_feedback_gain * axis_scale * socket_error_hand
                        feedback_norm = torch.linalg.vector_norm(feedback_hand, dim=-1, keepdim=True)
                        feedback_hand *= torch.clamp(
                            args_cli.policy_socket_feedback_max_action / feedback_norm.clamp_min(1.0e-8),
                            max=1.0,
                        )
                        feedback_hand *= feedback_active.unsqueeze(-1)
                        right_hand_axes_w = policy_state["hand_axes"][:, 1]
                        feedback_w = torch.sum(right_hand_axes_w * feedback_hand.unsqueeze(-1), dim=1)
                        right_root_quaternion = env.scene["robot_right"].data.root_quat_w.torch
                        feedback_b = math_utils.quat_apply_inverse(right_root_quaternion, feedback_w)
                        feedback_action = torch.zeros((policy_actions.shape[0], 6), device=env.device)
                        feedback_action[:, :3] = feedback_b
                        _add_term(policy_actions, env, "right_arm", feedback_action)
                        policy_actions.clamp_(-1.0, 1.0)
                    if args_cli.policy_right_arm_bias is not None and (
                        args_cli.policy_right_arm_bias_steps == 0 or step < args_cli.policy_right_arm_bias_steps
                    ):
                        _add_term(
                            policy_actions,
                            env,
                            "right_arm",
                            policy_actions.new_tensor(args_cli.policy_right_arm_bias).unsqueeze(0),
                        )
                        policy_actions.clamp_(-1.0, 1.0)
                    if policy_right_arm_bias_by_env is not None and (
                        args_cli.policy_right_arm_bias_steps == 0 or step < args_cli.policy_right_arm_bias_steps
                    ):
                        _add_term(policy_actions, env, "right_arm", policy_right_arm_bias_by_env)
                        policy_actions.clamp_(-1.0, 1.0)
                    right_regrasp_active = (right_regrasp_counters >= 0) & (
                        right_regrasp_counters < right_regrasp_steps
                    )
                    if args_cli.policy_right_regrasp_offset_hand is not None and bool(right_regrasp_active.any()):
                        right_regrasp_open = right_regrasp_active & (right_regrasp_counters < right_regrasp_insert_end)
                        zero_arm_action = torch.zeros((policy_actions.shape[0], 6), device=env.device)
                        _set_term_masked(policy_actions, env, "left_arm", right_regrasp_active, zero_arm_action)
                        _set_term_masked(policy_actions, env, "right_arm", right_regrasp_active, zero_arm_action)
                        _set_term_masked(policy_actions, env, "left_gripper", right_regrasp_active, -1.0)
                        _set_term_masked(policy_actions, env, "right_gripper", right_regrasp_open, 1.0)
                        _set_term_masked(
                            policy_actions, env, "right_gripper", right_regrasp_active & (~right_regrasp_open), -1.0
                        )
                        offset_hand = policy_actions.new_tensor(args_cli.policy_right_regrasp_offset_hand)
                        displacement_hand = torch.zeros((policy_actions.shape[0], 3), device=env.device)
                        if right_regrasp_has_clearance:
                            right_regrasp_retract = (
                                right_regrasp_active
                                & (right_regrasp_counters >= args_cli.policy_right_regrasp_open_steps)
                                & (right_regrasp_counters < right_regrasp_retract_end)
                            )
                            right_regrasp_move = (
                                right_regrasp_active
                                & (right_regrasp_counters >= right_regrasp_retract_end)
                                & (right_regrasp_counters < right_regrasp_move_end)
                            )
                            right_regrasp_insert = (
                                right_regrasp_active
                                & (right_regrasp_counters >= right_regrasp_move_end)
                                & (right_regrasp_counters < right_regrasp_insert_end)
                            )
                            displacement_hand[right_regrasp_retract, 2] = (
                                -args_cli.policy_right_regrasp_clearance / right_regrasp_retract_steps
                            )
                            displacement_hand[right_regrasp_move, :2] = (
                                offset_hand[:2] / args_cli.policy_right_regrasp_move_steps
                            )
                            displacement_hand[right_regrasp_insert, 2] = (
                                offset_hand[2] + args_cli.policy_right_regrasp_clearance
                            ) / right_regrasp_insert_steps
                            right_regrasp_motion = right_regrasp_retract | right_regrasp_move | right_regrasp_insert
                        else:
                            right_regrasp_motion = (
                                right_regrasp_active
                                & (right_regrasp_counters >= args_cli.policy_right_regrasp_open_steps)
                                & (right_regrasp_counters < right_regrasp_move_end)
                            )
                            displacement_hand[right_regrasp_motion] = (
                                offset_hand / args_cli.policy_right_regrasp_move_steps
                            )
                        right_hand_axes_w = policy_state["hand_axes"][:, 1]
                        displacement_w = torch.sum(right_hand_axes_w * displacement_hand.unsqueeze(-1), dim=1)
                        right_root_quaternion = env.scene["robot_right"].data.root_quat_w.torch
                        displacement_b = math_utils.quat_apply_inverse(right_root_quaternion, displacement_w)
                        right_arm_action = zero_arm_action.clone()
                        right_arm_action[:, :3] = displacement_b / arm_action_scale
                        _set_term_masked(policy_actions, env, "right_arm", right_regrasp_motion, right_arm_action)
                        policy_actions.clamp_(-1.0, 1.0)
                    if policy_actions.shape != actions.shape:
                        raise ValueError(
                            f"Checkpoint returned actions with shape {tuple(policy_actions.shape)}; "
                            f"expected {tuple(actions.shape)}."
                        )
                    inactive_actions = torch.zeros_like(policy_actions)
                    _set_term(inactive_actions, env, "left_gripper", -1.0)
                    _set_term(inactive_actions, env, "right_gripper", -1.0)
                    policy_actions = torch.where(active.unsqueeze(-1), policy_actions, inactive_actions)
                    _step(policy_actions, "checkpoint_policy")
                    final_policy_actions = policy_actions
                    completed_policy_steps = step + 1
                    current = _state(env)
                    policy_state = current
                    success_metrics = _success_metrics(env, success_params)
                    acquisition_candidate = active.unsqueeze(-1) & (
                        (current["distance"] <= acquisition_distance)
                        & (current["finger"] >= minimum_finger_position)
                        & (current["finger"] <= maximum_finger_position)
                    )
                    acquisition_candidate_steps = torch.where(
                        acquisition_candidate,
                        acquisition_candidate_steps + 1,
                        torch.zeros_like(acquisition_candidate_steps),
                    )
                    acquired_latched |= acquisition_candidate_steps >= acquisition_confirmation_steps
                    bilaterally_acquired |= acquired_latched.all(dim=1)
                    current_socket_errors = torch.linalg.vector_norm(
                        _tail_to_tcp_hand(current) - socket_targets,
                        dim=-1,
                    )
                    strict_socket_candidate = acquisition_candidate & (current_socket_errors <= maximum_socket_error)
                    strict_socket_candidate_steps = torch.where(
                        strict_socket_candidate,
                        strict_socket_candidate_steps + 1,
                        torch.zeros_like(strict_socket_candidate_steps),
                    )
                    strict_socket_acquired |= (strict_socket_candidate_steps >= acquisition_confirmation_steps).all(
                        dim=1
                    )
                    new_bilateral_acquisition = (first_bilateral_acquisition_steps < 0) & bilaterally_acquired
                    first_bilateral_acquisition_steps[new_bilateral_acquisition] = step + 1
                    acquisition_tail_to_tcp_hand[new_bilateral_acquisition] = _tail_to_tcp_hand(current)[
                        new_bilateral_acquisition
                    ]
                    acquisition_finger_positions[new_bilateral_acquisition] = current["finger"][
                        new_bilateral_acquisition
                    ]
                    for env_id in torch.nonzero(new_bilateral_acquisition, as_tuple=False).flatten().tolist():
                        acquisition_demonstration_lengths[env_id] = len(demonstration_observations)
                    right_regrasp_counters[right_regrasp_was_started] += 1
                    if args_cli.policy_right_regrasp_offset_hand is not None:
                        right_regrasp_counters[new_bilateral_acquisition] = 0
                        right_regrasp_start_tcp[new_bilateral_acquisition] = current["tcp"][
                            new_bilateral_acquisition, 1
                        ]
                        right_regrasp_start_hand_axes[new_bilateral_acquisition] = current["hand_axes"][
                            new_bilateral_acquisition, 1
                        ]
                    new_right_regrasp_completion = right_regrasp_counters == right_regrasp_steps
                    right_regrasp_completed |= new_right_regrasp_completion
                    right_regrasp_end_tcp[new_right_regrasp_completion] = current["tcp"][
                        new_right_regrasp_completion, 1
                    ]
                    right_regrasp_tail_to_tcp_hand[new_right_regrasp_completion] = _tail_to_tcp_hand(current)[
                        new_right_regrasp_completion
                    ]
                    right_regrasp_finger_positions[new_right_regrasp_completion] = current["finger"][
                        new_right_regrasp_completion
                    ]
                    retained = (
                        (current["distance"] <= retention_distance)
                        & (current["finger"] >= minimum_finger_position)
                        & (current["finger"] <= maximum_finger_position)
                    )
                    right_regrasp_grace = (right_regrasp_counters >= 0) & (
                        right_regrasp_counters < right_regrasp_steps + release_confirmation_steps
                    )
                    release_candidate = (
                        active
                        & bilaterally_acquired
                        & (~right_regrasp_grace)
                        & (acquired_latched & (~retained)).any(dim=1)
                    )
                    release_candidate_steps = torch.where(
                        release_candidate,
                        release_candidate_steps + 1,
                        torch.zeros_like(release_candidate_steps),
                    )
                    new_lost_grasp = (
                        active & (lost_grasp_policy_steps < 0) & (release_candidate_steps >= release_confirmation_steps)
                    )
                    lost_grasp_policy_steps[new_lost_grasp] = step + 1
                    new_success = active & (first_success_steps < 0) & success_metrics["success"]
                    first_success_steps[new_success] = step + 1
                    for env_id in torch.nonzero(new_success, as_tuple=False).flatten().tolist():
                        successful_demonstration_lengths[env_id] = len(demonstration_observations)
                    maximum_distance = torch.where(
                        active.unsqueeze(-1), torch.maximum(maximum_distance, current["distance"]), maximum_distance
                    )
                    minimum_socket_errors = torch.where(
                        active.unsqueeze(-1),
                        torch.minimum(minimum_socket_errors, current_socket_errors),
                        minimum_socket_errors,
                    )
                    minimum_throat_count = torch.where(
                        active,
                        torch.minimum(minimum_throat_count, success_metrics["throat_count"]),
                        minimum_throat_count,
                    )
                    maximum_tail_distances = torch.where(
                        active.unsqueeze(-1),
                        torch.maximum(maximum_tail_distances, success_metrics["tail_distances"]),
                        maximum_tail_distances,
                    )
                    maximum_tail_separation = torch.where(
                        active,
                        torch.maximum(maximum_tail_separation, success_metrics["tail_separation"]),
                        maximum_tail_separation,
                    )
                    if step + 1 in sample_steps:
                        policy_samples.append(
                            {
                                "step": step + 1,
                                "distance_mm": (current["distance"][0] * 1000).tolist(),
                                "finger_mm": (current["finger"][0] * 1000).tolist(),
                                "throat_count": int(success_metrics["throat_count"][0]),
                                "tail_to_knot_mm": (success_metrics["tail_distances"][0] * 1000).tolist(),
                                "tail_separation_mm": float(success_metrics["tail_separation"][0] * 1000),
                                "grasped": success_metrics["grasped"][0].tolist(),
                                "safe": bool(success_metrics["safe"][0]),
                                "success": bool(success_metrics["success"][0]),
                                "action": policy_actions[0].tolist(),
                            }
                        )
                    terminal_success = (
                        new_success
                        if args_cli.stop_on_success or args_cli.num_envs > 1
                        else torch.zeros_like(new_success)
                    )
                    active &= success_metrics["safe"] & (~terminal_success) & (~new_lost_grasp)
                    if not bool(active.any()):
                        break
                policy_end = _state(env)
                final_success_metrics = _success_metrics(env, success_params)
                first_bilateral_acquisition_step = int(first_bilateral_acquisition_steps[0])
                first_bilateral_acquisition_step = (
                    None if first_bilateral_acquisition_step < 0 else first_bilateral_acquisition_step
                )
                first_success_step = int(first_success_steps[0])
                first_success_step = None if first_success_step < 0 else first_success_step
                lost_grasp_policy_step = int(lost_grasp_policy_steps[0])
                lost_grasp_policy_step = None if lost_grasp_policy_step < 0 else lost_grasp_policy_step
                acquired_mask = first_bilateral_acquisition_steps >= 0
                success_mask = first_success_steps >= 0
                lost_grasp_mask = lost_grasp_policy_steps >= 0
                right_regrasp_tcp_delta_w = right_regrasp_end_tcp - right_regrasp_start_tcp
                right_regrasp_tcp_delta_hand = torch.sum(
                    right_regrasp_start_hand_axes * right_regrasp_tcp_delta_w.unsqueeze(-2), dim=-1
                )
                best_right_socket_env = int(torch.argmin(minimum_socket_errors[:, 1]).item())
                result = {
                    "run_label": args_cli.run_label,
                    "checkpoint": str(args_cli.policy_checkpoint.resolve()),
                    "num_envs": args_cli.num_envs,
                    "initial_level": initial_level,
                    "approach_waypoint_levels": args_cli.approach_waypoint_levels,
                    "joint_interpolation_levels": args_cli.joint_interpolation_levels,
                    "joint_interpolation_waypoints": args_cli.joint_interpolation_waypoints,
                    "final_waypoint_handoff_distance_m": args_cli.final_waypoint_handoff_distance,
                    "approach_alignment_roll_degrees": approach_alignment_roll_degrees,
                    "approach_orientation_steps": approach_orientation_step_count,
                    "approach_orientation_stage_result": approach_orientation_stage_result,
                    "pregrasp_clearance_m": args_cli.pregrasp_clearance,
                    "pregrasp_world_z_clearance_m": args_cli.pregrasp_world_z_clearance,
                    "freeze_approach_tail_targets": args_cli.freeze_approach_tail_targets,
                    "grasp_approach_close_step": args_cli.grasp_approach_close_step,
                    "grasp_approach_handoff_distance_m": args_cli.grasp_approach_handoff_distance,
                    "approach_duration_s": len(approach_trace[1:]) * policy_dt,
                    "policy_duration_s": completed_policy_steps * policy_dt,
                    "completed_policy_steps": completed_policy_steps,
                    "policy_handoff_close_steps": args_cli.policy_handoff_close_steps,
                    "policy_arm_action_scale": args_cli.policy_arm_action_scale,
                    "policy_stochastic": args_cli.policy_stochastic,
                    "policy_action_std": args_cli.policy_action_std,
                    "policy_force_grippers_closed": args_cli.policy_force_grippers_closed,
                    "policy_gripper_close_distance_m": args_cli.policy_gripper_close_distance,
                    "policy_gripper_close_socket_error_m": args_cli.policy_gripper_close_socket_error,
                    "policy_right_socket_feedback_gain_per_m": args_cli.policy_right_socket_feedback_gain,
                    "policy_right_socket_feedback_axis_scale": args_cli.policy_right_socket_feedback_axis_scale,
                    "policy_socket_feedback_activation_distance_m": (
                        args_cli.policy_socket_feedback_activation_distance
                    ),
                    "policy_socket_feedback_max_action": args_cli.policy_socket_feedback_max_action,
                    "policy_right_arm_bias": args_cli.policy_right_arm_bias,
                    "policy_right_arm_bias_uniform_scale": args_cli.policy_right_arm_bias_uniform_scale,
                    "policy_right_arm_bias_uniform_center": args_cli.policy_right_arm_bias_uniform_center,
                    "best_right_socket_env": best_right_socket_env,
                    "best_right_socket_bias": (
                        policy_right_arm_bias_by_env[best_right_socket_env].tolist()
                        if policy_right_arm_bias_by_env is not None
                        else None
                    ),
                    "batch_minimum_socket_error_mm": (minimum_socket_errors.amin(dim=0) * 1000).tolist(),
                    "policy_right_hand_translation_bias": args_cli.policy_right_hand_translation_bias,
                    "policy_right_arm_bias_steps": args_cli.policy_right_arm_bias_steps,
                    "policy_right_regrasp_offset_hand_m": args_cli.policy_right_regrasp_offset_hand,
                    "policy_right_regrasp_clearance_m": args_cli.policy_right_regrasp_clearance,
                    "policy_right_regrasp_stage_steps": [
                        args_cli.policy_right_regrasp_open_steps,
                        right_regrasp_retract_steps,
                        args_cli.policy_right_regrasp_move_steps,
                        right_regrasp_insert_steps,
                        args_cli.policy_right_regrasp_close_steps,
                    ],
                    "right_regrasp_completion": _acquisition_summary(
                        right_regrasp_completed,
                        right_regrasp_tail_to_tcp_hand,
                        right_regrasp_finger_positions,
                    ),
                    "right_regrasp_tcp_delta_hand_mean_mm": (
                        (right_regrasp_tcp_delta_hand[right_regrasp_completed].mean(dim=0) * 1000).tolist()
                        if bool(right_regrasp_completed.any())
                        else None
                    ),
                    "coupling_mode": args_cli.coupling_mode,
                    "num_substeps": physics_cfg.num_substeps,
                    "closed_position_m_per_finger": closed_position,
                    "open_position_m_per_finger": open_position,
                    "grasp_assist": args_cli.grasp_assist,
                    "seed": args_cli.seed,
                    "after_approach_distance_mm": (after_approach["distance"][0] * 1000).tolist(),
                    "after_approach_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(after_approach)[0] * 1000).tolist(),
                    "after_approach_finger_mm": (after_approach["finger"][0] * 1000).tolist(),
                    "after_handoff_close_distance_mm": (after_handoff_close["distance"][0] * 1000).tolist(),
                    "after_handoff_close_tail_to_tcp_hand_mm": (
                        _tail_to_tcp_hand(after_handoff_close)[0] * 1000
                    ).tolist(),
                    "after_handoff_close_finger_mm": (after_handoff_close["finger"][0] * 1000).tolist(),
                    "approach_stage_results": approach_stage_results,
                    "first_bilateral_acquisition_step": first_bilateral_acquisition_step,
                    "first_success_policy_step": first_success_step,
                    "lost_grasp_policy_step": lost_grasp_policy_step,
                    "bilateral_acquisition_count": int((first_bilateral_acquisition_steps >= 0).sum()),
                    "strict_socket_acquisition_count": int(strict_socket_acquired.sum()),
                    "success_count": len(successful_demonstration_lengths),
                    "lost_grasp_count": int((lost_grasp_policy_steps >= 0).sum()),
                    "acquisition_by_outcome": {
                        "success": _acquisition_summary(
                            acquired_mask & success_mask,
                            acquisition_tail_to_tcp_hand,
                            acquisition_finger_positions,
                        ),
                        "lost_grasp": _acquisition_summary(
                            acquired_mask & lost_grasp_mask,
                            acquisition_tail_to_tcp_hand,
                            acquisition_finger_positions,
                        ),
                        "other": _acquisition_summary(
                            acquired_mask & (~success_mask) & (~lost_grasp_mask),
                            acquisition_tail_to_tcp_hand,
                            acquisition_finger_positions,
                        ),
                    },
                    "minimum_throat_count": int(minimum_throat_count[0]),
                    "maximum_tail_to_knot_mm": (maximum_tail_distances[0] * 1000).tolist(),
                    "maximum_tail_separation_mm": float(maximum_tail_separation[0] * 1000),
                    "maximum_distance_mm": (maximum_distance[0] * 1000).tolist(),
                    "minimum_socket_error_mm": (minimum_socket_errors[0] * 1000).tolist(),
                    "final_distance_mm": (policy_end["distance"][0] * 1000).tolist(),
                    "final_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(policy_end)[0] * 1000).tolist(),
                    "final_finger_mm": (policy_end["finger"][0] * 1000).tolist(),
                    "final_action": final_policy_actions[0].tolist(),
                    "final_throat_count": int(final_success_metrics["throat_count"][0]),
                    "final_tail_to_knot_mm": (final_success_metrics["tail_distances"][0] * 1000).tolist(),
                    "final_tail_separation_mm": float(final_success_metrics["tail_separation"][0] * 1000),
                    "final_success_grasped": final_success_metrics["grasped"][0].tolist(),
                    "final_safe": bool(final_success_metrics["safe"][0]),
                    "final_success": first_success_step is not None,
                    "policy_samples": policy_samples,
                }
                if not args_cli.summary_only:
                    result["approach_samples"] = approach_samples
                if args_cli.save_demonstration is not None and successful_demonstration_lengths:
                    _save_demonstration(
                        args_cli.save_demonstration,
                        demonstration_observations,
                        demonstration_actions,
                        demonstration_phases,
                        successful_demonstration_lengths,
                        {**result, "demonstration_selection": "success"},
                    )
                    result["demonstration_path"] = str(args_cli.save_demonstration.resolve())
                else:
                    result["demonstration_path"] = None
                if args_cli.save_acquisition_demonstration is not None and acquisition_demonstration_lengths:
                    _save_demonstration(
                        args_cli.save_acquisition_demonstration,
                        demonstration_observations,
                        demonstration_actions,
                        demonstration_phases,
                        acquisition_demonstration_lengths,
                        {**result, "demonstration_selection": "bilateral_acquisition"},
                    )
                    result["acquisition_demonstration_path"] = str(args_cli.save_acquisition_demonstration.resolve())
                else:
                    result["acquisition_demonstration_path"] = None
                print("SHOELACE_HYBRID_RESULT=" + json.dumps(result, indent=None if args_cli.summary_only else 2))
                return

            _set_term(actions, env, "left_gripper", -1.0)
            _set_term(actions, env, "right_gripper", -1.0)
            close_samples = []
            for step in range(args_cli.close_steps):
                _step(actions, "close")
                current = _state(env)
                close_samples.append(
                    {
                        "step": step + 1,
                        "distance_mm": (current["distance"][0] * 1000).tolist(),
                        "finger_mm": (current["finger"][0] * 1000).tolist(),
                    }
                )
            after_close = _state(env)
            maximum_hold_distance = after_close["distance"].clone()
            hold_samples = []
            for step in range(args_cli.hold_steps):
                _step(actions, "hold")
                current = _state(env)
                maximum_hold_distance = torch.maximum(maximum_hold_distance, current["distance"])
                if step in {0, 4, 9, 19, 29, args_cli.hold_steps - 1}:
                    hold_samples.append(
                        {
                            "step": step + 1,
                            "distance_mm": (current["distance"][0] * 1000).tolist(),
                            "finger_mm": (current["finger"][0] * 1000).tolist(),
                        }
                    )
            pull_start = _state(env)

            directions_w = torch.tensor(
                (args_cli.left_pull_direction_w, args_cli.right_pull_direction_w),
                device=env.device,
                dtype=torch.float32,
            )
            directions_w /= torch.linalg.vector_norm(directions_w, dim=-1, keepdim=True)
            feedforward_displacement_w = args_cli.pull_speed * policy_dt * directions_w.unsqueeze(0)
            pull_start_tail_to_tcp_w = pull_start["tail"] - pull_start["tcp"]

            maximum_distance = pull_start["distance"].clone()
            minimum_follow_projection = torch.full((1, 2), float("inf"), device=env.device)
            first_success_step = None
            completed_pull_steps = 0
            samples = []
            for step in range(args_cli.pull_steps):
                current = _state(env)
                tail_to_tcp_error_w = (current["tail"] - current["tcp"]) - pull_start_tail_to_tcp_w
                displacement_w = feedforward_displacement_w + args_cli.pull_tracking_gain * tail_to_tcp_error_w
                displacement_norm = torch.linalg.vector_norm(displacement_w, dim=-1, keepdim=True)
                maximum_displacement = displacement_norm.new_full(
                    displacement_norm.shape, args_cli.pull_speed * policy_dt
                )
                displacement_w *= torch.minimum(displacement_norm, maximum_displacement) / displacement_norm.clamp_min(
                    1.0e-8
                )
                _set_tcp_translation(actions, env, displacement_w, arm_action_scale)
                _step(actions, "pull")
                current = _state(env)
                completed_pull_steps = step + 1
                success_metrics = _success_metrics(env, success_params)
                if first_success_step is None and bool(success_metrics["success"][0]):
                    first_success_step = step + 1
                maximum_distance = torch.maximum(maximum_distance, current["distance"])
                tcp_delta = current["tcp"] - pull_start["tcp"]
                tail_delta = current["tail"] - pull_start["tail"]
                relative_projection = torch.sum((tail_delta - tcp_delta) * directions_w.unsqueeze(0), dim=-1)
                minimum_follow_projection = torch.minimum(minimum_follow_projection, relative_projection)
                if step in {0, 4, 9, 19, 29, 44, args_cli.pull_steps - 1}:
                    samples.append(
                        {
                            "step": step + 1,
                            "tcp_mm": (torch.sum(tcp_delta * directions_w.unsqueeze(0), dim=-1)[0] * 1000).tolist(),
                            "tail_mm": (torch.sum(tail_delta * directions_w.unsqueeze(0), dim=-1)[0] * 1000).tolist(),
                            "distance_mm": (current["distance"][0] * 1000).tolist(),
                            "throat_count": int(success_metrics["throat_count"][0]),
                            "tail_to_knot_mm": (success_metrics["tail_distances"][0] * 1000).tolist(),
                            "tail_separation_mm": float(success_metrics["tail_separation"][0] * 1000),
                        }
                    )
                if args_cli.stop_on_success and first_success_step is not None:
                    break
            pull_end = _state(env)
            final_success_metrics = _success_metrics(env, success_params)

            tcp_delta = pull_end["tcp"] - pull_start["tcp"]
            tail_delta = pull_end["tail"] - pull_start["tail"]
            tcp_projection = torch.sum(tcp_delta * directions_w.unsqueeze(0), dim=-1)[0]
            tail_projection = torch.sum(tail_delta * directions_w.unsqueeze(0), dim=-1)[0]
            follow_ratio = tail_projection / tcp_projection.clamp_min(1.0e-8)
            relative_slip = torch.linalg.vector_norm(
                (pull_end["tail"] - pull_end["tcp"]) - (pull_start["tail"] - pull_start["tcp"]), dim=-1
            )[0]
            hold_relative_slip = torch.linalg.vector_norm(
                (pull_start["tail"] - pull_start["tcp"]) - (after_close["tail"] - after_close["tcp"]), dim=-1
            )[0]
            acquired_after_close = (
                (after_close["distance"][0] <= acquisition_distance)
                & (after_close["finger"][0] >= minimum_finger_position)
                & (after_close["finger"][0] <= maximum_finger_position)
            )
            retained_after_hold = (
                (pull_start["distance"][0] <= retention_distance)
                & (pull_start["finger"][0] >= minimum_finger_position)
                & (pull_start["finger"][0] <= maximum_finger_position)
            )
            retained = (
                (pull_end["distance"][0] <= retention_distance)
                & (pull_end["finger"][0] >= minimum_finger_position)
                & (pull_end["finger"][0] <= maximum_finger_position)
            )
            result = {
                "run_label": args_cli.run_label,
                "action_terms": dict(
                    zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True)
                ),
                "policy_dt_s": policy_dt,
                "gravity_m_s2": env_cfg.sim.gravity,
                "initial_level": initial_level,
                "outward_tilt_degrees": args_cli.outward_tilt_degrees,
                "hand_roll_degrees": args_cli.hand_roll_degrees,
                "mirrored_world_yaw_degrees": args_cli.mirrored_world_yaw_degrees,
                "orientation_advance_m": args_cli.orientation_advance_m,
                "tilt_duration_s": tilt_step_count * policy_dt,
                "tilt_error_degrees": tilt_error_degrees,
                "approach_duration_s": len(approach_trace[1:]) * policy_dt,
                "commanded_approach_speed_m_s": args_cli.approach_speed,
                "approach_tolerance_m": args_cli.approach_tolerance,
                "pregrasp_clearance_m": args_cli.pregrasp_clearance,
                "pregrasp_world_z_clearance_m": args_cli.pregrasp_world_z_clearance,
                "freeze_approach_tail_targets": args_cli.freeze_approach_tail_targets,
                "grasp_approach_close_step": args_cli.grasp_approach_close_step,
                "grasp_approach_handoff_distance_m": args_cli.grasp_approach_handoff_distance,
                "approach_reference_level": args_cli.approach_reference_level,
                "approach_waypoint_levels": args_cli.approach_waypoint_levels,
                "joint_interpolation_levels": args_cli.joint_interpolation_levels,
                "joint_interpolation_waypoints": args_cli.joint_interpolation_waypoints,
                "final_waypoint_handoff_distance_m": args_cli.final_waypoint_handoff_distance,
                "approach_alignment_roll_degrees": approach_alignment_roll_degrees,
                "approach_orientation_steps": approach_orientation_step_count,
                "approach_orientation_stage_result": approach_orientation_stage_result,
                "approach_orientation_error_degrees": approach_orientation_error_degrees,
                "left_tcp_offset_w_m": args_cli.left_tcp_offset_w,
                "right_tcp_offset_w_m": args_cli.right_tcp_offset_w,
                "left_tail_to_tcp_target_hand_m": args_cli.left_tail_to_tcp_target_hand,
                "right_tail_to_tcp_target_hand_m": args_cli.right_tail_to_tcp_target_hand,
                "left_grasp_joint_positions_rad": args_cli.left_grasp_joint_positions,
                "right_grasp_joint_positions_rad": args_cli.right_grasp_joint_positions,
                "close_duration_s": args_cli.close_steps * policy_dt,
                "hold_duration_s": args_cli.hold_steps * policy_dt,
                "pull_duration_s": completed_pull_steps * policy_dt,
                "commanded_pull_speed_m_s": args_cli.pull_speed,
                "pull_tracking_gain": args_cli.pull_tracking_gain,
                "pull_directions_w": directions_w.tolist(),
                "gripper_damping_n_s_m": args_cli.gripper_damping,
                "gripper_stiffness_n_m": env_cfg.scene.robot_left.actuators["panda_hand"].stiffness,
                "gripper_maximum_velocity_m_s": env_cfg.actions.left_gripper.maximum_velocity,
                "bend_stiffness": shoelace_env_module.BEND_STIFFNESS,
                "bend_stiffness_scale": args_cli.bend_stiffness_scale,
                "tail_bend_stiffness": shoelace_env_module.TAIL_BEND_STIFFNESS,
                "tail_bend_stiffness_scale": args_cli.tail_bend_stiffness_scale,
                "tail_bend_damping": shoelace_env_module.TAIL_BEND_DAMPING,
                "tail_bend_damping_scale": args_cli.tail_bend_damping_scale,
                "tail_stiff_core_length_m": shoelace_env_module.TAIL_STIFF_CORE_LENGTH,
                "tail_stiff_transition_length_m": shoelace_env_module.TAIL_STIFF_TRANSITION_LENGTH,
                "stretch_damping": shoelace_env_module.STRETCH_DAMPING,
                "bend_damping": shoelace_env_module.BEND_DAMPING,
                "cable_damping_scale": args_cli.cable_damping_scale,
                "closed_position_m_per_finger": closed_position,
                "open_position_m_per_finger": open_position,
                "acquisition_distance_m": acquisition_distance,
                "retention_distance_m": retention_distance,
                "minimum_grasp_position_m_per_finger": minimum_finger_position,
                "closed_threshold_m_per_finger": maximum_finger_position,
                "requested_lace_mu": args_cli.lace_mu,
                "requested_finger_mu": args_cli.finger_mu,
                "shape_materials": shape_materials,
                "num_substeps": physics_cfg.num_substeps,
                "vbd_iterations": args_cli.vbd_iterations,
                "coupling_mode": args_cli.coupling_mode,
                "grasp_assist": args_cli.grasp_assist,
                "seed": args_cli.seed,
                "coupler_iterations": coupler_cfg.iterations,
                "admm_rho": coupler_cfg.rho if isinstance(coupler_cfg, CouplerAdmmCfg) else None,
                "admm_gamma": coupler_cfg.gamma if isinstance(coupler_cfg, CouplerAdmmCfg) else None,
                "admm_baumgarte": coupler_cfg.baumgarte if isinstance(coupler_cfg, CouplerAdmmCfg) else None,
                "admm_contact_matching": (
                    coupler_cfg.rigid_contact_matching if isinstance(coupler_cfg, CouplerAdmmCfg) else None
                ),
                "proxy_mass_scale": proxy_cfg.mass_scale if proxy_cfg is not None else None,
                "proxy_mode": proxy_cfg.mode if proxy_cfg is not None else None,
                "collide_interval": args_cli.collide_interval if proxy_cfg is not None else None,
                "contact_history": args_cli.contact_history,
                "cuda_graph": physics_cfg.use_cuda_graph,
                "initial_distance_mm": (reset_state["distance"][0] * 1000).tolist(),
                "initial_tcp_to_tail_w_mm": (initial_tcp_to_tail_w[0] * 1000).tolist(),
                "initial_tcp_to_tail_hand_mm": (initial_tcp_to_tail_hand[0] * 1000).tolist(),
                "initial_finger_mm": (reset_state["finger"][0] * 1000).tolist(),
                "acquired_at_reset": acquired_at_reset.tolist(),
                "initial_tail_m": reset_state["tail"][0].tolist(),
                "initial_tail_speed_mm_s": (
                    torch.linalg.vector_norm(reset_state["tail_velocity"][0], dim=-1) * 1000
                ).tolist(),
                "initial_throat_count": int(initial_success_metrics["throat_count"][0]),
                "initial_tail_to_knot_mm": (initial_success_metrics["tail_distances"][0] * 1000).tolist(),
                "initial_tail_separation_mm": float(initial_success_metrics["tail_separation"][0] * 1000),
                "initial_hand_local_axes_w": reset_state["hand_axes"][0].tolist(),
                "after_tilt_distance_mm": (after_tilt["distance"][0] * 1000).tolist(),
                "after_tilt_reset_tail_distance_mm": (
                    torch.linalg.vector_norm(reset_state["tail"] - after_tilt["tcp"], dim=-1)[0] * 1000
                ).tolist(),
                "orientation_target_tcp_error_mm": (
                    torch.linalg.vector_norm(orientation_target_tcp - after_tilt["tcp"], dim=-1)[0] * 1000
                ).tolist(),
                "after_tilt_arm_joint_positions_rad": after_tilt["arm_joint"][0].tolist(),
                "after_tilt_hand_local_axes_w": after_tilt["hand_axes"][0].tolist(),
                "after_approach_distance_mm": (after_approach["distance"][0] * 1000).tolist(),
                "after_approach_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(after_approach)[0] * 1000).tolist(),
                "approach_stage_results": approach_stage_results,
                "after_approach_tail_m": after_approach["tail"][0].tolist(),
                "after_approach_tail_speed_mm_s": (
                    torch.linalg.vector_norm(after_approach["tail_velocity"][0], dim=-1) * 1000
                ).tolist(),
                "after_approach_cable_rms_speed_mm_s": (after_approach_cable_rms_speed[0] * 1000).item(),
                "saved_settled_state": (
                    str(args_cli.save_settled_state.resolve()) if args_cli.save_settled_state is not None else None
                ),
                "after_approach_arm_joint_positions_rad": after_approach["arm_joint"][0].tolist(),
                "after_approach_hand_local_axes_w": after_approach["hand_axes"][0].tolist(),
                "after_close_distance_mm": (after_close["distance"][0] * 1000).tolist(),
                "after_close_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(after_close)[0] * 1000).tolist(),
                "after_close_finger_mm": (after_close["finger"][0] * 1000).tolist(),
                "acquired_after_close": acquired_after_close.tolist(),
                "pull_start_distance_mm": (pull_start["distance"][0] * 1000).tolist(),
                "pull_start_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(pull_start)[0] * 1000).tolist(),
                "pull_start_finger_mm": (pull_start["finger"][0] * 1000).tolist(),
                "maximum_hold_distance_mm": (maximum_hold_distance[0] * 1000).tolist(),
                "hold_relative_slip_mm": (hold_relative_slip * 1000).tolist(),
                "retained_after_hold": retained_after_hold.tolist(),
                "tcp_pull_mm": (tcp_projection * 1000).tolist(),
                "tail_pull_mm": (tail_projection * 1000).tolist(),
                "follow_ratio": follow_ratio.tolist(),
                "relative_slip_mm": (relative_slip * 1000).tolist(),
                "final_distance_mm": (pull_end["distance"][0] * 1000).tolist(),
                "final_tail_to_tcp_hand_mm": (_tail_to_tcp_hand(pull_end)[0] * 1000).tolist(),
                "maximum_distance_mm": (maximum_distance[0] * 1000).tolist(),
                "retained_at_end": retained.tolist(),
                "first_success_pull_step": first_success_step,
                "final_success": bool(final_success_metrics["success"][0]),
                "final_throat_count": int(final_success_metrics["throat_count"][0]),
                "final_tail_to_knot_mm": (final_success_metrics["tail_distances"][0] * 1000).tolist(),
                "final_tail_separation_mm": float(final_success_metrics["tail_separation"][0] * 1000),
                "final_success_grasped": final_success_metrics["grasped"][0].tolist(),
                "final_safe": bool(final_success_metrics["safe"][0]),
                "pull_samples": samples,
            }
            if not args_cli.summary_only:
                result["curriculum_states"] = _calibrated_curriculum_states(
                    approach_trace=approach_trace,
                    reference_tail=reset_state["tail"],
                    level_count=curriculum["level_count"],
                    gripper_open_phase_fraction=env_cfg.events.reset_shoelace.params["gripper_open_phase_fraction"],
                    approach_phase_exponent=env_cfg.events.reset_shoelace.params["approach_phase_exponent"],
                    open_position=env_cfg.events.reset_shoelace.params["open_position"],
                    closed_position=closed_position,
                )
                result["approach_samples"] = approach_samples
                result["close_samples"] = close_samples
                result["hold_samples"] = hold_samples
            if args_cli.save_demonstration is not None and result["final_success"]:
                _save_demonstration(
                    args_cli.save_demonstration,
                    demonstration_observations,
                    demonstration_actions,
                    demonstration_phases,
                    {0: len(demonstration_observations)},
                    {**result, "demonstration_selection": "success"},
                )
                result["demonstration_path"] = str(args_cli.save_demonstration.resolve())
            else:
                result["demonstration_path"] = None
            result["acquisition_demonstration_path"] = None
            indent = None if args_cli.summary_only else 2
            print("SHOELACE_PULL_RESULT=" + json.dumps(result, indent=indent))
        finally:
            gym_env.close()


if __name__ == "__main__":
    main()
