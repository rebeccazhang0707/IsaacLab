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


def _set_term(actions: torch.Tensor, env, term_name: str, values: torch.Tensor | float) -> None:
    offset = 0
    for active_name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        if active_name == term_name:
            actions[:, offset : offset + dimension] = values
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


def _play_curriculum_levels(env, level_count: int, hold_seconds: float, cycle_count: int) -> None:
    """Display every calibrated reset level sequentially without advancing physics."""
    if cycle_count == 0 and not env.sim.has_active_visualizers():
        raise ValueError("Infinite curriculum playback requires an active visualizer.")

    env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    difficulty_term = env.curriculum_manager.cfg.pull_to_grasp.func
    cycle = 0
    while cycle_count == 0 or cycle < cycle_count:
        for level in range(level_count):
            if not env.sim.is_headless_or_exist_active_visualizer():
                return
            difficulty_term.levels[env_ids] = level
            env.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=0)
            env.scene.write_data_to_sim()
            env.sim.forward()
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
        "--initial_level",
        type=int,
        default=None,
        help="Reset curriculum level. By default, start from the full open-pregrasp pose.",
    )
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
    parser.add_argument("--num_substeps", type=int, default=8)
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
    if not np.isfinite(args_cli.left_pull_direction_w).all() or np.linalg.norm(args_cli.left_pull_direction_w) == 0.0:
        raise ValueError("Left pull direction must be finite and non-zero.")
    if not np.isfinite(args_cli.right_pull_direction_w).all() or np.linalg.norm(args_cli.right_pull_direction_w) == 0.0:
        raise ValueError("Right pull direction must be finite and non-zero.")
    if args_cli.left_tcp_offset_w is not None and not np.isfinite(args_cli.left_tcp_offset_w).all():
        raise ValueError("Left TCP offset must contain finite values.")
    if args_cli.right_tcp_offset_w is not None and not np.isfinite(args_cli.right_tcp_offset_w).all():
        raise ValueError("Right TCP offset must contain finite values.")
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

    torch.manual_seed(42)
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
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    has_orientation_target = (
        args_cli.outward_tilt_degrees is not None or args_cli.mirrored_world_yaw_degrees is not None
    )
    scripted_steps = (
        (args_cli.tilt_steps if has_orientation_target else 0)
        + args_cli.approach_steps
        + args_cli.close_steps
        + args_cli.hold_steps
        + args_cli.pull_steps
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
    retention_distance = env_cfg.terminations.lost_grasp.params["maximum_grasp_distance"]
    physics_cfg = env_cfg.sim.physics
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
    for termination_name in vars(env_cfg.terminations):
        setattr(env_cfg.terminations, termination_name, None)
    for reward_name in list(vars(env_cfg.rewards)):
        setattr(env_cfg.rewards, reward_name, None)

    with launch_simulation(env_cfg, args_cli):
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        env = gym_env.unwrapped
        try:
            gym_env.reset(seed=42)
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
            reset_state = _state(env)
            initial_tcp_to_tail_w = reset_state["tail"] - reset_state["tcp"]
            initial_tcp_to_tail_hand = torch.sum(reset_state["hand_axes"] * initial_tcp_to_tail_w.unsqueeze(-2), dim=-1)
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
                    gym_env.step(actions)
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
            left_tcp_target_w = None
            if args_cli.left_tcp_offset_w is not None:
                left_tcp_target_w = reset_state["tcp"][:, 0] + reset_state["tcp"].new_tensor(args_cli.left_tcp_offset_w)
            right_tcp_target_w = None
            if args_cli.right_tcp_offset_w is not None:
                right_tcp_target_w = reset_state["tcp"][:, 1] + reset_state["tcp"].new_tensor(
                    args_cli.right_tcp_offset_w
                )
            fixed_target_quaternions_w = None
            if left_tcp_target_w is not None or right_tcp_target_w is not None:
                fixed_target_quaternions_w = _mirrored_world_yaw_quaternions_w(env, 0.0)
            for step in range(args_cli.approach_steps):
                current = _state(env)
                if left_tcp_target_w is None and right_tcp_target_w is None:
                    error_w = current["tail"] - current["tcp"]
                else:
                    error_w = torch.zeros_like(current["tcp"])
                    if left_tcp_target_w is not None:
                        error_w[:, 0] = left_tcp_target_w - current["tcp"][:, 0]
                    if right_tcp_target_w is not None:
                        error_w[:, 1] = right_tcp_target_w - current["tcp"][:, 1]
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
                gym_env.step(actions)
                current = _state(env)
                approach_trace.append(current)
                if step in {0, 4, 9, 19, 29, 44, args_cli.approach_steps - 1}:
                    approach_samples.append(
                        {
                            "step": step + 1,
                            "distance_mm": (current["distance"][0] * 1000).tolist(),
                        }
                    )
                active_distance = (
                    current["distance"]
                    if right_tcp_target_w is None
                    else torch.linalg.vector_norm(right_tcp_target_w - current["tcp"][:, 1], dim=-1)
                )
                if bool(torch.all(active_distance <= args_cli.approach_tolerance)):
                    break
            after_approach = _state(env)
            after_approach_cable_rms_speed = _cable_rms_linear_speed(env)
            if args_cli.save_settled_state is not None:
                _save_settled_state(env, args_cli.save_settled_state)
            _set_tcp_translation(actions, env, torch.zeros_like(after_approach["tcp"]), arm_action_scale)

            _set_term(actions, env, "left_gripper", -1.0)
            _set_term(actions, env, "right_gripper", -1.0)
            close_samples = []
            for step in range(args_cli.close_steps):
                gym_env.step(actions)
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
                gym_env.step(actions)
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
            raw_speed = args_cli.pull_speed * policy_dt / arm_action_scale
            for arm, robot_name in enumerate(("robot_left", "robot_right")):
                root_quaternion = env.scene[robot_name].data.root_quat_w.torch
                local_direction = math_utils.quat_apply_inverse(root_quaternion, directions_w[arm].expand(1, -1))
                arm_action = torch.zeros((1, 6), device=env.device)
                arm_action[:, :3] = raw_speed * local_direction
                _set_term(actions, env, f"{robot_name.removeprefix('robot_')}_arm", arm_action)

            maximum_distance = pull_start["distance"].clone()
            minimum_follow_projection = torch.full((1, 2), float("inf"), device=env.device)
            first_success_step = None
            samples = []
            for step in range(args_cli.pull_steps):
                gym_env.step(actions)
                current = _state(env)
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
                        }
                    )
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
                "left_tcp_offset_w_m": args_cli.left_tcp_offset_w,
                "right_tcp_offset_w_m": args_cli.right_tcp_offset_w,
                "left_grasp_joint_positions_rad": args_cli.left_grasp_joint_positions,
                "right_grasp_joint_positions_rad": args_cli.right_grasp_joint_positions,
                "close_duration_s": args_cli.close_steps * policy_dt,
                "hold_duration_s": args_cli.hold_steps * policy_dt,
                "pull_duration_s": args_cli.pull_steps * policy_dt,
                "commanded_pull_speed_m_s": args_cli.pull_speed,
                "pull_directions_w": directions_w.tolist(),
                "gripper_damping_n_s_m": args_cli.gripper_damping,
                "gripper_stiffness_n_m": env_cfg.scene.robot_left.actuators["panda_hand"].stiffness,
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
                "num_substeps": args_cli.num_substeps,
                "vbd_iterations": args_cli.vbd_iterations,
                "coupling_mode": args_cli.coupling_mode,
                "grasp_assist": args_cli.grasp_assist,
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
                "after_close_finger_mm": (after_close["finger"][0] * 1000).tolist(),
                "acquired_after_close": acquired_after_close.tolist(),
                "pull_start_distance_mm": (pull_start["distance"][0] * 1000).tolist(),
                "pull_start_finger_mm": (pull_start["finger"][0] * 1000).tolist(),
                "maximum_hold_distance_mm": (maximum_hold_distance[0] * 1000).tolist(),
                "hold_relative_slip_mm": (hold_relative_slip * 1000).tolist(),
                "retained_after_hold": retained_after_hold.tolist(),
                "tcp_pull_mm": (tcp_projection * 1000).tolist(),
                "tail_pull_mm": (tail_projection * 1000).tolist(),
                "follow_ratio": follow_ratio.tolist(),
                "relative_slip_mm": (relative_slip * 1000).tolist(),
                "final_distance_mm": (pull_end["distance"][0] * 1000).tolist(),
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
            indent = None if args_cli.summary_only else 2
            print("SHOELACE_PULL_RESULT=" + json.dumps(result, indent=indent))
        finally:
            gym_env.close()


if __name__ == "__main__":
    main()
