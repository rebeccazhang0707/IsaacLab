# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a standalone multi-environment open-loop dual-Franka shoelace pull.

This demo directly builds the shoe, shoelace, and two Franka robots.
A batched state machine approaches both tails, closes the grippers, holds the
contacts, pulls in opposing directions, and releases the shoelace.

.. code-block:: bash

    uv run --frozen python scripts/demos/shoelace/franka_open_loop_untie_shoelace.py \\
        --device cuda:0 --visualizer newton_gl --num_envs 4 \\
        --approach_steps 80 --approach_speed 0.08 --approach_tolerance 0.007 \\
        --close_steps 2 --pull_steps 100 --pull_speed 0.1

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

import _newton_shoelace as shoelace  # noqa: E402
from isaaclab_newton.physics import (  # noqa: E402
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg  # noqa: E402

import isaaclab.envs.mdp as mdp  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.physics import PhysicsEvent  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerEntryCfg  # noqa: E402

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (  # noqa: E402
    ActionsCfg as FrankaSoftActionsCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftSceneCfg  # noqa: E402

SETTLED_STATE_ASSET = shoelace.ASSET_DIR / "settled_tail_clear_segment_poses.npz"

# Asset topology and scripted motion.
TAIL_REGIONS = ((357, 360), (0, 3))
TCP_OFFSET = (0.0, 0.0, 0.1034)
PULL_DIRECTIONS = ((-1.0, -0.35, 0.15), (1.0, -0.35, 0.15))
ARM_ACTION_SCALE = 0.005
ENV_SPACING = 1.5
SEED = 42
HOLD_STEPS = 30
RELEASE_STEPS = 10
GRIPPER_OPEN_POSITION = 0.01
GRIPPER_CLOSED_POSITION = 0.001
GRIPPER_STIFFNESS = 8000.0

# Shoelace-specific Newton solver parameters.
NEWTON_NUM_SUBSTEPS = 10
NEWTON_COLLISION_DECIMATION = 2
VBD_ITERATIONS = 20
ADMM_ITERATIONS = 5
ADMM_RHO = 400.0
ADMM_BAUMGARTE = 0.5
ADMM_CONTACT_MATCHING = "latest"

# Franka-specific distal-tail material.
TAIL_BEND_STIFFNESS = 100.0
TAIL_BEND_DAMPING = 2.0
TAIL_STIFF_CORE_LENGTH = 0.050
TAIL_STIFF_TRANSITION_LENGTH = 0.012

LEFT_ROBOT_POSITION = (-0.525248, 0.023338, -0.089901)
RIGHT_ROBOT_POSITION = (0.507963, -0.001890, -0.088518)
LEFT_ARM_JOINT_POSITIONS = {
    "panda_joint1": 0.267436,
    "panda_joint2": -0.276844,
    "panda_joint3": -0.509809,
    "panda_joint4": -2.681384,
    "panda_joint5": 0.871062,
    "panda_joint6": 2.543250,
    "panda_joint7": -0.079776,
}
RIGHT_ARM_JOINT_POSITIONS = {
    "panda_joint1": -0.379171,
    "panda_joint2": -0.306599,
    "panda_joint3": 0.461994,
    "panda_joint4": -2.726323,
    "panda_joint5": -0.930591,
    "panda_joint6": 2.645580,
    "panda_joint7": 1.577059,
}


def _load_settled_segment_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load validated gravity-settled cable segment poses [m, quaternion x-y-z-w]."""
    names = ("shoelace_left", "shoelace_right")
    with np.load(path, allow_pickle=False) as state:
        if tuple(state.files) != names:
            raise ValueError(f"Expected settled shoelace arrays {names}, got {tuple(state.files)}")
        poses = tuple(np.asarray(state[name], dtype=np.float32) for name in names)
    expected_shapes = (
        (shoelace.PINNED_FIRST + 1, 7),
        (shoelace.SHOELACE_SEGMENT_COUNT - shoelace.PINNED_LAST, 7),
    )
    for name, pose, expected_shape in zip(names, poses, expected_shapes, strict=True):
        if pose.shape != expected_shape or not np.isfinite(pose).all():
            raise ValueError(f"Invalid settled {name} poses: expected finite {expected_shape}, got {pose.shape}")
        if not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1.0, atol=2.0e-5):
            raise ValueError(f"Settled {name} poses contain non-unit quaternions")
    return poses


def _tail_joint_blend_weights(joint_count: int, segment_length: float, free_end_at_start: bool) -> np.ndarray:
    """Return smooth weights for the gravity-resistant distal tail material."""
    distances = segment_length * np.arange(1, joint_count + 1, dtype=np.float64)
    if TAIL_STIFF_TRANSITION_LENGTH > 0.0:
        weights = np.clip(
            (TAIL_STIFF_CORE_LENGTH + TAIL_STIFF_TRANSITION_LENGTH - distances) / TAIL_STIFF_TRANSITION_LENGTH,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)
    else:
        weights = (distances <= TAIL_STIFF_CORE_LENGTH).astype(np.float64)
    return weights if free_end_at_start else weights[::-1].copy()


def _franka_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    arm_joint_positions: dict[str, float],
) -> ArticulationCfg:
    """Build one fixed-base Franka in the deterministic open-pregrasp pose."""
    robot = FrankaSoftSceneCfg().default.robot.replace(prim_path=prim_path)
    robot.init_state.pos = position
    robot.init_state.rot = rotation
    robot.init_state.joint_pos.update(arm_joint_positions)
    robot.init_state.joint_pos["panda_finger_joint.*"] = GRIPPER_OPEN_POSITION
    robot.actuators["panda_hand"].actuator_velocity_limit = 0.04
    robot.actuators["panda_hand"].stiffness = GRIPPER_STIFFNESS
    return robot


def _arm_action(asset_name: str) -> mdp.DifferentialInverseKinematicsActionCfg:
    """Build one six-dimensional relative TCP action."""
    action = FrankaSoftActionsCfg().ik.arm_action
    action.asset_name = asset_name
    action.controller.use_relative_mode = True
    action.controller.ik_params = {"lambda_val": 0.01}
    action.scale = (ARM_ACTION_SCALE,) * 3 + (0.01,) * 3
    action.body_offset.pos = TCP_OFFSET
    return action


def _cable_cfg(prim_path: str) -> CableObjectCfg:
    """Build one cable whose exact geometry is filled from the shoelace asset at runtime."""
    return CableObjectCfg(prim_path=prim_path, spawn=shoelace.cable_spawn_cfg())


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Replicated dual-Franka shoe and shoelace scene."""

    robot_left = _franka_cfg(
        "{ENV_REGEX_NS}/RobotLeft",
        LEFT_ROBOT_POSITION,
        (0.0, 0.0, 0.0, 1.0),
        LEFT_ARM_JOINT_POSITIONS,
    )
    robot_right = _franka_cfg(
        "{ENV_REGEX_NS}/RobotRight",
        RIGHT_ROBOT_POSITION,
        (0.0, 0.0, 1.0, 0.0),
        RIGHT_ARM_JOINT_POSITIONS,
    )
    shoe = shoelace.shoe_asset_cfg(visible=False)
    shoe_collider = shoelace.shoe_collider_asset_cfg()
    shoe_visual = shoelace.shoe_visual_asset_cfg("/World/ShoeMaterials/shoes")
    tongue_upper = shoelace.tongue_upper_asset_cfg()
    shoelace_left = _cable_cfg("{ENV_REGEX_NS}/ShoelaceLeft")
    shoelace_pinned_visual = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ShoelacePinned")
    shoelace_right = _cable_cfg("{ENV_REGEX_NS}/ShoelaceRight")
    ground = shoelace.ground_asset_cfg(10.0)
    light = shoelace.light_asset_cfg()


@configclass
class ActionsCfg:
    """Relative Cartesian arm and binary gripper actions."""

    left_arm = _arm_action("robot_left")
    left_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": GRIPPER_CLOSED_POSITION},
    )
    right_arm = _arm_action("robot_right")
    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": GRIPPER_CLOSED_POSITION},
    )


@configclass
class EmptyObservationsCfg:
    """No observations are needed for the scripted controller."""

    pass


@configclass
class EventsCfg:
    """Restore every asset to its deterministic authored default state."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class ShoelaceVBDSolverCfg(VBDSolverCfg):
    """VBD settings retained from the validated shoelace configuration."""

    rigid_avbd_alpha: float = 0.0
    rigid_contact_hard: bool = True
    rigid_body_contact_buffer_size: int = shoelace.VBD_CONTACT_BUFFER


@configclass
class OpenLoopEnvCfg(ManagerBasedEnvCfg):
    """Standalone non-RL environment configuration for the scripted pull."""

    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=CouplerAdmmCfg(
                entries=[
                    CouplerEntryCfg(
                        name="robots",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=40,
                            integrator="implicitfast",
                            njmax=2048,
                            nconmax=256,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/(Robot(Left|Right)|Shoe)"],
                    ),
                    CouplerEntryCfg(
                        name="shoelace",
                        solver_cfg=ShoelaceVBDSolverCfg(
                            iterations=VBD_ITERATIONS,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/Shoelace(Left|Right)"],
                        include_static_shapes=True,
                    ),
                ],
                contact_pairs=[("robots", "shoelace")],
                iterations=ADMM_ITERATIONS,
                rho=ADMM_RHO,
                gamma=0.0,
                baumgarte=ADMM_BAUMGARTE,
                rigid_contact_matching=ADMM_CONTACT_MATCHING,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=shoelace.CONTACTS_PER_ENV * 4,
                max_triangle_pairs=shoelace.MIN_TRIANGLE_PAIRS,
            ),
            collision_decimation=NEWTON_COLLISION_DECIMATION,
            default_shape_cfg=NewtonShapeCfg(
                gap=shoelace.CONTACT_GAP,
                ke=2.5e4,
                kd=100.0,
                mu=10.0,
            ),
            num_substeps=NEWTON_NUM_SUBSTEPS,
            use_cuda_graph=True,
        ),
    )
    scene: SceneCfg = SceneCfg(num_envs=4, env_spacing=ENV_SPACING, replicate_physics=True)
    observations: EmptyObservationsCfg = EmptyObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    ui_window_class_type = None

    finger_mu = 40.0
    lace_mu = shoelace.LACE_MU
    shoe_mu = shoelace.SHOE_MU


class OpenLoopShoelacePhysics:
    """Configure the independent Newton cable model before solver construction."""

    def __init__(
        self,
        centerline: np.ndarray,
        cable_radius: float,
        num_envs: int,
        joint_stiffness_scale: float,
        finger_mu: float,
        lace_mu: float,
        shoe_mu: float,
    ) -> None:
        self.centerline = centerline
        self.cable_radius = cable_radius
        self.num_envs = num_envs
        self.joint_stiffness_scale = joint_stiffness_scale
        self.finger_mu = finger_mu
        self.lace_mu = lace_mu
        self.shoe_mu = shoe_mu
        self.cable_joints: list[list[int]] = []

    def register(self) -> None:
        """Register standalone Newton model lifecycle callbacks."""
        NewtonManager.register_callback(
            self._configure_builder,
            PhysicsEvent.MODEL_INIT,
            name="shoelace_open_loop_builder",
        )

    def _configure_builder(self, _: object) -> None:
        """Finalize cable material, topology, anchors, and collision filters."""
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton builder is unavailable during MODEL_INIT")
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp.values = {} if gravcomp.values is None else gravcomp.values
        for body, label in enumerate(builder.body_label):
            if "/RobotLeft/" in label or "/RobotRight/" in label:
                gravcomp.values[body] = 1.0
        build = shoelace.configure_shoelace_builder(
            builder,
            self.centerline,
            self.cable_radius,
            self.num_envs,
            lace_mu=self.lace_mu,
            shoe_mu=self.shoe_mu,
        )
        self.cable_joints = [left + right for left, right in build.joint_chains]
        for shape, label in enumerate(builder.shape_label):
            if "panda_leftfinger" in label or "panda_rightfinger" in label:
                builder.shape_material_mu[shape] = self.finger_mu

        for world_joints in build.joint_chains:
            for joints, free_end_at_start in zip(world_joints, (True, False), strict=True):
                weights = _tail_joint_blend_weights(len(joints), build.mean_segment_length, free_end_at_start)
                stiffness: list[float] = []
                damping: list[float] = []
                for weight in weights:
                    bend_stiffness = shoelace.BEND_STIFFNESS + weight * (TAIL_BEND_STIFFNESS - shoelace.BEND_STIFFNESS)
                    bend_damping = shoelace.BEND_DAMPING + weight * (TAIL_BEND_DAMPING - shoelace.BEND_DAMPING)
                    stiffness.extend((shoelace.STRETCH_STIFFNESS,) * 2 + (bend_stiffness,) * 2)
                    damping.extend((shoelace.STRETCH_DAMPING,) * 2 + (bend_damping,) * 2)
                dof_slice = shoelace.cable_joint_dof_slice(builder, joints)
                builder.joint_target_ke[dof_slice] = tuple(value * self.joint_stiffness_scale for value in stiffness)
                builder.joint_target_kd[dof_slice] = tuple(value * self.joint_stiffness_scale for value in damping)
        shoelace.configure_dahl(builder, self.cable_joints)


class OpenLoopShoelaceEnv(ManagerBasedEnv):
    """Construct the shared shoelace physics without any RL task managers."""

    cfg: OpenLoopEnvCfg

    def __init__(self, cfg: OpenLoopEnvCfg) -> None:
        self._centerline, self._cable_radius, authored_mean_segment_length = shoelace.load_shoelace()
        self._settled_segment_poses = _load_settled_segment_poses(SETTLED_STATE_ASSET)
        mean_segment_length = float(np.linalg.norm(np.diff(self._centerline, axis=0), axis=1).mean())
        self._configure_runtime_cfg(cfg, authored_mean_segment_length)
        self._physics = OpenLoopShoelacePhysics(
            self._centerline,
            self._cable_radius,
            cfg.scene.num_envs,
            authored_mean_segment_length / mean_segment_length,
            finger_mu=cfg.finger_mu,
            lace_mu=cfg.lace_mu,
            shoe_mu=cfg.shoe_mu,
        )
        super().__init__(cfg)
        self._install_settled_default_state()

    def _configure_runtime_cfg(self, cfg: OpenLoopEnvCfg, authored_mean_segment_length: float) -> None:
        """Fill asset-derived geometry and environment-count-dependent capacities."""
        cfg.scene.shoelace_pinned_visual.spawn = shoelace.pinned_shoelace_spawner(self._centerline, self._cable_radius)

        cable_normals = shoelace.parallel_transport_normals(self._centerline)
        cable_centerlines = (
            self._centerline[: shoelace.PINNED_FIRST + 2],
            self._centerline[shoelace.PINNED_LAST :],
        )
        cable_normal_chains = (
            cable_normals[: shoelace.PINNED_FIRST + 2],
            cable_normals[shoelace.PINNED_LAST :],
        )
        for cable_cfg, centerline, normals in zip(
            (cfg.scene.shoelace_left.spawn, cfg.scene.shoelace_right.spawn),
            cable_centerlines,
            cable_normal_chains,
            strict=True,
        ):
            shoelace.configure_cable_spawn(
                cable_cfg,
                centerline,
                normals,
                self._cable_radius,
                authored_mean_segment_length,
            )

        size = shoelace.ground_size(cfg.scene.num_envs, cfg.scene.env_spacing)
        cfg.scene.ground.spawn.size = (size, size)
        physics_cfg = cfg.sim.physics
        physics_cfg.collision_cfg.rigid_contact_max = shoelace.CONTACTS_PER_ENV * cfg.scene.num_envs
        physics_cfg.collision_cfg.max_triangle_pairs = max(
            shoelace.MIN_TRIANGLE_PAIRS,
            shoelace.TRIANGLE_PAIRS_PER_ENV * cfg.scene.num_envs,
        )

    def _install_settled_default_state(self) -> None:
        """Install the offline gravity-settled cable poses as reset defaults."""
        for cable_name, local_pose in zip(
            ("shoelace_left", "shoelace_right"),
            self._settled_segment_poses,
            strict=True,
        ):
            cable = self.scene[cable_name]
            default_pose = cable.data.default_segment_pose_w.torch
            segment_pose = default_pose.new_tensor(local_pose).unsqueeze(0).expand_as(default_pose).clone()
            segment_pose[..., :3] += self.scene.env_origins.unsqueeze(1)
            segment_velocity = torch.zeros_like(cable.data.default_segment_velocity_w.torch)
            default_pose.copy_(segment_pose)
            cable.data.default_segment_velocity_w.torch.copy_(segment_velocity)
            cable.write_segment_pose_to_sim_index(segment_pose=segment_pose)
            cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity)

    def _init_sim(self) -> None:
        """Register the shoe material and Newton callbacks before model creation."""
        shoelace.install_shoe_material_reference()
        self._physics.register()
        super()._init_sim()


def _tcp_positions(env: ManagerBasedEnv) -> torch.Tensor:
    """Return both TCP positions in each local environment frame [m]."""
    positions = []
    for robot_name in ("robot_left", "robot_right"):
        robot = env.scene[robot_name]
        body_ids, _ = robot.find_bodies("panda_hand")
        hand_position = robot.data.body_pos_w.torch[:, body_ids[0]] - env.scene.env_origins
        hand_quaternion = robot.data.body_quat_w.torch[:, body_ids[0]]
        offset = hand_position.new_tensor(TCP_OFFSET).expand_as(hand_position)
        positions.append(hand_position + math_utils.quat_apply(hand_quaternion, offset))
    return torch.stack(positions, dim=1)


def _tail_positions(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the three-segment tail centers targeted by the two grippers [m]."""
    left_chain = env.scene["shoelace_left"].data.segment_pose_w.torch[..., :3] - env.scene.env_origins.unsqueeze(1)
    right_chain = env.scene["shoelace_right"].data.segment_pose_w.torch[..., :3] - env.scene.env_origins.unsqueeze(1)
    right_tail_lower = TAIL_REGIONS[0][0] - shoelace.PINNED_LAST
    right_tail_upper = TAIL_REGIONS[0][1] - shoelace.PINNED_LAST
    left_tail_lower, left_tail_upper = TAIL_REGIONS[1]
    return torch.stack(
        (
            right_chain[:, right_tail_lower:right_tail_upper].mean(dim=1),
            left_chain[:, left_tail_lower:left_tail_upper].mean(dim=1),
        ),
        dim=1,
    )


def _state(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    tcp = _tcp_positions(env).clone()
    tail = _tail_positions(env).clone()
    return {
        "tcp": tcp,
        "tail": tail,
        "distance": torch.linalg.vector_norm(tail - tcp, dim=-1),
    }


def _set_action_term(
    actions: torch.Tensor,
    env: ManagerBasedEnv,
    term_name: str,
    values: torch.Tensor | float,
) -> None:
    offset = 0
    for active_name, dimension in zip(
        env.action_manager.active_terms,
        env.action_manager.action_term_dim,
        strict=True,
    ):
        if active_name == term_name:
            actions[:, offset : offset + dimension] = values
            return
        offset += dimension
    raise KeyError(f"Missing action term {term_name!r}; active terms: {env.action_manager.active_terms}")


def _set_tcp_translation(actions: torch.Tensor, env: ManagerBasedEnv, displacement_w: torch.Tensor) -> None:
    """Set batched world-frame TCP translations for both relative-IK terms [m]."""
    for arm, robot_name in enumerate(("robot_left", "robot_right")):
        root_quaternion = env.scene[robot_name].data.root_quat_w.torch
        displacement_b = math_utils.quat_apply_inverse(root_quaternion, displacement_w[:, arm])
        arm_action = torch.zeros((actions.shape[0], 6), device=env.device)
        arm_action[:, :3] = displacement_b / ARM_ACTION_SCALE
        _set_action_term(actions, env, f"{robot_name.removeprefix('robot_')}_arm", arm_action)


def _make_cfg(args: argparse.Namespace) -> OpenLoopEnvCfg:
    cfg = OpenLoopEnvCfg()
    cfg.seed = SEED
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = ENV_SPACING
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            eye=(0.0, -3.0, 2.5),
            lookat=(0.0, 0.0, 0.2),
        )
    ]
    return cfg


def run_state_machine(env: ManagerBasedEnv, args: argparse.Namespace) -> None:
    """Run the fixed-sequence open-loop state machine in all environments."""
    # State 0 - reset: restore the deterministic scene and command both grippers open.
    env.reset(seed=SEED)
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    _set_action_term(actions, env, "left_gripper", 1.0)
    _set_action_term(actions, env, "right_gripper", 1.0)

    # State 1 - approach: track the moving tail centers until every environment is within tolerance.
    maximum_approach_step = args.approach_speed * env.step_dt
    for _ in range(args.approach_steps):
        current = _state(env)
        error_w = current["tail"] - current["tcp"]
        distance = torch.linalg.vector_norm(error_w, dim=-1)
        step_distance = torch.minimum(distance, distance.new_full(distance.shape, maximum_approach_step))
        step_distance = torch.where(distance > args.approach_tolerance, step_distance, 0.0)
        displacement_w = error_w * (step_distance / distance.clamp_min(1.0e-8)).unsqueeze(-1)
        _set_tcp_translation(actions, env, displacement_w)
        env.step(actions)
        if bool(torch.all(_state(env)["distance"] <= args.approach_tolerance)):
            break
    after_approach = _state(env)
    _set_tcp_translation(actions, env, torch.zeros_like(after_approach["tcp"]))

    # State 2 - close: hold both TCPs still while the fingers close around the tails.
    _set_action_term(actions, env, "left_gripper", -1.0)
    _set_action_term(actions, env, "right_gripper", -1.0)
    for _ in range(args.close_steps):
        env.step(actions)

    # State 3 - hold: allow the closed finger-tail contacts to settle before pulling.
    for _ in range(HOLD_STEPS):
        env.step(actions)

    # State 4 - pull: move both TCPs along fixed opposing world-frame directions.
    directions_w = torch.tensor(
        PULL_DIRECTIONS,
        device=env.device,
        dtype=torch.float32,
    )
    directions_w /= torch.linalg.vector_norm(directions_w, dim=-1, keepdim=True)
    displacement_w = directions_w.unsqueeze(0).expand(env.num_envs, -1, -1) * (args.pull_speed * env.step_dt)
    _set_tcp_translation(actions, env, displacement_w)
    for _ in range(args.pull_steps):
        env.step(actions)

    # State 5 - release: stop both TCPs, open the grippers, and simulate the released cable.
    _set_tcp_translation(actions, env, torch.zeros_like(displacement_w))
    _set_action_term(actions, env, "left_gripper", 1.0)
    _set_action_term(actions, env, "right_gripper", 1.0)
    for _ in range(RELEASE_STEPS):
        env.step(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--approach_steps", type=int, default=80)
    parser.add_argument("--approach_speed", type=float, default=0.08, help="Maximum TCP approach speed [m/s].")
    parser.add_argument("--approach_tolerance", type=float, default=0.007, help="TCP-to-tail tolerance [m].")
    parser.add_argument("--close_steps", type=int, default=2)
    parser.add_argument("--pull_steps", type=int, default=100)
    parser.add_argument("--pull_speed", type=float, default=0.1, help="Commanded TCP pull speed [m/s].")
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = _make_cfg(args)

    with launch_simulation(cfg, args):
        env = OpenLoopShoelaceEnv(cfg)
        try:
            with torch.inference_mode():
                run_state_machine(env, args)
        finally:
            env.close()


if __name__ == "__main__":
    main()
