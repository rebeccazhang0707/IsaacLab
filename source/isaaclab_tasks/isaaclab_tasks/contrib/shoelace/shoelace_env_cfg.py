# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the minimal dual-Franka Newton shoelace RL task."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)

import isaaclab.envs.mdp as env_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerEntryCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    ActionsCfg as FrankaSoftActionsCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftSceneCfg

from . import mdp
from . import shoelace_physics as physics

TCP_OFFSET = physics.TCP_OFFSET
ARM_ACTION_SCALE = 0.005
ENV_SPACING = 1.5
GRIPPER_OPEN_POSITION = 0.01
GRIPPER_CLOSED_POSITION = 0.001
GRIPPER_STIFFNESS = 8000.0
CONTACT_OBSERVATION_HISTORY_LENGTH = 3
TAIL_SUCCESS_X_SEPARATION = 0.18

NEWTON_NUM_SUBSTEPS = 10
NEWTON_COLLISION_DECIMATION = 2
VBD_ITERATIONS = 12
ADMM_ITERATIONS = 2
ADMM_RHO = 500.0
ADMM_BAUMGARTE = 0.5
ADMM_CONTACT_MATCHING = "latest"

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

CABLE_CFGS = (SceneEntityCfg("shoelace_left"), SceneEntityCfg("shoelace_right"))
ROBOT_CFGS = (
    SceneEntityCfg("robot_left", joint_names=["panda_finger_joint1"], body_names=["panda_hand"]),
    SceneEntityCfg("robot_right", joint_names=["panda_finger_joint1"], body_names=["panda_hand"]),
)


def _franka_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    arm_joint_positions: dict[str, float],
) -> ArticulationCfg:
    """Build one fixed-base Franka in the nominal open-pregrasp pose."""
    robot = FrankaSoftSceneCfg().default.robot.replace(prim_path=prim_path)
    robot.init_state.pos = position
    robot.init_state.rot = rotation
    robot.init_state.joint_pos.update(arm_joint_positions)
    robot.init_state.joint_pos["panda_finger_joint.*"] = GRIPPER_OPEN_POSITION
    robot.actuators["panda_hand"].actuator_velocity_limit = 0.04
    robot.actuators["panda_hand"].stiffness = GRIPPER_STIFFNESS
    return robot


def _arm_action(asset_name: str) -> env_mdp.DifferentialInverseKinematicsActionCfg:
    """Build one six-dimensional relative TCP action."""
    action = FrankaSoftActionsCfg().ik.arm_action
    action.asset_name = asset_name
    action.controller.use_relative_mode = True
    action.controller.ik_params = {"lambda_val": 0.01}
    action.scale = (ARM_ACTION_SCALE,) * 3 + (0.01,) * 3
    action.body_offset.pos = TCP_OFFSET
    return action


def _cable_cfg(prim_path: str) -> CableObjectCfg:
    """Build one cable whose exact geometry is filled from the asset at runtime."""
    return CableObjectCfg(prim_path=prim_path, spawn=physics.cable_spawn_cfg())


@configclass
class ShoelaceSceneCfg(InteractiveSceneCfg):
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
    shoe = physics.shoe_asset_cfg(visible=False)
    shoe_collider = physics.shoe_collider_asset_cfg()
    shoe_visual = physics.shoe_visual_asset_cfg("/World/ShoeMaterials/shoes")
    tongue_upper = physics.tongue_upper_asset_cfg()
    shoelace_left = _cable_cfg("{ENV_REGEX_NS}/ShoelaceLeft")
    shoelace_pinned_visual = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Shoe/ShoelacePinned")
    shoelace_right = _cable_cfg("{ENV_REGEX_NS}/ShoelaceRight")
    ground = physics.ground_asset_cfg(10.0)
    light = physics.light_asset_cfg()


@configclass
class ActionsCfg:
    """Relative Cartesian arm and binary gripper actions."""

    left_arm = _arm_action("robot_left")
    left_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": GRIPPER_CLOSED_POSITION},
    )
    right_arm = _arm_action("robot_right")
    right_gripper = env_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": GRIPPER_CLOSED_POSITION},
    )


@configclass
class ObservationsCfg:
    """Basic dual-arm proprioception and free-tail state observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
        )
        right_joint_pos = ObsTerm(
            func=env_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
        )
        left_joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        right_joint_vel = ObsTerm(
            func=env_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        gripper_close_error = ObsTerm(
            func=mdp.gripper_close_error,
            params={"robot_cfgs": ROBOT_CFGS},
            clip=(0.0, GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION),
            scale=1.0 / (GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION),
        )
        tails_to_tcp = ObsTerm(
            func=mdp.tails_to_tcp,
            params={"cable_cfgs": CABLE_CFGS, "robot_cfgs": ROBOT_CFGS},
            scale=10.0,
        )
        finger_tail_signed_distance = ObsTerm(
            func=mdp.finger_tail_signed_distance,
            clip=(-physics.CONTACT_DISTANCE_CAP, physics.CONTACT_DISTANCE_CAP),
            scale=1.0 / physics.CONTACT_DISTANCE_CAP,
            history_length=CONTACT_OBSERVATION_HISTORY_LENGTH,
        )
        tail_tcp_relative_speed = ObsTerm(
            func=mdp.tail_tcp_relative_speed,
            params={"cable_cfgs": CABLE_CFGS, "robot_cfgs": ROBOT_CFGS},
            scale=10.0,
        )
        last_action = ObsTerm(func=env_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Restore defaults, then perturb arm joints and translate the shoe with its laces."""

    reset_scene = EventTerm(
        func=env_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_left_arm = EventTerm(
        func=mdp.reset_arm_joints,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint[1-7]"]),
        },
    )
    reset_right_arm = EventTerm(
        func=mdp.reset_arm_joints,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint[1-7]"]),
        },
    )
    reset_shoe = EventTerm(
        func=mdp.reset_shoe_position,
        mode="reset",
        params={"position_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}},
    )


@configclass
class RewardsCfg:
    """Acquisition-and-pull progress with small arm motion and action-change penalties."""

    dense_task = RewTerm(
        func=mdp.dense_task_reward,
        weight=10.0,
        params={
            "reach_std": 0.05,
            "contact_std": 5.0e-4,
            "relative_speed_std": 0.08,
            "grasp_filter_time_constant": 0.10,
            "open_position": GRIPPER_OPEN_POSITION,
            "closed_position": GRIPPER_CLOSED_POSITION,
            "success_x_separation": TAIL_SUCCESS_X_SEPARATION,
            "acquisition_weight": 0.3,
            "approach_fraction": 0.3,
            "cable_cfgs": CABLE_CFGS,
            "robot_cfgs": ROBOT_CFGS,
        },
    )
    arm_action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-0.001)
    arm_action_magnitude = RewTerm(func=mdp.arm_action_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """Two-tail X-separation success and episode timeout."""

    success = DoneTerm(
        func=mdp.tail_x_separation_success,
        params={"threshold": TAIL_SUCCESS_X_SEPARATION, "cable_cfgs": CABLE_CFGS},
    )
    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)


@configclass
class ShoelaceVBDSolverCfg(VBDSolverCfg):
    """VBD settings validated by the standalone Franka demo."""

    rigid_avbd_alpha: float = 0.0
    rigid_contact_hard: bool = True
    rigid_body_contact_buffer_size: int = physics.VBD_CONTACT_BUFFER


@configclass
class ShoelaceEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal trainable Newton environment for dual-Franka shoelace control."""

    seed: int | None = 42
    decimation: int = 4
    episode_length_s: float = 10.0
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
                        solver_cfg=ShoelaceVBDSolverCfg(iterations=VBD_ITERATIONS),
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
                rigid_contact_max=physics.CONTACTS_PER_ENV * 4,
                max_triangle_pairs=physics.MIN_TRIANGLE_PAIRS,
            ),
            collision_decimation=NEWTON_COLLISION_DECIMATION,
            default_shape_cfg=NewtonShapeCfg(
                gap=physics.CONTACT_GAP,
                ke=2.5e4,
                kd=100.0,
                mu=10.0,
            ),
            num_substeps=NEWTON_NUM_SUBSTEPS,
            use_cuda_graph=True,
        ),
    )
    scene: ShoelaceSceneCfg = ShoelaceSceneCfg(num_envs=4, env_spacing=ENV_SPACING, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    ui_window_class_type = None

    finger_mu: float = 40.0
    lace_mu: float = physics.LACE_MU
    shoe_mu: float = physics.SHOE_MU

    cable_inertia_regularization: float = physics.CABLE_INERTIA_REGULARIZATION
    """Isotropic inertia added to each dynamic cable segment [kg*m^2], without changing its mass."""
