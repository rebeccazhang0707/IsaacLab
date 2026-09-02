# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

import isaaclab_tasks.contrib.shoelace.mdp as mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

from .mdp.constants import TCP_OFFSET

_THROAT_RADIUS = 0.025
_MAXIMUM_THROAT_SEGMENTS = 52
_TAIL_SUCCESS_DISTANCE = 0.09
_TAIL_SUCCESS_SEPARATION = 0.18
_MAXIMUM_SUCCESS_GRASP_DISTANCE = 0.03
_GRASP_ACQUISITION_DISTANCE = 0.018
_MAXIMUM_GRASP_DISTANCE = 0.035
_GRIPPER_OPEN_POSITION = 0.04
_GRIPPER_CLOSED_POSITION = 0.0035
_GRIPPER_CLOSED_THRESHOLD = 0.02
_TARGET_PULL_SPEED = 0.04
_MINIMUM_LACE_HEIGHT = -0.003
_MAXIMUM_LACE_SPREAD = 0.6
_CURRICULUM_LEVEL_COUNT = 11
_GRIPPER_OPEN_PHASE_FRACTION = 0.4
_APPROACH_PHASE_EXPONENT = 2.0

_LEFT_ROBOT_POSITION = (-0.525248, 0.023338, -0.089901)
_RIGHT_ROBOT_POSITION = (0.507963, -0.001890, -0.088518)
_LEFT_FRANKA_ARM_JOINT_POSITIONS = {
    "panda_joint1": 0.267436,
    "panda_joint2": -0.276844,
    "panda_joint3": -0.509809,
    "panda_joint4": -2.681384,
    "panda_joint5": 0.871062,
    "panda_joint6": 2.543250,
    "panda_joint7": -0.079776,
}
_RIGHT_FRANKA_ARM_JOINT_POSITIONS = {
    "panda_joint1": -0.379171,
    "panda_joint2": -0.306599,
    "panda_joint3": 0.461994,
    "panda_joint4": -2.726323,
    "panda_joint5": -0.930591,
    "panda_joint6": 2.645580,
    "panda_joint7": 1.577059,
}
_LEFT_FRANKA_GRASP_JOINT_POSITIONS = (0.306502, -0.108321, -0.457832, -2.629056, 1.130974, 2.579587, -0.254864)
_RIGHT_FRANKA_GRASP_JOINT_POSITIONS = (
    -0.405931,
    -0.120616,
    0.399613,
    -2.651660,
    -1.227738,
    2.667932,
    1.787070,
)


def _rigid_material(friction: float, damping: float) -> list:
    """Build composed standard-friction and Newton-contact material fragments."""
    return [
        UsdPhysicsRigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=damping),
    ]


def _franka_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    arm_joint_positions: dict[str, float],
) -> ArticulationCfg:
    """Build one fixed-base Franka initialized above a free shoelace tail."""
    robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path=prim_path)
    robot.init_state.pos = position
    robot.init_state.rot = rotation
    robot.init_state.joint_pos.update(arm_joint_positions)
    robot.init_state.joint_pos["panda_finger_joint.*"] = _GRIPPER_OPEN_POSITION
    robot.spawn.rigid_props.disable_gravity = True
    robot.actuators = {
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
            joint_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
            stiffness={
                "panda_joint[1-4]": 600.0,
                "panda_joint5": 250.0,
                "panda_joint6": 150.0,
                "panda_joint7": 50.0,
            },
            damping={
                "panda_joint[1-4]": 50.0,
                "panda_joint5": 30.0,
                "panda_joint6": 25.0,
                "panda_joint7": 15.0,
            },
            armature={
                "panda_joint[1-2]": 0.6057,
                "panda_joint[3-4]": 0.4625,
                "panda_joint[5-7]": 0.2055,
            },
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            joint_effort_limit=500.0,
            actuator_velocity_limit=0.04,
            joint_velocity_limit=2.0,
            stiffness=1000.0,
            damping=100.0,
            armature=0.1,
        ),
        "panda_finger2_passive": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint2"],
            joint_effort_limit=1.0,
            actuator_velocity_limit=0.2,
            joint_velocity_limit=2.0,
            stiffness=0.0,
            damping=0.0,
            armature=0.1,
        ),
    }
    return robot


def _arm_action(asset_name: str) -> DifferentialInverseKinematicsActionCfg:
    """Build one six-dimensional relative TCP action."""
    return DifferentialInverseKinematicsActionCfg(
        asset_name=asset_name,
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=(0.005, 0.005, 0.005, 0.01, 0.01, 0.01),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
    )


def _shoelace_cable_cfg(prim_path: str) -> CableObjectCfg:
    """Build one runtime-configured dynamic shoelace chain."""
    return CableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.01)],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0),
                roughness=0.8,
            ),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.003,
                density=1150.0,
                stretch_stiffness=1.0e9,
                bend_stiffness=1.0e8,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
    )


@configclass
class ShoelaceSceneCfg(InteractiveSceneCfg):
    """Replicated dual-Franka shoe and shoelace scene."""

    robot_left = _franka_cfg(
        "{ENV_REGEX_NS}/RobotLeft",
        _LEFT_ROBOT_POSITION,
        (0.0, 0.0, 0.0, 1.0),
        _LEFT_FRANKA_ARM_JOINT_POSITIONS,
    )
    robot_right = _franka_cfg(
        "{ENV_REGEX_NS}/RobotRight",
        _RIGHT_ROBOT_POSITION,
        (0.0, 0.0, 1.0, 0.0),
        _RIGHT_FRANKA_ARM_JOINT_POSITIONS,
    )
    shoe = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="__runtime_shoe_collider__",
            visible=False,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)],
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
    )
    shoe_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path="__runtime_shoe_visual__",
            visual_material_bindings={"Model": "/World/ShoeMaterials/shoes"},
        ),
    )
    tongue_upper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shoe/TongueUpper",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.055, 0.006),
            visible=False,
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            physics_material=_rigid_material(0.2, 0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.008, 0.02, 0.10),
            rot=(
                math.cos(0.5 * math.radians(5.0)) * math.sin(0.5 * math.radians(28.0)),
                math.sin(0.5 * math.radians(5.0)) * math.cos(0.5 * math.radians(28.0)),
                -math.sin(0.5 * math.radians(5.0)) * math.sin(0.5 * math.radians(28.0)),
                math.cos(0.5 * math.radians(5.0)) * math.cos(0.5 * math.radians(28.0)),
            ),
        ),
    )
    shoelace_left = _shoelace_cable_cfg("{ENV_REGEX_NS}/ShoelaceLeft")
    shoelace_pinned_visual = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ShoelacePinned")
    shoelace_right = _shoelace_cable_cfg("{ENV_REGEX_NS}/ShoelaceRight")
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.08, 0.08, 0.08),
            size=(8.0, 8.0),
            physics_material=_rigid_material(0.8, 30.0),
        ),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 1.0)),
    )


@configclass
class ActionsCfg:
    """Relative Cartesian arm and binary gripper actions for both Frankas."""

    left_arm = _arm_action("robot_left")
    left_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": _GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": _GRIPPER_CLOSED_POSITION},
    )
    right_arm = _arm_action("robot_right")
    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": _GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": _GRIPPER_CLOSED_POSITION},
    )


_SHOELACE_ASSET_CFGS = (SceneEntityCfg("shoelace_left"), SceneEntityCfg("shoelace_right"))

_ROBOT_TERM_PARAMS = {
    "asset_cfgs": _SHOELACE_ASSET_CFGS,
    "left_robot_cfg": SceneEntityCfg("robot_left", body_names=["panda_hand"], joint_names=["panda_finger_joint1"]),
    "right_robot_cfg": SceneEntityCfg("robot_right", body_names=["panda_hand"], joint_names=["panda_finger_joint1"]),
}


@configclass
class ObservationsCfg:
    """Compact dual-arm proprioceptive and shoelace state observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        left_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_finger_joint1"])},
            scale=25.0,
        )
        right_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_finger_joint1"])},
            scale=25.0,
        )
        tails_to_tcp = ObsTerm(
            func=mdp.tails_to_tcp,
            params={
                "asset_cfgs": _SHOELACE_ASSET_CFGS,
                "left_robot_cfg": SceneEntityCfg("robot_left", body_names=["panda_hand"]),
                "right_robot_cfg": SceneEntityCfg("robot_right", body_names=["panda_hand"]),
            },
            scale=10.0,
        )
        tails_to_knot = ObsTerm(
            func=mdp.tails_to_knot,
            params={"asset_cfgs": _SHOELACE_ASSET_CFGS},
            scale=10.0,
        )
        tail_velocities = ObsTerm(func=mdp.tail_velocities, params={"asset_cfgs": _SHOELACE_ASSET_CFGS})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        throat_density = ObsTerm(
            func=mdp.throat_density,
            params={"throat_radius": _THROAT_RADIUS, "asset_cfgs": _SHOELACE_ASSET_CFGS},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Restore the authored shoelace and apply staged robot start poses."""

    reset_left_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left"),
        },
    )
    reset_right_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_right"),
        },
    )
    reset_shoelace = EventTerm(
        func=mdp.ResetShoelaceCurriculum,
        mode="reset",
        params={
            "difficulty_term_name": "pull_to_grasp",
            "asset_cfgs": _SHOELACE_ASSET_CFGS,
            "arm_cfgs": (
                SceneEntityCfg("robot_left", joint_names=["panda_joint[1-7]"]),
                SceneEntityCfg("robot_right", joint_names=["panda_joint[1-7]"]),
            ),
            "gripper_cfgs": (
                SceneEntityCfg("robot_left", joint_names=["panda_finger_joint.*"]),
                SceneEntityCfg("robot_right", joint_names=["panda_finger_joint.*"]),
            ),
            "grasp_joint_positions": (
                _LEFT_FRANKA_GRASP_JOINT_POSITIONS,
                _RIGHT_FRANKA_GRASP_JOINT_POSITIONS,
            ),
            "open_position": _GRIPPER_OPEN_POSITION,
            "closed_position": _GRIPPER_CLOSED_POSITION,
            "gripper_open_phase_fraction": _GRIPPER_OPEN_PHASE_FRACTION,
            "approach_phase_exponent": _APPROACH_PHASE_EXPONENT,
        },
    )


@configclass
class CurriculumCfg:
    """Success-driven progression from pull-only to full shoelace untying."""

    pull_to_grasp = CurrTerm(
        func=mdp.PullToGraspCurriculum,
        params={
            "level_count": _CURRICULUM_LEVEL_COUNT,
            "success_term_name": "success",
            "promotion_success_rate": 0.7,
            "minimum_episodes": 128,
            "current_level_fraction": 0.5,
            "current_level_fraction_schedule": (0.2, 0.35, 0.5),
            "terminal_level_fraction": 1.0,
            "terminal_level_fraction_schedule": (0.2, 0.35, 0.5, 0.75, 1.0),
            "fraction_increase_success_rate": 0.5,
            "fraction_backoff_success_rate": 0.1,
            "promotion_window_count": 2,
            "replay_level_weights": (0.5, 0.3, 0.2),
            "initial_level": 0,
        },
    )


@configclass
class RewardsCfg:
    """Meta-World-style dense task reward with completion and safety terms."""

    dense_task = RewTerm(
        func=mdp.shoelace_dense_reward,
        weight=10.0,
        params={
            **_ROBOT_TERM_PARAMS,
            "reach_std": 0.05,
            "grasp_std": _GRASP_ACQUISITION_DISTANCE,
            "throat_radius": _THROAT_RADIUS,
            "maximum_throat_segments": _MAXIMUM_THROAT_SEGMENTS,
            "tail_success_distance": _TAIL_SUCCESS_DISTANCE,
            "tail_success_separation": _TAIL_SUCCESS_SEPARATION,
            "target_speed": _TARGET_PULL_SPEED,
            "open_position": _GRIPPER_OPEN_POSITION,
            "closed_position": _GRIPPER_CLOSED_POSITION,
            "approach_weight": 0.1,
            "acquisition_weight": 0.25,
            "task_weight": 1.0,
            "pull_weight": 0.25,
        },
    )
    grasp_acquisition = RewTerm(
        func=mdp.grasp_acquisition_event,
        weight=10.0,
        params={
            **_ROBOT_TERM_PARAMS,
            "maximum_grasp_distance": _GRASP_ACQUISITION_DISTANCE,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "side_weights": (1.0, 1.0),
        },
    )
    success = RewTerm(
        func=mdp.termination_event_reward,
        weight=60.0,
        params={"term_keys": ["success"]},
    )
    failure = RewTerm(
        func=mdp.termination_event_reward,
        weight=-2.0,
        params={"term_keys": ["unsafe", "lost_grasp"]},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    left_joint_velocity = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot_left", joint_names=["panda_joint.*"]),
            "maximum_penalty": 100.0,
        },
    )
    right_joint_velocity = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot_right", joint_names=["panda_joint.*"]),
            "maximum_penalty": 100.0,
        },
    )


@configclass
class TerminationsCfg:
    """Untie success, grasp loss, safety, and time-limit terminations."""

    success = DoneTerm(
        func=mdp.shoelace_success,
        params={
            **_ROBOT_TERM_PARAMS,
            "throat_radius": _THROAT_RADIUS,
            "maximum_throat_segments": _MAXIMUM_THROAT_SEGMENTS,
            "tail_success_distance": _TAIL_SUCCESS_DISTANCE,
            "tail_success_separation": _TAIL_SUCCESS_SEPARATION,
            "maximum_success_grasp_distance": _MAXIMUM_SUCCESS_GRASP_DISTANCE,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "minimum_lace_height": _MINIMUM_LACE_HEIGHT,
            "maximum_lace_spread": _MAXIMUM_LACE_SPREAD,
        },
    )
    unsafe = DoneTerm(
        func=mdp.shoelace_unsafe,
        params={
            **_ROBOT_TERM_PARAMS,
            "minimum_lace_height": _MINIMUM_LACE_HEIGHT,
            "maximum_lace_spread": _MAXIMUM_LACE_SPREAD,
        },
    )
    lost_grasp = DoneTerm(
        func=mdp.lost_grasp,
        params={
            **_ROBOT_TERM_PARAMS,
            "acquisition_distance": _GRASP_ACQUISITION_DISTANCE,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "maximum_grasp_distance": _MAXIMUM_GRASP_DISTANCE,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ShoelaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for two Frankas untying an authored shoelace knot."""

    decimation = 4
    episode_length_s = 20.0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="robots",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            njmax=2048,
                            nconmax=256,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/(Robot(Left|Right)|Shoe)"],
                    ),
                    CouplerEntryCfg(
                        name="shoelace",
                        solver_cfg=VBDSolverCfg(
                            iterations=20,
                            rigid_contact_hard=True,
                            rigid_avbd_alpha=0.0,
                            rigid_body_contact_buffer_size=256,
                        ),
                        bodies=[r"/World/envs/env_[^/]+/Shoelace(Left|Right)"],
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="robots",
                        destination="shoelace",
                        bodies=[
                            r"/World/envs/env_[^/]+/Robot(Left|Right)/Geometry/.*panda_hand",
                            r"/World/envs/env_[^/]+/Robot(Left|Right)/Geometry/.*panda_(left|right)finger",
                            r"/World/envs/env_[^/]+/Shoe",
                        ],
                        collide_interval=2,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=512 * 32,
                max_triangle_pairs=32768 * 32,
            ),
            collision_decimation=2,
            default_shape_cfg=NewtonShapeCfg(gap=1.0e-4, ke=2.5e4, kd=100.0, mu=10.0),
            num_substeps=5,
            use_cuda_graph=True,
        ),
    )
    scene: ShoelaceSceneCfg = ShoelaceSceneCfg(
        num_envs=32,
        env_spacing=1.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    contacts_per_env = 512
    triangle_pairs_per_env = 8192
    grasp_assist_enabled = True
    grasp_assist_acquisition_distance = _GRASP_ACQUISITION_DISTANCE
    grasp_assist_release_distance = _MAXIMUM_GRASP_DISTANCE
    grasp_assist_acquisition_closed_separation = 0.0805
    grasp_assist_release_open_separation = 0.081
    grasp_assist_stiffness = 20.0
    grasp_assist_damping = 0.04
    grasp_assist_maximum_force = 2.0

    def __post_init__(self) -> None:
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.85, -0.85, 0.55),
            lookat=(0.0, 0.0, 0.08),
        )

    def play_mode(self) -> None:
        """Evaluate the complete authored approach-and-grasp task."""
        maximum_level = self.curriculum.pull_to_grasp.params["level_count"] - 1
        self.curriculum.pull_to_grasp.params["initial_level"] = maximum_level
        self.curriculum.pull_to_grasp.params["current_level_fraction"] = 1.0
        self.curriculum.pull_to_grasp.params["current_level_fraction_schedule"] = (1.0,)
        self.curriculum.pull_to_grasp.params["terminal_level_fraction"] = 1.0
        self.curriculum.pull_to_grasp.params["terminal_level_fraction_schedule"] = (1.0,)
