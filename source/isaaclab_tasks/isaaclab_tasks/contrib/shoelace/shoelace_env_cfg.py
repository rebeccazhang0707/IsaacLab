# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Literal

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
_MAXIMUM_SUCCESS_GRASP_DISTANCE = 0.015
_GRASP_ACQUISITION_DISTANCE = 0.012
_MAXIMUM_GRASP_DISTANCE = 0.020
_GRIPPER_OPEN_POSITION = 0.01
_GRIPPER_CLOSED_POSITION = 0.002
_GRIPPER_CLOSE_TARGET = 0.0015
_GRIPPER_MINIMUM_GRASP_POSITION = 0.0003
_GRIPPER_CLOSED_THRESHOLD = 0.0025
_GRIPPER_SUCCESS_THRESHOLD = 0.0035
_GRASP_CONFIRMATION_STEPS = 1
_GRASP_RELEASE_CONFIRMATION_STEPS = 6
_GRASP_ACQUISITION_DEADLINE_STEPS = 8
_SEPARATION_PROGRESS_DEADLINE_STEPS = 64
_MINIMUM_SEPARATION_PROGRESS = 0.5
_TARGET_PULL_SPEED = 0.04
_MINIMUM_LACE_HEIGHT = -0.003
_MAXIMUM_LACE_SPREAD = 0.6
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
_LEFT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS = (
    0.389244,
    0.011088,
    -0.521930,
    -2.569779,
    1.149809,
    2.555253,
    -0.150486,
)
_RIGHT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS = (
    -0.509291,
    -0.021903,
    0.501640,
    -2.588014,
    -1.197501,
    2.624648,
    1.603688,
)
_RIGHT_FRANKA_CONTACT_POSE_1_JOINT_POSITIONS = (
    -0.508908238594,
    0.001041418021,
    0.502844791584,
    -2.566197815865,
    -1.205718847000,
    2.614505814911,
    1.608019054752,
)
_RIGHT_FRANKA_CONTACT_POSE_2_JOINT_POSITIONS = (
    -0.508039408424,
    0.004605980527,
    0.503710989481,
    -2.561184597482,
    -1.203628728833,
    2.611603975973,
    1.606203600583,
)
_RIGHT_FRANKA_CONTACT_POSE_3_JOINT_POSITIONS = (
    -0.507449049679,
    0.019704037585,
    0.504276442333,
    -2.546659900388,
    -1.208664909196,
    2.604852518599,
    1.608572404884,
)
_RIGHT_FRANKA_APPROACH_BRIDGE_JOINT_POSITIONS = (
    (-0.507592004576, 0.017412899251, 0.504158130358, -2.549777760137, -1.210331670910, 2.606632777565, 1.610694055455),
    (-0.507656984074, 0.016371472736, 0.504104352187, -2.551194969114, -1.211089289871, 2.607441986187, 1.611658442079),
    (-0.507669979974, 0.016163187433, 0.504093596553, -2.551478410909, -1.211240813663, 2.607603827911, 1.611851319403),
)
_RIGHT_FRANKA_APPROACH_LEARNING_JOINT_POSITIONS = (
    (-0.507681936202, 0.015971564954, 0.504083701370, -2.551739177361, -1.211380215552, 2.607752722297, 1.612028766542),
    (-0.507690253578, 0.015838262360, 0.504076817764, -2.551920580110, -1.211477190779, 2.607856301001, 1.612152208029),
    (-0.507698570953, 0.015704959766, 0.504069934158, -2.552101982859, -1.211574166006, 2.607959879704, 1.612275649517),
)
_LEFT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS = (
    0.383982,
    -0.016711,
    -0.526566,
    -2.592917,
    1.140252,
    2.562327,
    -0.165460,
)
_RIGHT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS = (
    -0.510776,
    -0.033617,
    0.501523,
    -2.619221,
    -1.247455,
    2.646284,
    1.657949,
)
_APPROACH_BRIDGE_FRACTIONS = (
    (0.125, 0.0),
    (0.25, 0.0),
    (0.25, 0.01),
    (0.25, 0.0125),
    (0.5, 0.0125),
)
_LEFT_FRANKA_APPROACH_BRIDGE_JOINT_POSITIONS = tuple(
    tuple(
        start + fraction * (end - start)
        for start, end in zip(
            _LEFT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS,
            _LEFT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS,
            strict=True,
        )
    )
    for fraction, _ in _APPROACH_BRIDGE_FRACTIONS
)
_RIGHT_FRANKA_APPROACH_EXIT_BRIDGE_JOINT_POSITIONS = tuple(
    tuple(
        start + fraction * (end - start)
        for start, end in zip(
            _RIGHT_FRANKA_APPROACH_LEARNING_JOINT_POSITIONS[-1],
            _RIGHT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS,
            strict=True,
        )
    )
    for _, fraction in _APPROACH_BRIDGE_FRACTIONS
)
_CONTACT_CURRICULUM_LEVEL_COUNT = 81
_LEFT_FRANKA_CURRICULUM_JOINT_POSITIONS = (
    (_LEFT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS,) * 87
    + _LEFT_FRANKA_APPROACH_BRIDGE_JOINT_POSITIONS
    + (
        _LEFT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS,
        (0.374299, -0.059333, -0.531376, -2.610756, 1.098843, 2.563040, -0.152894),
        (0.354776, -0.120520, -0.533189, -2.634336, 1.037260, 2.562372, -0.133620),
        (0.320544, -0.194545, -0.527101, -2.659350, 0.959992, 2.557317, -0.108805),
        tuple(_LEFT_FRANKA_ARM_JOINT_POSITIONS.values()),
    )
)
_RIGHT_FRANKA_CURRICULUM_JOINT_POSITIONS = (
    (_RIGHT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS,) * 68
    + (_RIGHT_FRANKA_CONTACT_POSE_1_JOINT_POSITIONS,) * 10
    + (_RIGHT_FRANKA_CONTACT_POSE_2_JOINT_POSITIONS,)
    + (_RIGHT_FRANKA_CONTACT_POSE_3_JOINT_POSITIONS,) * 2
    + _RIGHT_FRANKA_APPROACH_BRIDGE_JOINT_POSITIONS
    + _RIGHT_FRANKA_APPROACH_LEARNING_JOINT_POSITIONS
    + _RIGHT_FRANKA_APPROACH_EXIT_BRIDGE_JOINT_POSITIONS
    + (
        _RIGHT_FRANKA_APPROACH_POSE_1_JOINT_POSITIONS,
        (-0.499383, -0.091087, 0.499334, -2.634953, -1.164799, 2.659881, 1.652890),
        (-0.477853, -0.149894, 0.497408, -2.662283, -1.103112, 2.660680, 1.633300),
        (-0.439454, -0.223472, 0.486224, -2.694287, -1.023344, 2.657568, 1.607431),
        tuple(_RIGHT_FRANKA_ARM_JOINT_POSITIONS.values()),
    )
)
_CURRICULUM_GRIPPER_JOINT_POSITIONS = (
    tuple(
        _GRIPPER_CLOSED_POSITION + fraction * (_GRIPPER_OPEN_POSITION - _GRIPPER_CLOSED_POSITION)
        for fraction in (
            0.0,
            0.0625,
            0.1171875,
            0.125,
            0.1875,
            0.19140625,
            0.19189453125,
            0.1923828125,
            0.19287109375,
            0.193359375,
            0.19384765625,
            0.1943359375,
            0.19482421875,
            0.1953125,
            0.19580078125,
            0.1962890625,
            0.19677734375,
            0.197265625,
            0.19775390625,
            0.1982421875,
            0.19873046875,
            0.19921875,
            0.19970703125,
            0.2001953125,
            0.20068359375,
            0.201171875,
            0.20166015625,
            0.2021484375,
            0.20263671875,
            0.203125,
            0.21484375,
            0.21875,
            0.2265625,
            0.234375,
            0.2421875,
            0.2470703125,
            0.25,
            0.25048828125,
            0.2509765625,
            0.25146484375,
            0.251953125,
            0.252197265625,
            0.2523193359375,
            0.25244140625,
            0.2525634765625,
            0.252685546875,
            0.2529296875,
            0.2530517578125,
            0.253173828125,
            0.2532958984375,
            0.25341796875,
            0.25390625,
            0.255859375,
            0.2572544642857143,
            0.25864955357142855,
            0.26004464285714285,
            0.26143973214285715,
            0.26283482142857145,
            0.2642299107142857,
            0.265625,
            0.26785714285714285,
            0.2700892857142857,
            0.27232142857142855,
            0.27455357142857145,
            0.2767857142857143,
            0.27901785714285715,
            0.28125,
            0.28515625,
            0.2890625,
            0.2900390625,
            0.291015625,
            0.375,
            0.3828125,
            0.390625,
            0.39453125,
            0.39501953125,
            0.3955078125,
            0.395751953125,
            0.5,
            0.75,
            1.0,
        )
    )
    + (_GRIPPER_OPEN_POSITION,) * 16
)
_CURRICULUM_LEVEL_COUNT = len(_CURRICULUM_GRIPPER_JOINT_POSITIONS)


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
            stiffness=6000.0,
            damping=60.0,
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


def _arm_action(asset_name: str) -> mdp.EMADifferentialInverseKinematicsActionCfg:
    """Build one six-dimensional relative TCP action."""
    return mdp.EMADifferentialInverseKinematicsActionCfg(
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
        body_offset=mdp.EMADifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
        alpha=0.75,
        warmup_steps=3,
        warmup_steps_by_curriculum_level=(3,) * _CONTACT_CURRICULUM_LEVEL_COUNT
        + (0,) * (_CURRICULUM_LEVEL_COUNT - _CONTACT_CURRICULUM_LEVEL_COUNT),
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
        close_command_expr={"panda_finger_joint1": _GRIPPER_CLOSE_TARGET},
    )
    right_arm = _arm_action("robot_right")
    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": _GRIPPER_OPEN_POSITION},
        close_command_expr={"panda_finger_joint1": _GRIPPER_CLOSE_TARGET},
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
        last_action = ObsTerm(func=mdp.filtered_last_action)

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
                _LEFT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS,
                _RIGHT_FRANKA_SETTLED_GRASP_JOINT_POSITIONS,
            ),
            "open_position": _GRIPPER_OPEN_POSITION,
            "closed_position": _GRIPPER_CLOSED_POSITION,
            "gripper_open_phase_fraction": _GRIPPER_OPEN_PHASE_FRACTION,
            "approach_phase_exponent": _APPROACH_PHASE_EXPONENT,
            "arm_joint_positions_by_level": (
                _LEFT_FRANKA_CURRICULUM_JOINT_POSITIONS,
                _RIGHT_FRANKA_CURRICULUM_JOINT_POSITIONS,
            ),
            "gripper_joint_positions_by_level": _CURRICULUM_GRIPPER_JOINT_POSITIONS,
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
            "promotion_success_rate": 0.5,
            "minimum_episodes": 128,
            "current_level_fraction": 0.5,
            "current_level_fraction_schedule": (0.2, 0.35, 0.5),
            "terminal_level_fraction": 1.0,
            "terminal_level_fraction_schedule": (0.2, 0.35, 0.5, 0.75, 1.0),
            "fraction_increase_success_rate": 0.5,
            "fraction_backoff_success_rate": 0.1,
            "promotion_window_count": 1,
            "replay_level_weights": (0.5, 0.3, 0.2),
            "initial_level": 0,
        },
    )


@configclass
class RewardsCfg:
    """Meta-World-style dense task reward with completion and safety terms."""

    dense_task = RewTerm(
        func=mdp.reset_relative_dense_reward,
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
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "maximum_grasp_distance": _MAXIMUM_GRASP_DISTANCE,
            "maximum_success_grasp_distance": _MAXIMUM_SUCCESS_GRASP_DISTANCE,
            "maximum_progress_rate": 3.0,
            "soft_min_temperature": 0.05,
            "soft_min_weight": 0.75,
            "confirmation_steps": _GRASP_CONFIRMATION_STEPS,
            "approach_weight": 0.1,
            "acquisition_weight": 0.25,
            "task_weight": 1.0,
            "pull_weight": 0.25,
        },
    )
    grasp_acquisition = RewTerm(
        func=mdp.bilateral_grasp_acquisition_event,
        weight=10.0,
        params={
            **_ROBOT_TERM_PARAMS,
            "maximum_grasp_distance": _GRASP_ACQUISITION_DISTANCE,
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "confirmation_steps": _GRASP_CONFIRMATION_STEPS,
        },
    )
    success = RewTerm(
        func=mdp.termination_event_reward,
        weight=60.0,
        params={"term_keys": ["success"]},
    )
    failure = RewTerm(
        func=mdp.termination_event_reward,
        weight=-12.0,
        params={
            "term_keys": ["unsafe", "lost_grasp", "missed_grasp", "insufficient_separation", "time_out"],
            "exclude_term_keys": ["success"],
            "include_time_outs": True,
        },
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
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_SUCCESS_THRESHOLD,
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
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "maximum_grasp_distance": _MAXIMUM_GRASP_DISTANCE,
            "confirmation_steps": _GRASP_CONFIRMATION_STEPS,
            "release_confirmation_steps": _GRASP_RELEASE_CONFIRMATION_STEPS,
        },
    )
    missed_grasp = DoneTerm(
        func=mdp.missed_grasp_acquisition,
        params={
            **_ROBOT_TERM_PARAMS,
            "acquisition_distance": _GRASP_ACQUISITION_DISTANCE,
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "deadline_steps": _GRASP_ACQUISITION_DEADLINE_STEPS,
        },
    )
    insufficient_separation = DoneTerm(
        func=mdp.insufficient_separation_progress,
        params={
            **_ROBOT_TERM_PARAMS,
            "acquisition_distance": _GRASP_ACQUISITION_DISTANCE,
            "minimum_finger_position": _GRIPPER_MINIMUM_GRASP_POSITION,
            "maximum_finger_position": _GRIPPER_CLOSED_THRESHOLD,
            "target_tail_separation": _TAIL_SUCCESS_SEPARATION,
            "minimum_progress_fraction": _MINIMUM_SEPARATION_PROGRESS,
            "deadline_steps": _SEPARATION_PROGRESS_DEADLINE_STEPS,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ShoelaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for two Frankas untying an authored shoelace knot."""

    is_finite_horizon = True
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
        env_spacing=0.25,
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
    coupling_mode: Literal["proxy", "admm"] = "proxy"
    """Coupled-solver mode selected when the environment is constructed."""
    admm_iterations: int = 5
    """Number of ADMM interface iterations per coupled step."""
    admm_rho: float = 400.0
    """ADMM penalty parameter [dimensionless]."""
    admm_gamma: float = 7.5e-5
    """ADMM proximal mass scaling parameter [dimensionless]."""
    admm_baumgarte: float = 0.75
    """ADMM position-error correction fraction [dimensionless]."""
    admm_contact_matching: Literal["disabled", "latest", "sticky"] = "latest"
    """Frame-to-frame matching mode for ADMM rigid contacts."""
    grasp_assist_enabled = False
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
