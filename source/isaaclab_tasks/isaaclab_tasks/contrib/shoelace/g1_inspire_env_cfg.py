# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Inspire hands manipulating a shoe on a narrow pedestal."""

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import clone, configclass

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from .shoelace_assets import ShoelaceUsdCfg
from .shoelace_env_cfg import ActionsCfg, ShoelaceEnvCfg, ShoelaceSceneCfg

_ARM_JOINTS = r"{side}_(shoulder_.*|elbow|wrist_.*)_joint"
_TCP_OFFSETS = ((0.185, -0.015, 0.060), (0.185, 0.015, 0.060))
_ROBOT_CFGS = (
    SceneEntityCfg("robot", joint_names=["L_index_proximal_joint"], body_names=["left_wrist_yaw_link"]),
    SceneEntityCfg("robot", joint_names=["R_index_proximal_joint"], body_names=["right_wrist_yaw_link"]),
)


def _robot_cfg() -> ArticulationCfg:
    """Configure fixed-base G1 with explicit MJWarp gravity compensation."""
    robot = clone(G1_INSPIRE_FTP_CFG)
    robot.spawn = ShoelaceUsdCfg(
        usd_path=robot.spawn.usd_path,
        rigid_props={"/.*": [robot.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]},
        articulation_props=robot.spawn.articulation_props,
        fix_root_link=True,
    )
    robot.init_state.pos = (0.0, -0.38, 0.795)
    robot.init_state.rot = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    # IK-measured open-hand pregrasp over the nominal settled free tails, 60 mm above them.
    robot.init_state.joint_pos = {
        ".*(hip|knee|ankle|waist).*": 0.0,
        "[LR]_.*_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0086,
        "left_shoulder_roll_joint": -0.0590,
        "left_shoulder_yaw_joint": -0.1207,
        "left_elbow_joint": 0.0720,
        "left_wrist_roll_joint": 0.0286,
        "left_wrist_pitch_joint": 0.0472,
        "left_wrist_yaw_joint": -0.0744,
        "right_shoulder_pitch_joint": -0.1606,
        "right_shoulder_roll_joint": 0.0721,
        "right_shoulder_yaw_joint": 0.1153,
        "right_elbow_joint": 0.1573,
        "right_wrist_roll_joint": -0.0341,
        "right_wrist_pitch_joint": 0.1447,
        "right_wrist_yaw_joint": 0.0628,
    }
    return robot


def _arm_action(side: str, offset: tuple[float, float, float]) -> env_mdp.DifferentialInverseKinematicsActionCfg:
    """Control one wrist with a relative pose command in its pinch frame."""
    return env_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[_ARM_JOINTS.format(side=side)],
        body_name=f"{side}_wrist_yaw_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
        scale=(0.003, 0.003, 0.003, 0.01, 0.01, 0.01),
        body_offset=env_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=offset),
    )


def _hand_action(prefix: str) -> env_mdp.BinaryJointPositionActionCfg:
    """Use a bounded thumb-index pinch while retaining all authored finger joints."""
    return env_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"{prefix}_.*_joint"],
        open_command_expr={f"{prefix}_.*_joint": 0.0},
        close_command_expr={
            f"{prefix}_index_proximal_joint": 1.2,
            f"{prefix}_index_intermediate_joint": 1.2,
            f"{prefix}_(middle|ring|pinky)_.*_joint": 0.2,
            f"{prefix}_thumb_proximal_yaw_joint": 0.9,
            f"{prefix}_thumb_proximal_pitch_joint": 0.4,
            f"{prefix}_thumb_intermediate_joint": 0.5,
            f"{prefix}_thumb_distal_joint": 0.7,
        },
    )


@configclass
class G1ShoelaceSceneCfg(ShoelaceSceneCfg):
    """One humanoid and an 8 by 26 cm shoe support, with room on both sides."""

    robot_left = None
    robot_right = None
    robot = _robot_cfg()
    pedestal_column = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PedestalColumn",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.76),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=10.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.30)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
    )
    pedestal_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PedestalTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.26, 0.02),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.23)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.77)),
    )

    def __post_init__(self) -> None:
        """Raise the complete shoe assembly and bind both hands' contact bodies."""
        self.shoelace_asset.init_state.pos = (0.0, 0.0, 0.78)
        self.shoe.init_state.pos = (0.0, 0.0, 0.78)
        self.finger_tail_contacts.robot_prim_names = ("Robot", "Robot")
        self.finger_tail_contacts.finger_body_names = (
            ("L_thumb_distal", "L_index_intermediate"),
            ("R_thumb_distal", "R_index_intermediate"),
        )


@configclass
class G1ShoelaceActionsCfg(ActionsCfg):
    """Two Cartesian arms and two thumb-index pinch commands, totalling 14 actions."""

    left_arm = _arm_action("left", _TCP_OFFSETS[0])
    left_gripper = _hand_action("L")
    right_arm = _arm_action("right", _TCP_OFFSETS[1])
    right_gripper = _hand_action("R")


@configclass
class G1ShoelaceEnvCfg(ShoelaceEnvCfg):
    """G1 Inspire variant sharing the cable, reset, reward and bilateral success definitions.

    The pelvis is fixed while both arms and all hand joints move. The initial hand action uses
    an open/pinch synergy; this does not expose independent policy control of every finger.
    The shoe remains kinematic, as in the Franka baseline. Its X axis is the humanoid's lateral axis.
    """

    scene: G1ShoelaceSceneCfg = G1ShoelaceSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)
    actions: G1ShoelaceActionsCfg = G1ShoelaceActionsCfg()
    tcp_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = _TCP_OFFSETS
    """Left/right wrist-to-pinch offsets in the respective wrist frames [m]."""

    def __post_init__(self) -> None:
        """Bind the shared managers to one articulation with increasing finger closure angles."""
        policy = self.observations.policy
        for side in ("left", "right"):
            joint_cfg = SceneEntityCfg("robot", joint_names=[_ARM_JOINTS.format(side=side)])
            getattr(policy, f"{side}_joint_pos").params["asset_cfg"] = joint_cfg
            getattr(policy, f"{side}_joint_vel").params["asset_cfg"] = clone(joint_cfg)
            getattr(self.events, f"reset_{side}_arm").params["asset_cfg"] = clone(joint_cfg)
        for name in ("gripper_close_error", "tails_to_tcp", "tail_tcp_relative_speed"):
            getattr(policy, name).params["robot_cfgs"] = tuple(clone(cfg) for cfg in _ROBOT_CFGS)
        policy.gripper_close_error.params["closing_direction"] = 1.0
        policy.gripper_close_error.clip = (0.0, 1.2)
        policy.gripper_close_error.scale = 1.0 / 1.2
        for term in (
            self.rewards.dense_task,
            self.rewards.pregrasp,
            self.rewards.grasp_hold,
            self.terminations.success,
        ):
            term.params.update(
                robot_cfgs=tuple(clone(cfg) for cfg in _ROBOT_CFGS), open_position=0.0, closed_position=1.2
            )
        self.events.reset_shoe.params["position_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
        self.sim.physics.solver_cfg.entries[0].bodies = [
            r"/World/envs/env_[^/]+/(Robot|ShoelaceScene/Shoe|Pedestal(Column|Top))"
        ]
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(1.1, 1.3, 1.15), lookat=(0.0, -0.15, 0.8), focal_length=20.0, max_visible_envs=1)
        ]
