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
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import clone, configclass

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from .mdp.inspire_hand_cfg import InspireFingerActionCfg, InspirePinchIKActionCfg
from .mdp.rewards import binary_gripper_switch_rate
from .shoelace_assets import ShoelaceUsdCfg
from .shoelace_env_cfg import ObservationsCfg, RewardsCfg, ShoelaceEnvCfg, ShoelaceSceneCfg

_ARM_JOINTS = r"{side}_(shoulder_.*|elbow|wrist_.*)_joint"
# Finger targets and near-tip surface landmarks from a constrained fit around a 2.8 mm target.
# TCP is the live surface-point midpoint; dynamic grasp/hold validation remains necessary.
_INDEX_CLOSED = (0.802548, 0.853050)
_THUMB_YAW = (1.191681, 1.246639)
_THUMB_CLOSED = (0.265063, 0.262787)
_INDEX_OPEN = tuple(value - 0.2 for value in _INDEX_CLOSED)
_THUMB_OPEN = tuple(value - 0.05 for value in _THUMB_CLOSED)
_FINGER_CFGS = tuple(
    SceneEntityCfg("robot", joint_names=[joint], body_names=[body])
    for joint, body in (
        ("L_thumb_proximal_pitch_joint", "L_thumb_distal"),
        ("L_index_proximal_joint", "L_index_intermediate"),
        ("R_thumb_proximal_pitch_joint", "R_thumb_distal"),
        ("R_index_proximal_joint", "R_index_intermediate"),
    )
)
_PAD_OFFSETS = (
    (0.019816, -0.013389, -0.004498),
    (-0.005060, -0.037262, -0.003575),
    (0.014990, 0.014964, -0.007629),
    (-0.005540, 0.035395, -0.006249),
)
_PAD_BODY_NAMES = (("L_thumb_distal", "L_index_intermediate"), ("R_thumb_distal", "R_index_intermediate"))
_TCP_BODY_OFFSETS = (_PAD_OFFSETS[:2], _PAD_OFFSETS[2:])
_ROBOT_CFGS = tuple(
    SceneEntityCfg(
        "robot", joint_names=[f"{prefix}_index_proximal_joint"], body_names=list(bodies), preserve_order=True
    )
    for prefix, bodies in zip(("L", "R"), _PAD_BODY_NAMES, strict=True)
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
        "[LR]_(middle|ring|pinky)_.*_joint": 0.0,
        "left_shoulder_pitch_joint": -0.012014,
        "left_shoulder_roll_joint": 0.083006,
        "left_shoulder_yaw_joint": -0.022565,
        "left_elbow_joint": -0.125478,
        "left_wrist_roll_joint": -0.105378,
        "left_wrist_pitch_joint": 0.257075,
        "left_wrist_yaw_joint": -0.155942,
        "right_shoulder_pitch_joint": -0.224297,
        "right_shoulder_roll_joint": -0.062120,
        "right_shoulder_yaw_joint": 0.027560,
        "right_elbow_joint": -0.006563,
        "right_wrist_roll_joint": 0.079811,
        "right_wrist_pitch_joint": 0.361425,
        "right_wrist_yaw_joint": 0.140190,
    }
    for arm, prefix in enumerate(("L", "R")):
        robot.init_state.joint_pos.update(_thumb_targets(prefix, _THUMB_OPEN[arm], _THUMB_YAW[arm]))
        robot.init_state.joint_pos[f"{prefix}_index_proximal_joint"] = _INDEX_OPEN[arm]
        robot.init_state.joint_pos[f"{prefix}_index_intermediate_joint"] = _INDEX_OPEN[arm]
    return robot


def _arm_action(side: str, arm: int) -> InspirePinchIKActionCfg:
    """Control the live two-finger midpoint with relative translation and wrist rotation."""
    return InspirePinchIKActionCfg(
        asset_name="robot",
        joint_names=[_ARM_JOINTS.format(side=side)],
        body_name=f"{side}_wrist_yaw_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
        scale=(0.003, 0.003, 0.003, 0.01, 0.01, 0.01),
        body_offset=env_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(),
        pad_body_names=_PAD_BODY_NAMES[arm],
        pad_offsets=_TCP_BODY_OFFSETS[arm],
    )


def _thumb_targets(prefix: str, pitch: float, yaw: float) -> dict[str, float]:
    """Respect the USD thumb mimic ratios, 1.6 and 2.4 times the primary flexion angle."""
    return {
        f"{prefix}_thumb_proximal_yaw_joint": yaw,
        f"{prefix}_thumb_proximal_pitch_joint": pitch,
        f"{prefix}_thumb_intermediate_joint": 1.6 * pitch,
        f"{prefix}_thumb_distal_joint": 2.4 * pitch,
    }


def _pinch_action(prefix: str, arm: int) -> InspireFingerActionCfg:
    """Synchronously open/close the thumb and index chains with one binary command."""
    opened = _thumb_targets(prefix, _THUMB_OPEN[arm], _THUMB_YAW[arm])
    closed = _thumb_targets(prefix, _THUMB_CLOSED[arm], _THUMB_YAW[arm])
    opened.update({f"{prefix}_index_{part}_joint": _INDEX_OPEN[arm] for part in ("proximal", "intermediate")})
    closed.update({f"{prefix}_index_{part}_joint": _INDEX_CLOSED[arm] for part in ("proximal", "intermediate")})
    return InspireFingerActionCfg(
        asset_name="robot",
        joint_names=[f"{prefix}_(thumb|index)_.*_joint"],
        open_command_expr=opened,
        close_command_expr=closed,
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
class G1ShoelaceActionsCfg:
    """Two Cartesian arms and two synchronous thumb/index commands, totalling 14 actions."""

    left_arm = _arm_action("left", 0)
    left_gripper = _pinch_action("L", 0)
    right_arm = _arm_action("right", 1)
    right_gripper = _pinch_action("R", 1)


@configclass
class G1ShoelaceObservationsCfg(ObservationsCfg):
    """Expose actual thumb/index configurations, target errors and uncapped pad-to-tail vectors."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        gripper_close_error = ObservationTermCfg(
            func=f"{__package__}.mdp.inspire_hand:inspire_finger_close_error",
            params={
                "finger_cfgs": _FINGER_CFGS,
                "motion_ranges": (0.05, 0.2, 0.05, 0.2),
            },
            clip=(-1.0, 1.0),
        )
        finger_joint_pos = ObservationTermCfg(
            func=env_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["[LR]_(thumb|index)_.*_joint"])},
        )
        finger_joint_vel = ObservationTermCfg(
            func=env_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["[LR]_(thumb|index)_.*_joint"])},
            scale=0.1,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1ShoelaceRewardsCfg(RewardsCfg):
    """Shared grasp rewards with a cost per binary hand command reversal."""

    finger_action_switch = RewardTermCfg(func=binary_gripper_switch_rate, weight=-0.002)


@configclass
class G1ShoelaceEnvCfg(ShoelaceEnvCfg):
    """G1 Inspire variant sharing the cable, reset, reward and bilateral success definitions.

    The pelvis is fixed while both arms, thumbs and index fingers move. Each hand has one
    synchronous thumb/index open/close action; the other three fingers retain straight targets.
    The shoe remains kinematic, as in the Franka baseline. Its X axis is the humanoid's lateral axis.
    """

    scene: G1ShoelaceSceneCfg = G1ShoelaceSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)
    actions: G1ShoelaceActionsCfg = G1ShoelaceActionsCfg()
    observations: G1ShoelaceObservationsCfg = G1ShoelaceObservationsCfg()
    rewards: G1ShoelaceRewardsCfg = G1ShoelaceRewardsCfg()
    tcp_body_offsets = _TCP_BODY_OFFSETS
    """Thumb/index surface offsets in their own body frames; the live mean defines each TCP [m]."""
    finger_body_offsets = _PAD_OFFSETS
    """The same surface landmarks used by the shared grasp rewards [m]."""
    grasp_open_apertures = (0.028813161, 0.025744588)
    """Measured left/right surface spacing at the ready pose [m]."""

    def __post_init__(self) -> None:
        """Bind the shared managers to one articulation with increasing finger closure angles."""
        policy = self.observations.policy
        for side in ("left", "right"):
            joint_cfg = SceneEntityCfg("robot", joint_names=[_ARM_JOINTS.format(side=side)])
            getattr(policy, f"{side}_joint_pos").params["asset_cfg"] = joint_cfg
            getattr(policy, f"{side}_joint_vel").params["asset_cfg"] = clone(joint_cfg)
            getattr(self.events, f"reset_{side}_arm").params["asset_cfg"] = clone(joint_cfg)
        policy.tails_to_tcp.params["robot_cfgs"] = tuple(clone(cfg) for cfg in _ROBOT_CFGS)
        for name in ("pad_to_tail", "finger_tail_relative_velocity"):
            getattr(policy, name).params["finger_cfgs"] = tuple(clone(cfg) for cfg in _FINGER_CFGS)
        for term in (
            self.rewards.pregrasp,
            self.rewards.dense_task,
            self.rewards.grasp_hold,
            self.terminations.success,
        ):
            term.params.update(
                finger_cfgs=tuple(clone(cfg) for cfg in _FINGER_CFGS),
            )
        self.events.reset_shoe.params["position_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
        self.sim.physics.solver_cfg.entries[0].bodies = [
            r"/World/envs/env_[^/]+/(Robot|ShoelaceScene/Shoe|Pedestal(Column|Top))"
        ]
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(1.1, 1.3, 1.15), lookat=(0.0, -0.15, 0.8), focal_length=20.0, max_visible_envs=1)
        ]
