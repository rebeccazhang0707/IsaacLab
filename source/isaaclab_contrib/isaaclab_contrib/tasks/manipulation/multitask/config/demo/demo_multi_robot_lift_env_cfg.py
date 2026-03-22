# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot lift env: OpenArm + Franka, each lifting its own cube.

Each robot type occupies its own env-id group.  The
:class:`ActionManager` automatically shares action columns
across disjoint groups.

Lift-specific batched MDP classes iterate ``robot_meta`` at init
time to discover ``robot_cfg``, ``object_cfg``, ``ee_frame_cfg``,
and ``command_name`` for each robot group and use the
gather-first-compute-once pattern for efficient multi-robot
computation.

Layout (2 groups, evenly split):
    Group 0:  OpenArm  -- Lift cube
    Group 1:  Franka   -- Lift cube
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import LiftGroupCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

TASK_OPENARM = "openarm_lift"
TASK_FRANKA = "franka_lift"

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)

_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    ),
)

_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultiRobotLiftSceneCfg(InteractiveSceneCfg):
    """Two robot types, each lifting its own cube."""

    task_groups = {
        TASK_OPENARM: 1,
        TASK_FRANKA: 1,
    }

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )

    # ── OpenArm ──────────────────────────────────────────────
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
        task_group=TASK_OPENARM,
    )

    openarm_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
        task_group=TASK_OPENARM,
    )
    openarm_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
        task_group=TASK_OPENARM,
    )

    # ── Franka ───────────────────────────────────────────────
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
        task_group=TASK_FRANKA,
    )
    franka_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
        task_group=TASK_FRANKA,
    )
    franka_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
        task_group=TASK_FRANKA,
    )


# ===========================================================
# Actions  (DiffIK + binary gripper, columns shared)
# ===========================================================

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class MultiRobotLiftActionsCfg:
    """Per-robot arm IK + gripper actions."""

    openarm_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        body_name="openarm_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    openarm_gripper = BinaryJointPositionActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )

    franka_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="franka_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    franka_gripper = BinaryJointPositionActionCfg(
        asset_name="franka_robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ===========================================================
# Commands
# ===========================================================


@configclass
class MultiRobotLiftCommandsCfg:
    """Per-group lift command targets (goal position for the object)."""

    openarm_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="openarm_robot",
        body_name="openarm_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
    )
    franka_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="franka_robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )


# ===========================================================
# Observations
# ===========================================================


@configclass
class MultiRobotLiftObsCfg:
    """Proprioceptive + object observations, batched across robot groups.

    Batched observation classes iterate ``robot_meta`` to discover
    ``robot_cfg``, ``object_cfg``, ``command_name`` per robot group.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose = ObsTerm(func=mdp.batched_ee_pose)
        object_pos = ObsTerm(func=mdp.batched_object_pos_in_robot_frame)
        target_object_pos = ObsTerm(func=mdp.batched_generated_commands)
        ee_object_pos_error = ObsTerm(func=mdp.batched_ee_object_pos_error)
        object_target_pos_error = ObsTerm(func=mdp.batched_object_target_pos_error)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards
# ===========================================================


@configclass
class MultiRobotLiftRewardsCfg:
    """Lift rewards batched across all robot groups.

    Batched reward classes iterate ``robot_meta`` and use the
    gather-first-compute-once pattern for ``robot_cfg``,
    ``object_cfg``, ``ee_frame_cfg``, and ``command_name``.
    """

    reaching_object = RewTerm(
        func=mdp.batched_object_ee_distance,
        weight=1.0,
        params={"std": 0.1},
    )
    lifting_object = RewTerm(
        func=mdp.batched_object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.04},
    )
    object_goal_tracking = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04},
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.04},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.batched_joint_vel_l2,
        weight=-1e-4,
    )


# ===========================================================
# Terminations
# ===========================================================


@configclass
class MultiRobotLiftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.batched_object_height_below_minimum,
        params={"minimum_height": -0.05},
    )


# ===========================================================
# Events
# ===========================================================


@configclass
class MultiRobotLiftEventsCfg:
    """Reset events batched across all robot groups.

    Batched event classes iterate ``robot_meta`` to discover robot
    groups and dispatch reset logic per group with filtered ``env_ids``.
    """

    reset_to_default = EventTerm(
        func=mdp.batched_reset_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_joints = EventTerm(
        func=mdp.batched_reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_object = EventTerm(
        func=mdp.batched_reset_object_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotLiftCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot lift: OpenArm + Franka, each lifting its own cube.

    Group 0: OpenArm (7 arm DoF + 2 finger DoF)
    Group 1: Franka  (7 arm DoF + 2 finger DoF)

    Action dim: max(7, 7) = 6 (IK shared) + 1 (gripper shared) = 7
    (IK dim=6, gripper dim=1, columns shared across disjoint groups)

    The ``robot_meta`` dict declares ``asset_cfg``, ``command_name``,
    ``robot_cfg``, ``object_cfg``, and ``ee_frame_cfg`` — used by
    batched MDP term classes.
    """

    scene: MultiRobotLiftSceneCfg = MultiRobotLiftSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    # Per-robot metadata used by batched MDP term classes.
    # Each key is a scene asset name.  Batched classes iterate these entries at init time:
    #   asset_cfg    – SceneEntityCfg identifying the EE body and arm+gripper joints.
    #   robot_cfg    – SceneEntityCfg for the robot articulation root.
    #   object_cfg   – SceneEntityCfg for the rigid object to lift.
    #   ee_frame_cfg – SceneEntityCfg for the FrameTransformer sensor.
    #   command_name – name of the UniformPoseCommandCfg for the object goal.
    robot_meta = {
        TASK_OPENARM: LiftGroupCfg(
            asset_cfg=SceneEntityCfg(
                "openarm_robot",
                body_names=["openarm_hand"],
                joint_names=["openarm_joint.*", "openarm_finger_joint.*"],
            ),
            robot_cfg=SceneEntityCfg("openarm_robot"),
            object_cfg=SceneEntityCfg("openarm_object"),
            ee_frame_cfg=SceneEntityCfg("openarm_ee_frame"),
            command_name="openarm_object_pose",
        ),
        TASK_FRANKA: LiftGroupCfg(
            asset_cfg=SceneEntityCfg(
                "franka_robot",
                body_names=["panda_hand"],
                joint_names=["panda_joint.*", "panda_finger.*"],
            ),
            robot_cfg=SceneEntityCfg("franka_robot"),
            object_cfg=SceneEntityCfg("franka_object"),
            ee_frame_cfg=SceneEntityCfg("franka_ee_frame"),
            command_name="franka_object_pose",
        ),
    }

    actions: MultiRobotLiftActionsCfg = MultiRobotLiftActionsCfg()
    commands: MultiRobotLiftCommandsCfg = MultiRobotLiftCommandsCfg()
    observations: MultiRobotLiftObsCfg = MultiRobotLiftObsCfg()
    rewards: MultiRobotLiftRewardsCfg = MultiRobotLiftRewardsCfg()
    terminations: MultiRobotLiftTerminationsCfg = MultiRobotLiftTerminationsCfg()
    events: MultiRobotLiftEventsCfg = MultiRobotLiftEventsCfg()
    curriculum: MultiRobotLiftCurriculumCfg = MultiRobotLiftCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotLiftEnvCfg_PLAY(MultiRobotLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
