# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Multi-robot multi-task env: OpenArm-lift, Franka-cabinet, UR10-reach.

Three robot types, each performing a different manipulation task.
Each robot-task pair occupies its own env-id group.  The
:class:`ActionManager` automatically shares action columns across
disjoint groups.

Task-specific MDP functions use the ``task_group`` mechanism on
each term config, with explicit per-term parameters referencing
the correct scene entities.  This combines the multi-robot layout
(different robots per group) with multi-task MDP logic (different
observation/reward functions per group).

Layout (3 groups, evenly split):
    Group 0:  OpenArm  -- Lift cube
    Group 1:  Franka   -- Open cabinet drawer
    Group 2:  UR10     -- Track 6D pose command
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.managers import RobotGroupCfg, SceneEntityCfg
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

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

TASK_OPENARM_LIFT = "openarm_lift"
TASK_FRANKA_CABINET = "franka_cabinet"
TASK_UR10_REACH = "ur10_reach"

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

_OPENARM_MARKER_CFG = FRAME_MARKER_CFG.copy()
_OPENARM_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_OPENARM_MARKER_CFG.prim_path = "/Visuals/OpenArmEEFrame"

_FRANKA_MARKER_CFG = FRAME_MARKER_CFG.copy()
_FRANKA_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_FRANKA_MARKER_CFG.prim_path = "/Visuals/FrankaEEFrame"

_CABINET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_CABINET_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_CABINET_MARKER_CFG.prim_path = "/Visuals/CabinetFrame"

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultiRobotMultiTaskSceneCfg(InteractiveSceneCfg):
    """Three robot types, each performing a different task."""

    task_groups = {
        TASK_OPENARM_LIFT: 1,
        TASK_FRANKA_CABINET: 1,
        TASK_UR10_REACH: 1,
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

    # ── OpenArm Lift ─────────────────────────────────────────
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArmTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_OPENARM_LIFT,
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
        task_group=TASK_OPENARM_LIFT,
    )
    openarm_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
        task_group=TASK_OPENARM_LIFT,
    )
    openarm_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_OPENARM_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
        task_group=TASK_OPENARM_LIFT,
    )

    # ── Franka Cabinet ───────────────────────────────────────
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
        task_group=TASK_FRANKA_CABINET,
    )
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.4),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
        task_group=TASK_FRANKA_CABINET,
    )
    franka_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_FRANKA_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
        task_group=TASK_FRANKA_CABINET,
    )
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=False,
        visualizer_cfg=_CABINET_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, -0.5, -0.5, 0.5),
                ),
            ),
        ],
        task_group=TASK_FRANKA_CABINET,
    )

    # ── UR10 Reach ───────────────────────────────────────────
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_UR10_REACH,
    )
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
        task_group=TASK_UR10_REACH,
    )


# ===========================================================
# Actions  (IK + gripper per robot, columns shared)
# ===========================================================


@configclass
class MultiRobotMultiTaskActionsCfg:
    """Per-robot actions shared through disjoint task groups.

    Action dim: max(6+1, 6+1, 6) = 7 (IK + gripper columns shared).
    """

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

    ur10_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="ur10_robot",
        joint_names=[".*"],
        body_name="ee_link",
        controller=_IK_CTRL,
        scale=0.5,
    )


# ===========================================================
# Commands
# ===========================================================


@configclass
class MultiRobotMultiTaskCommandsCfg:
    """Task-specific commands for lift and reach groups.

    The cabinet task does not use a pose command — the goal is
    defined by the drawer joint position.
    """

    openarm_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="openarm_robot",
        body_name="openarm_hand",
        task_group=TASK_OPENARM_LIFT,
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
    ur10_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="ur10_robot",
        body_name="ee_link",
        task_group=TASK_UR10_REACH,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-3.14, 3.14),
        ),
    )


# ===========================================================
# Observations
# ===========================================================


@configclass
class MultiRobotMultiTaskObsCfg:
    """Shared and task-specific observations packed into one policy vector.

    Global terms (``task_onehot``, ``actions``) are computed for all
    environments.  Task-specific terms use ``task_group`` scoping and
    are zero-padded for environments outside their group.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        task_onehot = ObsTerm(func=mdp.multi_task_onehot)
        ee_pose = ObsTerm(func=mdp.ee_pose_b, per_robot=True)
        actions = ObsTerm(func=mdp.last_action)

        # ── OpenArm Lift ─────────────────────────────────
        openarm_object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            task_group=TASK_OPENARM_LIFT,
            params={
                "robot_cfg": SceneEntityCfg("openarm_robot"),
                "object_cfg": SceneEntityCfg("openarm_object"),
            },
        )
        openarm_target_pose = ObsTerm(
            func=mdp.generated_commands,
            task_group=TASK_OPENARM_LIFT,
            params={"command_name": "openarm_object_pose"},
        )
        openarm_ee_object_error = ObsTerm(
            func=mdp.ee_object_pos_error,
            task_group=TASK_OPENARM_LIFT,
            params={
                "robot_cfg": SceneEntityCfg("openarm_robot"),
                "object_cfg": SceneEntityCfg("openarm_object"),
                "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame"),
            },
        )
        openarm_object_target_error = ObsTerm(
            func=mdp.object_target_pos_error,
            task_group=TASK_OPENARM_LIFT,
            params={
                "command_name": "openarm_object_pose",
                "robot_cfg": SceneEntityCfg("openarm_robot"),
                "object_cfg": SceneEntityCfg("openarm_object"),
            },
        )

        # ── Franka Cabinet ───────────────────────────────
        franka_cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            task_group=TASK_FRANKA_CABINET,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        franka_cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            task_group=TASK_FRANKA_CABINET,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        franka_cabinet_handle_error = ObsTerm(
            func=mdp.cabinet_rel_ee_drawer_distance,
            task_group=TASK_FRANKA_CABINET,
            params={
                "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
                "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
            },
        )

        # ── UR10 Reach ───────────────────────────────────
        ur10_reach_command = ObsTerm(
            func=mdp.generated_commands,
            task_group=TASK_UR10_REACH,
            params={"command_name": "ur10_ee_pose"},
        )
        ur10_reach_ee_error = ObsTerm(
            func=mdp.ee_pos_error,
            task_group=TASK_UR10_REACH,
            params={
                "command_name": "ur10_ee_pose",
                "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards
# ===========================================================


@configclass
class MultiRobotMultiTaskRewardsCfg:
    """Task-specific rewards for OpenArm-lift, Franka-cabinet, and UR10-reach."""

    # ── OpenArm Lift ─────────────────────────────────
    lift_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        task_group=TASK_OPENARM_LIFT,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("openarm_object"),
            "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame"),
        },
    )
    lift_lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        task_group=TASK_OPENARM_LIFT,
        weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("openarm_object")},
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        task_group=TASK_OPENARM_LIFT,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "openarm_object_pose",
            "robot_cfg": SceneEntityCfg("openarm_robot"),
            "object_cfg": SceneEntityCfg("openarm_object"),
        },
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        task_group=TASK_OPENARM_LIFT,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "openarm_object_pose",
            "robot_cfg": SceneEntityCfg("openarm_robot"),
            "object_cfg": SceneEntityCfg("openarm_object"),
        },
    )

    # ── Franka Cabinet ───────────────────────────────
    cabinet_approach_ee_handle = RewTerm(
        func=mdp.cabinet_approach_ee_handle,
        task_group=TASK_FRANKA_CABINET,
        weight=2.0,
        params={
            "threshold": 0.2,
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.cabinet_align_ee_handle,
        task_group=TASK_FRANKA_CABINET,
        weight=0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.cabinet_approach_gripper_handle,
        task_group=TASK_FRANKA_CABINET,
        weight=5.0,
        params={
            "offset": 0.04,
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.cabinet_align_grasp_around_handle,
        task_group=TASK_FRANKA_CABINET,
        weight=0.125,
        params={
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.cabinet_grasp_handle,
        task_group=TASK_FRANKA_CABINET,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_finger_.*"]),
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.cabinet_open_drawer_bonus,
        task_group=TASK_FRANKA_CABINET,
        weight=7.5,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.cabinet_multi_stage_open_drawer,
        task_group=TASK_FRANKA_CABINET,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )

    # ── UR10 Reach ───────────────────────────────────
    reach_ee_pos_tracking = RewTerm(
        func=mdp.position_command_error,
        task_group=TASK_UR10_REACH,
        weight=-0.2,
        params={
            "command_name": "ur10_ee_pose",
            "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
        },
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        task_group=TASK_UR10_REACH,
        weight=0.1,
        params={
            "std": 0.1,
            "command_name": "ur10_ee_pose",
            "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
        },
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.orientation_command_error,
        task_group=TASK_UR10_REACH,
        weight=-0.2,
        params={
            "command_name": "ur10_ee_pose",
            "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
        },
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        task_group=TASK_UR10_REACH,
        weight=0.1,
        params={
            "std": 0.1,
            "command_name": "ur10_ee_pose",
            "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
        },
    )

    # ── Global penalties ─────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        per_robot=True,
    )


# ===========================================================
# Terminations
# ===========================================================


@configclass
class MultiRobotMultiTaskTerminationsCfg:
    """Task-specific terminations for the multi-robot multi-task demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.object_height_below_minimum,
        task_group=TASK_OPENARM_LIFT,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("openarm_object")},
    )
    cabinet_success = DoneTerm(
        func=mdp.cabinet_drawer_opened,
        task_group=TASK_FRANKA_CABINET,
        params={"threshold": 0.39, "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )


# ===========================================================
# Events
# ===========================================================


@configclass
class MultiRobotMultiTaskEventsCfg:
    """Reset events for the multi-robot multi-task layout."""

    # ── Shared robot resets ───────────────────────────
    reset_to_default = EventTerm(
        func=mdp.reset_asset_to_default,
        mode="reset",
        per_robot=True,
        params={"reset_joint_targets": True},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        per_robot=True,
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )
    # ── OpenArm Lift resets ───────────────────────────
    reset_openarm_object = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        task_group=TASK_OPENARM_LIFT,
        params={
            "object_cfg": SceneEntityCfg("openarm_object"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )

    # ── Franka Cabinet resets ────────────────────────
    reset_cabinet = EventTerm(
        func=mdp.reset_asset_to_default,
        mode="reset",
        task_group=TASK_FRANKA_CABINET,
        params={"asset_cfg": SceneEntityCfg("cabinet")},
    )


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotMultiTaskCurriculumCfg:
    """Gradually increase action-rate and joint-vel penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 100000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 100000},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot multi-task: OpenArm-lift, Franka-cabinet, UR10-reach.

    Group 0: OpenArm (7 arm DoF + 2 finger DoF) – lift a cube
    Group 1: Franka  (7 arm DoF + 2 finger DoF) – open a cabinet drawer
    Group 2: UR10    (6 arm DoF)                – track a 6D pose command

    Action dim: max(6+1, 6+1, 6) = 7 (IK + gripper columns shared).

    Each group uses ``task_group`` scoping on its MDP terms, with
    explicit per-term parameters referencing the correct scene
    entities.
    """

    scene: MultiRobotMultiTaskSceneCfg = MultiRobotMultiTaskSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    robot_meta = {
        "openarm_robot": RobotGroupCfg(
            asset_cfg=SceneEntityCfg("openarm_robot", body_names=["openarm_hand"]),
        ),
        "franka_robot": RobotGroupCfg(
            asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"]),
        ),
        "ur10_robot": RobotGroupCfg(
            asset_cfg=SceneEntityCfg("ur10_robot", body_names=["ee_link"]),
        ),
    }

    actions: MultiRobotMultiTaskActionsCfg = MultiRobotMultiTaskActionsCfg()
    commands: MultiRobotMultiTaskCommandsCfg = MultiRobotMultiTaskCommandsCfg()
    observations: MultiRobotMultiTaskObsCfg = MultiRobotMultiTaskObsCfg()
    rewards: MultiRobotMultiTaskRewardsCfg = MultiRobotMultiTaskRewardsCfg()
    terminations: MultiRobotMultiTaskTerminationsCfg = MultiRobotMultiTaskTerminationsCfg()
    events: MultiRobotMultiTaskEventsCfg = MultiRobotMultiTaskEventsCfg()
    curriculum: MultiRobotMultiTaskCurriculumCfg = MultiRobotMultiTaskCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotMultiTaskEnvCfg_PLAY(MultiRobotMultiTaskEnvCfg):
    """Play config with fewer environments and no observation corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
