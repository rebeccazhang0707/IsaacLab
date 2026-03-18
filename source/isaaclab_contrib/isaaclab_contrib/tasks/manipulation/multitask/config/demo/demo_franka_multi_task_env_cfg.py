# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Single-robot Franka multitask env with lift, cabinet, and reach groups.

This demo keeps one shared Franka articulation across all environments while
splitting environments into three task groups:

* ``lift``: lift a cube
* ``cabinet``: open a drawer
* ``reach``: track a pose command

Task-specific assets and terms are scoped with ``task_group``. Since the robot
is shared across all environments, task-specific observation/reward wrappers
slice the shared robot state using the layout's global environment ids.
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

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

TASK_LIFT = "lift"
TASK_CABINET = "cabinet"
TASK_REACH = "reach"


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

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_EE_MARKER_CFG.prim_path = "/Visuals/FrankaMultiTaskEEFrame"

_CABINET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_CABINET_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_CABINET_MARKER_CFG.prim_path = "/Visuals/FrankaMultiTaskCabinetFrame"

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class FrankaMultiTaskSceneCfg(InteractiveSceneCfg):
    """Single Franka scene with task-group assets for lift, cabinet, and reach."""

    task_groups = {
        TASK_LIFT: 1,
        TASK_CABINET: 1,
        TASK_REACH: 1,
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

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    lift_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LiftTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_LIFT,
    )
    lift_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
        task_group=TASK_LIFT,
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
        task_group=TASK_CABINET,
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
        task_group=TASK_CABINET,
    )

    reach_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReachTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_REACH,
    )


@configclass
class FrankaMultiTaskActionsCfg:
    """Shared Franka actions across all task groups."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaMultiTaskCommandsCfg:
    """Task-specific commands for lift and reach groups."""

    lift_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        task_group=TASK_LIFT,
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
    reach_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        task_group=TASK_REACH,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class FrankaMultiTaskObservationsCfg:
    """Shared and task-specific observations packed into one policy vector."""

    @configclass
    class PolicyCfg(ObsGroup):
        # proprioceptive observations
        task_onehot = ObsTerm(func=mdp.multi_task_onehot)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pose = ObsTerm(
            func=mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        # lift task
        lift_object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            task_group=TASK_LIFT,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("lift_object"),
            },
        )
        lift_target_object_pose = ObsTerm(
            func=mdp.generated_commands,
            task_group=TASK_LIFT,
            params={"command_name": "lift_object_pose"},
        )
        lift_ee_object_error = ObsTerm(
            func=mdp.ee_object_pos_error,
            task_group=TASK_LIFT,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("lift_object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        lift_object_target_pos_error = ObsTerm(
            func=mdp.object_target_pos_error,
            task_group=TASK_LIFT,
            params={
                "command_name": "lift_object_pose",
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("lift_object"),
            },
        )
        # cabinet task
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            task_group=TASK_CABINET,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            task_group=TASK_CABINET,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_handle_error = ObsTerm(
            func=mdp.cabinet_rel_ee_drawer_distance,
            task_group=TASK_CABINET,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
            },
        )
        # reach task
        reach_command = ObsTerm(
            func=mdp.generated_commands,
            task_group=TASK_REACH,
            params={"command_name": "reach_ee_pose"},
        )
        reach_ee_error = ObsTerm(
            func=mdp.ee_pos_error,
            task_group=TASK_REACH,
            params={
                "command_name": "reach_ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaMultiTaskRewardsCfg:
    """Task-specific rewards for lift, cabinet, and reach."""

    lift_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        task_group=TASK_LIFT,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("lift_object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    lift_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        task_group=TASK_LIFT,
        weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("lift_object")},
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        task_group=TASK_LIFT,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "lift_object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("lift_object"),
        },
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        task_group=TASK_LIFT,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "lift_object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("lift_object"),
        },
    )

    cabinet_approach_ee_handle = RewTerm(
        func=mdp.cabinet_approach_ee_handle,
        task_group=TASK_CABINET,
        weight=2.0,
        params={
            "threshold": 0.2,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.cabinet_align_ee_handle,
        task_group=TASK_CABINET,
        weight=0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.cabinet_approach_gripper_handle,
        task_group=TASK_CABINET,
        weight=5.0,
        params={
            "offset": 0.04,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.cabinet_align_grasp_around_handle,
        task_group=TASK_CABINET,
        weight=0.125,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.cabinet_grasp_handle,
        task_group=TASK_CABINET,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_.*"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.cabinet_open_drawer_bonus,
        task_group=TASK_CABINET,
        weight=7.5,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.cabinet_multi_stage_open_drawer,
        task_group=TASK_CABINET,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame"),
        },
    )

    reach_ee_pos_tracking = RewTerm(
        func=mdp.position_command_error,
        task_group=TASK_REACH,
        weight=-0.2,
        params={
            "command_name": "reach_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        task_group=TASK_REACH,
        weight=0.1,
        params={
            "std": 0.1,
            "command_name": "reach_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.orientation_command_error,
        task_group=TASK_REACH,
        weight=-0.3,
        params={
            "command_name": "reach_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        task_group=TASK_REACH,
        weight=0.1,
        params={
            "std": 0.2,
            "command_name": "reach_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class FrankaMultiTaskTerminationsCfg:
    """Task-specific terminations for the multitask Franka demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.object_height_below_minimum,
        task_group=TASK_LIFT,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("lift_object")},
    )
    cabinet_success = DoneTerm(
        func=mdp.cabinet_drawer_opened,
        task_group=TASK_CABINET,
        params={"threshold": 0.39, "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )


@configclass
class FrankaMultiTaskEventsCfg:
    """Reset events for shared-robot multitask layouts."""

    reset_robot_to_default = EventTerm(
        func=mdp.reset_asset_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "reset_joint_targets": True},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_lift_object = EventTerm(
        func=mdp.reset_object_state_uniform,
        mode="reset",
        task_group=TASK_LIFT,
        params={
            "object_cfg": SceneEntityCfg("lift_object"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    reset_cabinet = EventTerm(
        func=mdp.reset_asset_to_default,
        mode="reset",
        task_group=TASK_CABINET,
        params={"asset_cfg": SceneEntityCfg("cabinet")},
    )


@configclass
class FrankaMultiTaskCurriculumCfg:
    """Gradually increase global action penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 100000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 100000},
    )


@configclass
class FrankaMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Single Franka multitask RL env covering lift, cabinet, and reach."""

    scene: FrankaMultiTaskSceneCfg = FrankaMultiTaskSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    actions: FrankaMultiTaskActionsCfg = FrankaMultiTaskActionsCfg()
    commands: FrankaMultiTaskCommandsCfg = FrankaMultiTaskCommandsCfg()
    observations: FrankaMultiTaskObservationsCfg = FrankaMultiTaskObservationsCfg()
    rewards: FrankaMultiTaskRewardsCfg = FrankaMultiTaskRewardsCfg()
    terminations: FrankaMultiTaskTerminationsCfg = FrankaMultiTaskTerminationsCfg()
    events: FrankaMultiTaskEventsCfg = FrankaMultiTaskEventsCfg()
    curriculum: FrankaMultiTaskCurriculumCfg = FrankaMultiTaskCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class FrankaMultiTaskEnvCfg_PLAY(FrankaMultiTaskEnvCfg):
    """Play config with fewer environments and no observation corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
