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

All MDP terms use **explicit batched classes** that iterate ``robot_meta``
and fill their output tensors via ``layout.env_slice(group_key)``.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import cloner_strategies
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
from isaaclab.scene import CloneCfg, InclusionSet, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import (
    CabinetGroupCfg,
    LiftGroupCfg,
    PoseCommandRanges,
    ReachGroupCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

TASK_LIFT = "lift"
TASK_CABINET = "cabinet"
TASK_REACH = "reach"

ROBOT_META = {
    TASK_LIFT: LiftGroupCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_joint.*", "panda_finger.*"]),
        command_name="ee_pose",
        command_ranges=PoseCommandRanges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
        robot_cfg=SceneEntityCfg("robot"),
        object_cfg=SceneEntityCfg("lift_object"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
    ),
    TASK_CABINET: CabinetGroupCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger.*"]),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        cabinet_frame_cfg=SceneEntityCfg("cabinet_frame"),
        cabinet_asset_cfg=SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
    ),
    TASK_REACH: ReachGroupCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_joint.*"]),
        command_name="ee_pose",
        command_ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    ),
}

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
    """Single Franka scene with clone groups for lift, cabinet, and reach."""

    clone_cfg = CloneCfg(
        # sequential strategy creates slices of environments, it is more efficient with large number of environments 
        # despite visually looks cluttered.
        clone_strategy=cloner_strategies.sequential,
        clone_groups={
            TASK_LIFT: InclusionSet(assets=["lift_table", "lift_object"], weight=1),
            TASK_CABINET: InclusionSet(assets=["cabinet", "cabinet_frame"], weight=1),
            TASK_REACH: InclusionSet(assets=["reach_table"], weight=1),
        },
    )

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
    )
    lift_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
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
    )
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
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
    )

    reach_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReachTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )


@configclass
class FrankaMultiTaskActionsCfg:
    """Shared Franka actions across all clone groups."""

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
    """Single batched pose command — per-group ranges live in ``robot_meta``."""

    ee_pose = mdp.BatchedPoseCommandCfg(
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        robot_meta=ROBOT_META,
    )


@configclass
class FrankaMultiTaskObservationsCfg:
    """Batched observation space for all clone groups.

    Every term is a batched :class:`ManagerTermBase` that internally
    iterates ``robot_meta`` entries and fills a single output tensor
    covering all environments.  Task-specific terms (e.g. object
    position for lift, cabinet joint state) are zero-padded for groups
    where they do not apply.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── shared proprioception ─────────────────────────
        task_onehot = ObsTerm(func=mdp.multi_task_onehot, params={"robot_meta": ROBOT_META})
        joint_pos_rel = ObsTerm(func=mdp.batched_joint_pos_rel, params={"robot_meta": ROBOT_META})
        joint_vel = ObsTerm(func=mdp.batched_joint_vel, params={"robot_meta": ROBOT_META})
        ee_pose = ObsTerm(func=mdp.batched_ee_pose, params={"robot_meta": ROBOT_META})
        actions = ObsTerm(func=mdp.last_action)

        # ── lift observations (zero-padded for non-lift) ──
        object_pos = ObsTerm(func=mdp.batched_object_pos_in_robot_frame, params={"robot_meta": ROBOT_META})
        object_target_pos_error = ObsTerm(func=mdp.batched_object_target_pos_error, params={"robot_meta": ROBOT_META})
        ee_object_pos_error = ObsTerm(func=mdp.batched_ee_object_pos_error, params={"robot_meta": ROBOT_META})

        # ── cabinet observations (zero-padded for non-cab) ─
        cabinet_joint_pos = ObsTerm(func=mdp.batched_cabinet_joint_pos, params={"robot_meta": ROBOT_META})
        cabinet_joint_vel = ObsTerm(func=mdp.batched_cabinet_joint_vel, params={"robot_meta": ROBOT_META})
        cabinet_handle_error = ObsTerm(
            func=mdp.batched_cabinet_rel_ee_drawer_distance, params={"robot_meta": ROBOT_META}
        )

        # ── reach + lift commands (zero-padded for cabinet) ─
        commands = ObsTerm(func=mdp.batched_generated_commands, params={"robot_meta": ROBOT_META})
        ee_pos_error = ObsTerm(func=mdp.batched_ee_pos_error, params={"robot_meta": ROBOT_META})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaMultiTaskRewardsCfg:
    """Rewards for lift, cabinet, and reach clone groups.

    Each batched class internally selects groups via ``isinstance``
    checks on the ``robot_meta`` entries.
    """

    # ── Lift ──────────────────────────────────────────
    lift_reaching_object = RewTerm(
        func=mdp.batched_object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "robot_meta": ROBOT_META},
    )
    lift_lifting_object = RewTerm(
        func=mdp.batched_object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.04, "robot_meta": ROBOT_META},
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04, "robot_meta": ROBOT_META},
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.04, "robot_meta": ROBOT_META},
    )

    # ── Cabinet ───────────────────────────────────────
    cabinet_approach_ee_handle = RewTerm(
        func=mdp.batched_cabinet_approach_ee_handle,
        weight=2.0,
        params={"threshold": 0.2, "robot_meta": ROBOT_META},
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.batched_cabinet_align_ee_handle,
        weight=0.5,
        params={"robot_meta": ROBOT_META},
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.batched_cabinet_approach_gripper_handle,
        weight=5.0,
        params={"offset": 0.04, "robot_meta": ROBOT_META},
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.batched_cabinet_align_grasp_around_handle,
        weight=0.125,
        params={"robot_meta": ROBOT_META},
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.batched_cabinet_grasp_handle,
        weight=0.5,
        params={"threshold": 0.03, "open_joint_pos": 0.04, "robot_meta": ROBOT_META},
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.batched_cabinet_open_drawer_bonus,
        weight=7.5,
        params={"robot_meta": ROBOT_META},
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.batched_cabinet_multi_stage_open_drawer,
        weight=1.0,
        params={"robot_meta": ROBOT_META},
    )

    # ── Reach ─────────────────────────────────────────
    reach_ee_pos_tracking = RewTerm(
        func=mdp.batched_position_command_error,
        weight=-0.2,
        params={"robot_meta": ROBOT_META},
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.batched_position_command_error_tanh,
        weight=0.1,
        params={"std": 0.1, "robot_meta": ROBOT_META},
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.batched_orientation_command_error,
        weight=-0.3,
        params={"robot_meta": ROBOT_META},
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.batched_orientation_command_error_tanh,
        weight=0.1,
        params={"std": 0.2, "robot_meta": ROBOT_META},
    )

    # ── Global penalties ──────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.batched_joint_vel_l2, weight=-1e-4, params={"robot_meta": ROBOT_META})


@configclass
class FrankaMultiTaskTerminationsCfg:
    """Terminations for the multitask Franka demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.batched_object_height_below_minimum,
        params={"minimum_height": -0.05, "robot_meta": ROBOT_META},
    )
    cabinet_success = DoneTerm(
        func=mdp.batched_cabinet_drawer_opened,
        params={"threshold": 0.39, "robot_meta": ROBOT_META},
    )


@configclass
class FrankaMultiTaskEventsCfg:
    """Reset events for shared-robot multitask layouts."""

    reset_to_default = EventTerm(
        func=mdp.batched_reset_to_default,
        mode="reset",
        params={"reset_joint_targets": True, "robot_meta": ROBOT_META},
    )
    reset_joints = EventTerm(
        func=mdp.batched_reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "robot_meta": ROBOT_META,
        },
    )
    reset_lift_object = EventTerm(
        func=mdp.batched_reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "robot_meta": ROBOT_META,
        },
    )
    reset_cabinet = EventTerm(
        func=mdp.batched_reset_cabinet,
        mode="reset",
        params={"robot_meta": ROBOT_META},
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
    """Single Franka multitask RL env covering lift, cabinet, and reach.

    ``robot_meta`` maps task-group names to typed group configs.
    All batched MDP classes iterate these entries to fill their
    output tensors by group slice.
    """

    scene: FrankaMultiTaskSceneCfg = FrankaMultiTaskSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
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
