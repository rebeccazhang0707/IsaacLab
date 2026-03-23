# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Multi-robot multi-task env: OpenArm-lift, Franka-cabinet, UR10-reach.

Three robot types, each performing a different manipulation task.
Each robot-task pair occupies its own env-id group.  The
:class:`ActionManager` automatically shares action columns across
disjoint groups.

All MDP terms use **explicit batched classes** that iterate
``robot_meta`` (keyed by task-group name) and fill their output
tensors via ``layout.env_slice(group_key)``.

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
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

TASK_OPENARM_LIFT = "openarm_lift"
TASK_FRANKA_CABINET = "franka_cabinet"
TASK_UR10_REACH = "ur10_reach"

# Robot metadata - defined at module level so term configs can reference it.
# SceneEntityCfg instances are auto-resolved when passed via term params.
ROBOT_META = {
    TASK_OPENARM_LIFT: LiftGroupCfg(
        asset_cfg=SceneEntityCfg(
            "openarm_robot",
            body_names=["openarm_hand"],
            joint_names=["openarm_joint.*", "openarm_finger_joint.*"],
        ),
        command_name="ee_pose",
        command_ranges=PoseCommandRanges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
        robot_cfg=SceneEntityCfg("openarm_robot"),
        object_cfg=SceneEntityCfg("openarm_object"),
        ee_frame_cfg=SceneEntityCfg("openarm_ee_frame"),
    ),
    TASK_FRANKA_CABINET: CabinetGroupCfg(
        asset_cfg=SceneEntityCfg(
            "franka_robot",
            body_names=["panda_hand"],
            joint_names=["panda_joint.*", "panda_finger.*"],
        ),
        ee_frame_cfg=SceneEntityCfg("franka_ee_frame"),
        cabinet_frame_cfg=SceneEntityCfg("cabinet_frame"),
        cabinet_asset_cfg=SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
    ),
    TASK_UR10_REACH: ReachGroupCfg(
        asset_cfg=SceneEntityCfg("ur10_robot", body_names=["ee_link"], joint_names=[".*"]),
        command_name="ee_pose",
        command_ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
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

    clone_cfg = CloneCfg(
        clone_groups={
            TASK_OPENARM_LIFT: InclusionSet(
                assets=["openarm_table", "openarm_robot", "openarm_object", "openarm_ee_frame"], weight=1
            ),
            TASK_FRANKA_CABINET: InclusionSet(
                assets=["franka_robot", "cabinet", "franka_ee_frame", "cabinet_frame"], weight=1
            ),
            TASK_UR10_REACH: InclusionSet(assets=["ur10_table", "ur10_robot"], weight=1),
        }
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

    # ── OpenArm Lift ─────────────────────────────────────────
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArmTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
    )
    openarm_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
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
    )

    # ── Franka Cabinet ───────────────────────────────────────
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
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
    )

    # ── UR10 Reach ───────────────────────────────────────────
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
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
    """Single batched pose command — per-group ranges live in ``robot_meta``.

    The cabinet task does not use a pose command — the goal is
    defined by the drawer joint position.
    """

    ee_pose = mdp.BatchedPoseCommandCfg(
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        robot_meta=ROBOT_META,
    )


# ===========================================================
# Observations
# ===========================================================


@configclass
class MultiRobotMultiTaskObsCfg:
    """Batched observation space for all robot-task groups.

    Every term is a batched :class:`ManagerTermBase` that internally
    iterates ``robot_meta`` entries and fills a single output tensor
    covering all environments.  Task-specific terms are zero-padded
    for groups where they do not apply.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── shared proprioception ─────────────────────────
        task_onehot = ObsTerm(func=mdp.multi_task_onehot)
        joint_pos_rel = ObsTerm(func=mdp.batched_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.batched_joint_vel)
        ee_pose = ObsTerm(func=mdp.batched_ee_pose)
        actions = ObsTerm(func=mdp.last_action)

        # ── lift observations (zero-padded for non-lift) ──
        object_pos = ObsTerm(func=mdp.batched_object_pos_in_robot_frame)
        object_target_pos_error = ObsTerm(func=mdp.batched_object_target_pos_error)
        ee_object_pos_error = ObsTerm(func=mdp.batched_ee_object_pos_error)

        # ── cabinet observations (zero-padded for non-cab) ─
        cabinet_joint_pos = ObsTerm(func=mdp.batched_cabinet_joint_pos)
        cabinet_joint_vel = ObsTerm(func=mdp.batched_cabinet_joint_vel)
        cabinet_handle_error = ObsTerm(func=mdp.batched_cabinet_rel_ee_drawer_distance)

        # ── reach + lift commands (zero-padded for cabinet) ─
        commands = ObsTerm(func=mdp.batched_generated_commands)
        ee_pos_error = ObsTerm(func=mdp.batched_ee_pos_error)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards
# ===========================================================


@configclass
class MultiRobotMultiTaskRewardsCfg:
    """Task-specific rewards for OpenArm-lift, Franka-cabinet, and UR10-reach.

    Each batched class internally selects groups via ``isinstance``
    checks on the ``robot_meta`` entries.
    """

    # ── OpenArm Lift ─────────────────────────────────
    lift_reaching_object = RewTerm(
        func=mdp.batched_object_ee_distance,
        weight=1.0,
        params={"std": 0.1},
    )
    lift_lifting_object = RewTerm(
        func=mdp.batched_object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.04},
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04},
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.batched_object_goal_distance,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.04},
    )

    # ── Franka Cabinet ───────────────────────────────
    cabinet_approach_ee_handle = RewTerm(
        func=mdp.batched_cabinet_approach_ee_handle,
        weight=2.0,
        params={"threshold": 0.2},
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.batched_cabinet_align_ee_handle,
        weight=0.5,
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.batched_cabinet_approach_gripper_handle,
        weight=5.0,
        params={"offset": 0.04},
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.batched_cabinet_align_grasp_around_handle,
        weight=0.125,
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.batched_cabinet_grasp_handle,
        weight=0.5,
        params={"threshold": 0.03, "open_joint_pos": 0.04},
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.batched_cabinet_open_drawer_bonus,
        weight=7.5,
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.batched_cabinet_multi_stage_open_drawer,
        weight=1.0,
    )

    # ── UR10 Reach ───────────────────────────────────
    reach_ee_pos_tracking = RewTerm(
        func=mdp.batched_position_command_error,
        weight=-0.2,
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.batched_position_command_error_tanh,
        weight=0.1,
        params={"std": 0.1},
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.batched_orientation_command_error,
        weight=-0.2,
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.batched_orientation_command_error_tanh,
        weight=0.1,
        params={"std": 0.1},
    )

    # ── Global penalties ─────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.batched_joint_vel_l2, weight=-1e-4)


# ===========================================================
# Terminations
# ===========================================================


@configclass
class MultiRobotMultiTaskTerminationsCfg:
    """Task-specific terminations for the multi-robot multi-task demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.batched_object_height_below_minimum,
        params={"minimum_height": -0.05},
    )
    cabinet_success = DoneTerm(
        func=mdp.batched_cabinet_drawer_opened,
        params={"threshold": 0.39},
    )


# ===========================================================
# Events
# ===========================================================


@configclass
class MultiRobotMultiTaskEventsCfg:
    """Reset events for heterogeneous multi-robot layout.

    Uses batched terms that handle dual-indexing (global for env_origins,
    local for asset data).
    """

    reset_to_default = EventTerm(
        func=mdp.batched_reset_to_default,
        mode="reset",
        params={"robot_meta": ROBOT_META, "reset_joint_targets": True},
    )
    reset_joints = EventTerm(
        func=mdp.batched_reset_joints,
        mode="reset",
        params={"robot_meta": ROBOT_META, "position_range": (0.5, 1.25), "velocity_range": (0.0, 0.0)},
    )
    reset_object = EventTerm(
        func=mdp.batched_reset_object_uniform,
        mode="reset",
        params={
            "robot_meta": ROBOT_META,
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    reset_cabinet = EventTerm(
        func=mdp.batched_reset_cabinet,
        mode="reset",
        params={"robot_meta": ROBOT_META},
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

    Group 0: OpenArm (7 arm DoF + 2 finger DoF) -- lift a cube
    Group 1: Franka  (7 arm DoF + 2 finger DoF) -- open a cabinet drawer
    Group 2: UR10    (6 arm DoF)                -- track a 6D pose command

    Action dim: max(6+1, 6+1, 6) = 7 (IK + gripper columns shared).

    ``robot_meta`` maps task-group names to typed group configs.
    All batched MDP classes iterate these entries to fill their
    output tensors by group slice.
    """

    scene: MultiRobotMultiTaskSceneCfg = MultiRobotMultiTaskSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

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
