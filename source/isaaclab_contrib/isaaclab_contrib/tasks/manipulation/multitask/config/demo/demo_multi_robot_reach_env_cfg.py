# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot reach env: three robot types, all reach.

Each robot type occupies its own env-id group.  The
:class:`ActionManager` automatically shares action columns
across disjoint groups, so three DiffIK terms (each dim=6)
produce ``total_action_dim = 6`` instead of 18.

Observations are task-space-centric: EE pose, command target,
and position error are the same dimension for all robots
(no padding).  Joint-space terms are auto-padded to the
maximum arm DoF across all registered robots.

Layout (3 groups, evenly split):
    Group 0:  OpenArm -- Reach (7 arm DoF)
    Group 1:  Franka  -- Reach (7 arm DoF)
    Group 2:  UR10    -- Reach (6 arm DoF)
"""

from __future__ import annotations

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import ReachGroupCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

TASK_OPENARM = "openarm_reach"
TASK_FRANKA = "franka_reach"
TASK_UR10 = "ur10_reach"

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultitaskPhysicsCfg(PresetCfg):
    """Physics backend presets for the single-robot multitask environment."""

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=60,
            nconmax=80,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )


@configclass
class MultiRobotReachSceneCfg(InteractiveSceneCfg):
    """Three robot types, all doing reach (no objects)."""

    task_groups = {
        TASK_OPENARM: 1,
        TASK_FRANKA: 1,
        TASK_UR10: 1,
    }

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.05),
        ),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=_TABLE_SPAWN,
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
        task_group=TASK_OPENARM,
    )
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
        task_group=TASK_FRANKA,
    )
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
        task_group=TASK_UR10,
    )


# ===========================================================
# Actions  (3 DiffIK terms, columns shared → action_dim=6)
# ===========================================================


_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class MultiRobotReachActionsCfg:
    """Three DiffIK terms (one per robot group).

    Because the ActionManager shares action columns across
    disjoint groups, total_action_dim = max(6, 6, 6) = 6.
    """

    openarm_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        body_name="openarm_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    franka_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="franka_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
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
class MultiRobotReachCommandsCfg:
    """Per-group reach command targets."""

    openarm_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="openarm_robot",
        body_name="openarm_hand",
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.35),
            pos_y=(-0.2, 0.2),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
    )
    franka_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="franka_robot",
        body_name="panda_hand",
        resampling_time_range=(3.0, 3.0),
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
    ur10_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="ur10_robot",
        body_name="ee_link",
        resampling_time_range=(3.0, 3.0),
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
class MultiRobotReachObsCfg:
    """Task-space + proprioceptive observations.

    Batched observation classes iterate ``robot_meta`` at init time
    to discover robot groups and scatter per-group results (with
    zero-padding) into a single ``(num_envs, max_feat)`` tensor.

    Task-space terms (EE pose, command, error) have the same
    dimension regardless of robot DoF.  Joint-space terms are
    auto-padded to the maximum across robot groups.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.batched_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.batched_joint_vel)
        ee_pose = ObsTerm(func=mdp.batched_ee_pose)
        ee_command = ObsTerm(func=mdp.batched_generated_commands)
        ee_pos_error = ObsTerm(func=mdp.batched_ee_pos_error)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards / Terminations / Events
# ===========================================================


@configclass
class MultiRobotReachRewardsCfg:
    """Reach rewards batched across all robot groups.

    Batched reward classes iterate ``robot_meta`` at init time and
    use the gather-first-compute-once pattern to reduce CUDA
    kernel launches.
    """

    ee_pos_tracking = RewTerm(
        func=mdp.batched_position_command_error,
        weight=-0.2,
    )
    ee_pos_tracking_fine = RewTerm(
        func=mdp.batched_position_command_error_tanh,
        weight=0.1,
        params={"std": 0.1},
    )
    ee_ori_tracking = RewTerm(
        func=mdp.batched_orientation_command_error,
        weight=-0.1,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.batched_joint_vel_l2,
        weight=-0.0001,
    )


@configclass
class MultiRobotReachTerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class MultiRobotReachEventsCfg:
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


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotReachCurriculumCfg:
    """Gradually increase action-rate and joint-vel penalties to suppress jitter."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 12000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 12000},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotReachEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot reach: 3 groups, shared 6D action columns.

    Group 0: OpenArm (7 arm DoF)
    Group 1: Franka  (7 arm DoF)
    Group 2: UR10    (6 arm DoF)

    Action dim: 6 (columns shared across disjoint groups).
    """

    scene: MultiRobotReachSceneCfg = MultiRobotReachSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=False,
    )
    # Per-robot metadata used by batched MDP term classes.
    # Each key is a scene asset name.  Batched classes iterate these entries at init time to
    # discover robot groups and pre-allocate staging buffers:
    #   asset_cfg    – SceneEntityCfg identifying the EE body and arm joints.
    #   command_name – name of the UniformPoseCommandCfg that generates the reach target.
    robot_meta = {
        TASK_OPENARM: ReachGroupCfg(
            asset_cfg=SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], joint_names=["openarm_joint.*"]),
            command_name="openarm_ee_pose",
        ),
        TASK_FRANKA: ReachGroupCfg(
            asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"], joint_names=["panda_joint.*"]),
            command_name="franka_ee_pose",
        ),
        TASK_UR10: ReachGroupCfg(
            asset_cfg=SceneEntityCfg("ur10_robot", body_names=["ee_link"], joint_names=[".*"]),
            command_name="ur10_ee_pose",
        ),
    }

    actions: MultiRobotReachActionsCfg = MultiRobotReachActionsCfg()
    commands: MultiRobotReachCommandsCfg = MultiRobotReachCommandsCfg()
    observations: MultiRobotReachObsCfg = MultiRobotReachObsCfg()
    rewards: MultiRobotReachRewardsCfg = MultiRobotReachRewardsCfg()
    terminations: MultiRobotReachTerminationsCfg = MultiRobotReachTerminationsCfg()
    events: MultiRobotReachEventsCfg = MultiRobotReachEventsCfg()
    curriculum: MultiRobotReachCurriculumCfg = MultiRobotReachCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotReachEnvCfg_PLAY(MultiRobotReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
