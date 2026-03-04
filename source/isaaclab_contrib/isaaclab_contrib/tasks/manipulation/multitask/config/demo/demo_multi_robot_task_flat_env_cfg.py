# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat (hand-written) multi-robot multi-task env config.

Demonstrates how to write a heterogeneous multi-robot environment *without*
the ``MultiTaskRegistryConfig`` composition machinery.  Every scene asset,
action term, and prim-path is specified explicitly so the user has full control.

Scene assets use placeholder tokens ``{GROUP0}``, ``{GROUP1}``, ... in their
``prim_path``. The env-level ``__post_init__`` partitions ``num_envs`` across
groups and walks every scene asset to replace the tokens with the concrete
``(0|1|...)`` regex, so the config works for any ``num_envs``.

Layout (3 groups, evenly split):
    Group 0:  OpenArm  -- Lift Cube          (joint-position actions)
    Group 1:  Franka   -- Stack Cubes        (joint-position actions)
    Group 2:  UR10     -- Reach (no objects)  (differential-IK actions)
"""

from __future__ import annotations

import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

from isaaclab_contrib.tasks.manipulation.multitask import mdp

NUM_GROUPS = 3

_GROUP_TOKEN_RE = re.compile(r"\{GROUP(\d+)\}")


def _partition_env_ids(num_envs: int, num_groups: int) -> list[list[int]]:
    """Split *num_envs* indices as evenly as possible across *num_groups*."""
    base, remainder = divmod(num_envs, num_groups)
    groups: list[list[int]] = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def _resolve_group_tokens(prim_path: str, groups: list[list[int]]) -> str:
    """Replace ``{GROUP<N>}`` tokens in *prim_path* with ``(<id>|<id>|...)``."""

    def _replacer(m: re.Match) -> str:
        idx = int(m.group(1))
        return "(" + "|".join(str(i) for i in groups[idx]) + ")"

    return _GROUP_TOKEN_RE.sub(_replacer, prim_path)


_LIFT_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_LIFT_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=_LIFT_CUBE_RIGID_PROPS,
)

_STACK_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_STACK_CUBE_1_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_2_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_3_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
)


@configclass
class FlatMultiRobotSceneCfg(InteractiveSceneCfg):
    """Scene with three robot groups, each in its own env-id subset.

    Prim paths use ``{GROUP0}`` / ``{GROUP1}`` / ``{GROUP2}`` placeholders
    resolved to concrete env-id regexes by the env config's ``__post_init__``.
    """

    # -- shared (all envs) ---------------------------------------------------
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- Group 0: OpenArm Lift -----------------------------------------------
    openarm_robot = OPENARM_UNI_CFG.replace(prim_path="/World/envs/env_{GROUP0}/Robot")
    openarm_table = AssetBaseCfg(
        prim_path="/World/envs/env_{GROUP0}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    openarm_cube = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP0}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_LIFT_CUBE_SPAWN,
    )

    # -- Group 1: Franka Stack (joint-position) ------------------------------
    franka_stack_robot = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_{GROUP1}/Robot")
    franka_stack_table = AssetBaseCfg(
        prim_path="/World/envs/env_{GROUP1}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    franka_cube_1 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_1_SPAWN,
    )
    franka_cube_2 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_2_SPAWN,
    )
    franka_cube_3 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_3_SPAWN,
    )

    # -- Group 2: UR10 Reach (IK controller, no gripper, no objects) ----------
    ur10_reach_robot = UR10_CFG.replace(prim_path="/World/envs/env_{GROUP2}/Robot")
    ur10_reach_table = AssetBaseCfg(
        prim_path="/World/envs/env_{GROUP2}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )


@configclass
class FlatMultiRobotActionsCfg:
    """Per-group action terms.

    Groups 0 & 1 use joint-position control.
    Group 2 uses differential inverse kinematics with a UR10 (no gripper).
    """

    # Group 0: OpenArm (joint-position)
    openarm_arm = JointPositionActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    openarm_gripper = BinaryJointPositionActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )

    # Group 1: Franka Stack (joint-position)
    franka_stack_arm = JointPositionActionCfg(
        asset_name="franka_stack_robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    franka_stack_gripper = BinaryJointPositionActionCfg(
        asset_name="franka_stack_robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

    # Group 2: UR10 Reach (differential IK, no gripper)
    ur10_reach_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="ur10_reach_robot",
        joint_names=[".*"],
        body_name="ee_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )


@configclass
class FlatObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        sim_time_s = ObsTerm(func=mdp.current_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlatRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class FlatTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FlatEventsCfg:
    reset_scene_to_default = EventTerm(func=mdp.reset_multitask_scene_to_default, mode="reset")


@configclass
class FlatMultiRobotLiftStackEnvCfg(ManagerBasedRLEnvCfg):
    """Hand-written multi-robot env with three heterogeneous groups.

    No ``MultiTaskRegistryConfig`` -- everything is explicit.
    The scene uses ``{GROUP<N>}`` placeholder tokens in prim paths.
    ``__post_init__`` partitions ``num_envs`` across groups and walks every
    scene asset to resolve the placeholders into concrete env-id regexes.

    Group 0: OpenArm  -- lift cube          (joint-position actions)
    Group 1: Franka   -- stack 3 cubes      (joint-position actions)
    Group 2: UR10     -- reach (no objects) (differential-IK actions)
    """

    scene: FlatMultiRobotSceneCfg = FlatMultiRobotSceneCfg(
        num_envs=24, env_spacing=2.0, replicate_physics=False
    )
    actions: FlatMultiRobotActionsCfg = FlatMultiRobotActionsCfg()
    observations: FlatObservationsCfg = FlatObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: FlatTerminationsCfg = FlatTerminationsCfg()
    events: FlatEventsCfg = FlatEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # --- resolve {GROUP<N>} placeholders in scene asset prim_paths -------
        groups = _partition_env_ids(self.scene.num_envs, NUM_GROUPS)
        for attr_name in vars(self.scene):
            cfg = getattr(self.scene, attr_name)
            prim_path = getattr(cfg, "prim_path", None)
            if prim_path is not None and _GROUP_TOKEN_RE.search(prim_path):
                cfg.prim_path = _resolve_group_tokens(prim_path, groups)
