# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat (hand-written) single-robot multi-task env config.

Demonstrates how to write a heterogeneous multi-task environment that uses
**one robot type** (Franka) across all environments, sharing a single
``ArticulationView``, while each group of environments runs a different task.
No ``MultiTaskRegistryConfig`` is used -- everything is specified explicitly.

The Franka robot is declared once with ``prim_path="{ENV_REGEX_NS}/Robot"``,
which creates a single PhysX ``ArticulationView`` covering every environment.
Because the robot and its action space are identical for all groups, a single
set of action terms (``arm_action`` + ``gripper_action``) applies to all envs.

Per-task objects (lift cube, stacking cubes) use ``{GROUP<N>}`` placeholder
tokens in their ``prim_path``; the env-level ``__post_init__`` resolves these
into concrete ``(0|1|...)`` env-id regexes. The reach group simply has no
objects.

Group / task mapping is stored in ``_group_env_ids`` and ``_task_id_per_env``
so that any MDP term can query which task an environment belongs to.  The
stack task uses a different Franka default joint pose, which is applied
per-env via ``_default_joint_pos_per_env`` and the corresponding reset event.

Layout (3 groups, evenly split):
    Group 0 (LIFT):   Franka -- Lift Cube        (1 cube)
    Group 1 (STACK):  Franka -- Stack Cubes      (3 cubes)
    Group 2 (REACH):  Franka -- Reach            (no objects)
"""

from __future__ import annotations

import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab_contrib.tasks.manipulation.multitask import mdp

# ---------------------------------------------------------------------------
# Task / group constants
# ---------------------------------------------------------------------------

TASK_LIFT = 0
TASK_STACK = 1
TASK_REACH = 2
NUM_GROUPS = 3

_GROUP_TOKEN_RE = re.compile(r"\{GROUP(\d+)\}")

# Stack task uses a different default arm pose than lift/reach.
# (From the official Franka stack env config.)
_STACK_DEFAULT_JOINT_POS = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]

# Default Franka joint pose from FRANKA_PANDA_CFG (lift / reach tasks).
_DEFAULT_JOINT_POS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-group event functions
# ---------------------------------------------------------------------------


def reset_default_joint_pos_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Overwrite ``default_joint_pos`` from per-env buffer stored on env.cfg.

    This is the flat equivalent of ``set_default_joint_pose`` but supports
    heterogeneous default poses across environments (e.g. stack envs use a
    different arm pose than lift/reach envs).
    """
    per_env = getattr(env.cfg, "_default_joint_pos_per_env", None)
    if per_env is None:
        return
    asset = env.scene[asset_cfg.name]
    per_env = per_env.to(device=asset.device)
    asset.data.default_joint_pos[env_ids] = per_env[env_ids]


# ---------------------------------------------------------------------------
# Shared spawn helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotSceneCfg(InteractiveSceneCfg):
    """Scene for a single-robot multi-task environment.

    **One Franka** is placed in every environment (``{ENV_REGEX_NS}/Robot``).
    This results in a single ``ArticulationView`` spanning all ``num_envs``.

    Per-task objects use ``{GROUP<N>}`` placeholders in their ``prim_path``
    so that they only exist in the environments assigned to their group.
    Group 2 (reach) has no objects at all.
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

    # -- Franka robot (ALL envs, one ArticulationView) -----------------------
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -- Table (ALL envs) ----------------------------------------------------
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )

    # -- Group 0: Lift – single cube ----------------------------------------
    lift_cube = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP0}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_LIFT_CUBE_SPAWN,
    )

    # -- Group 1: Stack – three cubes ----------------------------------------
    stack_cube_1 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_1_SPAWN,
    )
    stack_cube_2 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_2_SPAWN,
    )
    stack_cube_3 = RigidObjectCfg(
        prim_path="/World/envs/env_{GROUP1}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_3_SPAWN,
    )

    # -- Group 2: Reach – no objects -----------------------------------------
    # (nothing to declare)


# ---------------------------------------------------------------------------
# Actions  (single set – same robot, same action space for every env)
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotActionsCfg:
    """Joint-position actions applied to the shared ``robot`` asset.

    Because every environment uses the same Franka with the same action space,
    only one ``arm_action`` and one ``gripper_action`` are needed.  The action
    terms reference ``asset_name="robot"`` whose ``prim_path`` is
    ``{ENV_REGEX_NS}/Robot`` – this means ``_assigned_envs`` is empty (all
    envs), so the actions are dispatched to every environment.
    """

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ---------------------------------------------------------------------------
# Observations / Rewards / Terminations / Events
# ---------------------------------------------------------------------------


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
    """Events including per-env default joint pose reset for heterogeneous tasks."""

    reset_scene_to_default = EventTerm(func=mdp.reset_multitask_scene_to_default, mode="reset")
    reset_default_joint_pos = EventTerm(
        func=reset_default_joint_pos_per_env,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Flat single-robot multi-task env: one Franka, three tasks.

    No ``MultiTaskRegistryConfig`` -- everything is explicit.

    The Franka robot has ``prim_path="{ENV_REGEX_NS}/Robot"``, so it is
    instantiated in every environment and backed by **one** PhysX
    ``ArticulationView``.  Actions (``arm_action``, ``gripper_action``)
    reference ``asset_name="robot"`` and therefore apply to all envs.

    Per-task objects (cubes) use ``{GROUP<N>}`` placeholder tokens that are
    resolved in ``__post_init__`` into concrete env-id regexes.

    Group tracking attributes set on this cfg instance:

    * ``_group_env_ids``  – ``list[list[int]]`` mapping group index to env IDs.
    * ``_task_id_per_env`` – ``torch.Tensor`` of shape ``(num_envs,)`` with the
      task/group index for each environment.
    * ``_default_joint_pos_per_env`` – ``torch.Tensor`` of shape
      ``(num_envs, num_joints)`` holding the per-env default joint positions
      (stack envs get a different pose than lift/reach envs).

    Group 0 (LIFT):  Franka -- lift 1 cube       (joint-position actions)
    Group 1 (STACK): Franka -- stack 3 cubes     (joint-position actions)
    Group 2 (REACH): Franka -- reach (no objects) (joint-position actions)
    """

    scene: FlatSingleRobotSceneCfg = FlatSingleRobotSceneCfg(
        num_envs=24, env_spacing=2.0, replicate_physics=False
    )
    actions: FlatSingleRobotActionsCfg = FlatSingleRobotActionsCfg()
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

        # --- partition env IDs across tasks ----------------------------------
        groups = _partition_env_ids(self.scene.num_envs, NUM_GROUPS)

        # Expose the partition so any MDP term can look up task membership.
        self._group_env_ids: list[list[int]] = groups

        # Per-env task index tensor (shape: [num_envs]).
        task_ids = torch.zeros(self.scene.num_envs, dtype=torch.long)
        for group_idx, env_ids in enumerate(groups):
            task_ids[env_ids] = group_idx
        self._task_id_per_env: torch.Tensor = task_ids

        # --- per-env default joint positions ---------------------------------
        # The stack task uses a different starting arm pose.
        num_joints = len(_DEFAULT_JOINT_POS)
        default_jpos = torch.zeros(self.scene.num_envs, num_joints, dtype=torch.float32)
        default_jpos[:] = torch.tensor(_DEFAULT_JOINT_POS, dtype=torch.float32)
        default_jpos[groups[TASK_STACK]] = torch.tensor(_STACK_DEFAULT_JOINT_POS, dtype=torch.float32)
        self._default_joint_pos_per_env: torch.Tensor = default_jpos

        # --- resolve {GROUP<N>} placeholders in per-task object prim_paths ---
        for attr_name in vars(self.scene):
            cfg = getattr(self.scene, attr_name)
            prim_path = getattr(cfg, "prim_path", None)
            if prim_path is not None and _GROUP_TOKEN_RE.search(prim_path):
                cfg.prim_path = _resolve_group_tokens(prim_path, groups)
