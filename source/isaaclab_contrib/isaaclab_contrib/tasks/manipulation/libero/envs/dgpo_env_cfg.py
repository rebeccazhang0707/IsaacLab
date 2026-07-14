# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a harvest+CloneCfg LIBERO env with DGPO observation / action layout.

This is the **first-class DGPO training path** on the harvest / cloner API
(not a temporary checkpoint-only bridge).

Goals
-----

1. Keep the efficient harvest scene (``CloneCfg`` / ``InclusionSet`` / shared
   prototypes).
2. Use OSC so ``action_dim == 7`` (pose_rel + binary gripper).
3. Install observation groups whose concatenated widths match actor=324 /
   critic=572 (DGPO layout in :mod:`~...dgpo_layout`).
4. When demos are available (``LIBERO_ASSEMBLED_DATASET_DIR``), install
   :class:`~...mdp.demos.commands.SourceLiberoCommand` so privileged
   EE/joint/object diffs are real, and reset objects/robot from demo
   ``initial_state``.

Quaternion order
----------------

* Sim math is always XYZW (Isaac Lab).
* ``policy_quat_order`` (default ``wxyz``) converts EE pose terms at the
  observation boundary — set to ``xyzw`` for a native Isaac Lab 3.x policy.
* ``demo_quat_order`` (default ``wxyz``) converts demo ``obs/ee_states`` on load.

Gym ids: ``Isaac-Libero-All-Dgpo-Osc-v0`` (train) and
``Isaac-Libero-All-Dgpo-Osc-Play-v0`` plus suite-subset ``*-Play-v0`` variants
(see :mod:`~...config.franka`).

Environment variables
---------------------

* ``LIBERO_ASSETS_DATA_DIR`` / ``LIBERO_CONFIG_DIR`` — USD + per-task JSON
  (required to build the harvest scene).
* ``LIBERO_ASSEMBLED_DATASET_DIR`` — HDF5 demo root
  (``{suite}_task{id}_*_demo.hdf5``). Defaults try RobotLearningLab
  ``sim2sim_dataset`` if present.
* ``LIBERO_COMPAT_REQUIRE_DEMOS=1`` — fail hard when demos are missing.
* ``LIBERO_COMPAT_MAX_DEMOS_PER_TASK`` — optional positive int; load at most
  N demos per task (cuts HDF5 + pose-cache cost for play/debug).
* ``LIBERO_EVALUATION=1`` — disable random demo start-timestep sampling.
* ``ROBOT_INIT_NOISE_STD`` — optional Gaussian noise on demo robot arm joints
  at reset (default ``0.0`` for eval; playground often uses ``0.05``).
* ``DGPO_ABC_CHECKPOINT`` — optional path for dim-contract checkpoint tests.

Train from scratch (harvest + OSC + DGPO obs)::

    export LIBERO_ASSETS_DATA_DIR=/path/to/libero/USD
    export LIBERO_CONFIG_DIR=/path/to/libero/config
    # optional demos for privileged critic / demo reset:
    export LIBERO_ASSEMBLED_DATASET_DIR=/path/to/sim2sim_dataset
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Libero-All-Dgpo-Osc-v0 --num_envs 2560 presets=physx

Eval smoke (once assets + demos are mounted)::

    export LIBERO_EVALUATION=1
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Libero-All-Dgpo-Osc-Play-v0 --num_envs 40 \\
        --checkpoint /path/to/model.pt presets=physx --headless
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as env_mdp
from isaaclab.cloner import sequential
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask.config.demo.physics_cfg import MultitaskPhysicsCfg
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry

from ..dgpo_layout import (
    DGPO_ABC_DECIMATION,
    DGPO_ABC_EPISODE_LENGTH_S,
    DGPO_ABC_HARVEST_SUITES,
    DGPO_ABC_SIM_DT,
    dgpo_abc_contract,
    verify_contract_arithmetic,
)
from ..mdp import observations as dgpo_mdp  # DGPO obs terms
from ..mdp.combined import libero_task_success, reset_libero_prototypes
from ..mdp.demos import events as dgpo_events
from ..mdp.demos.commands import make_dgpo_commands_cfg
from ..mdp.quat import DEFAULT_DEMO_QUAT_ORDER, DEFAULT_POLICY_QUAT_ORDER, QuatOrder
from ..robots.franka_osc import FRANKA_OSC
from ..tasks.harvest.combined import build_combined_tasks

# Match harvest All env geometric success proxy (object near rest).
_SPEED_THRESHOLD = 0.4


@configclass
class DgpoPolicyCfg(ObsGroup):
    """Policy group: last_action + multi-hot + pose buffer → 277."""

    last_action = ObsTerm(func=env_mdp.last_action)
    task_multi_hot = ObsTerm(
        func=dgpo_mdp.task_assignment_multi_hot_encoding,
        params={
            "full_sequence": True,
            "overlap_tags": {
                "libero_10::0": ("alphabet_soup_to_basket", "tomato_sauce_to_basket"),
                "libero_10::1": ("cream_cheese_to_basket", "butter_to_basket"),
                "libero_10::7": ("alphabet_soup_to_basket", "cream_cheese_to_basket"),
                "libero_10::2": ("turn_on_flat_stove", "moka_pot_on_stove"),
                "libero_10::8": ("moka_pot_on_stove",),
                "libero_object::0": ("alphabet_soup_to_basket",),
                "libero_object::5": ("tomato_sauce_to_basket",),
                "libero_object::1": ("cream_cheese_to_basket",),
                "libero_object::6": ("butter_to_basket",),
                "libero_goal::7": ("turn_on_flat_stove",),
            },
            "shared_subtask_order": (
                "alphabet_soup_to_basket",
                "tomato_sauce_to_basket",
                "cream_cheese_to_basket",
                "butter_to_basket",
                "turn_on_flat_stove",
                "moka_pot_on_stove",
            ),
        },
    )
    buffer_pose = ObsTerm(
        func=dgpo_mdp.ObjectTargetPoseBuffer,
        params={
            "include_objects": True,
            "include_targets": True,
            "use_base_frame": True,
            "focused_only": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoProprioCfg(ObsGroup):
    """Proprio group including gripper force → 47 (DGPO actor path)."""

    eef_pose = ObsTerm(func=dgpo_mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=dgpo_mdp.gripper_pos)
    joint_pos = ObsTerm(
        func=env_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=env_mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"])},
    )
    gripper_force = ObsTerm(
        func=dgpo_mdp.contact_force_in_world_frame,
        params={"contact_sensor_name": "contact_gripper", "history_length": 4},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoPrivilegedProprioCfg(ObsGroup):
    """Privileged critic group without gripper force → 248.

    EE/joint/object diffs are real when :class:`~.commands.SourceLiberoCommand`
    is installed; otherwise zeros with locked widths.
    """

    eef_pose_diff = ObsTerm(
        func=dgpo_mdp.ee_frame_pose_diff_in_base_frame,
        params={"command_name": "ee_pose"},
    )
    joint_pos_diff = ObsTerm(
        func=dgpo_mdp.joint_pos_diff,
        params={
            "command_name": "joint_state",
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"]),
        },
    )
    object_target_pose_diff = ObsTerm(
        func=dgpo_mdp.ObjectTargetPoseDiff,
        params={
            "command_name": "source_action",
            "include_objects": True,
            "include_targets": True,
            "use_base_frame": True,
            "focused_only": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DgpoObservationsCfg:
    """Observation groups matching DGPO ``agent.yaml`` ``obs_groups``."""

    policy: DgpoPolicyCfg = DgpoPolicyCfg()
    proprio: DgpoProprioCfg = DgpoProprioCfg()
    privileged_proprio: DgpoPrivilegedProprioCfg = DgpoPrivilegedProprioCfg()


# Backward-compatible aliases (gym / tests / launch may still say Compat*).
CompatPolicyCfg = DgpoPolicyCfg
CompatProprioCfg = DgpoProprioCfg
CompatPrivilegedProprioCfg = DgpoPrivilegedProprioCfg
CompatObservationsCfg = DgpoObservationsCfg


def _env_spacing_for(suites: tuple[tuple[str, str], ...]) -> float:
    return 3.0 if any(name == "libero_long" for name, _ in suites) else 2.5


def _install_single_franka_osc_actions(env_cfg) -> None:
    """Replace registry ScatteredActionTerm duplicates with one OSC + gripper.

    The factory registers FRANKA_OSC once per LIBERO task; the shared
    ``franka_robot`` then yields N identical OSC/gripper sub-terms. Install a
    single ``OperationalSpaceControllerAction`` + binary gripper instead.
    """
    specs = FRANKA_OSC.action_specs()
    # Preserve arm → gripper column order (6 + 1 = 7).
    env_cfg.actions.arm = specs["arm"][1]
    env_cfg.actions.gripper = specs["gripper"][1]


def make_libero_dgpo_env_cfg(
    suites: tuple[tuple[str, str], ...] = DGPO_ABC_HARVEST_SUITES,
    num_envs: int = 40,
    *,
    policy_quat_order: QuatOrder = DEFAULT_POLICY_QUAT_ORDER,
    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER,
) -> type:
    """Assemble harvest scene + OSC + DGPO observation groups (primary training path).

    Args:
        suites: ``(suite_name, prefix)`` pairs in task-index order.  Default is
            :data:`~...dgpo_layout.DGPO_ABC_HARVEST_SUITES` (libero_10 / object /
            spatial / goal) so sequential ``env i → task i % n`` matches the
            DGPO assignment order.
        num_envs: Number of parallel environments (must be >= number of tasks).
        policy_quat_order: Quat order emitted in EE pose / EE pose-diff obs
            (default ``wxyz`` for DGPO; use ``xyzw`` for native Isaac Lab 3.x).
        demo_quat_order: Quat order of demo ``obs/ee_states`` before conversion
            to sim XYZW (default ``wxyz``).

    Returns:
        A :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` **subclass** with CloneCfg
        harvest layout, OSC actions (dim 7), and obs groups sized for
        actor=324 / critic=572.
    """
    if policy_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"policy_quat_order must be 'wxyz' or 'xyzw', got {policy_quat_order!r}.")
    if demo_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"demo_quat_order must be 'wxyz' or 'xyzw', got {demo_quat_order!r}.")

    verify_contract_arithmetic()
    contract = dgpo_abc_contract()

    task_cfgs, prototypes, bindings = build_combined_tasks(suites)
    if num_envs < len(task_cfgs):
        raise ValueError(f"num_envs ({num_envs}) must be >= number of tasks ({len(task_cfgs)}).")

    registry = MultiTaskRegistry()
    for task_cfg in task_cfgs:
        registry.register(FRANKA_OSC, task_cfg, group_name=task_cfg.name)

    base_cls = registry.build_env_cfg(
        num_envs=num_envs,
        env_spacing=_env_spacing_for(suites),
        replicate_physics=True,
        physics=MultitaskPhysicsCfg(),
        decimation=DGPO_ABC_DECIMATION,
        episode_length_s=DGPO_ABC_EPISODE_LENGTH_S,
        sim_dt=DGPO_ABC_SIM_DT,
    )
    parent_post_init = base_cls.__post_init__
    commands_cfg = make_dgpo_commands_cfg(bindings, num_envs, demo_quat_order=demo_quat_order)
    demos_wired = commands_cfg is not None and commands_cfg.libero_demo is not None
    meta = {
        "name": contract.name,
        "action_dim": contract.action_dim,
        "actor_obs_dim": contract.actor_obs_dim,
        "critic_obs_dim": contract.critic_obs_dim,
        "policy_dim": contract.policy.dim,
        "proprio_dim": contract.proprio.dim,
        "privileged_proprio_dim": contract.privileged_proprio.dim,
        "num_prototypes": len(prototypes),
        "num_tasks": len(task_cfgs),
        "policy_quat_order": policy_quat_order,
        "demo_quat_order": demo_quat_order,
        "mdp_status": {
            "multi_hot": "real",
            "buffer_pose": "real",
            "proprio": "real",
            "gripper_force": "real_if_contact_sensor",
            "privileged_ee_joint_diff": "real_if_demos" if demos_wired else "zero_until_demos",
            "object_target_pose_diff": "real_if_demos" if demos_wired else "zero_until_demos",
            "demo_initial_state_reset": "real_if_demos" if demos_wired else "authored_prototypes",
            "demos_wired": demos_wired,
        },
    }

    def _post_init(self):
        parent_post_init(self)
        # Deterministic env→task map (same as harvest All env).
        self.scene.clone_cfg.clone_strategy = sequential
        # Match DGPO control rate (sim.dt * decimation = 0.05 s).
        self.decimation = DGPO_ABC_DECIMATION
        self.sim.dt = DGPO_ABC_SIM_DT
        self.sim.render_interval = DGPO_ABC_DECIMATION
        self.episode_length_s = DGPO_ABC_EPISODE_LENGTH_S
        # One OSC + gripper on the shared Franka (not N scattered copies).
        _install_single_franka_osc_actions(self)
        self.observations = DgpoObservationsCfg()
        if commands_cfg is not None:
            self.commands = commands_cfg
        # First-class quat-order flags (obs / demo boundaries).
        self.policy_quat_order = policy_quat_order
        self.demo_quat_order = demo_quat_order
        self.dgpo_layout = meta
        self.dgpo_task_bindings = bindings
        # Short-lived aliases for older call sites.
        self.compat_dim_contract = meta
        self.compat_task_bindings = bindings

        # Harvest prototype tasks leave termination_terms empty; registry only
        # installs time_out. Install geometric success so success ends the
        # episode and ManagerBasedRLEnv.step resets (play/eval + training).
        self.terminations.libero_success = DoneTerm(
            func=libero_task_success,
            params={"tasks": bindings, "speed_threshold": _SPEED_THRESHOLD},
        )

        # Reset path: demo initial_state when traj bank is present; else authored poses.
        if demos_wired:
            # Drop per-group object jitter so demo poses are not overwritten.
            for attr_name in list(vars(self.events).keys()):
                if attr_name.endswith("_reset_objects_default") or attr_name.endswith("_reset_primary_uniform"):
                    setattr(self.events, attr_name, None)
            noise = float(os.getenv("ROBOT_INIT_NOISE_STD", "0.0"))
            self.events.libero_demo_reset = EventTerm(
                func=dgpo_events.reset_libero_scene_to_demo_initial_state,
                mode="reset",
                params={
                    "command_name": "source_action",
                    "include_articulations": True,
                    "include_rigid_objects": True,
                    "robot_joint_noise_std": noise,
                },
            )
        else:
            self.events.libero_reset_objects = EventTerm(
                func=reset_libero_prototypes,
                mode="reset",
                params={"tasks": bindings, "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)}},
            )

    return configclass(type("_LiberoDgpoOscEnvCfg", (base_cls,), {"__post_init__": _post_init}))


def make_libero_dgpo_play_cfg(
    suites: tuple[tuple[str, str], ...] = DGPO_ABC_HARVEST_SUITES,
    *,
    policy_quat_order: QuatOrder = DEFAULT_POLICY_QUAT_ORDER,
    demo_quat_order: QuatOrder = DEFAULT_DEMO_QUAT_ORDER,
) -> type:
    """Play/eval variant: one env per task, no observation corruption."""
    num_tasks = sum(10 for _ in suites)  # each suite contributes 10 tasks
    # Prefer exact count from harvest when configs are available.
    try:
        task_cfgs, _, _ = build_combined_tasks(suites)
        num_tasks = len(task_cfgs)
    except FileNotFoundError:
        pass
    base_cls = make_libero_dgpo_env_cfg(
        suites=suites,
        num_envs=num_tasks,
        policy_quat_order=policy_quat_order,
        demo_quat_order=demo_quat_order,
    )
    parent_post_init = base_cls.__post_init__

    def _post_init(self):
        parent_post_init(self)
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
        self.observations.privileged_proprio.enable_corruption = False

    return configclass(type("_LiberoDgpoOscPlayEnvCfg", (base_cls,), {"__post_init__": _post_init}))


# Stable aliases used by gym registration / older docs.
make_libero_compat_env_cfg = make_libero_dgpo_env_cfg
make_libero_compat_play_cfg = make_libero_dgpo_play_cfg
