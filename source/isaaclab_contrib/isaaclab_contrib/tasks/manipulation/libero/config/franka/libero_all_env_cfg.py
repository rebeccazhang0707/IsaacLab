# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combined multi-suite LIBERO env with object-level prototype sharing.

Any subset of the four suites (``libero_spatial`` / ``libero_goal`` /
``libero_object`` / ``libero_long``, 10 tasks each) can be trained together in
one scene via :func:`make_libero_combined_env_cfg`.  ``Isaac-Libero-All-*`` uses
all 40 tasks; :data:`SUITE_SPECS` names the selectable suites and a couple of
representative subset combos are registered below.

Identical object models are harvested into shared prototypes
(:mod:`...tasks.harvest.prototypes`) so each distinct USD is spawned once and cloned -- as
a single :class:`~isaaclab.assets.AssetView` -- into the union of the envs whose
task uses it (e.g. ``alphabet_soup`` is one entity shared by every selected task
that uses it).

The scene, Franka robot, differential-IK + gripper actions, robot observations,
and penalty curriculum are assembled by the shared
:class:`~...multitask.registry.MultiTaskRegistry` (one registration per selected
task).  Because a shared AssetView spans envs of different tasks, all
*task-dependent* MDP is supplied per env by task id via :mod:`...mdp.combined`: a
deterministic :func:`~isaaclab.cloner.sequential` clone map assigns env ``i`` to
the ``i % n_tasks``-th selected task, driving the per-env ``task_onehot`` (width =
number of selected tasks), object observations, the three reward modes, the
success termination, and the per-task reset.

Constraint: ``num_envs`` must be >= the number of selected tasks (a multiple gives
an even per-task split).  USD assets load from ``LIBERO_ASSETS_DATA_DIR`` and
per-task layouts from ``LIBERO_CONFIG_DIR`` (or the ``config`` dir sibling to the
asset dir).
"""

from __future__ import annotations

import warnings

from isaaclab.cloner import sequential
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.tasks.manipulation.multitask.config.demo.physics_cfg import MultitaskPhysicsCfg
from isaaclab_contrib.tasks.manipulation.multitask.registry import MultiTaskRegistry

from ... import mdp
from ...robots import FRANKA
from ...tasks.common.base import resolve_reward_mode
from ...tasks.harvest.combined import build_combined_tasks

# Selectable suites: short name -> ``(suite_name, per-task scene-key prefix)``.
SUITE_SPECS: dict[str, tuple[str, str]] = {
    "spatial": ("libero_spatial", "spatial"),
    "goal": ("libero_goal", "goal"),
    "object": ("libero_object", "object"),
    "long": ("libero_long", "libero_long"),
}
_ALL_SUITES: tuple[str, ...] = ("spatial", "goal", "object", "long")
_DEFAULT_REWARD_MODE = "goal"

# Default object-position jitter [m] applied on top of each task's authored pose.
_OBJECT_JITTER = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}
# Reward weights (mirror the per-suite LiberoManipulationTaskCfg defaults).
_SUCCESS_WEIGHT = 10.0
_REACH_WEIGHT = 1.0
_LIFT_WEIGHT = 5.0
_PLACE_WEIGHT = 3.0
_SPEED_THRESHOLD = 0.4


def _env_spacing_for(suites: tuple[str, ...]) -> float:
    """Grid spacing [m] sized to the largest fixture footprint in the selected suites.

    The ``long`` suite includes ``study_table`` (~2.8 m in Y), so it needs a
    wider grid than the 2.5 m that fits the (shrunk) floor and kitchen table.
    """
    return 3.0 if "long" in suites else 2.5


def _augment(cfg, reward_mode: str, bindings, object_proto_names: list[str], num_tasks: int) -> None:
    """Replace the registry's per-group task MDP with per-env task-gather terms."""
    # Object-level prototype sharing needs the deterministic env->task map.
    cfg.scene.clone_cfg.clone_strategy = sequential

    # ── Observations: per-env task one-hot + per-prototype object positions ──
    policy = cfg.observations.policy
    policy.task_onehot = ObsTerm(func=mdp.libero_task_onehot, params={"num_tasks": num_tasks})
    policy.libero_objects = ObsTerm(func=mdp.libero_object_positions, params={"proto_names": object_proto_names})

    # ── Rewards: sparse success always; dense shaping for metaworld_dense ────
    mode = resolve_reward_mode(reward_mode)
    cfg.rewards.libero_success = RewTerm(
        func=mdp.libero_success_bonus,
        weight=_SUCCESS_WEIGHT,
        params={"tasks": bindings, "speed_threshold": _SPEED_THRESHOLD},
    )
    if mode == "metaworld_dense":
        cfg.rewards.libero_reach = RewTerm(
            func=mdp.libero_reach_reward, weight=_REACH_WEIGHT, params={"tasks": bindings, "std": 0.1}
        )
        cfg.rewards.libero_lift = RewTerm(func=mdp.libero_lift_reward, weight=_LIFT_WEIGHT, params={"tasks": bindings})
        cfg.rewards.libero_place = RewTerm(
            func=mdp.libero_place_reward, weight=_PLACE_WEIGHT, params={"tasks": bindings, "std": 0.1}
        )
    elif mode == "world_prediction":
        warnings.warn(
            "LIBERO reward mode 'world_prediction' needs a per-step demo reference trajectory, which is "
            "unavailable in isaaclab_contrib; only the sparse success bonus is used. Use "
            "LIBERO_REWARD_MODE=metaworld_dense for a demo-free dense reward.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Terminations: per-env success + object dropped ──────────────────────
    cfg.terminations.libero_success = DoneTerm(
        func=mdp.libero_task_success, params={"tasks": bindings, "speed_threshold": _SPEED_THRESHOLD}
    )
    cfg.terminations.libero_object_dropped = DoneTerm(func=mdp.libero_object_dropped, params={"tasks": bindings})

    # ── Events: per-task object reset (writes each task's authored poses) ────
    cfg.events.libero_reset_objects = EventTerm(
        func=mdp.reset_libero_prototypes,
        mode="reset",
        params={"tasks": bindings, "pose_range": _OBJECT_JITTER},
    )


def make_libero_combined_env_cfg(
    suites: tuple[str, ...] = _ALL_SUITES,
    num_envs: int = 2560,
    reward_mode: str = _DEFAULT_REWARD_MODE,
) -> type:
    """Build a combined env cfg class from an arbitrary subset of LIBERO suites.

    Args:
        suites: Short suite names to include (subset of :data:`SUITE_SPECS`), in
            task-index order.
        num_envs: Total envs; must be >= the number of selected tasks (a multiple
            gives an even per-task split under the sequential clone map).
        reward_mode: Default reward mode; ``LIBERO_REWARD_MODE`` overrides it.

    Returns:
        A :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` subclass.

    Raises:
        KeyError: If a suite name is not in :data:`SUITE_SPECS`.
        ValueError: If ``num_envs`` is smaller than the number of selected tasks.
    """
    specs = tuple(SUITE_SPECS[name] for name in suites)
    tasks, prototypes, bindings = build_combined_tasks(specs)
    num_tasks = len(tasks)
    if num_envs < num_tasks:
        raise ValueError(f"num_envs ({num_envs}) must be >= number of selected tasks ({num_tasks}).")
    object_proto_names = [proto.name for proto in prototypes if not proto.is_fixture]

    registry = MultiTaskRegistry()
    for task in tasks:
        registry.register(FRANKA, task, group_name=task.name)

    base_cls = registry.build_env_cfg(
        num_envs=num_envs,
        env_spacing=_env_spacing_for(suites),
        replicate_physics=True,
        physics=MultitaskPhysicsCfg(),
        decimation=2,
        episode_length_s=8.0,
        sim_dt=1.0 / 60.0,
    )
    parent_post_init = base_cls.__post_init__

    def _post_init(self):
        parent_post_init(self)
        _augment(self, reward_mode, bindings, object_proto_names, num_tasks)

    return configclass(type("_LiberoComboEnvCfg", (base_cls,), {"__post_init__": _post_init}))


def make_libero_combined_play_cfg(suites: tuple[str, ...] = _ALL_SUITES) -> type:
    """Return a play-mode combo cfg class (one env per selected task, no corruption)."""
    num_tasks = len(suites) * 10
    base_cls = make_libero_combined_env_cfg(suites=suites, num_envs=num_tasks)
    parent_post_init = base_cls.__post_init__

    def _play_post_init(self):
        parent_post_init(self)
        self.observations.policy.enable_corruption = False

    return configclass(type("_LiberoComboPlayEnvCfg", (base_cls,), {"__post_init__": _play_post_init}))


# ── Registered combos ───────────────────────────────────────────────────────
LiberoAllEnvCfg = make_libero_combined_env_cfg(suites=_ALL_SUITES, num_envs=2560)
"""Combined 40-task LIBERO env cfg (all four suites, 35 shared prototypes)."""

LiberoAllEnvCfg_PLAY = make_libero_combined_play_cfg(suites=_ALL_SUITES)
"""Play variant of the 40-task env (one env per task, no observation corruption)."""

LiberoObjectLongEnvCfg = make_libero_combined_env_cfg(suites=("object", "long"), num_envs=1280)
"""Cross-suite combo: ``libero_object`` + ``libero_long`` (20 tasks)."""

LiberoObjectLongEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("object", "long"))
"""Play variant of the object+long combo (one env per task, no corruption)."""

LiberoSpatialGoalEnvCfg = make_libero_combined_env_cfg(suites=("spatial", "goal"), num_envs=2560)
"""Cross-suite combo: ``libero_spatial`` + ``libero_goal`` (20 tasks)."""

LiberoSpatialGoalEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("spatial", "goal"))
"""Play variant of the spatial+goal combo (one env per task, no corruption)."""

# ── Single-suite harvest envs ────────────────────────────────────────────────
# Built through the same factory (object-level prototype sharing + per-env task
# gather); relational / articulation goals (e.g. ``libero_goal`` open-cabinet /
# turn-on-stove tasks with no manipulated rigid object) are handled per env by
# task id, so there is no per-suite ``primary_key=None`` special case.
LiberoSpatialEnvCfg = make_libero_combined_env_cfg(suites=("spatial",), num_envs=2560)
"""Spatial suite alone (10 tasks) via the harvest+selector factory."""

LiberoSpatialEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("spatial",))
"""Play variant of the spatial-only env (one env per task, no observation corruption)."""

LiberoGoalEnvCfg = make_libero_combined_env_cfg(suites=("goal",), num_envs=2560)
"""Goal suite alone (10 tasks) via the harvest+selector factory."""

LiberoGoalEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("goal",))
"""Play variant of the goal-only env (one env per task, no observation corruption)."""

LiberoObjectEnvCfg = make_libero_combined_env_cfg(suites=("object",), num_envs=2560)
"""Object suite alone (10 tasks) via the harvest+selector factory."""

LiberoObjectEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("object",))
"""Play variant of the object-only env (one env per task, no observation corruption)."""

LiberoLongEnvCfg = make_libero_combined_env_cfg(suites=("long",), num_envs=2560)
"""Long-horizon suite alone (10 tasks) via the harvest+selector factory."""

LiberoLongEnvCfg_PLAY = make_libero_combined_play_cfg(suites=("long",))
"""Play variant of the long-only env (one env per task, no observation corruption)."""
