# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg

from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .action_manager import ActionTerm
    from .command_manager import CommandTerm
    from .manager_base import ManagerTermBase
    from .recorder_manager import RecorderTerm


@configclass
class RobotGroupCfg:
    """Typed metadata for a robot group in heterogeneous multi-robot environments.

    Declares the scene entities and command association for a single
    robot type.  Used as the value type in
    :attr:`~isaaclab.envs.ManagerBasedEnvCfg.robot_meta` so that the
    ``per_robot`` auto-injection mechanism has typed, IDE-discoverable
    fields instead of an opaque ``dict[str, Any]``.

    Only non-``None`` fields are injected into MDP term functions.
    If a task requires additional per-robot metadata beyond these
    common fields, subclass this configclass and add extra fields —
    the injection mechanism iterates all dataclass fields
    automatically.

    Example::

        robot_meta = {
            "franka_robot": RobotGroupCfg(
                asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"]),
                command_name="franka_ee_pose",
            ),
        }
    """

    asset_cfg: SceneEntityCfg = MISSING
    """SceneEntityCfg identifying the robot articulation.

    Typically includes ``body_names`` for the end-effector and
    ``joint_names`` for the arm (and optionally gripper) joints.
    """

    command_name: str | None = None
    """Name of the command term that generates targets for this robot."""

    robot_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the robot articulation root.

    Used by reward/observation functions that need the root pose
    separately from the EE body (e.g. ``object_ee_distance``).
    When ``None``, functions that accept ``robot_cfg`` will not
    receive an auto-injected value.
    """

    object_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the manipulation object (e.g. a cube to lift)."""

    ee_frame_cfg: SceneEntityCfg | None = None
    """SceneEntityCfg for the end-effector FrameTransformer sensor."""


@configclass
class ManagerTermBaseCfg:
    """Configuration for a manager term."""

    func: Callable | ManagerTermBase = MISSING
    """The function or class to be called for the term.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`ManagerTermBase` class
    and implement the required methods.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """

    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """

    task_group: str | None = None
    """Task group name this term belongs to.

    When set, the term is scoped to environments belonging to the named group declared in
    :attr:`~isaaclab.scene.InteractiveSceneCfg.task_groups`.

    The exact semantics depend on the manager:

    * **Reward / Termination / Observation** — the function returns ``(group_envs, ...)`` and the manager
      scatters the result into the full-sized buffer (non-group rows filled with zero / False).
    * **Event** — the manager filters ``env_ids`` to group-local indices before calling the function.

    **Auto-injection**: if the function signature accepts a ``task_group``
    parameter, the value is automatically injected from this field —
    there is no need to duplicate it in :attr:`params`.

    Can be combined with :attr:`per_robot`.  When both are set,
    ``per_robot`` dispatch is filtered to only the robots whose scene
    group matches this task group.
    """

    per_robot: bool = False
    """Automatically dispatch this term once per robot group.

    When ``True``, the manager iterates
    :attr:`ManagerBasedEnvCfg.robot_meta` and calls the function once
    for each robot entry.  Any metadata key that matches a function
    parameter name (and is not already provided in :attr:`params`) is
    auto-injected.  Values of type :class:`SceneEntityCfg` are
    automatically resolved against the scene before injection.

    ``robot_meta`` maps each asset name to a
    :class:`RobotGroupCfg` (or a plain ``dict[str, Any]``).  Common
    keys include ``asset_cfg`` (:class:`SceneEntityCfg`) and
    ``command_name`` (str).

    This allows reuse of standard term functions (e.g.
    :func:`~isaaclab.envs.mdp.rewards.joint_vel_l2`,
    :func:`~isaaclab.envs.mdp.events.reset_joints_by_scale`) without
    writing multi-robot wrapper functions.

    Results are scattered / filtered by the manager — the function
    itself stays layout-unaware.

    Can be combined with :attr:`task_group` to restrict dispatch to
    robots belonging to a specific task group.
    """


##
# Recorder manager.
##


@configclass
class RecorderTermCfg:
    """Configuration for an recorder term."""

    class_type: type[RecorderTerm] = MISSING
    """The associated recorder term class.

    The class should inherit from :class:`isaaclab.managers.recorder_manager.RecorderTerm`.
    """


##
# Action manager.
##


@configclass
class ActionTermCfg:
    """Configuration for an action term."""

    class_type: type[ActionTerm] = MISSING
    """The associated action term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.ActionTerm`.
    """

    asset_name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""


##
# Command manager.
##


@configclass
class CommandTermCfg:
    """Configuration for a command generator term."""

    class_type: type[CommandTerm] = MISSING
    """The associated command term class to use.

    The class should inherit from :class:`isaaclab.managers.command_manager.CommandTerm`.
    """

    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    task_group: str | None = None
    """Task group name this command term belongs to. Defaults to None.

    When set, the command term is automatically scoped to the
    environments belonging to the named group declared in
    :attr:`~isaaclab.scene.InteractiveSceneCfg.task_groups`.

    If ``None``, the term falls back to the layout key of the
    referenced asset (resolved via ``asset_name``).  If the asset also
    covers all environments, the command applies everywhere.
    """


##
# Curriculum manager.
##


@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term."""

    func: Callable[..., float | dict[str, float] | None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input and return the curriculum state for
    logging purposes. If the function returns None, the curriculum state
    is not logged.
    """


##
# Observation manager.
##


@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    modifiers: list[ModifierCfg] | None = None
    """The list of data modifiers to apply to the observation in order. Defaults to None,
    in which case no modifications will be applied.

    Modifiers are applied in the order they are specified in the list. They can be stateless
    or stateful, and can be used to apply transformations to the observation data. For example,
    a modifier can be used to normalize the observation data or to apply a rolling average.

    For more information on modifiers, see the :class:`~isaaclab.utils.modifiers.ModifierCfg` class.
    """

    noise: NoiseCfg | NoiseModelCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""

    scale: tuple[float, ...] | float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`).

    We leverage PyTorch broadcasting to scale the observation tensor with the provided value. If a tuple is provided,
    please make sure the length of the tuple matches the dimensions of the tensor outputted from the term.
    """

    history_length: int = 0
    """Number of past observations to store in the observation buffers. Defaults to 0, meaning no history.

    Observation history initializes to empty, but is filled with the first append after reset or initialization.
    Subsequent history only adds a single entry to the history buffer. If flatten_history_dim is set to True,
    the source data of shape (N, H, D, ...) where N is the batch dimension and H is the history length will
    be reshaped to a 2-D tensor of shape (N, H*D*...). Otherwise, the data will be returned as is.
    """

    flatten_history_dim: bool = True
    """Whether or not the observation manager should flatten history-based observation terms to a 2-D (N, D) tensor.
    Defaults to True."""


@configclass
class ObservationGroupCfg:
    """Configuration for an observation group."""

    concatenate_terms: bool = True
    """Whether to concatenate the observation terms in the group. Defaults to True.

    If true, the observation terms in the group are concatenated along the dimension specified through
    :attr:`concatenate_dim`. Otherwise, they are kept separate and returned as a dictionary.

    If the observation group contains terms of different dimensions, it must be set to False.
    """

    concatenate_dim: int = -1
    """Dimension along to concatenate the different observation terms. Defaults to -1, which
    means the last dimension of the observation terms.

    If :attr:`concatenate_terms` is True, this parameter specifies the dimension along which the observation
    terms are concatenated. The indicated dimension depends on the shape of the observations. For instance,
    for a 2-D RGB image of shape (H, W, C), the dimension 0 means concatenating along the height, 1 along the
    width, and 2 along the channels. The offset due to the batched environment is handled automatically.
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the observation group. Defaults to False.

    If true, the observation terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """

    history_length: int | None = None
    """Number of past observation to store in the observation buffers for all observation terms in group.

    This parameter will override :attr:`ObservationTermCfg.history_length` if set. Defaults to None.
    If None, each terms history will be controlled on a per term basis. See :class:`ObservationTermCfg`
    for details on :attr:`ObservationTermCfg.history_length` implementation.
    """

    flatten_history_dim: bool = True
    """Flag to flatten history-based observation terms to a 2-D (num_env, D) tensor for all observation terms in group.
    Defaults to True.

    This parameter will override all :attr:`ObservationTermCfg.flatten_history_dim` in the group if
    ObservationGroupCfg.history_length is set.
    """


##
# Event manager
##


@configclass
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for a event term."""

    func: Callable[..., None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """

    mode: str = MISSING
    """The mode in which the event term is applied.

    Note:
        The mode name ``"interval"`` is a special mode that is handled by the
        manager Hence, its name is reserved and cannot be used for other modes.
    """

    interval_range_s: tuple[float, float] | None = None
    """The range of time in seconds at which the term is applied. Defaults to None.

    Based on this, the interval is sampled uniformly between the specified
    range for each environment instance. The term is applied on the environment
    instances where the current time hits the interval time.

    Note:
        This is only used if the mode is ``"interval"``.
    """

    is_global_time: bool = False
    """Whether randomization should be tracked on a per-environment basis. Defaults to False.

    If True, the same interval time is used for all the environment instances.
    If False, the interval time is sampled independently for each environment instance
    and the term is applied when the current time hits the interval time for that instance.

    Note:
        This is only used if the mode is ``"interval"``.
    """

    min_step_count_between_reset: int = 0
    """The number of environment steps after which the term is applied since its last application. Defaults to 0.

    When the mode is "reset", the term is only applied if the number of environment steps since
    its last application exceeds this quantity. This helps to avoid calling the term too often,
    thereby improving performance.

    If the value is zero, the term is applied on every call to the manager with the mode "reset".

    Note:
        This is only used if the mode is ``"reset"``.
    """


##
# Reward manager.
##


@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """


##
# Termination manager.
##


@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the termination signals as torch boolean tensors of
    shape (num_envs,).
    """

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """
