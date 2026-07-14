# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-conditioned reset events for the LIBERO harvest + DGPO training path.

On reset, objects / articulations / robot joints are restored from the demo
``initial_state`` of the trajectory reserved by
:class:`~.commands.SourceLiberoCommand` (same traj the privileged diffs track).

Asset names in the HDF5 bank are playground-canonical (``alphabet_soup_1``,
``robot``). Harvest scenes use shared prototype names (``akita_black_bowl``,
``franka_robot``). Mapping uses each env's :class:`TaskBinding` plus
``selector.filter_reset_ids`` for view-local writes.

Demo poses are loaded already translated by harvest ``workspace_shift``
(``ROBOT_BASE_KITCHEN - task_robot_base``) inside the command bank so they
match raised fixtures; this module only adds ``env_origins`` for world writes.

Quaternion order: assembled demos store ``root_pose`` as WXYZ. Legacy HDF5
loads convert them to XYZW via :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler`
before this module writes through ``write_root_pose_to_sim_index``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .commands import SourceLiberoCommand, TrajectoryState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ...tasks.harvest.prototypes import TaskBinding


logger = logging.getLogger(__name__)

_ROBOT_DEMO_NAMES = frozenset({"robot", "franka_robot"})
_ROBOT_SCENE_NAME = "franka_robot"
# Env-local z below this after workspace shift usually means the demo pose was
# not aligned with harvest kitchen-base normalisation (objects sit on the floor).
_SUSPICIOUS_ENV_LOCAL_Z_M = 0.05


def canonical_demo_asset_name(asset_name: str) -> str:
    """Strip playground group suffixes / ``object_`` prefix from a demo asset key."""
    base_name = asset_name.split("/", 1)[0]
    base_name = re.sub(r"_group_\d+$", "", base_name)
    if base_name.startswith("object_"):
        base_name = base_name[len("object_") :]
    return base_name


def build_canonical_to_proto_map(binding: TaskBinding) -> dict[str, str]:
    """Map demo-canonical object names → harvest prototype scene names for one task."""
    out: dict[str, str] = {}
    for obj in binding.object_bindings:
        out[obj.canonical_name] = obj.proto_name
        # Also accept already-canonicalised / stripped keys.
        out[canonical_demo_asset_name(obj.canonical_name)] = obj.proto_name
    return out


def resolve_demo_asset_to_scene(
    asset_name: str,
    *,
    entity_type: str,
    canonical_to_proto: dict[str, str],
) -> str | None:
    """Resolve a demo ``initial_state`` asset key to a harvest scene entity name.

    Args:
        asset_name: Key under ``initial_state/{rigid_object|articulation}/``.
        entity_type: ``\"rigid_object\"`` or ``\"articulation\"``.
        canonical_to_proto: Per-task map from :func:`build_canonical_to_proto_map`.

    Returns:
        Scene entity name, or ``None`` if the asset is not present in this task.
    """
    if entity_type == "articulation" and asset_name in _ROBOT_DEMO_NAMES:
        return _ROBOT_SCENE_NAME
    canonical = canonical_demo_asset_name(asset_name)
    if asset_name in canonical_to_proto:
        return canonical_to_proto[asset_name]
    if canonical in canonical_to_proto:
        return canonical_to_proto[canonical]
    # Direct hit when demos already store harvest prototype names.
    if asset_name in canonical_to_proto.values():
        return asset_name
    return None


def _resolve_env_ids(env_ids_arg, total_envs: int, device: torch.device) -> torch.Tensor:
    if env_ids_arg is None:
        return torch.arange(total_envs, device=device, dtype=torch.long)
    if isinstance(env_ids_arg, slice):
        start, stop, step = env_ids_arg.indices(total_envs)
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    if torch.is_tensor(env_ids_arg):
        return env_ids_arg.to(device=device, dtype=torch.long)
    if hasattr(env_ids_arg, "__iter__"):
        return torch.tensor([int(v) for v in env_ids_arg], device=device, dtype=torch.long)
    return torch.tensor([int(env_ids_arg)], device=device, dtype=torch.long)


def _task_bindings(env: ManagerBasedEnv) -> list[TaskBinding]:
    bindings = getattr(env.cfg, "dgpo_task_bindings", None)
    if bindings is None:
        raise RuntimeError("Demo reset requires env.cfg.dgpo_task_bindings (set by make_libero_dgpo_env_cfg).")
    return bindings


def _squeeze_pose(root_pose: torch.Tensor) -> torch.Tensor:
    pose = root_pose
    if pose.ndim >= 2 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim > 1:
        pose = pose.reshape(-1)
    return pose


def _resolve_robot_arm_gripper_ids(env: ManagerBasedEnv, articulation) -> tuple[list[int], list[int]]:
    """Resolve Franka arm / gripper joint ids for demo joint writes."""
    arm_joint_ids: list[int] = []
    gripper_joint_ids: list[int] = []

    action_manager = getattr(env, "action_manager", None)
    action_terms = getattr(action_manager, "_terms", None)
    if action_terms is not None:
        arm_action = action_terms.get("arm") or action_terms.get("arm_action")
        runtime_arm = getattr(arm_action, "_joint_ids", None) if arm_action is not None else None
        if runtime_arm is not None and not isinstance(runtime_arm, slice):
            arm_joint_ids = [int(j) for j in runtime_arm]
        gripper_action = action_terms.get("gripper") or action_terms.get("gripper_action")
        runtime_grip = getattr(gripper_action, "_joint_ids", None) if gripper_action is not None else None
        if runtime_grip is not None and not isinstance(runtime_grip, slice):
            gripper_joint_ids = [int(j) for j in runtime_grip]

    if not arm_joint_ids:
        found, _ = articulation.find_joints(["panda_joint.*"], preserve_order=True)
        arm_joint_ids = [int(j) for j in found]
    if not gripper_joint_ids:
        found, _ = articulation.find_joints(["panda_finger.*"], preserve_order=True)
        gripper_joint_ids = [int(j) for j in found]
    return arm_joint_ids, gripper_joint_ids


def _write_root_pose_velocity(
    env: ManagerBasedEnv,
    asset_name: str,
    env_idx: int,
    root_pose: torch.Tensor | None,
    *,
    zero_velocity: bool = True,
) -> None:
    selector = env.scene.selector
    env_tensor = torch.tensor([env_idx], dtype=torch.long, device=env.device)
    global_ids, view_ids = selector.filter_reset_ids(asset_name, env_tensor)
    if global_ids.numel() == 0 or asset_name not in env.scene.keys():
        return
    asset = env.scene[asset_name]
    origin = env.scene.env_origins[global_ids]
    if root_pose is not None:
        pose = _squeeze_pose(root_pose.to(device=env.device).float()).unsqueeze(0).clone()
        if pose.shape[-1] < 7:
            return
        pose = pose[:, :7].clone()
        # Demo poses are env-local (harvest kitchen frame after command-bank shift).
        env_local_z = float(pose[0, 2].item())
        if env_local_z < _SUSPICIOUS_ENV_LOCAL_Z_M:
            logger.warning(
                "Demo reset wrote %s env=%d with env-local z=%.4f m (< %.2f). "
                "Check workspace_shift alignment (objects may fall to the floor).",
                asset_name,
                env_idx,
                env_local_z,
                _SUSPICIOUS_ENV_LOCAL_Z_M,
            )
        pose[:, :3] += origin
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_ids)
    if zero_velocity:
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((view_ids.shape[0], 6), device=env.device), env_ids=view_ids
        )


def _write_articulation_joints(
    env: ManagerBasedEnv,
    asset_name: str,
    env_idx: int,
    joint_position: torch.Tensor | None,
    joint_velocity: torch.Tensor | None,
    *,
    is_robot: bool,
    robot_joint_noise_std: float,
) -> None:
    if joint_position is None:
        return
    selector = env.scene.selector
    env_tensor = torch.tensor([env_idx], dtype=torch.long, device=env.device)
    global_ids, view_ids = selector.filter_reset_ids(asset_name, env_tensor)
    if global_ids.numel() == 0 or asset_name not in env.scene.keys():
        return
    asset = env.scene[asset_name]
    jp = joint_position.to(device=env.device).float()
    jv = joint_velocity.to(device=env.device).float() if joint_velocity is not None else torch.zeros_like(jp)
    if jp.ndim == 1:
        jp = jp.unsqueeze(0)
    if jv.ndim == 1:
        jv = jv.unsqueeze(0)

    if is_robot:
        arm_ids, grip_ids = _resolve_robot_arm_gripper_ids(env, asset)
        arm_dim = len(arm_ids)
        if arm_dim == 0 or jp.shape[-1] < arm_dim:
            return
        arm_pos = jp[..., :arm_dim].clone()
        arm_vel = torch.zeros_like(arm_pos)
        if robot_joint_noise_std > 0.0:
            arm_pos = arm_pos + torch.randn_like(arm_pos) * robot_joint_noise_std
            import warp as wp

            soft = asset.data.soft_joint_pos_limits
            soft_t = soft if isinstance(soft, torch.Tensor) else wp.to_torch(soft)
            lim = soft_t[view_ids][:, arm_ids]
            arm_pos = arm_pos.clamp_(lim[..., 0], lim[..., 1])
        asset.write_joint_position_to_sim_index(position=arm_pos, joint_ids=arm_ids, env_ids=view_ids)
        asset.write_joint_velocity_to_sim_index(velocity=arm_vel, joint_ids=arm_ids, env_ids=view_ids)
        asset.set_joint_position_target_index(target=arm_pos, joint_ids=arm_ids, env_ids=view_ids)
        asset.set_joint_velocity_target_index(target=arm_vel, joint_ids=arm_ids, env_ids=view_ids)
        if grip_ids and jp.shape[-1] >= arm_dim + len(grip_ids):
            grip_pos = jp[..., arm_dim : arm_dim + len(grip_ids)].clone().abs()
            grip_vel = torch.zeros_like(grip_pos)
            asset.write_joint_position_to_sim_index(position=grip_pos, joint_ids=grip_ids, env_ids=view_ids)
            asset.write_joint_velocity_to_sim_index(velocity=grip_vel, joint_ids=grip_ids, env_ids=view_ids)
            asset.set_joint_position_target_index(target=grip_pos, joint_ids=grip_ids, env_ids=view_ids)
        return

    # Non-robot articulation (cabinet / stove): match asset joint count (pad/truncate).
    import warp as wp

    default_jp = asset.data.default_joint_pos
    default_t = default_jp if isinstance(default_jp, torch.Tensor) else wp.to_torch(default_jp)
    n_asset = int(default_t.shape[-1])
    out_pos = default_t[view_ids].clone()
    count = min(int(jp.shape[-1]), n_asset)
    out_pos[:, :count] = jp[:, :count]
    zero_vel = torch.zeros_like(out_pos)
    asset.write_joint_position_to_sim_index(position=out_pos, env_ids=view_ids)
    asset.write_joint_velocity_to_sim_index(velocity=zero_vel, env_ids=view_ids)
    asset.set_joint_position_target_index(target=out_pos, env_ids=view_ids)


def _apply_initial_state_to_env(
    env: ManagerBasedEnv,
    env_idx: int,
    state: TrajectoryState,
    *,
    canonical_to_proto: dict[str, str],
    include_articulations: bool,
    include_rigid_objects: bool,
    robot_joint_noise_std: float,
) -> None:
    if include_articulations and "articulation" in state:
        for asset_name, asset_state in state["articulation"].items():
            if not isinstance(asset_state, dict):
                continue
            scene_name = resolve_demo_asset_to_scene(
                asset_name, entity_type="articulation", canonical_to_proto=canonical_to_proto
            )
            if scene_name is None:
                continue
            is_robot = scene_name == _ROBOT_SCENE_NAME
            root_pose = asset_state.get("root_pose")
            # Keep robot base from scene default; only joints come from the demo.
            if not is_robot and root_pose is not None and torch.is_tensor(root_pose):
                _write_root_pose_velocity(env, scene_name, env_idx, root_pose, zero_velocity=True)
            _write_articulation_joints(
                env,
                scene_name,
                env_idx,
                asset_state.get("joint_position"),
                asset_state.get("joint_velocity"),
                is_robot=is_robot,
                robot_joint_noise_std=robot_joint_noise_std if is_robot else 0.0,
            )

    if include_rigid_objects and "rigid_object" in state:
        for asset_name, asset_state in state["rigid_object"].items():
            if not isinstance(asset_state, dict):
                continue
            scene_name = resolve_demo_asset_to_scene(
                asset_name, entity_type="rigid_object", canonical_to_proto=canonical_to_proto
            )
            if scene_name is None:
                continue
            root_pose = asset_state.get("root_pose")
            if root_pose is not None and torch.is_tensor(root_pose):
                _write_root_pose_velocity(env, scene_name, env_idx, root_pose, zero_velocity=True)


def reset_libero_scene_to_demo_initial_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | Sequence[int] | None,
    command_name: str = "source_action",
    include_articulations: bool = True,
    include_rigid_objects: bool = True,
    robot_joint_noise_std: float = 0.0,
) -> None:
    """Restore scene entities from the demo ``initial_state`` of the reserved traj.

    Must run in ``mode=\"reset\"`` **before** :meth:`CommandManager.reset` so that
    :meth:`~.commands.SourceLiberoCommand.reserve_trajectories_for_envs` can pin
    the same trajectory the privileged diffs will advance.
    """
    if not command_name:
        raise ValueError("Parameter 'command_name' must be provided.")
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return
    try:
        term = command_manager.get_term(command_name)
    except Exception:
        return
    if not isinstance(term, SourceLiberoCommand):
        return

    requested = _resolve_env_ids(env_ids, env.scene.num_envs, env.device)
    if requested.numel() == 0:
        return

    bindings = _task_bindings(env)
    n_tasks = len(bindings)
    reservations = term.reserve_trajectories_for_envs(requested)

    for env_idx in requested.tolist():
        state = reservations.get(int(env_idx))
        if state is None:
            continue
        task = bindings[int(env_idx) % n_tasks]
        canonical_to_proto = build_canonical_to_proto_map(task)
        _apply_initial_state_to_env(
            env,
            int(env_idx),
            state,
            canonical_to_proto=canonical_to_proto,
            include_articulations=include_articulations,
            include_rigid_objects=include_rigid_objects,
            robot_joint_noise_std=robot_joint_noise_std,
        )
