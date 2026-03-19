# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
__all__ = [
    "cabinet_align_ee_handle",
    "cabinet_align_grasp_around_handle",
    "cabinet_approach_ee_handle",
    "cabinet_approach_gripper_handle",
    "cabinet_grasp_handle",
    "cabinet_multi_stage_open_drawer",
    "cabinet_open_drawer_bonus",
    "cabinet_rel_ee_drawer_distance",
    "object_target_pos_error",
    "ee_pose_b",
    "ee_pos_error",
    "ee_object_pos_error",
    "multi_task_onehot",
    "reset_asset_to_default",
    "reset_object_state_uniform",
    "cabinet_drawer_opened",
    "object_height_below_minimum",
    "orientation_command_error_tanh",
    "batched_ee_pose",
    "batched_joint_pos_rel",
    "batched_joint_vel",
    "batched_ee_pos_error",
    "batched_generated_commands",
    "batched_object_pos_in_robot_frame",
    "batched_ee_object_pos_error",
    "batched_object_target_pos_error",
    "batched_joint_vel_l2",
    "batched_position_command_error",
    "batched_position_command_error_tanh",
    "batched_orientation_command_error",
    "batched_object_ee_distance",
    "batched_object_is_lifted",
    "batched_object_goal_distance",
    "batched_reset_to_default",
    "batched_reset_joints_by_scale",
    "batched_reset_object_state_uniform",
    "batched_object_height_below_minimum",
]

from .batched_events import (
    batched_reset_to_default,
    batched_reset_joints_by_scale,
    batched_reset_object_state_uniform,
)
from .batched_obs import (
    batched_ee_pose,
    batched_joint_pos_rel,
    batched_joint_vel,
    batched_ee_pos_error,
    batched_generated_commands,
    batched_object_pos_in_robot_frame,
    batched_ee_object_pos_error,
    batched_object_target_pos_error,
)
from .batched_rewards import (
    batched_joint_vel_l2,
    batched_position_command_error,
    batched_position_command_error_tanh,
    batched_orientation_command_error,
    batched_object_ee_distance,
    batched_object_is_lifted,
    batched_object_goal_distance,
)
from .batched_terminations import batched_object_height_below_minimum
from .events import reset_asset_to_default, reset_object_state_uniform
from .observations import (
    cabinet_rel_ee_drawer_distance,
    ee_object_pos_error,
    ee_pos_error,
    ee_pose_b,
    multi_task_onehot,
    object_target_pos_error,
)
from .rewards import (
    orientation_command_error_tanh,
    cabinet_align_ee_handle,
    cabinet_align_grasp_around_handle,
    cabinet_approach_ee_handle,
    cabinet_approach_gripper_handle,
    cabinet_grasp_handle,
    cabinet_multi_stage_open_drawer,
    cabinet_open_drawer_bonus,
)
from .terminations import cabinet_drawer_opened, object_height_below_minimum

# Standard MDP symbols from isaaclab core and single-task packages.
# These work transparently in multitask configs thanks to ScopedEnv.
from isaaclab.envs.mdp import *
from isaaclab_tasks.manager_based.manipulation.lift.mdp import *
from isaaclab_tasks.manager_based.manipulation.reach.mdp import *
from isaaclab_tasks.manager_based.manipulation.cabinet.mdp import *
