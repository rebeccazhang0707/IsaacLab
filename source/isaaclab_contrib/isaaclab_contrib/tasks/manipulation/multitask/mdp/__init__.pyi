# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
__all__ = [
    # utils
    "RobotGroupCfg",
    "ReachGroupCfg",
    "LiftGroupCfg",
    "CabinetGroupCfg",
    # batched observations
    "batched_ee_pose",
    "batched_ee_pos_error",
    "batched_object_pos_in_robot_frame",
    "batched_ee_object_pos_error",
    "batched_object_target_pos_error",
    "batched_joint_pos_rel",
    "batched_joint_vel",
    "batched_generated_commands",
    "batched_cabinet_joint_pos",
    "batched_cabinet_joint_vel",
    "batched_cabinet_rel_ee_drawer_distance",
    # batched rewards
    "batched_joint_vel_l2",
    "batched_position_command_error",
    "batched_position_command_error_tanh",
    "batched_orientation_command_error",
    "batched_orientation_command_error_tanh",
    "batched_object_ee_distance",
    "batched_object_is_lifted",
    "batched_object_goal_distance",
    "batched_cabinet_approach_ee_handle",
    "batched_cabinet_align_ee_handle",
    "batched_cabinet_align_grasp_around_handle",
    "batched_cabinet_approach_gripper_handle",
    "batched_cabinet_grasp_handle",
    "batched_cabinet_open_drawer_bonus",
    "batched_cabinet_multi_stage_open_drawer",
    # batched terminations
    "batched_object_height_below_minimum",
    "batched_cabinet_drawer_opened",
    # batched events
    "batched_reset_to_default",
    "batched_reset_joints_by_scale",
    "batched_reset_object_state_uniform",
    "batched_reset_cabinet_to_default",
    # observations (non-batched)
    "multi_task_onehot",
    # re-exported from isaaclab core
    "UniformPoseCommandCfg",
    "action_rate_l2",
    "last_action",
    "modify_reward_weight",
    "time_out",
]

from .batched_events import (
    batched_reset_cabinet_to_default,
    batched_reset_joints_by_scale,
    batched_reset_object_state_uniform,
    batched_reset_to_default,
)
from .batched_obs import (
    batched_cabinet_joint_pos,
    batched_cabinet_joint_vel,
    batched_cabinet_rel_ee_drawer_distance,
    batched_ee_object_pos_error,
    batched_ee_pos_error,
    batched_ee_pose,
    batched_generated_commands,
    batched_joint_pos_rel,
    batched_joint_vel,
    batched_object_pos_in_robot_frame,
    batched_object_target_pos_error,
)
from .batched_rewards import (
    batched_cabinet_align_ee_handle,
    batched_cabinet_align_grasp_around_handle,
    batched_cabinet_approach_ee_handle,
    batched_cabinet_approach_gripper_handle,
    batched_cabinet_grasp_handle,
    batched_cabinet_multi_stage_open_drawer,
    batched_cabinet_open_drawer_bonus,
    batched_joint_vel_l2,
    batched_object_ee_distance,
    batched_object_goal_distance,
    batched_object_is_lifted,
    batched_orientation_command_error,
    batched_orientation_command_error_tanh,
    batched_position_command_error,
    batched_position_command_error_tanh,
)
from .batched_terminations import batched_cabinet_drawer_opened, batched_object_height_below_minimum
from .observations import multi_task_onehot
from .utils import CabinetGroupCfg, LiftGroupCfg, ReachGroupCfg, RobotGroupCfg

from isaaclab.envs.mdp import (
    UniformPoseCommandCfg,
    action_rate_l2,
    last_action,
    modify_reward_weight,
    time_out,
)
