# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def reset_shoelace_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shoelace"),
) -> None:
    """Restore selected cables to their authored knotted pose with zero velocity."""
    cable = env.scene[asset_cfg.name]
    segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
    segment_velocity = cable.data.default_segment_velocity_w.torch[env_ids].clone()
    cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
    cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)
