# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for legacy rsl-rl checkpoint migration (model_state_dict → split)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from isaaclab_rl.rsl_rl import ensure_rsl_rl_checkpoint_compatible, migrate_legacy_rsl_rl_checkpoint

_DGPO_CKPT_CANDIDATES = (
    Path("/data/Projects/Robotics/IsaacLab/reb_isaaclab/_RobotLearningLab/logs/paper_plots/DGPO+ABC/model_32500.pt"),
    Path("/data/Projects/Robotics/IsaacLab/RobotLearningLab/logs/paper_plots/DGPO+ABC/model_32500.pt"),
)


def _dgpo_checkpoint() -> Path | None:
    env_path = os.environ.get("DGPO_ABC_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for path in _DGPO_CKPT_CANDIDATES:
        if path.is_file():
            return path
    return None


def _make_legacy_checkpoint() -> dict:
    """Minimal synthetic ActorCritic-style checkpoint."""
    return {
        "model_state_dict": {
            "log_std": torch.zeros(7),
            "actor.0.weight": torch.randn(512, 324),
            "actor.0.bias": torch.randn(512),
            "actor.2.weight": torch.randn(256, 512),
            "actor.2.bias": torch.randn(256),
            "actor.4.weight": torch.randn(128, 256),
            "actor.4.bias": torch.randn(128),
            "actor.6.weight": torch.randn(7, 128),
            "actor.6.bias": torch.randn(7),
            "actor_obs_normalizer._mean": torch.zeros(1, 324),
            "actor_obs_normalizer._var": torch.ones(1, 324),
            "actor_obs_normalizer._std": torch.ones(1, 324),
            "actor_obs_normalizer.count": torch.tensor(1.0),
            "critic.0.weight": torch.randn(512, 572),
            "critic.0.bias": torch.randn(512),
            "critic.2.weight": torch.randn(256, 512),
            "critic.2.bias": torch.randn(256),
            "critic.4.weight": torch.randn(128, 256),
            "critic.4.bias": torch.randn(128),
            "critic.6.weight": torch.randn(1, 128),
            "critic.6.bias": torch.randn(1),
            "critic_obs_normalizer._mean": torch.zeros(1, 572),
            "critic_obs_normalizer._var": torch.ones(1, 572),
            "critic_obs_normalizer._std": torch.ones(1, 572),
            "critic_obs_normalizer.count": torch.tensor(1.0),
        },
        "optimizer_state_dict": {"param_groups": []},
        "iter": 32500,
        "infos": None,
        "importance_aware_ppo_state": {"dummy": 1},
    }


def test_migrate_legacy_synthetic_checkpoint_keys_and_shapes():
    converted = migrate_legacy_rsl_rl_checkpoint(_make_legacy_checkpoint())

    assert "actor_state_dict" in converted
    assert "critic_state_dict" in converted
    assert "model_state_dict" not in converted
    assert converted["importance_aware_ppo_state"]["dummy"] == 1

    actor = converted["actor_state_dict"]
    critic = converted["critic_state_dict"]
    assert actor["mlp.0.weight"].shape == (512, 324)
    assert critic["mlp.0.weight"].shape == (512, 572)
    assert "distribution.log_std_param" in actor
    assert "obs_normalizer._mean" in actor
    assert "obs_normalizer._mean" in critic
    assert actor["obs_normalizer._mean"].shape == (1, 324)
    assert critic["obs_normalizer._mean"].shape == (1, 572)


def test_migrate_is_noop_for_new_format():
    new_ckpt = {
        "actor_state_dict": {"mlp.0.weight": torch.randn(2, 2)},
        "critic_state_dict": {"mlp.0.weight": torch.randn(2, 2)},
        "iter": 1,
        "infos": None,
    }
    assert migrate_legacy_rsl_rl_checkpoint(new_ckpt) is new_ckpt


@pytest.mark.skipif(_dgpo_checkpoint() is None, reason="DGPO+ABC checkpoint not on disk")
def test_migrate_dgpo_abc_checkpoint_shapes():
    path = _dgpo_checkpoint()
    assert path is not None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    converted = migrate_legacy_rsl_rl_checkpoint(loaded)

    assert converted["actor_state_dict"]["mlp.0.weight"].shape == (512, 324)
    assert converted["critic_state_dict"]["mlp.0.weight"].shape == (512, 572)
    assert "distribution.log_std_param" in converted["actor_state_dict"]

    migrated_path = ensure_rsl_rl_checkpoint_compatible(str(path))
    assert migrated_path != str(path)
    reloaded = torch.load(migrated_path, map_location="cpu", weights_only=False)
    assert "actor_state_dict" in reloaded
    assert reloaded["actor_state_dict"]["mlp.0.weight"].shape == (512, 324)
    os.remove(migrated_path)
