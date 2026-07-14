# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for LIBERO DGPO (OSC) multitask environments.

Every environment is built by
:func:`~...envs.dgpo_env_cfg.make_libero_dgpo_env_cfg` /
:func:`~...envs.dgpo_env_cfg.make_libero_dgpo_play_cfg`: harvest+cloner scene,
OSC actions (dim 7), and DGPO obs groups (actor=324 / critic=572). Suite order
for All matches :data:`~...dgpo_layout.DGPO_ABC_HARVEST_SUITES`
(long → object → spatial → goal). Importing this package registers:

* ``Isaac-Libero-All-Dgpo-Osc-v0`` — train all four suites (40 tasks).
* ``Isaac-Libero-All-Dgpo-Osc-Play-v0`` — play/eval all suites.
* ``Isaac-Libero-{Long,Object,Spatial,Goal}-Dgpo-Osc-Play-v0`` — single-suite
  play (10 tasks).
* ``Isaac-Libero-Spatial-Goal-Dgpo-Osc-Play-v0`` — spatial + goal play
  (20 tasks).
* ``Isaac-Libero-Object-Long-Dgpo-Osc-Play-v0`` — long + object play
  (20 tasks; DGPO relative order).

Play variants use one env per task and disable observation corruption.
Subset train configs still exist on :mod:`.libero_dgpo_env_cfg` for
programmatic use (no separate gym ids).
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

_DGPO_ENTRY = "isaaclab_contrib.tasks.manipulation.libero.envs.dgpo_env:DgpoManagerBasedRLEnv"
_DGPO_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:LiberoAllDgpoPPORunnerCfg"
_CFG = f"{__name__}.libero_dgpo_env_cfg"


def _register_dgpo(gym_id: str, env_cfg_attr: str) -> None:
    """Register a DGPO OSC env (train or play) with the shared runner cfg."""
    gym.register(
        id=gym_id,
        entry_point=_DGPO_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG}:{env_cfg_attr}",
            "rsl_rl_cfg_entry_point": _DGPO_AGENT,
        },
    )


# ---------------------------------------------------------------------------
# Train: all suites (40 tasks)
# ---------------------------------------------------------------------------

_register_dgpo("Isaac-Libero-All-Dgpo-Osc-v0", "LiberoAllDgpoOscEnvCfg")

# ---------------------------------------------------------------------------
# Play / eval: all suites + suite subsets
# ---------------------------------------------------------------------------

_register_dgpo("Isaac-Libero-All-Dgpo-Osc-Play-v0", "LiberoAllDgpoOscEnvCfg_PLAY")

_register_dgpo("Isaac-Libero-Long-Dgpo-Osc-Play-v0", "LiberoLongDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Libero-Object-Dgpo-Osc-Play-v0", "LiberoObjectDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Libero-Spatial-Dgpo-Osc-Play-v0", "LiberoSpatialDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Libero-Goal-Dgpo-Osc-Play-v0", "LiberoGoalDgpoOscEnvCfg_PLAY")

_register_dgpo("Isaac-Libero-Spatial-Goal-Dgpo-Osc-Play-v0", "LiberoSpatialGoalDgpoOscEnvCfg_PLAY")
_register_dgpo("Isaac-Libero-Object-Long-Dgpo-Osc-Play-v0", "LiberoObjectLongDgpoOscEnvCfg_PLAY")
