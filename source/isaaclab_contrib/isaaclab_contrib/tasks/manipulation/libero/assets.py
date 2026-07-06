# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared asset definitions for the LIBERO manipulation suites.

The two LIBERO suites (``libero_spatial`` and ``libero_goal``) share the same
Franka Panda robot and the same physics presets for their manipulable objects.
Centralising these here avoids duplicating the ~50-line robot config across the
scene configs, the robot module, and the task modules.

USD assets are loaded from the directory pointed to by the
``LIBERO_ASSETS_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def libero_assets_dir() -> str:
    """Return the LIBERO USD asset directory from ``LIBERO_ASSETS_DATA_DIR``.

    Returns:
        The asset root path, or an empty string when the variable is unset.
    """
    return os.getenv("LIBERO_ASSETS_DATA_DIR", "")


def libero_config_dir() -> str:
    """Return the LIBERO per-task JSON config directory.

    Uses ``LIBERO_CONFIG_DIR`` when set; otherwise falls back to the ``config``
    directory sitting next to the USD asset directory (``LIBERO_ASSETS_DATA_DIR``
    points at ``.../libero/USD``, so the configs live in ``.../libero/config``).

    Returns:
        The config directory path, or an empty string when neither the config
        variable nor the asset variable is set.
    """
    explicit = os.getenv("LIBERO_CONFIG_DIR", "")
    if explicit:
        return explicit
    assets = os.getenv("LIBERO_ASSETS_DATA_DIR", "")
    if assets:
        return os.path.join(os.path.dirname(assets.rstrip("/")), "config")
    return ""


# ---------------------------------------------------------------------------
# Shared physics properties for manipulable rigid objects
# ---------------------------------------------------------------------------

OBJECT_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
"""Rigid-body solver properties shared by all LIBERO manipulable objects."""


# ---------------------------------------------------------------------------
# Franka Panda robot (LIBERO PD gains + ready pose)
# ---------------------------------------------------------------------------

LIBERO_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.66, 0.0, 0.912),
        joint_pos={
            "panda_joint1": -0.019882839432642387,
            "panda_joint2": -0.18734066496238144,
            "panda_joint3": 0.0076694004538321505,
            "panda_joint4": -2.4034025985475256,
            "panda_joint5": 0.004964681607500244,
            "panda_joint6": 2.2453365042123963,
            "panda_joint7": 0.7948478983158621,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka Panda articulation with LIBERO-specific PD gains and ready-pose joints."""
