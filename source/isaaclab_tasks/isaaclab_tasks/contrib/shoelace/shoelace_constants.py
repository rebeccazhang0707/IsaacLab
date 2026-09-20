# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset paths and physical defaults shared by authoring and the RL configuration."""

import math
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "data"
CURVE_ASSET = ASSET_DIR / "curve.usd"
MODEL_ASSET = ASSET_DIR / "model.usd"
COLLIDER_ASSET = ASSET_DIR / "collider_simplified.usd"

TCP_OFFSET = (0.0, 0.0, 0.1034)

# Shared Franka-validated cable topology.
AUTHORED_SEGMENT_COUNT = 450
SHOELACE_SEGMENT_COUNT = 360
PINNED_FIRST = 78
PINNED_LAST = 281
PINNED_TUBE_SIDES = 6

# Shared Franka-validated cable material and contact defaults.
CABLE_DENSITY = 1150.0
CABLE_INERTIA_REGULARIZATION = 1.0e-6  # Additive isotropic proxy inertia [kg*m^2].
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 5.0
BEND_DAMPING = 1.0
DAHL_MAX_STRAIN = 0.20
DAHL_DECAY = 0.35
CONTACT_GAP = 1.0e-4
CONTACT_DISTANCE_CAP = 2.0e-3
CONTACT_KE = 1.0e6
CONTACT_KD = 0.0
GROUND_CONTACT_KD = 30.0
LACE_MU = 0.1
SHOE_MU = 0.02
GROUND_MU = 0.8
COLLISION_GROUP = 1
VBD_CONTACT_BUFFER = 256
CONTACTS_PER_ENV = 512
TRIANGLE_PAIRS_PER_ENV = 8192
MIN_TRIANGLE_PAIRS = 1_000_000

# Shared shoe geometry and display defaults.
TONGUE_UPPER_CENTER = (-0.008, 0.02, 0.10)
TONGUE_UPPER_SIZE = (0.05, 0.055, 0.006)
TONGUE_UPPER_PITCH = math.radians(28.0)
TONGUE_UPPER_Y_ROTATION = math.radians(5.0)
CABLE_COLOR = (112.0 / 255.0, 65.0 / 255.0, 39.0 / 255.0)


TAIL_BEND_STIFFNESS = 100.0
TAIL_BEND_DAMPING = 2.0
TAIL_STIFF_CORE_LENGTH = 0.050
TAIL_STIFF_TRANSITION_LENGTH = 0.012
