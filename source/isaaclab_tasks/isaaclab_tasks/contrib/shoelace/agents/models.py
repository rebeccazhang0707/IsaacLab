# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models used for shoelace checkpoint consolidation."""

from rsl_rl.models import MLPModel
from tensordict import TensorDict


class FixedObservationStatisticsMLPModel(MLPModel):
    """MLP model that preserves observation statistics loaded from a checkpoint."""

    def update_normalization(self, obs: TensorDict) -> None:
        """Keep the loaded observation normalization statistics unchanged."""
        del obs
