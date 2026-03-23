# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone strategies for environment-to-combo assignment.

Each strategy assigns environments to combos from a weighted combo space.
A "combo" represents a complete source configuration (group + prototype variants).
The strategy controls **both** which group an env belongs to **and** which
prototype variants it receives - these are unified as a single assignment.

Strategy signature::

    def strategy(weights: torch.Tensor, num_envs: int, device: str) -> torch.Tensor:
        '''
        Args:
            weights: [num_combos] weight per combo (determines distribution)
            num_envs: number of environments to assign
            device: torch device

        Returns:
            [num_envs] tensor where output[env_id] = combo_index
        '''
"""

import torch


def _allocate_counts(weights: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Allocate integer env counts to combos based on weights (vectorized)."""
    probs = weights / weights.sum()
    counts = (probs * num_envs).floor().long()
    remainder = num_envs - counts.sum().item()
    if remainder > 0:
        fracs = (probs * num_envs) - counts.float()
        _, top_indices = fracs.topk(min(remainder, len(weights)))
        counts[top_indices] += 1
    return counts


def random(weights: torch.Tensor, num_envs: int, device: str) -> torch.Tensor:
    """Randomly assign environments to combos respecting weights.

    Env counts per combo are deterministic (from weights), but which specific
    env IDs go to each combo is randomized.

    Args:
        weights: [num_combos] weight per combo.
        num_envs: Number of environments to assign.
        device: Torch device.

    Returns:
        [num_envs] tensor where ``output[env_id] = combo_index``.
    """
    counts = _allocate_counts(weights, num_envs)
    sequential_assign = torch.repeat_interleave(torch.arange(len(weights), device=device), counts.to(device))
    perm = torch.randperm(num_envs, device=device)
    assignment = torch.empty(num_envs, dtype=torch.long, device=device)
    assignment[perm] = sequential_assign
    return assignment


def sequential(weights: torch.Tensor, num_envs: int, device: str) -> torch.Tensor:
    """Sequentially assign environments to combos in contiguous blocks.

    Env 0..count[0]-1 go to combo 0, then count[0]..count[0]+count[1]-1 to
    combo 1, etc.

    Args:
        weights: [num_combos] weight per combo.
        num_envs: Number of environments to assign.
        device: Torch device.

    Returns:
        [num_envs] tensor where ``output[env_id] = combo_index``.
    """
    counts = _allocate_counts(weights, num_envs)
    return torch.repeat_interleave(torch.arange(len(weights), device=device), counts.to(device))


def interleaved(weights: torch.Tensor, num_envs: int, device: str) -> torch.Tensor:
    """Assign environments to combos in an interleaved (round-robin) pattern.

    Cycles through combos: env 0 -> combo 0, env 1 -> combo 1, ...,
    then wraps back. Respects weights for total count distribution.

    Args:
        weights: [num_combos] weight per combo.
        num_envs: Number of environments to assign.
        device: Torch device.

    Returns:
        [num_envs] tensor where ``output[env_id] = combo_index``.
    """
    counts = _allocate_counts(weights, num_envs)
    num_combos = len(weights)
    assignment = torch.empty(num_envs, dtype=torch.long, device=device)
    combo_remaining = counts.clone().to(device)
    combo_idx = 0
    for env_idx in range(num_envs):
        # Find next combo with remaining capacity
        while combo_remaining[combo_idx] == 0:
            combo_idx = (combo_idx + 1) % num_combos
        assignment[env_idx] = combo_idx
        combo_remaining[combo_idx] -= 1
        combo_idx = (combo_idx + 1) % num_combos
    return assignment
