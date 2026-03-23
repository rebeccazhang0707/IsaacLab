# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for environment cloning partitioning."""

from dataclasses import MISSING

from isaaclab.cloner.cloner_strategies import random as random_strategy
from isaaclab.utils.configclass import configclass


@configclass
class InclusionSet:
    """One clone group defined by explicitly listing included assets.

    Each entry maps to a named group in the :class:`CloneCfg`.  The
    :attr:`assets` list references scene-config attribute names (the
    same strings used as keys when accessing ``scene[name]``).  The
    :attr:`weight` controls relative environment count when the total
    is divided across groups.

    Example::

        InclusionSet(assets=["lift_table", "lift_object"], weight=1)
    """

    assets: list[str] = MISSING
    """Asset names (matching scene cfg attribute names) to include in this group."""

    weight: int = 1
    """Relative weight for env partitioning."""


@configclass
class CloneCfg:
    """Describes how to partition environments into groups for selective cloning.

    Each entry in :attr:`clone_groups` maps a group name to a set
    descriptor (currently :class:`InclusionSet`).  Future set types
    (e.g. ``ExclusionSet``) can be added without changing the
    top-level structure.

    Assets **not** mentioned in any group are cloned into **all**
    environments (the default behaviour).  An asset name appearing
    in multiple groups is cloned into the **union** of those groups'
    environments.

    The :attr:`clone_strategy` controls how environments are assigned to
    groups and how prototype variants are selected:

    - ``random``: Shuffled assignment (env IDs randomly distributed)
    - ``sequential``: Contiguous blocks (env 0..n to group 0, etc.)
    - ``interleaved``: Alternating/cycling pattern (env 0 to group 0, env 1 to group 1, ...)

    Example::

        from isaaclab.cloner import random, sequential, interleaved

        CloneCfg(
            clone_strategy=random,  # shuffled env assignment
            clone_groups={
                "lift": InclusionSet(assets=["lift_table", "lift_object"], weight=1),
                "cabinet": InclusionSet(assets=["cabinet", "cabinet_frame"], weight=1),
                "reach": InclusionSet(assets=["reach_table"], weight=1),
            },
        )
    """

    clone_groups: dict[str, InclusionSet] = MISSING
    """Mapping from group name to its set descriptor."""

    clone_strategy: callable = random_strategy
    """Strategy for env-to-group assignment. Default is :func:`random` (shuffled)."""
