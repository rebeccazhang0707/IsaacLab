# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils import configclass


@dataclass
class TemplateClonePlan:
    """Runtime clone plan resolved from scene layout information.

    ``InteractiveScene`` prepares this plan from :class:`~isaaclab.scene.CloneCfg`
    using the flat prototype structure where each prototype is at
    ``/World/template/prototype_N``. The :attr:`group_mask` directly specifies
    which prototype to clone for each partition (no variant selection needed).
    """

    prototype_paths: tuple[str, ...]
    """Flat prototype paths: ``("/World/template/prototype_0", "/World/template/prototype_1", ...)``."""

    dest_paths: tuple[str, ...]
    """Destination relative paths for each prototype.

    Example: ``("Robot/", "Table/", "Robot/panda_hand/wrist_cam")`` maps each prototype
    to its location within each environment (relative to env root).
    """

    partition_assignment: torch.Tensor
    """Per-environment partition index, shape ``[num_envs]``.

    Maps each environment to its partition index. Used to build clone masking via
    ``group_mask[partition_assignment].T``.
    """

    group_mask: torch.Tensor
    """Boolean mask specifying which prototypes to clone for each partition.

    Shape: ``[num_partitions, num_prototypes]``. For partition P, the prototypes
    with ``group_mask[P, proto] == True`` are cloned to the environments in that partition.
    """


@configclass
class TemplateCloneCfg:
    """Configuration for template-based cloning.

    This configuration is consumed by :func:`~isaaclab.scene.cloner.clone_from_template` to
    replicate one or more "prototype" prims authored under a template root into multiple
    per-environment destinations. It supports both USD-spec replication and PhysX replication.

    The cloning flow is:

    1. Prototypes are spawned under :attr:`template_root` with names like
       ``proto_asset_0``, ``proto_asset_1``, etc.
    2. The :attr:`clone_plan` specifies which prototypes to clone to which environments.
    3. Prototypes are stamped to destinations derived from :attr:`clone_regex`.
    4. Optionally PhysX replication is performed for the same mapping.

    When using :class:`~isaaclab.scene.InteractiveScene`, the :attr:`clone_plan` is built
    automatically from :class:`~isaaclab.scene.CloneCfg` via :class:`ClonePlanBuilder`.
    """

    template_root: str = "/World/template"
    """Root path under which template prototypes are authored."""

    template_prototype_identifier: str = "proto_asset"
    """Name prefix used to identify prototype prims under :attr:`template_root`."""

    clone_regex: str = "/World/envs/env_.*"
    """Destination template for per-environment paths.

    The substring ``".*"`` is replaced with ``"{}"`` internally and formatted with the
    environment index (e.g., ``/World/envs/env_0``, ``/World/envs/env_1``).
    """

    clone_usd: bool = True
    """Enable USD-spec replication to author cloned prims and optional transforms."""

    clone_physics: bool = True
    """Enable PhysX replication for the same mapping to speed up physics setup."""

    physics_clone_fn: callable | None = None
    """Function used to perform physics replication."""

    visualizer_clone_fn: callable | None = None
    """Optional function used to build precomputed visualizer artifacts from the clone plan."""

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric for PhysX replication. Default is False."""

    clone_plan: TemplateClonePlan | None = None
    """Tensorized clone plan specifying what to clone and where.

    ``InteractiveScene`` populates this automatically from
    :class:`~isaaclab.scene.CloneCfg`.
    """
