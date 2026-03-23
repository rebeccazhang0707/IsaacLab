# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from isaaclab.physics.scene_data_requirements import SceneDataRequirement, VisualizerPrebuiltArtifacts

from .cloner_cfg import TemplateClonePlan

if TYPE_CHECKING:
    from isaaclab.scene.clone_cfg import CloneCfg

    from .cloner_cfg import TemplateCloneCfg

logger = logging.getLogger(__name__)


class ClonePlanBuilder:
    """Accumulates prototype metadata during scene spawning and builds TemplateClonePlan.

    This class encapsulates all the state and logic needed to build a clone plan:
    - Registering prototypes as assets are spawned
    - Expanding partitions based on prototype variants
    - Computing the vectorized group_mask
    - Returning the final TemplateClonePlan

    Example:
        >>> builder = ClonePlanBuilder("cuda:0")
        >>> # During asset spawning:
        >>> start_idx = builder.register("robot", "Robot/", num_variants=1)
        >>> start_idx = builder.register("object", "Object/", num_variants=3)
        >>> # After spawning:
        >>> group_assignment, group_names = builder.finalize(cloner_cfg, clone_cfg, num_envs=64)
        >>> # cloner_cfg.clone_plan is now populated
    """

    def __init__(self, device: str):
        """Initialize an empty builder.

        Args:
            device: Torch device for tensor allocation.
        """
        self._device = device
        self._proto_mapping: dict[str, list[int]] = {}
        self._dest_paths: list[str] = []
        self._asset_names: list[str] = []
        self._asset_num_variants: list[int] = []
        self._proto_asset_idx: list[int] = []
        self._proto_variant: list[int] = []
        self._counter = 0

    def _extend_protos(self, asset_idx: int, dest_path: str, num_variants: int, variant_start: int = 0) -> int:
        """Extend prototype lists for an asset. Returns starting proto index."""
        start = self._counter
        self._dest_paths.extend([dest_path] * num_variants)
        self._proto_asset_idx.extend([asset_idx] * num_variants)
        self._proto_variant.extend(range(variant_start, variant_start + num_variants))
        self._counter += num_variants
        return start

    def register(self, asset_name: str, dest_path: str, num_variants: int = 1) -> int:
        """Register prototypes for an asset.

        Args:
            asset_name: Unique name for the asset.
            dest_path: Destination path relative to environment root (e.g., "Robot/").
            num_variants: Number of prototype variants for this asset.

        Returns:
            Starting prototype index for this asset.
        """
        asset_idx = len(self._asset_names)
        self._asset_names.append(asset_name)
        self._asset_num_variants.append(num_variants)
        start = self._extend_protos(asset_idx, dest_path, num_variants)
        self._proto_mapping[asset_name] = list(range(start, start + num_variants))
        return start

    def register_into_collection(self, collection_name: str, dest_path: str, num_variants: int = 1) -> int:
        """Register prototypes into a named collection incrementally.

        Used for RigidObjectCollection where each object registers separately.
        Creates the collection on first call, extends it on subsequent calls.

        Args:
            collection_name: Unique name for the collection.
            dest_path: Destination path relative to environment root.
            num_variants: Number of prototype variants for this object.

        Returns:
            Starting prototype index for this object.
        """
        if collection_name not in self._proto_mapping:
            return self.register(collection_name, dest_path, num_variants)

        # Extend existing collection
        asset_idx = self._asset_names.index(collection_name)
        prev_variants = self._asset_num_variants[asset_idx]
        self._asset_num_variants[asset_idx] += num_variants
        start = self._extend_protos(asset_idx, dest_path, num_variants, prev_variants)
        self._proto_mapping[collection_name].extend(range(start, start + num_variants))
        return start

    @property
    def proto_count(self) -> int:
        """Total number of registered prototypes."""
        return self._counter

    @property
    def proto_mapping(self) -> dict[str, list[int]]:
        """Mapping from asset name to prototype indices."""
        return self._proto_mapping

    @property
    def dest_paths(self) -> list[str]:
        """Destination paths for all prototypes."""
        return self._dest_paths

    def finalize(
        self, cloner_cfg: TemplateCloneCfg, clone_cfg: CloneCfg | None, num_envs: int
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Finalize the clone plan and populate the cloner config.

        Args:
            cloner_cfg: The cloner config to populate with the built plan.
            clone_cfg: Clone configuration specifying groups and strategy, or None for homogeneous.
            num_envs: Number of environments to clone.

        Returns:
            Tuple of (group_assignment, group_names):
            - group_assignment: Per-env group index tensor [num_envs]
            - group_names: Ordered tuple of group names
        """
        n_protos, dev = self._counter, self._device
        pid = cloner_cfg.template_prototype_identifier
        proto_paths = tuple(f"{cloner_cfg.template_root}/{pid}_{i}" for i in range(n_protos))
        dest_paths = tuple(self._dest_paths)

        # Homogeneous case
        if clone_cfg is None or n_protos == 0:
            zeros = torch.zeros(num_envs, dtype=torch.long, device=dev)
            homogeneous_mask = torch.ones((1, n_protos), dtype=torch.bool, device=dev)
            cloner_cfg.clone_plan = TemplateClonePlan(proto_paths, dest_paths, zeros, homogeneous_mask)
            return zeros, ()

        # Build asset membership and max variants per group
        groups = clone_cfg.clone_groups
        group_names = tuple(groups.keys())
        n_groups, n_assets = len(group_names), len(self._asset_names)
        asset_idx = {name: i for i, name in enumerate(self._asset_names)}
        asset_in_groups = torch.zeros((n_assets, n_groups), dtype=torch.bool, device=dev)
        max_vars, weights = [], []
        for g, (name, inc) in enumerate(groups.items()):
            weights.append(inc.weight)
            mv = 1
            for a in inc.assets:
                if a in asset_idx:
                    asset_in_groups[asset_idx[a], g] = True
                    mv = max(mv, self._asset_num_variants[asset_idx[a]])
            max_vars.append(mv)

        # Partition expansion and strategy
        part_var = torch.cat([torch.arange(m, device=dev) for m in max_vars])
        max_vars = torch.tensor(max_vars, dtype=torch.long, device=dev)
        weights = torch.tensor(weights, dtype=torch.float32, device=dev)
        part_group = torch.repeat_interleave(torch.arange(n_groups, device=dev), max_vars)
        assignment = clone_cfg.clone_strategy(torch.repeat_interleave(weights / max_vars, max_vars), num_envs, dev)

        # Group mask computation
        proto_asset = torch.tensor(self._proto_asset_idx, dtype=torch.long, device=dev)
        proto_var = torch.tensor(self._proto_variant, dtype=torch.long, device=dev)
        asset_vars = torch.tensor(self._asset_num_variants, dtype=torch.long, device=dev)
        proto_grp = asset_in_groups[proto_asset]
        is_global = ~proto_grp.any(dim=1)
        grp_match = proto_grp[:, part_group].T
        var_match = proto_var == (part_var.unsqueeze(1) % asset_vars[proto_asset])
        mask = is_global | (grp_match & var_match)

        cloner_cfg.clone_plan = TemplateClonePlan(proto_paths, dest_paths, assignment, mask)
        return part_group[assignment], group_names


def clone_from_template(stage: Usd.Stage, num_clones: int, template_clone_cfg: TemplateCloneCfg) -> None:
    """Clone assets from a template root into per-environment destinations.

    Uses the flat prototype structure where each prototype is at
    ``/World/template/prototype_N``. The clone plan's ``group_mask`` directly
    specifies which prototypes to clone for each partition, and ``dest_paths``
    specifies where each prototype should be placed in the cloned environments.

    Args:
        stage: The USD stage to author into.
        num_clones: Number of environments to clone to.
        template_clone_cfg: Configuration describing template location, destination pattern,
            and replication/mapping behavior.
    """
    cfg: TemplateCloneCfg = template_clone_cfg
    clone_plan = cfg.clone_plan
    world_indices = torch.arange(num_clones, device=cfg.device)
    clone_path_fmt = cfg.clone_regex.replace(".*", "{}")

    # Build destination paths using the plan's dest_paths (relative to env)
    src_paths = list(clone_plan.prototype_paths)
    dest_paths = [f"{clone_path_fmt}/{dp}" for dp in clone_plan.dest_paths]

    # group_mask[partition_assignment] -> [num_envs, num_protos], transpose to [num_protos, num_envs]
    clone_masking = clone_plan.group_mask[clone_plan.partition_assignment].T

    is_homogeneous = clone_plan.group_mask.shape[0] == 1 and bool(clone_plan.group_mask.all().item())

    # Spawn the first instance of clones from prototypes, then deactivate the prototypes
    proto_idx_per_src = clone_masking.to(torch.int32).argmax(dim=1)
    proto_mask = torch.zeros_like(clone_masking)
    proto_mask.scatter_(1, proto_idx_per_src.view(-1, 1).to(torch.long), clone_masking.any(dim=1, keepdim=True))
    usd_replicate(stage, src_paths, dest_paths, world_indices, proto_mask)
    stage.GetPrimAtPath(cfg.template_root).SetActive(False)
    get_pos = lambda path: stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()  # noqa: E731
    positions = torch.tensor([get_pos(clone_path_fmt.format(i)) for i in world_indices])

    # If all prototypes map to env_0 and the plan is homogeneous,
    # clone whole env_0 to all envs; otherwise clone per-object
    if torch.all(proto_idx_per_src == 0) and is_homogeneous:
        mapping = clone_masking.new_ones(1, num_clones)
        replicate_args = [clone_path_fmt.format(0)], [clone_path_fmt], world_indices, mapping
        usd_positions = positions
    else:
        selected_src = [tpl.format(int(idx)) for tpl, idx in zip(dest_paths, proto_idx_per_src.tolist())]
        replicate_args = selected_src, dest_paths, world_indices, clone_masking
        usd_positions = None

    if cfg.clone_physics and cfg.physics_clone_fn is not None:
        cfg.physics_clone_fn(stage, *replicate_args, positions=positions, device=cfg.device)
    if cfg.visualizer_clone_fn is not None:
        cfg.visualizer_clone_fn(stage, *replicate_args, positions=positions, device=cfg.device)
    if cfg.clone_usd:
        usd_replicate(stage, *replicate_args, positions=usd_positions)


def usd_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
) -> None:
    """Replicate USD prims to per-environment destinations.

    Copies each source prim spec to destination templates for selected environments
    (``mask``). Optionally authors translate/orient from position/quaternion buffers.
    Replication runs in path-depth order (parents before children) for robust composition.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination formattable templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions (``[E, 3]``) -> ``xformOp:translate``.
        quaternions: Optional orientations (``[E, 4]``) in ``xyzw`` -> ``xformOp:orient``.

    """
    rl = stage.GetRootLayer()

    # Group replication by destination path depth so ancestors land before deeper paths.
    # This avoids composition issues for nested or interdependent specs.
    def dp_depth(template: str) -> int:
        """Return destination prim path depth for stable parent-first replication."""
        dp = template.format(0)
        return Sdf.Path(dp).pathElementCount

    order = sorted(range(len(sources)), key=lambda i: dp_depth(destinations[i]))

    # Process in layers of equal depth, committing at each depth to stabilize composition
    depth_to_indices: dict[int, list[int]] = {}
    for i in order:
        d = dp_depth(destinations[i])
        depth_to_indices.setdefault(d, []).append(i)

    for depth in sorted(depth_to_indices.keys()):
        with Sdf.ChangeBlock():
            for i in depth_to_indices[depth]:
                src = sources[i]
                tmpl = destinations[i]
                # Select target environments for this source (supports None, [E], or [S, E])
                target_envs = env_ids if mask is None else env_ids[mask[i]]
                for wid in target_envs.tolist():
                    dp = tmpl.format(wid)
                    Sdf.CreatePrimInLayer(rl, dp)
                    if src == dp:
                        pass  # self-copy: CreatePrimInLayer already ensures it exists; CopySpec would be destructive
                    else:
                        Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                    if positions is not None or quaternions is not None:
                        ps = rl.GetPrimAtPath(dp)
                        op_names = []
                        if positions is not None:
                            p = positions[wid]
                            t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                            if t_attr is None:
                                t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                            t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                            op_names.append("xformOp:translate")
                        if quaternions is not None:
                            q = quaternions[wid]
                            o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                            if o_attr is None:
                                o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                            # xyzw convention: q[3] is w, q[0:3] is xyz
                            o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                            op_names.append("xformOp:orient")
                        # Only author xformOpOrder for the ops we actually authored
                        if op_names:
                            op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                            )
                            op_order.default = Vt.TokenArray(op_names)


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: list[str] = [],
) -> None:
    """Create inverted collision groups for clones (PhysX only).

    Sets PhysX scene attributes and collision groups on the prim at ``physicsscene_path``
    (no PhysxSchema import). Call only when the physics backend is PhysX; Newton uses
    its own collision/world handling and does not use USD PhysX collision groups.

    Creates one PhysicsCollisionGroup per prim under ``collision_root_path``, enabling
    inverted filtering so clones don't collide across groups. Optionally adds a global
    group that collides with all.

    Args:
        stage: USD stage.
        physicsscene_path: Path to PhysicsScene prim.
        collision_root_path: Root scope for collision groups.
        prim_paths: Per-clone prim paths.
        global_paths: Optional global-collider paths.

    """

    scene_prim = stage.GetPrimAtPath(physicsscene_path)
    # We invert the collision group filters for more efficient collision filtering across environments
    invert_attr = scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool)
    invert_attr.Set(True)

    # Make sure we create the collision_scope in the RootLayer since the edit target
    # may be a live layer in the case of Live Sync.
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        UsdGeom.Scope.Define(stage, collision_root_path)

    with Sdf.ChangeBlock():
        if len(global_paths) > 0:
            global_collision_group_path = collision_root_path + "/global_group"
            # add collision group prim
            global_collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                "global_group",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            global_collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                global_collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            global_includes_rel = Sdf.RelationshipSpec(global_collision_group, "collection:colliders:includes", False)
            for global_path in global_paths:
                global_includes_rel.targetPathList.Append(global_path)

            # filteredGroups rel
            global_filtered_groups = Sdf.RelationshipSpec(global_collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            global_filtered_groups.targetPathList.Append(global_collision_group_path)

        # set collision groups and filters
        for i, prim_path in enumerate(prim_paths):
            collision_group_path = collision_root_path + f"/group{i}"
            # add collision group prim
            collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                f"group{i}",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            includes_rel.targetPathList.Append(prim_path)

            # filteredGroups rel
            filtered_groups = Sdf.RelationshipSpec(collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            filtered_groups.targetPathList.Append(collision_group_path)
            if len(global_paths) > 0:
                filtered_groups.targetPathList.Append(global_collision_group_path)
                global_filtered_groups.targetPathList.Append(collision_group_path)


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
    """Create a centered grid of transforms for ``N`` instances.

    Computes ``(x, y)`` coordinates in a roughly square grid centered at the origin
    with the provided spacing, places the third coordinate according to ``up_axis``,
    and returns identity orientations. This matches the grid layout used by
    :class:`isaaclab.terrains.TerrainImporter` for consistent environment positioning.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions.
        up_axis: Up axis for positions ("z", "y", or "x").
        device: Torch device for returned tensors.

    Returns:
        A tuple ``(pos, ori)`` where:
            - ``pos`` is a tensor of shape ``(N, 3)`` with positions.
            - ``ori`` is a tensor of shape ``(N, 4)`` with identity quaternions in ``(x, y, z, w)``.
    """
    # Match terrain_importer._compute_env_origins_grid layout for consistency
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))

    # Create meshgrid matching terrain's "ij" indexing
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=device, dtype=torch.float32),
        torch.arange(num_cols, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Flatten and take first N elements
    ii = ii.flatten()[:N]
    jj = jj.flatten()[:N]

    # Match terrain's coordinate system: X from rows (negated), Y from cols
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    z0 = torch.zeros(N, device=device)

    # place on plane based on up_axis
    if up_axis.lower() == "z":
        pos = torch.stack([x, y, z0], dim=1)
    elif up_axis.lower() == "y":
        pos = torch.stack([x, z0, y], dim=1)
    else:  # up_axis == "x"
        pos = torch.stack([z0, x, y], dim=1)

    # identity orientations (x,y,z,w)
    ori = torch.zeros((N, 4), device=device)
    ori[:, 3] = 1.0  # w=1 for identity quaternion
    return pos, ori


def resolve_visualizer_clone_fn(
    physics_backend: str,
    requirements: SceneDataRequirement,
    stage,
    set_visualizer_artifact: Callable[[VisualizerPrebuiltArtifacts | None], None],
):
    """Return an optional visualizer prebuild hook for clone workflows.

    Args:
        physics_backend: Active physics backend name.
        requirements: Aggregated scene-data requirements.
        stage: USD stage used by the clone callback.
        set_visualizer_artifact: Callback for storing prebuilt visualizer artifacts.

    Returns:
        Clone callback when the prebuild path is supported; otherwise ``None``.
    """
    if "physx" not in physics_backend or not requirements.requires_newton_model:
        return None
    try:
        from isaaclab_newton.cloner.newton_replicate import (
            create_newton_visualizer_prebuild_clone_fn,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("Visualizer prebuild hook unavailable: failed to import backend helper.")
        logger.debug("Visualizer prebuild import failure details: %s", exc)
        return None

    return create_newton_visualizer_prebuild_clone_fn(
        stage=stage,
        set_visualizer_artifact=set_visualizer_artifact,
    )
