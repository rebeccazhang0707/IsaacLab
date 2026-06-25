# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The two clone engines that turn harvested tasks into a running heterogeneous scene.

A :class:`CloneEngine` consumes the harvested ``(tasks, prototypes)`` (see
:mod:`registry_harvest`). The base class owns everything that is *the same regardless
of cloning API* -- the run loop, the whole-view reset, and the joint drive -- and
leaves just the cloning itself to subclasses:

* :class:`InteractiveSceneEngine` -- hand a declarative
  :class:`~isaaclab.scene.InteractiveSceneCfg` (prototypes + a heterogeneous
  :class:`~isaaclab.cloner.CloneCfg`) to :class:`~isaaclab.scene.InteractiveScene`,
  which lays out envs, clones, and builds physics views for you.
* :class:`ManualCloneEngine` -- do it by hand: ``grid_transforms`` for env origins,
  ``usd_replicate`` to lay out env containers, then a
  :class:`~isaaclab.cloner.ReplicateSession` (``make_clone_plan`` + ``replicate``) that
  spawns each prototype once in ``env_0`` and clones it into the envs that use it.

Both author into the same ``/World/envs/env_*`` containers and publish a single clone
plan, so a scene uses exactly one engine. Import after ``AppLauncher`` has started.
"""

from __future__ import annotations

import torch
import warp as wp
from registry_harvest import Prototype, TaskGroup

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, InclusionSet, ReplicateSession, make_valid_clone_combinations, sequential
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

# Env-namespace prefix the harvested prototype prim paths are bound to. It differs per engine
# (InteractiveScene resolves the ``{ENV_REGEX_NS}`` macro itself; the manual path needs the
# expanded regex), so each engine exposes its own via ``PRIM_PREFIX``.
PRIM_PREFIX_INTERACTIVE_SCENE = "{ENV_REGEX_NS}"
PRIM_PREFIX_MANUAL = "/World/envs/env_.*"

# Random joint-perturbation magnitude per robot family; legged platforms wiggle less than
# arms so they stay upright while remaining visually expressive.
ARM_NOISE = 0.4
LEG_NOISE = 0.06
DEFAULT_NOISE = 0.15
LEGGED_HINTS = ("anymal", "unitree", "spot", "a1", "go1", "go2", "cassie", "digit", "h1", "g1", "humanoid", "ant")
ARM_HINTS = ("panda", "franka", "ur5", "ur10", "kinova", "sawyer", "flexiv", "allegro", "shadow", "robot", "arm")


def noise_scale(asset_name: str) -> float:
    """Per-asset joint-perturbation magnitude inferred from the asset name."""
    lname = asset_name.lower()
    if any(h in lname for h in LEGGED_HINTS):
        return LEG_NOISE
    if any(h in lname for h in ARM_HINTS):
        return ARM_NOISE
    return DEFAULT_NOISE


def _is_clone_group(cfg) -> bool:
    """Whether ``make_clone_plan`` treats this cfg as a clone group.

    Mirrors the plan's own filter: an env-scoped cfg that carries a spawner. Assets
    without a spawn (e.g. a :class:`~isaaclab.assets.SurfaceGripperCfg`, whose prims
    live inside their owner's USD) are not replicated as their own group, so they must
    not become columns of the ``valid_set`` handed to :func:`make_valid_clone_combinations`.
    """
    prim_path = getattr(cfg, "prim_path", None)
    return bool(prim_path) and getattr(cfg, "spawn", None) is not None and "/World/envs/" in prim_path


def _global_assets() -> dict[str, AssetBaseCfg]:
    """The two global shared assets (a single ground + dome-light prim, never cloned)."""
    return {
        "ground": AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=GroundPlaneCfg(),
        ),
        "light": AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        ),
    }


class CloneEngine:
    """Build a heterogeneous scene from ``(tasks, prototypes)`` and drive it.

    Subclasses implement only the cloning API -- :meth:`build` plus a few one-line
    accessors. The run loop, the whole-view reset, and the joint drive live here and are
    identical for both engines. Every env is reset and driven together, so a prototype's
    whole physics view is addressed at once via the analytic ``Prototype.env_ids`` map.
    """

    RESET_INTERVAL = 60  # steps between full (all-env) resets
    PRIM_PREFIX = ""  # set by each subclass; the caller harvests with it

    def __init__(
        self, sim, simulation_app, tasks, prototypes, num_envs, env_spacing, device, clone_strategy=sequential
    ):
        self.sim = sim
        self.simulation_app = simulation_app
        self.tasks: list[TaskGroup] = tasks
        self.prototypes: list[Prototype] = prototypes
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.device = device
        self.clone_strategy = clone_strategy
        self.origins: torch.Tensor | None = None  # [num_envs, 3] world env origins (set in build)
        self._by_name = {p.name: p for p in prototypes}

    # ----------------------------------------------------------------
    # API-specific hooks -- the only thing the two engines differ in
    # ----------------------------------------------------------------

    def build(self) -> None:
        """Spawn + clone the scene, create physics views, and set :attr:`origins`."""
        raise NotImplementedError

    def iter_articulations(self):
        """Yield ``(Prototype, articulation)`` for every articulation in the scene."""
        raise NotImplementedError

    def _instance(self, proto: Prototype):
        """Return the live Articulation/RigidObject backing ``proto``."""
        raise NotImplementedError

    def flush(self) -> None:
        """Push buffered writes to the simulation."""
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Refresh asset data after a physics step."""
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Shared driving -- identical across engines, no grouping
    # ----------------------------------------------------------------

    def run(self) -> None:
        """Build once, then drive every environment together forever."""
        self.build()
        dt = self.sim.get_physics_dt()
        self.reset()
        self.flush()
        step = 0
        while self.simulation_app.is_running():
            if step % self.RESET_INTERVAL == 0:
                self.reset()
                print(f"[step {step:>5d}] reset all {self.num_envs} envs together")
            self.apply_actions()
            self.flush()
            self.sim.step()
            step += 1
            self.update(dt)

    def reset(self) -> None:
        """Re-pose every prototype across its *whole* physics view at once.

        No per-env subset: a prototype's view holds exactly the envs it
        was cloned into, ordered by ascending env id, which equals
        :attr:`~registry_harvest.Prototype.env_ids`. We write the full view (local ids
        ``0..k-1``) and fetch each instance's world origin via that forward map. A shared
        prototype still gets each env's task-specific init pose, looked up forward by
        ``env -> task (env % n_tasks)``.
        """
        n_tasks = len(self.tasks)
        for proto in self.prototypes:
            if not proto.resettable or not proto.env_ids:
                continue
            env_ids = proto.env_ids  # global envs, ascending == this prototype's view order
            world_pos = self.origins[env_ids]  # forward map: view instance -> world origin [k, 3]
            view_idx = torch.arange(len(env_ids), device=world_pos.device, dtype=torch.long)
            self._write_init_state(proto, view_idx, world_pos, env_ids, n_tasks)

    def apply_actions(self) -> None:
        """Hold default joint targets everywhere, with a small random offset, on all envs."""
        for proto, articulation in self.iter_articulations():
            default = wp.to_torch(articulation.data.default_joint_pos)
            noise = noise_scale(proto.name) * torch.randn(default.shape[0], default.shape[1], device=default.device)
            articulation.set_joint_position_target_index(target=default + noise)

    def _write_init_state(
        self, proto: Prototype, view_idx: torch.Tensor, world_pos: torch.Tensor, env_ids: list[int], n_tasks: int
    ) -> None:
        """Write the per-env root pose (+ default joints for articulations) across the whole view."""
        obj = self._instance(proto)
        dev, dtype = world_pos.device, world_pos.dtype
        k = view_idx.shape[0]
        pose = torch.zeros((k, 7), device=dev, dtype=dtype)
        for i, env in enumerate(env_ids):
            # env -> owning task -> that task's intended pose for this prototype (forward lookup).
            init = getattr(self.tasks[env % n_tasks].init_cfgs.get(proto.name), "init_state", None)
            pos = getattr(init, "pos", None) or (0.0, 0.0, 0.0)
            rot = getattr(init, "rot", None) or (1.0, 0.0, 0.0, 0.0)
            pose[i, :3] = torch.tensor(pos, device=dev, dtype=dtype) + world_pos[i]
            pose[i, 3:7] = torch.tensor(rot, device=dev, dtype=dtype)
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_idx)
        obj.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((k, 6), device=dev, dtype=dtype), env_ids=view_idx
        )
        if proto.kind == "ArticulationCfg":
            jpos = wp.to_torch(obj.data.default_joint_pos)[view_idx].clone()
            jvel = wp.to_torch(obj.data.default_joint_vel)[view_idx].clone()
            obj.write_joint_position_to_sim_index(position=jpos, env_ids=view_idx)
            obj.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=view_idx)


# ======================================================================
# Recipe A -- declarative cfg handed to InteractiveScene (high level)
# ======================================================================


class InteractiveSceneEngine(CloneEngine):
    """High-level cloning: assemble a scene cfg and let ``InteractiveScene`` do the rest."""

    PRIM_PREFIX = PRIM_PREFIX_INTERACTIVE_SCENE

    def build(self) -> None:
        # --- 1. assemble one declarative scene cfg --------------------------
        # One clone combination per task; a shared prototype name appears in every task
        # that uses it, so the cloner clones it into their union.
        clone_combinations = [InclusionSet(assets=list(t.prototype_names), weight=1) for t in self.tasks]

        namespace = _global_assets()
        namespace["clone_cfg"] = CloneCfg(clone_strategy=self.clone_strategy, clone_combinations=clone_combinations)
        for proto in self.prototypes:
            namespace[proto.name] = proto.cfg
        scene_cls = configclass(type("RegistrySceneCfg", (InteractiveSceneCfg,), namespace))

        # --- 2. hand it over: InteractiveScene lays out envs, clones, and builds ---
        # --- physics views; sim.reset() starts physics.                          ---
        cfg = scene_cls(num_envs=self.num_envs, env_spacing=self.env_spacing, replicate_physics=False)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.origins = self.scene.env_origins

    def iter_articulations(self):
        for proto in self.prototypes:
            if proto.name in self.scene.articulations:
                yield proto, self.scene.articulations[proto.name]

    def _instance(self, proto):
        if proto.name in self.scene.articulations:
            return self.scene.articulations[proto.name]
        return self.scene.rigid_objects[proto.name]

    def flush(self):
        self.scene.write_data_to_sim()

    def update(self, dt):
        self.scene.update(dt)


# ======================================================================
# Recipe B -- grid_transforms + usd_replicate + ReplicateSession (low level)
# ======================================================================


class ManualCloneEngine(CloneEngine):
    """Low-level cloning: lay out envs and clone every prototype by hand."""

    PRIM_PREFIX = PRIM_PREFIX_MANUAL

    def build(self) -> None:
        stage = self.sim.stage

        # --- 1. shared globals: one ground + light prim, spawned directly ---
        for cfg in _global_assets().values():
            cfg.spawn.func(cfg.prim_path, cfg.spawn)

        # --- 2. lay out env containers: clone an empty env_0 Xform to env_1..N ---
        stage.DefinePrim("/World/envs/env_0", "Xform")
        all_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.origins, _ = cloner.grid_transforms(self.num_envs, self.env_spacing, device=self.device)
        with cloner.disabled_fabric_change_notifies(stage, restore=False):
            cloner.usd_replicate(
                stage, ["/World/envs/env_0"], ["/World/envs/env_{}"], all_indices, positions=self.origins
            )

        # --- 3. heterogeneous valid set: one clone combination per task ---
        # valid_set has one column per clone group, and make_clone_plan only groups env-scoped
        # cfgs with a spawner -- so restrict columns (and each task's inclusion set) to clonable
        # prototypes to stay in lockstep with the plan. Non-clonable assets (e.g. surface
        # grippers) ride along inside their owner's env and are constructed below.
        clonable = [p for p in self.prototypes if _is_clone_group(p.cfg)]
        clonable_names = {p.name for p in clonable}
        valid_set = make_valid_clone_combinations(
            [p.name for p in clonable],
            [1] * len(clonable),  # multi-asset spawners were collapsed to one variant
            [InclusionSet(assets=[n for n in t.prototype_names if n in clonable_names], weight=1) for t in self.tasks],
            self.device,
        )

        # --- 4. spawn each prototype in env_0 and let the clone plan replicate it ---
        with ReplicateSession(
            [p.cfg for p in self.prototypes],
            num_clones=self.num_envs,
            env_spacing=self.env_spacing,
            device=self.device,
            stage=stage,
            clone_strategy=self.clone_strategy,
            valid_set=valid_set,
        ):
            # make_clone_plan (in __enter__) has pointed each cfg's spawn_path at env_0.
            for proto in self.prototypes:
                cfg = proto.cfg
                if cfg.class_type is not None:
                    # Articulation / RigidObject: constructor spawns + registers replication.
                    proto.instance = cfg.class_type(cfg)
                elif cfg.spawn is not None:
                    # Static prop (no physics view): spawn in env_0 + queue its USD copy.
                    init = getattr(cfg, "init_state", None)
                    cfg.spawn.func(
                        cfg.spawn.spawn_path,
                        cfg.spawn,
                        translation=getattr(init, "pos", None),
                        orientation=getattr(init, "rot", None),
                    )
                    cloner.queue_usd_replication(cfg)

        self.sim.reset()  # initialise physics views on every spawned prototype
        self._live = [p.instance for p in self.prototypes if p.instance is not None]

    def iter_articulations(self):
        for proto in self.prototypes:
            if proto.kind == "ArticulationCfg" and proto.instance is not None:
                yield proto, proto.instance

    def _instance(self, proto):
        return proto.instance

    def flush(self):
        for instance in self._live:
            instance.write_data_to_sim()

    def update(self, dt):
        for instance in self._live:
            instance.update(dt)
