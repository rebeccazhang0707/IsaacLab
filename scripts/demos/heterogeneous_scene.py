# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a multi-robot scene from task configs and step physics only.

The pipeline is: gather registered task scenes, filter scenes whose floor is
not at level 0, fold them together with :func:`~isaaclab.scene.add`
while skipping every task light and floor, then add one Dome light and one
shared ground plane. No task environments or MDP managers are constructed;
the demo owns generic PhysX simulation settings. ``-Play`` task variants are
excluded up front; Newton scenes and scenes without a declarative level-0
floor are reported and skipped.

.. note::
    The heterogeneous composition (each environment holds one task's assets) requires
    the PhysX backend. Newton's ``ArticulationView`` requires every asset to exist in
    every environment with an identical topology. With ``--physics newton_mjwarp`` the
    demo therefore collapses to a single environment that holds every task, shifting
    each task's assets to a per-task sub-origin so the tasks sit side by side.

.. code-block:: bash

    # Usage with every supported registered task scene.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py

    # Usage with a smaller composition.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py --num_task 3 --num_envs 3

    # Kitless Newton (MJWarp) physics with the Newton visualizer (no Isaac Sim).
    # Note: collapses to a single environment holding every task at per-task offsets.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py \
        --physics newton_mjwarp --visualizer newton

"""

from __future__ import annotations

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Demo: clone-only multi-robot multi-task scene.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0, help="Physics timestep [s].")
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of tasks to use from the default order. Omit to use all tasks.",
)
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()
# strip consumed args so hydra-based task-config resolution does not re-parse them
sys.argv = [sys.argv[0], *hydra_args]

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import grid_transforms, sequential
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add
from isaaclab.terrains import TerrainImporterCfg

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip

from isaaclab_tasks.utils import resolve_task_config


def _registered_task_ids() -> list[str]:
    """Return registered task IDs owned by isaaclab_tasks, skipping ``-Play`` variants."""
    task_ids = []
    for task_spec in gym.registry.values():
        if task_spec.id.endswith("-Play"):
            continue
        entry_point = task_spec.kwargs.get("env_cfg_entry_point")
        if isinstance(entry_point, str):
            module_name = entry_point.split(":", maxsplit=1)[0]
        else:
            module_name = getattr(entry_point, "__module__", type(entry_point).__module__)
        if module_name.startswith("isaaclab_tasks."):
            task_ids.append(task_spec.id)
    return sorted(task_ids)


def reject_scene(env_cfg: object) -> str | None:
    """Return why one resolved task config is outside this demo's scope."""
    # inspect cfg metadata only: importing a backend package to type-check would
    # defeat the resolvable class_type pattern (backends load via cfg resolution).
    physics_cfg = env_cfg.sim.physics
    if physics_cfg is not None and any(
        cls.__module__.startswith("isaaclab_newton.") for cls in type(physics_cfg).__mro__
    ):
        return f"Newton physics config {type(physics_cfg).__name__}"

    scene_cfg = env_cfg.scene
    fields = [value for name, value in vars(scene_cfg).items() if name not in InteractiveSceneCfg.__dataclass_fields__]
    # flat terrain importers always place their plane at the stage origin
    floor_levels = [0.0 for value in fields if isinstance(value, TerrainImporterCfg) and value.terrain_type == "plane"]
    floor_levels += [
        value.init_state.pos[2]
        for value in fields
        if isinstance(value, AssetBaseCfg) and isinstance(value.spawn, sim_utils.GroundPlaneCfg)
    ]
    if not floor_levels:
        return "no declarative flat floor"
    mismatched = [level for level in floor_levels if abs(level) > 1e-6]
    if mismatched:
        return f"floor level {mismatched[0]} is not 0"
    return None


def _load_task_scenes() -> tuple[list[str], list[InteractiveSceneCfg]]:
    """Gather every registered task scene and filter unsupported ones."""
    accepted_ids, accepted_scenes, skipped = [], [], []
    for task_id in _registered_task_ids():
        env_cfg, _ = resolve_task_config(task_id, "")
        reason = reject_scene(env_cfg)
        if reason is not None:
            skipped.append((task_id, reason))
            continue
        accepted_ids.append(task_id)
        accepted_scenes.append(env_cfg.scene)
        if args_cli.num_task is not None and len(accepted_ids) == args_cli.num_task:
            break

    if skipped:
        print("\n[INFO] Skipped task scenes outside the composition scope:")
        for task_id, reason in skipped:
            print(f"  {task_id}: {reason}")
    if len(accepted_ids) < 2:
        raise ValueError("Select at least two supported task scenes.")
    return accepted_ids, accepted_scenes


def main() -> None:
    """Gather task scenes, filter by floor level, compose, add light and floor, simulate."""
    # Resolve and compose every task scene before Kit launches: config resolution is
    # simulator-free, and the launch swaps module state that must not interleave with it.
    task_ids, task_scene_cfgs = _load_task_scenes()
    print(f"\n[INFO] Composing task scenes: {task_ids}")
    for task_scene_cfg in task_scene_cfgs:
        task_scene_cfg.env_spacing = args_cli.env_spacing

    # Newton's ArticulationView requires every asset to exist in every environment, so
    # the per-task heterogeneous clone combinations cannot be used. Instead, tile every
    # task inside each environment: shift each task's assets to a per-task sub-origin.
    newton_homogeneous = args_cli.physics == "newton_mjwarp"
    if newton_homogeneous:
        print("[INFO] Newton backend: tiling every task at a per-task offset inside one environment.")
        task_offsets, _ = grid_transforms(len(task_scene_cfgs), args_cli.env_spacing)
        for task_scene_cfg, offset in zip(task_scene_cfgs, task_offsets):
            for name, value in vars(task_scene_cfg).items():
                if name in InteractiveSceneCfg.__dataclass_fields__ or not isinstance(value, AssetBaseCfg):
                    continue
                pos = value.init_state.pos
                value.init_state.pos = (pos[0] + float(offset[0]), pos[1] + float(offset[1]), pos[2])

    scene_cfg = task_scene_cfgs[0]

    def is_global_asset(a: AssetBaseCfg) -> bool:
        return isinstance(a.spawn, (sim_utils.LightCfg, sim_utils.GroundPlaneCfg))

    for task_scene_cfg in task_scene_cfgs[1:]:
        scene_cfg = scene_add(scene_cfg, task_scene_cfg, asset_skip=is_global_asset)
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.replicate_physics = True
    scene_cfg.clone_cfg.clone_strategy = sequential
    if newton_homogeneous:
        scene_cfg.clone_cfg.clone_combinations = []
        scene_cfg.num_envs = 1
        print(f"[INFO] Newton backend: single environment holding all {len(task_ids)} tasks at per-task offsets.")

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # The default newton mjwarp solver configuration needs to be tuned for this demo.
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            # Some task assets (e.g. the sorting-scale screen frame) carry planar mesh colliders
            # that MuJoCo's contact generation rejects; let Newton's collision pipeline supply instead.
            physics_cfg.solver_cfg.use_mujoco_contacts = False
            physics_cfg.solver_cfg.nconmax = 1024
            physics_cfg.solver_cfg.njmax = 2048
            # The default explicit euler integrator diverges on the small-mass, contact-rich
            # task robots (e.g. Ant) at this timestep; match the task presets and arms.py.
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.num_substeps = 2

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print(f"[INFO] Composed {len(task_ids)} task scenes into {scene_cfg.num_envs} environments. Stepping physics.")

        sim_dt = sim.get_physics_dt()
        # Step while a visualizer window is still open (or none exist, e.g. headless).
        while sim.is_headless_or_exist_active_visualizer():
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)


if __name__ == "__main__":
    main()
