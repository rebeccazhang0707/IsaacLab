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

.. code-block:: bash

    # Usage with every supported registered task scene.
    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py

    # Usage with a smaller composition.
    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py --num_task 3 --num_envs 3

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
parser.add_argument("--physics", default="physx", choices=["physx"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()
# strip consumed args so hydra-based task-config resolution does not re-parse them
sys.argv = [sys.argv[0], *hydra_args]

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add
from isaaclab.terrains import TerrainImporterCfg

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

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print(f"[INFO] Composed {len(task_ids)} task scenes into {args_cli.num_envs} environments. Stepping physics.")

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
