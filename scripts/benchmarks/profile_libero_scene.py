# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark LIBERO scene construction (DGPO harvest+OSC).

Scenes use the ``harvest`` implementation -- object-level prototype sharing:
identical object models are de-duplicated into one shared
:class:`~isaaclab.assets.AssetView`, and all task-dependent MDP is gathered per
env by task id. Profiling always builds the DGPO OSC env
(``Isaac-Libero-*-Dgpo-Osc-*`` layout; optional demos).

``--impl harvest`` and ``--impl compat_osc`` are aliases for the same DGPO path
(kept so historical ``libero_scene_profiling.md`` / ``run_libero_profile.sh``
commands still work).

Usage (PhysX backend, headless)::

    LIBERO_ASSETS_DATA_DIR=<usd> LIBERO_CONFIG_DIR=<config> \
    ./isaaclab.sh -p scripts/benchmarks/profile_libero_scene.py \
        --suite object --impl harvest --num_envs 64 \
        --num_steps 300 --warmup_steps 50 --repeats 3 --seed 0 --headless presets=physx

    LIBERO_ASSETS_DATA_DIR=<usd> LIBERO_CONFIG_DIR=<config> \
    ./isaaclab.sh -p scripts/benchmarks/profile_libero_scene.py \
        --suite all --impl compat_osc --num_envs 40 \
        --num_steps 50 --warmup_steps 10 --repeats 1 --seed 0 --headless presets=physx
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import setup_preset_cli

# ── argparse + AppLauncher args BEFORE any sim/torch import ──────────────────
parser = argparse.ArgumentParser(description="Profile LIBERO scene construction (DGPO harvest+OSC).")
parser.add_argument(
    "--suite",
    type=str,
    default="object",
    choices=("spatial", "goal", "object", "long", "all"),
    help="LIBERO suite (use 'all' for the 40-task DGPO env).",
)
parser.add_argument(
    "--impl",
    type=str,
    default="harvest",
    choices=("harvest", "compat_osc"),
    help="Alias for the DGPO harvest+OSC path (both choices build the same env).",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments (must be >= suite task count).")
parser.add_argument("--num_steps", type=int, default=300, help="Measured steps per repeat.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps (excluded from timing).")
parser.add_argument("--repeats", type=int, default=3, help="Measurement repeats (each emits its own JSON line).")
parser.add_argument("--seed", type=int, default=0, help="Seed for the fixed action stream (shared across impls).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# ── Heavy imports (after argv is finalized) ─────────────────────────────────
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402  (registers LIBERO gyms + imports the package)

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402


def _build_env_cfg_cls(suite: str, num_envs: int):
    """Return a DGPO OSC env cfg class for the selected suite."""
    from isaaclab_contrib.tasks.manipulation.libero.dgpo_layout import DGPO_ABC_HARVEST_SUITES
    from isaaclab_contrib.tasks.manipulation.libero.envs.dgpo_env_cfg import make_libero_dgpo_env_cfg

    if suite == "all":
        return make_libero_dgpo_env_cfg(suites=DGPO_ABC_HARVEST_SUITES, num_envs=num_envs)
    by_short = {prefix: (suite_name, prefix) for suite_name, prefix in DGPO_ABC_HARVEST_SUITES}
    selected = by_short.get(suite)
    if selected is None:
        raise ValueError(f"Unknown suite {suite!r} (use long|object|spatial|goal|all).")
    return make_libero_dgpo_env_cfg(suites=(selected,), num_envs=num_envs)


# Register the selected implementation as a temporary gym id so it resolves
# through the same hydra/preset pipeline as the shipped envs.
_TASK_ID = f"Isaac-Libero-Profile-{args_cli.suite.title()}-{args_cli.impl}-v0"
_ENV_CFG_CLS = _build_env_cfg_cls(args_cli.suite, args_cli.num_envs)
_ENTRY = "isaaclab_contrib.tasks.manipulation.libero.envs.dgpo_env:DgpoManagerBasedRLEnv"
gym.register(
    id=_TASK_ID,
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": _ENV_CFG_CLS},
)


def _count_asset_views(scene) -> dict[str, int]:
    """Count spawned AssetViews (articulations + rigid objects) in the scene."""
    n_art = len(getattr(scene, "articulations", {}))
    n_rigid = len(getattr(scene, "rigid_objects", {}))
    return {"articulations": n_art, "rigid_objects": n_rigid, "total_asset_views": n_art + n_rigid}


def main(env_cfg) -> None:
    """Build the env, warm up, then time ``--num_steps`` with a fixed action stream."""
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    device = env_cfg.sim.device

    torch.manual_seed(args_cli.seed)

    # ── Build (scene construction) ──────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    build_t0 = time.perf_counter()
    env = gym.make(_TASK_ID, cfg=env_cfg)
    torch.cuda.synchronize()
    build_time_s = time.perf_counter() - build_t0

    unwrapped = env.unwrapped
    asset_views = _count_asset_views(unwrapped.scene)
    action_dim = unwrapped.action_manager.total_action_dim
    num_envs = unwrapped.num_envs

    # Fixed action stream: identical actions across impls for the same seed.
    generator = torch.Generator(device=device).manual_seed(args_cli.seed)
    total_steps = args_cli.warmup_steps + args_cli.repeats * args_cli.num_steps
    actions = 2.0 * torch.rand((total_steps, num_envs, action_dim), generator=generator, device=device) - 1.0

    env.reset()

    # ── Warmup (excluded) ───────────────────────────────────────────────────
    with torch.inference_mode():
        for i in range(args_cli.warmup_steps):
            env.step(actions[i])
    torch.cuda.synchronize()

    # ── Measured repeats ────────────────────────────────────────────────────
    idx = args_cli.warmup_steps
    results = []
    for repeat in range(args_cli.repeats):
        step_times_ms: list[float] = []
        with torch.inference_mode():
            for _ in range(args_cli.num_steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                env.step(actions[idx])
                torch.cuda.synchronize()
                step_times_ms.append((time.perf_counter() - t0) * 1e3)
                idx += 1
        total_step_time_s = sum(step_times_ms) / 1e3
        throughput = num_envs * args_cli.num_steps / total_step_time_s
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        summary = {
            "suite": args_cli.suite,
            "impl": args_cli.impl,
            "num_envs": num_envs,
            "num_steps": args_cli.num_steps,
            "repeat": repeat,
            "seed": args_cli.seed,
            "action_dim": action_dim,
            "build_time_s": round(build_time_s, 4),
            "step_time_mean_ms": round(statistics.mean(step_times_ms), 4),
            "step_time_median_ms": round(statistics.median(step_times_ms), 4),
            "step_time_std_ms": round(statistics.pstdev(step_times_ms), 4),
            "throughput_env_steps_per_s": round(throughput, 2),
            "peak_gpu_mem_mb": round(peak_mem_mb, 2),
            **asset_views,
        }
        results.append(summary)
        print("LIBERO_PROFILE_JSON " + json.dumps(summary), flush=True)

    env.close()

    mean_tp = statistics.mean(r["throughput_env_steps_per_s"] for r in results)
    mean_step = statistics.mean(r["step_time_mean_ms"] for r in results)
    print(
        f"\n[SUMMARY] suite={args_cli.suite} impl={args_cli.impl} num_envs={num_envs} "
        f"asset_views={asset_views['total_asset_views']} build={build_time_s:.2f}s "
        f"step={mean_step:.3f}ms throughput={mean_tp:.0f} env-steps/s",
        flush=True,
    )


if __name__ == "__main__":
    env_cfg, _ = resolve_task_config(_TASK_ID, None)
    # Guard: if no ``presets=physx`` was passed, still collapse the physics
    # PresetCfg to PhysX so launch_simulation boots the Kit/PhysX runtime.
    from isaaclab_tasks.utils import PresetCfg

    if isinstance(getattr(env_cfg.sim, "physics", None), PresetCfg):
        resolve_presets(env_cfg, selected=("physx",))

    with launch_simulation(env_cfg, args_cli):
        main(env_cfg)
