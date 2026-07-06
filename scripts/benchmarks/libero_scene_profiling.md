# LIBERO scene-construction profiling: `harvest` vs `per_task_group`

> **Note:** Based on the results below, the `per_task_group` implementation was
> subsequently **removed** — only `harvest` remains in the codebase (all LIBERO
> gym ids now resolve to the harvest+selector factory). This document is retained
> as the historical record of the comparison that motivated that decision.

## What was compared

Two scene-construction implementations for the LIBERO suites, benchmarked
per single suite with [`profile_libero_scene.py`](./profile_libero_scene.py):

- **`harvest`** — object-level prototype sharing (the heterogeneous
  cloner + selector approach). Identical object models across tasks are
  de-duplicated into a single shared
  [`AssetView`](../../source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/libero/tasks/harvest/prototypes.py);
  all task-dependent MDP is gathered per env by task id
  (`make_libero_combined_env_cfg`).
- **`per_task_group`** — the earlier approach: every task is its own clone
  group with its own (un-shared) fixture + object `AssetView`s
  (`tasks.per_task_group.data_driven.build_*_tasks`).

Both configurations share the same Franka robot, differential-IK + gripper
actions, robot observations, penalty curriculum, and the `goal` reward mode;
they differ **only** in scene construction and where the task-dependent MDP is
scoped. `env_spacing` is matched per suite (3.0 m for `long` to fit
`study_table`, else 2.5 m) so the comparison is fair.

## Environment

| Item | Value |
|---|---|
| GPU | single NVIDIA RTX 5880 Ada |
| Physics | `presets=physx` (PhysX backend) |
| Reward mode | `goal` (sparse success bonus) |
| `num_envs` | 800 |
| Measured steps | 300 (per repeat) |
| Warmup steps | 50 (excluded) |
| Seeds / repeats | 3 seeds (`--seed 0/1/2`, one repeat each) |
| Mode | headless |

The `num_envs=1000` run was attempted first but hit a PhysX
`CUDA error: illegal memory access` in `GpuRigidBodyView`, so the sweep was
stepped down to `num_envs=800`, the reliable ceiling on this machine.

## Reproduction

Set the LIBERO asset/config locations, then run the driver script (which loops
4 suites × 2 impls × 3 seeds at `num_envs=800`):

```bash
export LIBERO_ASSETS_DATA_DIR=<path>/libero/USD
export LIBERO_CONFIG_DIR=<path>/libero/config
./scripts/benchmarks/run_libero_profile.sh
```

A single measurement can be reproduced directly:

```bash
LIBERO_ASSETS_DATA_DIR=<usd> LIBERO_CONFIG_DIR=<config> \
./isaaclab.sh -p scripts/benchmarks/profile_libero_scene.py \
    --suite goal --impl harvest --num_envs 800 \
    --num_steps 300 --warmup_steps 50 --repeats 1 --seed 0 \
    --headless presets=physx
```

## Results (`num_envs=800`, mean over 3 seeds)

| suite | impl | AssetViews | build [s] | step [ms] | throughput [env-steps/s] | GPU [MB] |
|---|---|---:|---:|---:|---:|---:|
| spatial | harvest | 8 | 4.29 | 498.9 | 1605 | 19.9 |
| spatial | per_task_group | 71 | 5.96 | 843.8 | 948 | 21.7 |
| goal | harvest | 8 | 4.25 | 174.9 | 4575 | 19.9 |
| goal | per_task_group | 71 | 5.85 | 1114.4 | 718 | 21.7 |
| object | harvest | 12 | 3.90 | 50.1 | 15981 | 19.9 |
| object | per_task_group | 71 | 5.21 | 61.7 | 12978 | 22.2 |
| long | harvest | 26 | 4.51 | 79.9 | 10014 | 20.6 |
| long | per_task_group | 47 | 4.98 | 77.2 | 10359 | 21.2 |

## Throughput speedup (harvest ÷ per_task_group)

Comparison of the speedup at `num_envs=800` against an earlier `num_envs=64`
run to test the "advantage narrows at scale" hypothesis:

| suite | speedup @ 64 | speedup @ 800 | trend |
|---|---|---|---|
| spatial | 1.39× | 1.69× | grows |
| goal | n/a (`per_task_group` previously crashed) | 6.37× | n/a → dramatic |
| object | 1.28× | 1.23× | ~flat |
| long | 0.97× | 0.97× | parity |

## Conclusion

The **"advantage narrows at scale" hypothesis is REFUTED for shared-object
suites.** The `harvest` advantage persists (`object`) or grows (`spatial`),
and is dramatic for `goal` (6.37×). Only `long` — the suite with the least
object reuse (26 vs 47 views, versus 8 vs 71 for spatial/goal) — stays at
parity.

- **`goal` is the extreme case (6.37×)** because `per_task_group` simulates
  **21 articulations** (10 cabinets + 10 stoves + the robot) plus 50 rigid
  objects = 71 `AssetView`s, whereas `harvest` collapses the shared kitchen
  set to just **8** views. Articulation simulation dominates step time, so the
  view-count blow-up hurts `goal` the most.
- **Mechanism:** the per-`AssetView` fixed cost (kernel dispatch +
  reset/articulation simulation) grows roughly linearly with view count and is
  **not** amortized by batched physics at 800 envs. Fewer, wider (shared)
  views therefore keep winning as env count rises.
- **`long` stays flat** because its tasks reuse few models across tasks, so the
  two implementations end up with a similar view count (26 vs 47) and similar
  per-step cost.

Build time and peak GPU memory also favor `harvest` across every suite (fewer
distinct scene entities to spawn and track), though the differences there are
small (sub-2 s build, ~2 MB memory).

## Caveats

- Single GPU (RTX 5880 Ada), 3 seeds only.
- `num_envs=1000` is unstable on this machine: PhysX raises
  `CUDA error: illegal memory access` in `GpuRigidBodyView`. `num_envs=800` is
  the reliable ceiling used for all numbers here.
- `goal` / `per_task_group` has a large step-time standard deviation
  (~830 ms), so its mean step time is noisy; the throughput ranking is
  nonetheless unambiguous.
- These are scene-construction / stepping microbenchmarks with a fixed random
  action stream (identical actions per seed across impls); they do not measure
  learning throughput or final task success.

## Appendix: raw `LIBERO_PROFILE_JSON` lines (`num_envs=800`)

Extracted from `/tmp/libero_profile_800.log` (`grep LIBERO_PROFILE_JSON`).
Each line is one seed's single-repeat measurement; the table above is the mean
over the three seeds per (suite, impl).

```json
{"suite": "spatial", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 4.186, "step_time_mean_ms": 511.9314, "step_time_median_ms": 536.1201, "step_time_std_ms": 169.4429, "throughput_env_steps_per_s": 1562.71, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "spatial", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 4.1751, "step_time_mean_ms": 478.0429, "step_time_median_ms": 485.6347, "step_time_std_ms": 153.9827, "throughput_env_steps_per_s": 1673.49, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "spatial", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 4.5126, "step_time_mean_ms": 506.6866, "step_time_median_ms": 479.2244, "step_time_std_ms": 203.3048, "throughput_env_steps_per_s": 1578.89, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "spatial", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 5.8372, "step_time_mean_ms": 839.0215, "step_time_median_ms": 859.7013, "step_time_std_ms": 300.416, "throughput_env_steps_per_s": 953.49, "peak_gpu_mem_mb": 21.71, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "spatial", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 5.9419, "step_time_mean_ms": 858.8523, "step_time_median_ms": 887.7877, "step_time_std_ms": 306.2578, "throughput_env_steps_per_s": 931.48, "peak_gpu_mem_mb": 21.7, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "spatial", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 6.0945, "step_time_mean_ms": 833.5332, "step_time_median_ms": 875.0208, "step_time_std_ms": 296.2747, "throughput_env_steps_per_s": 959.77, "peak_gpu_mem_mb": 21.68, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "goal", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 4.2484, "step_time_mean_ms": 175.9383, "step_time_median_ms": 166.6859, "step_time_std_ms": 32.33, "throughput_env_steps_per_s": 4547.05, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "goal", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 4.2396, "step_time_mean_ms": 174.8754, "step_time_median_ms": 165.4058, "step_time_std_ms": 32.2806, "throughput_env_steps_per_s": 4574.69, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "goal", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 4.2539, "step_time_mean_ms": 173.7968, "step_time_median_ms": 165.5387, "step_time_std_ms": 31.588, "throughput_env_steps_per_s": 4603.08, "peak_gpu_mem_mb": 19.9, "articulations": 3, "rigid_objects": 5, "total_asset_views": 8}
{"suite": "goal", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 5.7879, "step_time_mean_ms": 1110.6609, "step_time_median_ms": 1077.0867, "step_time_std_ms": 834.253, "throughput_env_steps_per_s": 720.29, "peak_gpu_mem_mb": 21.67, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "goal", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 5.969, "step_time_mean_ms": 1119.8084, "step_time_median_ms": 1090.012, "step_time_std_ms": 834.6166, "throughput_env_steps_per_s": 714.41, "peak_gpu_mem_mb": 21.66, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "goal", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 5.8005, "step_time_mean_ms": 1112.7328, "step_time_median_ms": 1083.6506, "step_time_std_ms": 830.7578, "throughput_env_steps_per_s": 718.95, "peak_gpu_mem_mb": 21.64, "articulations": 21, "rigid_objects": 50, "total_asset_views": 71}
{"suite": "object", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 4.0227, "step_time_mean_ms": 48.9181, "step_time_median_ms": 42.1451, "step_time_std_ms": 18.1683, "throughput_env_steps_per_s": 16353.85, "peak_gpu_mem_mb": 19.9, "articulations": 1, "rigid_objects": 11, "total_asset_views": 12}
{"suite": "object", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 3.8211, "step_time_mean_ms": 51.5001, "step_time_median_ms": 43.5874, "step_time_std_ms": 19.6948, "throughput_env_steps_per_s": 15533.96, "peak_gpu_mem_mb": 19.9, "articulations": 1, "rigid_objects": 11, "total_asset_views": 12}
{"suite": "object", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 3.8683, "step_time_mean_ms": 49.8244, "step_time_median_ms": 42.9336, "step_time_std_ms": 18.7319, "throughput_env_steps_per_s": 16056.4, "peak_gpu_mem_mb": 19.9, "articulations": 1, "rigid_objects": 11, "total_asset_views": 12}
{"suite": "object", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 5.229, "step_time_mean_ms": 62.748, "step_time_median_ms": 50.8668, "step_time_std_ms": 29.1196, "throughput_env_steps_per_s": 12749.42, "peak_gpu_mem_mb": 22.18, "articulations": 1, "rigid_objects": 70, "total_asset_views": 71}
{"suite": "object", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 5.2134, "step_time_mean_ms": 60.344, "step_time_median_ms": 50.3514, "step_time_std_ms": 26.6004, "throughput_env_steps_per_s": 13257.34, "peak_gpu_mem_mb": 22.17, "articulations": 1, "rigid_objects": 70, "total_asset_views": 71}
{"suite": "object", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 5.19, "step_time_mean_ms": 61.8838, "step_time_median_ms": 51.2669, "step_time_std_ms": 27.5294, "throughput_env_steps_per_s": 12927.46, "peak_gpu_mem_mb": 22.13, "articulations": 1, "rigid_objects": 70, "total_asset_views": 71}
{"suite": "long", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 4.3725, "step_time_mean_ms": 80.1532, "step_time_median_ms": 73.4803, "step_time_std_ms": 16.089, "throughput_env_steps_per_s": 9980.89, "peak_gpu_mem_mb": 20.61, "articulations": 4, "rigid_objects": 22, "total_asset_views": 26}
{"suite": "long", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 4.8156, "step_time_mean_ms": 78.492, "step_time_median_ms": 72.1393, "step_time_std_ms": 15.5318, "throughput_env_steps_per_s": 10192.13, "peak_gpu_mem_mb": 20.61, "articulations": 4, "rigid_objects": 22, "total_asset_views": 26}
{"suite": "long", "impl": "harvest", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 4.3297, "step_time_mean_ms": 81.0724, "step_time_median_ms": 73.0324, "step_time_std_ms": 17.3165, "throughput_env_steps_per_s": 9867.72, "peak_gpu_mem_mb": 20.61, "articulations": 4, "rigid_objects": 22, "total_asset_views": 26}
{"suite": "long", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 0, "action_dim": 7, "build_time_s": 4.9553, "step_time_mean_ms": 75.8077, "step_time_median_ms": 67.9056, "step_time_std_ms": 20.0129, "throughput_env_steps_per_s": 10553.01, "peak_gpu_mem_mb": 21.24, "articulations": 5, "rigid_objects": 42, "total_asset_views": 47}
{"suite": "long", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 1, "action_dim": 7, "build_time_s": 5.0003, "step_time_mean_ms": 77.539, "step_time_median_ms": 69.0999, "step_time_std_ms": 20.7698, "throughput_env_steps_per_s": 10317.38, "peak_gpu_mem_mb": 21.23, "articulations": 5, "rigid_objects": 42, "total_asset_views": 47}
{"suite": "long", "impl": "per_task_group", "num_envs": 800, "num_steps": 300, "repeat": 0, "seed": 2, "action_dim": 7, "build_time_s": 4.9923, "step_time_mean_ms": 78.388, "step_time_median_ms": 69.7668, "step_time_std_ms": 22.0641, "throughput_env_steps_per_s": 10205.65, "peak_gpu_mem_mb": 21.2, "articulations": 5, "rigid_objects": 42, "total_asset_views": 47}
```
