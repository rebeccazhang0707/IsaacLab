#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Driver for the LIBERO scene-construction benchmark: sweeps 4 suites x 3 seeds
# at num_envs=800 (harvest+selector) and appends the machine-readable JSON
# summary lines to a log. See scripts/benchmarks/libero_scene_profiling.md for
# the analysis (which also compares against the now-removed per_task_group impl).
#
# Requires LIBERO_ASSETS_DATA_DIR / LIBERO_CONFIG_DIR to point at the LIBERO
# USD assets and per-task JSON config (defaults below match the local dataset).
set -u
cd "$(dirname "$0")/../.." || exit 1
export LIBERO_ASSETS_DATA_DIR="${LIBERO_ASSETS_DATA_DIR:-_RobotLearningLab/benchmarks/datasets/libero/USD}"
export LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-_RobotLearningLab/benchmarks/datasets/libero/config}"
LOG="${LIBERO_PROFILE_LOG:-/tmp/libero_profile_800.log}"
: > "$LOG"
NUM_ENVS=800
NUM_STEPS=300
WARMUP=50
for suite in spatial goal object long; do
  for rep in 0 1 2; do
    echo "=== RUN suite=$suite impl=harvest rep=$rep ==="
    ./isaaclab.sh -p scripts/benchmarks/profile_libero_scene.py \
      --suite "$suite" --impl harvest --num_envs "$NUM_ENVS" \
      --num_steps "$NUM_STEPS" --warmup_steps "$WARMUP" --repeats 1 \
      --seed "$rep" --headless presets=physx 2>&1 \
      | grep -E "LIBERO_PROFILE_JSON|\[SUMMARY\]|out of memory|CUDA error" | tee -a "$LOG"
  done
done
echo "=== DRIVER_DONE ==="
