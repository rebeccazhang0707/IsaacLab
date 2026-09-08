#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/model_20.pt" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly CHECKPOINT_PATH="$(realpath -- "$1")"
readonly BASELINE_COMMIT="1c89eb4382cd5dda79302055ca44af58d4891fa2"
readonly EXPECTED_CHECKPOINT_SHA256="17e78e49e28417bd6fcdad38a53587e71fc89da34ea24d1a6badf56ab08ce502"

readonly SHOELACE_GPU="${SHOELACE_GPU:-0}"
readonly SHOELACE_LEVEL="${SHOELACE_LEVEL:-91}"
readonly SHOELACE_NUM_ENVS="${SHOELACE_NUM_ENVS:-1}"
readonly SHOELACE_SEED="${SHOELACE_SEED:-1220}"
readonly SHOELACE_VISUALIZER="${SHOELACE_VISUALIZER:-newton_gl}"

actual_checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | cut -d ' ' -f 1)"
if [[ "${actual_checkpoint_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "Warning: checkpoint SHA256 does not match the validated model_20.pt." >&2
    echo "Expected: ${EXPECTED_CHECKPOINT_SHA256}" >&2
    echo "Actual:   ${actual_checkpoint_sha256}" >&2
fi

# Evaluate against the exact 62-observation source even when the active branch uses the 68-observation interface.
eval_worktree="$(mktemp -d "${TMPDIR:-/tmp}/shoelace-model20-eval.XXXXXX")"
rmdir -- "${eval_worktree}"
git -C "${REPO_ROOT}" worktree add --quiet --detach "${eval_worktree}" "${BASELINE_COMMIT}"

cleanup() {
    git -C "${REPO_ROOT}" worktree remove --force "${eval_worktree}" >/dev/null
}
trap cleanup EXIT

cd -- "${eval_worktree}"

PYTHONPATH="${eval_worktree}/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}" \
uv run --project "${REPO_ROOT}" isaaclab play \
    --rl_library rsl_rl \
    --task IsaacContrib-Shoelace-DualFranka \
    --checkpoint "${CHECKPOINT_PATH}" \
    --num_envs "${SHOELACE_NUM_ENVS}" \
    --seed "${SHOELACE_SEED}" \
    --train_env_cfg \
    --device "cuda:${SHOELACE_GPU}" \
    --visualizer "${SHOELACE_VISUALIZER}" \
    agent.actor.class_name=isaaclab_tasks.contrib.shoelace.agents.models:FrozenGripperOutputHeadFixedObservationStatisticsMLPModel \
    agent.critic.class_name=isaaclab_tasks.contrib.shoelace.agents.models:FixedObservationStatisticsMLPModel \
    env.coupling_mode=admm \
    env.curriculum.pull_to_grasp.params.initial_level="${SHOELACE_LEVEL}" \
    env.curriculum.pull_to_grasp.params.minimum_episodes=1000000 \
    env.curriculum.pull_to_grasp.params.current_level_fraction=1.0 \
    'env.curriculum.pull_to_grasp.params.current_level_fraction_schedule=[1.0]'
