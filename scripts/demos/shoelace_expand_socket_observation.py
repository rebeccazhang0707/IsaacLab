# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expand a 62-value shoelace checkpoint for the appended contact-socket observation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

_INSERT_INDEX = 36
_ADDED_DIMENSIONS = 6


def _insert_columns(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Insert the socket observation columns before the tail-to-knot features."""
    inserted_shape = (*tensor.shape[:-1], _ADDED_DIMENSIONS)
    inserted = torch.full(inserted_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor[..., :_INSERT_INDEX], inserted, tensor[..., _INSERT_INDEX:]), dim=-1)


def expand_checkpoint(checkpoint: dict) -> dict:
    """Return a checkpoint whose new socket inputs initially leave actor and critic outputs unchanged."""
    expanded = copy.deepcopy(checkpoint)
    actor_state = expanded["actor_state_dict"]
    critic_state = expanded["critic_state_dict"]
    expected_shapes = {
        "actor observation mean": (1, 62),
        "actor input weight": (256, 62),
        "critic observation mean": (1, 63),
        "critic input weight": (256, 63),
    }
    actual_shapes = {
        "actor observation mean": tuple(actor_state["obs_normalizer._mean"].shape),
        "actor input weight": tuple(actor_state["mlp.0.weight"].shape),
        "critic observation mean": tuple(critic_state["obs_normalizer._mean"].shape),
        "critic input weight": tuple(critic_state["mlp.0.weight"].shape),
    }
    mismatches = {
        name: (actual_shapes[name], expected)
        for name, expected in expected_shapes.items()
        if actual_shapes[name] != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint does not use the 62-value shoelace observation contract: {mismatches}")

    for state in (actor_state, critic_state):
        state["obs_normalizer._mean"] = _insert_columns(state["obs_normalizer._mean"], 0.0)
        state["obs_normalizer._var"] = _insert_columns(state["obs_normalizer._var"], 1.0)
        state["obs_normalizer._std"] = _insert_columns(state["obs_normalizer._std"], 1.0)
        state["mlp.0.weight"] = _insert_columns(state["mlp.0.weight"], 0.0)

    expanded["optimizer_state_dict"]["state"] = {}
    expanded["iter"] = 0
    previous_infos = expanded.get("infos")
    expanded["infos"] = dict(previous_infos) if isinstance(previous_infos, dict) else {}
    expanded["infos"]["shoelace_socket_observation"] = {
        "insert_index": _INSERT_INDEX,
        "added_dimensions": _ADDED_DIMENSIONS,
        "initial_normalizer_mean": 0.0,
        "initial_normalizer_variance": 1.0,
        "initial_input_weights": 0.0,
    }
    return expanded


def main() -> None:
    """Parse arguments and save an expanded RSL-RL checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Output must not overwrite the source checkpoint.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expanded = expand_checkpoint(checkpoint)
    expanded["infos"]["shoelace_socket_observation"]["source_checkpoint"] = str(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, args.output)
    print(
        "SHOELACE_SOCKET_OBSERVATION_RESULT="
        + json.dumps(
            {
                "source": str(args.checkpoint.resolve()),
                "output": str(args.output.resolve()),
                "actor_observations": expanded["actor_state_dict"]["mlp.0.weight"].shape[1],
                "critic_observations": expanded["critic_state_dict"]["mlp.0.weight"].shape[1],
            }
        )
    )


if __name__ == "__main__":
    main()
