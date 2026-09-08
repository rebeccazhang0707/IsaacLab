# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit a shoelace actor to an approach demonstration while retaining a contact policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
_GRIPPER_INDICES = (6, 13)
_FINGER_OBSERVATION_INDICES = (28, 29)
_MLP_KEYS = (
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
)


class _ShoelaceActor(nn.Module):
    """Standard shoelace MLP with fixed checkpoint observation statistics."""

    def __init__(self, actor_state: dict[str, torch.Tensor]):
        super().__init__()
        missing_keys = [
            key for key in ("obs_normalizer._mean", "obs_normalizer._std", *_MLP_KEYS) if key not in actor_state
        ]
        if missing_keys:
            raise ValueError(f"Checkpoint does not contain the standard shoelace actor keys: {missing_keys}")
        input_dim = actor_state["mlp.0.weight"].shape[1]
        first_hidden_dim = actor_state["mlp.0.weight"].shape[0]
        second_hidden_dim = actor_state["mlp.2.weight"].shape[0]
        output_dim = actor_state["mlp.4.weight"].shape[0]
        if output_dim != 14:
            raise ValueError(f"Expected a 14-action shoelace actor, got {output_dim} outputs.")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.ELU(),
            nn.Linear(first_hidden_dim, second_hidden_dim),
            nn.ELU(),
            nn.Linear(second_hidden_dim, output_dim),
        )
        self.register_buffer("observation_mean", actor_state["obs_normalizer._mean"].clone())
        self.register_buffer("observation_std", actor_state["obs_normalizer._std"].clone())
        with torch.no_grad():
            for key in _MLP_KEYS:
                self.mlp.state_dict()[key.removeprefix("mlp.")].copy_(actor_state[key])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return unscaled policy MLP outputs."""
        normalized = (observations - self.observation_mean) / (self.observation_std + 1.0e-2)
        return self.mlp(normalized)


def _load_dataset(path: Path) -> dict:
    """Load and validate one shoelace trajectory dataset."""
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = ("observations", "actions", "phases", "trajectory_lengths")
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        raise ValueError(f"Dataset {path} is missing keys: {missing_keys}")
    observations = dataset["observations"]
    actions = dataset["actions"]
    phases = dataset["phases"]
    trajectory_lengths = dataset["trajectory_lengths"]
    if observations.ndim != 2 or observations.shape[1] != 62:
        raise ValueError(f"Dataset {path} observations must have shape [N, 62], got {tuple(observations.shape)}.")
    if actions.shape != (observations.shape[0], 14):
        raise ValueError(f"Dataset {path} actions must have shape [N, 14], got {tuple(actions.shape)}.")
    if len(phases) != observations.shape[0] or sum(trajectory_lengths) != observations.shape[0]:
        raise ValueError(f"Dataset {path} trajectory metadata does not align with its samples.")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError(f"Dataset {path} contains non-finite observations or actions.")
    return dataset


def _combine_datasets(datasets: list[dict]) -> dict:
    """Concatenate datasets while preserving trajectory boundaries."""
    return {
        "observations": torch.cat([dataset["observations"] for dataset in datasets]),
        "actions": torch.cat([dataset["actions"] for dataset in datasets]),
        "phases": [phase for dataset in datasets for phase in dataset["phases"]],
        "trajectory_lengths": [length for dataset in datasets for length in dataset["trajectory_lengths"]],
    }


def _trajectory_split(
    trajectory_lengths: list[int],
    validation_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sample indices by complete trajectories."""
    if len(trajectory_lengths) < 2:
        raise ValueError("At least two trajectories are required for a train/validation split.")
    validation_count = min(max(round(len(trajectory_lengths) * validation_fraction), 1), len(trajectory_lengths) - 1)
    validation_trajectories = set(
        torch.randperm(len(trajectory_lengths), generator=generator)[:validation_count].tolist()
    )
    train_indices = []
    validation_indices = []
    offset = 0
    for trajectory_id, length in enumerate(trajectory_lengths):
        target = validation_indices if trajectory_id in validation_trajectories else train_indices
        target.append(torch.arange(offset, offset + length))
        offset += length
    return torch.cat(train_indices), torch.cat(validation_indices)


def _deterministic_actions(outputs: torch.Tensor, arm_action_scale: float) -> torch.Tensor:
    """Apply the task's deterministic hybrid-action transform."""
    actions = outputs.clone()
    actions[..., _ARM_INDICES] *= arm_action_scale
    return actions


def _sample(indices: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """Sample dataset indices with replacement using a CPU generator."""
    return indices[torch.randint(indices.numel(), (count,), generator=generator)]


@torch.inference_mode()
def _evaluate(
    actor: _ShoelaceActor,
    reference_actor: _ShoelaceActor,
    approach_observations: torch.Tensor,
    approach_actions: torch.Tensor,
    retention_observations: torch.Tensor,
    arm_action_scale: float,
    gripper_logit_scale: float,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate approach imitation and retained-policy drift."""
    actor.eval()
    approach_squared_error = 0.0
    approach_gripper_correct = 0
    approach_gripper_count = 0
    approach_open_correct = 0
    approach_open_count = 0
    approach_close_correct = 0
    approach_close_count = 0
    approach_open_bce = 0.0
    approach_close_bce = 0.0
    for start in range(0, approach_observations.shape[0], batch_size):
        observations = approach_observations[start : start + batch_size]
        targets = approach_actions[start : start + batch_size]
        outputs = actor(observations)
        actions = _deterministic_actions(outputs, arm_action_scale)
        approach_squared_error += F.mse_loss(actions[:, _ARM_INDICES], targets[:, _ARM_INDICES], reduction="sum").item()
        gripper_targets = (targets[:, _GRIPPER_INDICES] > 0.0).to(outputs.dtype)
        gripper_logits = gripper_logit_scale * outputs[:, _GRIPPER_INDICES]
        gripper_prediction = gripper_logits > 0.0
        gripper_target = gripper_targets.bool()
        gripper_correct = gripper_prediction == gripper_target
        gripper_bce = F.binary_cross_entropy_with_logits(gripper_logits, gripper_targets, reduction="none")
        open_mask = gripper_target
        close_mask = ~gripper_target
        approach_open_bce += gripper_bce[open_mask].sum().item()
        approach_close_bce += gripper_bce[close_mask].sum().item()
        approach_gripper_correct += int(gripper_correct.sum())
        approach_gripper_count += gripper_targets.numel()
        approach_open_correct += int(gripper_correct[open_mask].sum())
        approach_open_count += int(open_mask.sum())
        approach_close_correct += int(gripper_correct[close_mask].sum())
        approach_close_count += int(close_mask.sum())

    retention_squared_error = 0.0
    retention_gripper_correct = 0
    retention_gripper_count = 0
    for start in range(0, retention_observations.shape[0], batch_size):
        observations = retention_observations[start : start + batch_size]
        actions = _deterministic_actions(actor(observations), arm_action_scale)
        reference_actions = _deterministic_actions(reference_actor(observations), arm_action_scale)
        retention_squared_error += F.mse_loss(actions, reference_actions, reduction="sum").item()
        retention_gripper_correct += int(
            ((actions[:, _GRIPPER_INDICES] > 0.0) == (reference_actions[:, _GRIPPER_INDICES] > 0.0)).sum()
        )
        retention_gripper_count += actions[:, _GRIPPER_INDICES].numel()

    approach_arm_count = approach_observations.shape[0] * len(_ARM_INDICES)
    retention_action_count = retention_observations.shape[0] * 14
    if approach_open_count == 0 or approach_close_count == 0:
        raise ValueError("Approach validation data must contain both open and close gripper labels.")
    approach_open_accuracy = approach_open_correct / approach_open_count
    approach_close_accuracy = approach_close_correct / approach_close_count
    return {
        "approach_arm_rmse": math.sqrt(approach_squared_error / approach_arm_count),
        "approach_gripper_balanced_bce": 0.5
        * (approach_open_bce / approach_open_count + approach_close_bce / approach_close_count),
        "approach_gripper_accuracy": approach_gripper_correct / approach_gripper_count,
        "approach_open_accuracy": approach_open_accuracy,
        "approach_close_accuracy": approach_close_accuracy,
        "approach_gripper_balanced_accuracy": 0.5 * (approach_open_accuracy + approach_close_accuracy),
        "retention_action_rmse": math.sqrt(retention_squared_error / retention_action_count),
        "retention_gripper_agreement": retention_gripper_correct / retention_gripper_count,
    }


def main():  # noqa: C901
    """Run behavior cloning and write an RSL-RL-compatible checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--approach_dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--retention_dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--approach_batch_fraction", type=float, default=0.5)
    parser.add_argument("--retention_weight", type=float, default=10.0)
    parser.add_argument("--gripper_weight", type=float, default=0.05)
    parser.add_argument("--parameter_delta_weight", type=float, default=1.0e-4)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--arm_action_scale", type=float, default=0.7)
    parser.add_argument("--gripper_logit_scale", type=float, default=2.0)
    parser.add_argument(
        "--approach_finger_observation_offsets",
        type=float,
        nargs="+",
        default=None,
        help="Per-dataset offsets added to the two scaled finger observations.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size < 2 or args.patience <= 0:
        raise ValueError("Epochs and patience must be positive; batch size must be at least two.")
    if args.learning_rate <= 0.0 or args.arm_action_scale <= 0.0 or args.gripper_logit_scale <= 0.0:
        raise ValueError("Learning rate and action scales must be positive.")
    approach_finger_offsets = args.approach_finger_observation_offsets or [0.0] * len(args.approach_dataset)
    if len(approach_finger_offsets) != len(args.approach_dataset):
        raise ValueError("Provide one approach finger observation offset per approach dataset.")
    if not all(math.isfinite(offset) for offset in approach_finger_offsets):
        raise ValueError("Approach finger observation offsets must be finite.")
    if not 0.0 < args.approach_batch_fraction < 1.0:
        raise ValueError("Approach batch fraction must lie strictly between zero and one.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("Validation fraction must lie strictly between zero and one.")
    if min(args.retention_weight, args.gripper_weight, args.parameter_delta_weight) < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    for path in (args.checkpoint, *args.approach_dataset, args.retention_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Output must not overwrite the source checkpoint.")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = _ShoelaceActor(checkpoint["actor_state_dict"]).to(args.device)
    reference_actor = copy.deepcopy(actor).eval()
    for parameter in reference_actor.parameters():
        parameter.requires_grad_(False)
    reference_parameters = [parameter.detach().clone() for parameter in actor.parameters()]

    approach_datasets = []
    for path, finger_offset in zip(args.approach_dataset, approach_finger_offsets, strict=True):
        dataset = _load_dataset(path)
        dataset["observations"] = dataset["observations"].clone()
        dataset["observations"][:, _FINGER_OBSERVATION_INDICES] += finger_offset
        approach_datasets.append(dataset)
    approach_dataset = _combine_datasets(approach_datasets)
    retention_dataset = _load_dataset(args.retention_dataset)
    approach_train, approach_validation = _trajectory_split(
        approach_dataset["trajectory_lengths"], args.validation_fraction, generator
    )
    retention_train, retention_validation = _trajectory_split(
        retention_dataset["trajectory_lengths"], args.validation_fraction, generator
    )
    approach_observations = approach_dataset["observations"].to(args.device)
    approach_actions = approach_dataset["actions"].to(args.device)
    retention_observations = retention_dataset["observations"].to(args.device)
    approach_train = approach_train.to(args.device)
    approach_validation = approach_validation.to(args.device)
    retention_train = retention_train.to(args.device)
    retention_validation = retention_validation.to(args.device)
    approach_train_gripper_targets = (approach_actions[approach_train][:, _GRIPPER_INDICES] > 0.0).to(
        approach_actions.dtype
    )
    open_fraction = approach_train_gripper_targets.mean(dim=0)
    if bool(torch.any((open_fraction <= 0.0) | (open_fraction >= 1.0))):
        raise ValueError("Approach training data must contain both open and close labels for each gripper.")
    open_gripper_weights = 0.5 / open_fraction
    close_gripper_weights = 0.5 / (1.0 - open_fraction)

    approach_batch_size = round(args.batch_size * args.approach_batch_fraction)
    retention_batch_size = args.batch_size - approach_batch_size
    steps_per_epoch = max(
        math.ceil(approach_train.numel() / approach_batch_size),
        math.ceil(retention_train.numel() / retention_batch_size),
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    validation_inputs = (
        approach_observations[approach_validation],
        approach_actions[approach_validation],
        retention_observations[retention_validation],
    )
    initial_metrics = _evaluate(
        actor,
        reference_actor,
        *validation_inputs,
        args.arm_action_scale,
        args.gripper_logit_scale,
        args.batch_size,
    )
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(actor.state_dict())
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        actor.train()
        training_loss = 0.0
        for _ in range(steps_per_epoch):
            approach_indices = _sample(approach_train.cpu(), approach_batch_size, generator).to(args.device)
            retention_indices = _sample(retention_train.cpu(), retention_batch_size, generator).to(args.device)
            approach_outputs = actor(approach_observations[approach_indices])
            approach_predictions = _deterministic_actions(approach_outputs, args.arm_action_scale)
            approach_targets = approach_actions[approach_indices]
            approach_arm_loss = F.mse_loss(approach_predictions[:, _ARM_INDICES], approach_targets[:, _ARM_INDICES])
            gripper_targets = (approach_targets[:, _GRIPPER_INDICES] > 0.0).to(approach_outputs.dtype)
            gripper_loss = F.binary_cross_entropy_with_logits(
                args.gripper_logit_scale * approach_outputs[:, _GRIPPER_INDICES],
                gripper_targets,
                reduction="none",
            )
            gripper_weights = torch.where(gripper_targets.bool(), open_gripper_weights, close_gripper_weights)
            approach_gripper_loss = (gripper_weights * gripper_loss).mean()
            retention_inputs = retention_observations[retention_indices]
            retention_outputs = actor(retention_inputs)
            with torch.no_grad():
                retention_targets = reference_actor(retention_inputs)
            retention_loss = F.mse_loss(retention_outputs, retention_targets)
            parameter_delta_loss = sum(
                F.mse_loss(parameter, reference)
                for parameter, reference in zip(actor.parameters(), reference_parameters, strict=True)
            )
            loss = (
                approach_arm_loss
                + args.gripper_weight * approach_gripper_loss
                + args.retention_weight * retention_loss
                + args.parameter_delta_weight * parameter_delta_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            training_loss += loss.item()

        metrics = _evaluate(
            actor,
            reference_actor,
            *validation_inputs,
            args.arm_action_scale,
            args.gripper_logit_scale,
            args.batch_size,
        )
        score = (
            metrics["approach_arm_rmse"] ** 2
            + args.gripper_weight * metrics["approach_gripper_balanced_bce"]
            + args.retention_weight * metrics["retention_action_rmse"] ** 2
        )
        history.append({"epoch": epoch, "training_loss": training_loss / steps_per_epoch, **metrics})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print("SHOELACE_BC_EPOCH=" + json.dumps(history[-1]), flush=True)
        if epochs_without_improvement >= args.patience:
            break

    actor.load_state_dict(best_state)
    final_metrics = _evaluate(
        actor,
        reference_actor,
        *validation_inputs,
        args.arm_action_scale,
        args.gripper_logit_scale,
        args.batch_size,
    )
    output_checkpoint = copy.deepcopy(checkpoint)
    actor_state = output_checkpoint["actor_state_dict"]
    for key in _MLP_KEYS:
        actor_state[key] = actor.mlp.state_dict()[key.removeprefix("mlp.")].detach().cpu().clone()
    output_checkpoint["optimizer_state_dict"]["state"] = {}
    output_checkpoint["iter"] = 0
    summary = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "approach_datasets": [str(path.resolve()) for path in args.approach_dataset],
        "retention_dataset": str(args.retention_dataset.resolve()),
        "train_approach_samples": approach_train.numel(),
        "validation_approach_samples": approach_validation.numel(),
        "train_retention_samples": retention_train.numel(),
        "validation_retention_samples": retention_validation.numel(),
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "approach_batch_fraction": args.approach_batch_fraction,
            "retention_weight": args.retention_weight,
            "gripper_weight": args.gripper_weight,
            "parameter_delta_weight": args.parameter_delta_weight,
            "approach_finger_observation_offsets": approach_finger_offsets,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        },
    }
    previous_infos = output_checkpoint.get("infos")
    output_checkpoint["infos"] = dict(previous_infos) if isinstance(previous_infos, dict) else {}
    output_checkpoint["infos"]["shoelace_behavior_cloning"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print("SHOELACE_BC_RESULT=" + json.dumps({**summary, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
