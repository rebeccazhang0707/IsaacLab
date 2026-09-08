# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models and action distributions used for shoelace training."""

import copy
import math

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict
from torch import nn
from torch.distributions import Bernoulli, Normal


class ShoelaceHybridActionDistribution(GaussianDistribution):
    """Gaussian arm exploration with Bernoulli binary-gripper actions.

    The shoelace action layout contains six continuous arm actions followed by one binary gripper action for each
    robot. Modeling every dimension as Gaussian makes an arbitrarily small change around zero flip a gripper command,
    while the Gaussian likelihood scales that update by the inverse variance. This distribution retains Gaussian arm
    actions and interprets the two gripper MLP outputs as Bernoulli logits instead.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1.0e-6, 1.0e6),
        std_type: str = "scalar",
        learn_std: bool = True,
        arm_action_scale: float = 1.0,
        gripper_logit_scale: float = 1.0,
    ) -> None:
        """Initialize the hybrid distribution.

        Args:
            output_dim: Total action dimension. The shoelace policy requires 14 actions.
            init_std: Initial standard deviation of the continuous arm actions.
            std_range: Minimum and maximum continuous-action standard deviations.
            std_type: Continuous standard-deviation parameterization.
            learn_std: Whether the continuous standard deviation is learnable.
            arm_action_scale: Scale applied to continuous arm means and standard deviations.
            gripper_logit_scale: Scale applied to the two gripper MLP outputs before constructing Bernoulli logits.

        Raises:
            ValueError: If the action dimension does not match the dual-Franka action layout.
        """
        if output_dim != 14:
            raise ValueError(f"ShoelaceHybridActionDistribution requires 14 actions, got {output_dim}.")
        if not math.isfinite(arm_action_scale) or arm_action_scale <= 0.0:
            raise ValueError("arm_action_scale must be finite and positive.")
        if not math.isfinite(gripper_logit_scale) or gripper_logit_scale <= 0.0:
            raise ValueError("gripper_logit_scale must be finite and positive.")
        super().__init__(output_dim, init_std, std_range, std_type, learn_std)
        self.arm_action_scale = arm_action_scale
        self.gripper_logit_scale = gripper_logit_scale
        self._arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        self._gripper_indices = [6, 13]
        self._mlp_output: torch.Tensor | None = None
        self._arm_distribution: Normal | None = None
        self._gripper_distribution: Bernoulli | None = None

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian arm and Bernoulli gripper distributions."""
        self._mlp_output = self._scale_arm_output(mlp_output)
        if self.std_type == "scalar":
            std = self.std_param.clamp(self.std_range[0], self.std_range[1])
        else:
            log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
        self._arm_distribution = Normal(
            self._mlp_output[..., self._arm_indices],
            self.arm_action_scale * std[self._arm_indices],
        )
        self._gripper_distribution = Bernoulli(logits=self.gripper_logit_scale * mlp_output[..., self._gripper_indices])

    def sample(self) -> torch.Tensor:
        """Sample continuous arm actions and signed binary gripper actions."""
        actions = torch.empty_like(self._mlp_output)
        actions[..., self._arm_indices] = self._arm_distribution.sample()
        gripper_sign = 2.0 * self._gripper_distribution.sample() - 1.0
        gripper_magnitude = self._mlp_output[..., self._gripper_indices].abs().clamp_min(1.0e-6)
        actions[..., self._gripper_indices] = gripper_sign * gripper_magnitude
        return actions

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Scale continuous arms while preserving gripper command signs."""
        return self._scale_arm_output(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly deterministic action transform."""
        return _ShoelaceDeterministicActionScale(self.arm_action_scale)

    @property
    def mean(self) -> torch.Tensor:
        """Return arm means and expected signed gripper actions."""
        mean = self._mlp_output.clone()
        gripper_magnitude = self._mlp_output[..., self._gripper_indices].abs().clamp_min(1.0e-6)
        mean[..., self._gripper_indices] = (2.0 * self._gripper_distribution.probs - 1.0) * gripper_magnitude
        return mean

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of every emitted action."""
        std = torch.empty_like(self._mlp_output)
        std[..., self._arm_indices] = self._arm_distribution.stddev
        gripper_magnitude = self._mlp_output[..., self._gripper_indices].abs().clamp_min(1.0e-6)
        std[..., self._gripper_indices] = 2.0 * self._gripper_distribution.stddev * gripper_magnitude
        return std

    @property
    def entropy(self) -> torch.Tensor:
        """Return the joint arm-and-gripper entropy."""
        return self._arm_distribution.entropy().sum(dim=-1) + self._gripper_distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the action logits or means and their continuous standard deviations."""
        return self._mlp_output, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the joint log probability of arm and binary gripper actions."""
        arm_log_prob = self._arm_distribution.log_prob(outputs[..., self._arm_indices]).sum(dim=-1)
        gripper_values = (outputs[..., self._gripper_indices] > 0.0).to(outputs.dtype)
        gripper_log_prob = self._gripper_distribution.log_prob(gripper_values).sum(dim=-1)
        return arm_log_prob + gripper_log_prob

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return ``KL(old || new)`` for the Gaussian and Bernoulli factors."""
        old_output, old_std = old_params
        new_output, new_std = new_params
        arm_kl = torch.distributions.kl_divergence(
            Normal(old_output[..., self._arm_indices], old_std[..., self._arm_indices]),
            Normal(new_output[..., self._arm_indices], new_std[..., self._arm_indices]),
        ).sum(dim=-1)
        gripper_kl = torch.distributions.kl_divergence(
            Bernoulli(logits=self.gripper_logit_scale * old_output[..., self._gripper_indices]),
            Bernoulli(logits=self.gripper_logit_scale * new_output[..., self._gripper_indices]),
        ).sum(dim=-1)
        return arm_kl + gripper_kl

    def _scale_arm_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return policy outputs with the continuous arm dimensions scaled."""
        output = mlp_output.clone()
        output[..., self._arm_indices] *= self.arm_action_scale
        return output


class _ShoelaceDeterministicActionScale(nn.Module):
    """Apply the shoelace arm scale during deterministic policy export."""

    def __init__(self, arm_action_scale: float) -> None:
        super().__init__()
        self.arm_action_scale = arm_action_scale

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        """Scale continuous arm outputs and preserve gripper outputs."""
        return torch.cat(
            (
                output[..., :6] * self.arm_action_scale,
                output[..., 6:7],
                output[..., 7:13] * self.arm_action_scale,
                output[..., 13:14],
            ),
            dim=-1,
        )


class FixedObservationStatisticsMLPModel(MLPModel):
    """MLP model that preserves observation statistics loaded from a checkpoint."""

    def update_normalization(self, obs: TensorDict) -> None:
        """Keep the loaded observation normalization statistics unchanged."""
        del obs


class FrozenBackboneFixedObservationStatisticsMLPModel(FixedObservationStatisticsMLPModel):
    """Checkpoint-consolidation actor that trains only its distribution and final linear layer."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the model and freeze all MLP layers except its output layer.

        Args:
            obs: Observation dictionary used to determine input dimensions.
            obs_groups: Observation groups consumed by each model role.
            obs_set: Model role whose observation groups are selected.
            output_dim: Number of policy outputs.
            hidden_dims: Hidden-layer dimensions.
            activation: Hidden-layer activation name.
            obs_normalization: Whether to construct an observation normalizer.
            distribution_cfg: Optional stochastic-output distribution configuration.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        for parameter in self.mlp[-1].parameters():
            parameter.requires_grad_(True)


class FrozenGripperOutputHeadFixedObservationStatisticsMLPModel(FrozenBackboneFixedObservationStatisticsMLPModel):
    """Output-head actor that preserves the two binary gripper rows.

    This model keeps the ordinary MLP checkpoint layout while masking gradients for output rows 6 and 13. Use a
    fresh optimizer state so that moments restored from an earlier run cannot move the masked rows.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the model and mask gradients for its gripper output rows.

        Args:
            obs: Observation dictionary used to determine input dimensions.
            obs_groups: Observation groups consumed by each model role.
            obs_set: Model role whose observation groups are selected.
            output_dim: Number of policy outputs. The shoelace actor requires 14 outputs.
            hidden_dims: Hidden-layer dimensions.
            activation: Hidden-layer activation name.
            obs_normalization: Whether to construct an observation normalizer.
            distribution_cfg: Optional stochastic-output distribution configuration.

        Raises:
            ValueError: If the output dimension does not match the dual-Franka action layout.
        """
        if output_dim != 14:
            raise ValueError(f"FrozenGripperOutputHead model requires 14 actions, got {output_dim}.")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        output_layer = self.mlp[-1]
        output_layer.weight.register_hook(_mask_gripper_output_rows)
        output_layer.bias.register_hook(_mask_gripper_output_rows)


class FrozenPolicyFixedObservationStatisticsMLPModel(FixedObservationStatisticsMLPModel):
    """Critic-warm-up actor that preserves every policy parameter."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the model and freeze its MLP and output distribution.

        Args:
            obs: Observation dictionary used to determine input dimensions.
            obs_groups: Observation groups consumed by each model role.
            obs_set: Model role whose observation groups are selected.
            output_dim: Number of policy outputs.
            hidden_dims: Hidden-layer dimensions.
            activation: Hidden-layer activation name.
            obs_normalization: Whether to construct an observation normalizer.
            distribution_cfg: Optional stochastic-output distribution configuration.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class SplitBackboneFixedObservationStatisticsMLPModel(FixedObservationStatisticsMLPModel):
    """Checkpoint-consolidation actor with independent arm and gripper MLP backbones."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize two independently trainable copies of the configured MLP.

        Args:
            obs: Observation dictionary used to determine input dimensions.
            obs_groups: Observation groups consumed by each model role.
            obs_set: Model role whose observation groups are selected.
            output_dim: Number of policy outputs.
            hidden_dims: Hidden-layer dimensions.
            activation: Hidden-layer activation name.
            obs_normalization: Whether to construct an observation normalizer.
            distribution_cfg: Optional stochastic-output distribution configuration.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.mlp = _ShoelaceSplitActionMLP(self.mlp)


class FrozenGripperSplitBackboneFixedObservationStatisticsMLPModel(SplitBackboneFixedObservationStatisticsMLPModel):
    """Checkpoint-consolidation actor that trains its arm MLP while preserving its gripper policy."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize independent arm and frozen gripper MLP backbones.

        Args:
            obs: Observation dictionary used to determine input dimensions.
            obs_groups: Observation groups consumed by each model role.
            obs_set: Model role whose observation groups are selected.
            output_dim: Number of policy outputs.
            hidden_dims: Hidden-layer dimensions.
            activation: Hidden-layer activation name.
            obs_normalization: Whether to construct an observation normalizer.
            distribution_cfg: Optional stochastic-output distribution configuration.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        for parameter in self.mlp.gripper_mlp.parameters():
            parameter.requires_grad_(False)


def _mask_gripper_output_rows(gradient: torch.Tensor) -> torch.Tensor:
    """Return an output-head gradient with both gripper rows cleared."""
    gradient = gradient.clone()
    gradient[[6, 13]] = 0.0
    return gradient


class _ShoelaceSplitActionMLP(nn.Module):
    """Merge independent arm and gripper MLP outputs into the shoelace action layout."""

    def __init__(self, mlp: nn.Module) -> None:
        """Create identical, independently trainable arm and gripper MLPs."""
        super().__init__()
        self.arm_mlp = mlp
        self.gripper_mlp = copy.deepcopy(mlp)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return arm outputs from one MLP and gripper outputs from the other."""
        arm_output = self.arm_mlp(latent)
        gripper_output = self.gripper_mlp(latent)
        output = arm_output.clone()
        output[..., [6, 13]] = gripper_output[..., [6, 13]]
        return output
