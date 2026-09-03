"""Restricted Boltzmann machines used for layer-wise pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

HiddenType = Literal["bernoulli", "gaussian"]


@dataclass(frozen=True)
class RBMTrainingConfig:
    """Hyperparameters from the authors' published MATLAB implementation."""

    epochs: int = 50
    batch_size: int = 100
    learning_rate: float = 0.1
    weight_decay: float = 0.0002
    initial_momentum: float = 0.5
    final_momentum: float = 0.9
    momentum_switch_epoch: int = 5


class RestrictedBoltzmannMachine:
    """Bernoulli-visible RBM trained using one-step contrastive divergence.

    Visible values may be probabilities in ``[0, 1]``, as in the paper. Hidden
    units are Bernoulli in intermediate layers and unit-variance Gaussian in
    the final, linear code layer.
    """

    def __init__(
        self,
        visible_units: int,
        hidden_units: int,
        *,
        hidden_type: HiddenType = "bernoulli",
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if visible_units <= 0 or hidden_units <= 0:
            raise ValueError("visible_units and hidden_units must be positive")
        if hidden_type not in ("bernoulli", "gaussian"):
            raise ValueError(f"unsupported hidden type: {hidden_type}")

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.hidden_type = hidden_type
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.weights = 0.1 * torch.randn(
            visible_units,
            hidden_units,
            generator=self.generator,
            device=self.device,
        )
        self.visible_bias = torch.zeros(visible_units, device=self.device)
        self.hidden_bias = torch.zeros(hidden_units, device=self.device)

        self._weight_velocity = torch.zeros_like(self.weights)
        self._visible_bias_velocity = torch.zeros_like(self.visible_bias)
        self._hidden_bias_velocity = torch.zeros_like(self.hidden_bias)

    def hidden_mean(self, visible: torch.Tensor) -> torch.Tensor:
        activation = visible @ self.weights + self.hidden_bias
        return torch.sigmoid(activation) if self.hidden_type == "bernoulli" else activation

    def sample_hidden(self, mean: torch.Tensor) -> torch.Tensor:
        if self.hidden_type == "bernoulli":
            return torch.bernoulli(mean, generator=self.generator)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=self.generator,
        )
        return mean + noise

    def visible_mean(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(hidden @ self.weights.T + self.visible_bias)

    @torch.no_grad()
    def cd1_step(
        self,
        visible: torch.Tensor,
        *,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> float:
        """Perform the exact positive/negative phases of the paper's CD-1 rule."""
        visible = visible.to(self.device)
        batch_size = visible.shape[0]
        if visible.ndim != 2 or visible.shape[1] != self.visible_units:
            raise ValueError(
                f"expected [batch, {self.visible_units}], got {tuple(visible.shape)}"
            )
        if batch_size == 0:
            raise ValueError("cannot train on an empty batch")

        positive_hidden = self.hidden_mean(visible)
        hidden_states = self.sample_hidden(positive_hidden)
        negative_visible = self.visible_mean(hidden_states)
        negative_hidden = self.hidden_mean(negative_visible)

        positive_products = visible.T @ positive_hidden
        negative_products = negative_visible.T @ negative_hidden
        scale = 1.0 / batch_size

        self._weight_velocity.mul_(momentum).add_(
            learning_rate
            * (scale * (positive_products - negative_products) - weight_decay * self.weights)
        )
        self._visible_bias_velocity.mul_(momentum).add_(
            learning_rate * scale * (visible - negative_visible).sum(dim=0)
        )
        self._hidden_bias_velocity.mul_(momentum).add_(
            learning_rate * scale * (positive_hidden - negative_hidden).sum(dim=0)
        )

        self.weights.add_(self._weight_velocity)
        self.visible_bias.add_(self._visible_bias_velocity)
        self.hidden_bias.add_(self._hidden_bias_velocity)

        return float((visible - negative_visible).square().sum().cpu())

    def fit(
        self,
        data: torch.Tensor,
        config: RBMTrainingConfig,
        *,
        verbose: bool = True,
    ) -> list[float]:
        """Train on a fixed set of mini-batches and return SSE for every epoch."""
        if data.ndim != 2 or data.shape[1] != self.visible_units:
            raise ValueError(f"expected data with {self.visible_units} columns")
        if data.shape[0] == 0:
            raise ValueError("cannot train on an empty dataset")

        history: list[float] = []
        for epoch in range(config.epochs):
            momentum = (
                config.final_momentum
                if epoch >= config.momentum_switch_epoch
                else config.initial_momentum
            )
            epoch_error = 0.0
            for start in range(0, data.shape[0], config.batch_size):
                batch = data[start : start + config.batch_size]
                epoch_error += self.cd1_step(
                    batch,
                    learning_rate=config.learning_rate,
                    momentum=momentum,
                    weight_decay=config.weight_decay,
                )
            history.append(epoch_error)
            if verbose:
                print(
                    f"    epoch {epoch + 1:3d}/{config.epochs}: "
                    f"reconstruction SSE={epoch_error:.3f}",
                    flush=True,
                )
        return history

    @torch.no_grad()
    def transform(self, data: torch.Tensor, *, batch_size: int = 2048) -> torch.Tensor:
        """Return deterministic feature means, kept on CPU between RBM layers."""
        outputs = []
        for start in range(0, data.shape[0], batch_size):
            batch = data[start : start + batch_size].to(self.device)
            outputs.append(self.hidden_mean(batch).cpu())
        return torch.cat(outputs, dim=0)

    def state_dict(self) -> dict[str, torch.Tensor | str | int]:
        return {
            "visible_units": self.visible_units,
            "hidden_units": self.hidden_units,
            "hidden_type": self.hidden_type,
            "weights": self.weights.detach().cpu(),
            "visible_bias": self.visible_bias.detach().cpu(),
            "hidden_bias": self.hidden_bias.detach().cpu(),
        }


def pretrain_stack(
    data: torch.Tensor,
    layer_sizes: list[int],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device | str,
    seed: int,
    verbose: bool = True,
) -> tuple[list[RestrictedBoltzmannMachine], list[list[float]]]:
    """Greedily pretrain an encoder; the last layer has a linear Gaussian code."""
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain input and code dimensions")
    if data.shape[1] != layer_sizes[0]:
        raise ValueError("data width does not match the input layer")

    representation = data.cpu()
    rbms: list[RestrictedBoltzmannMachine] = []
    histories: list[list[float]] = []

    for index, (visible_units, hidden_units) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        hidden_type: HiddenType = (
            "gaussian" if index == len(layer_sizes) - 2 else "bernoulli"
        )
        learning_rate = 0.001 if hidden_type == "gaussian" else 0.1
        if verbose:
            print(
                f"Pretraining RBM {index + 1}/{len(layer_sizes) - 1}: "
                f"{visible_units} -> {hidden_units} ({hidden_type})",
                flush=True,
            )
        rbm = RestrictedBoltzmannMachine(
            visible_units,
            hidden_units,
            hidden_type=hidden_type,
            device=device,
            seed=seed + index,
        )
        history = rbm.fit(
            representation,
            RBMTrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            ),
            verbose=verbose,
        )
        rbms.append(rbm)
        histories.append(history)
        representation = rbm.transform(representation)

    return rbms, histories
