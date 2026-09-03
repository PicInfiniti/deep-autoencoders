"""Symmetric deep autoencoder obtained by unrolling pretrained RBMs."""

from __future__ import annotations

import torch
from torch import nn

from .rbm import RestrictedBoltzmannMachine


class DeepAutoencoder(nn.Module):
    """Logistic deep autoencoder with a linear central code layer."""

    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__()
        if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive dimensions")
        self.layer_sizes = list(layer_sizes)
        self.encoder = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        reversed_sizes = list(reversed(layer_sizes))
        self.decoder = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(reversed_sizes[:-1], reversed_sizes[1:])
        )

    @classmethod
    def from_rbms(cls, rbms: list[RestrictedBoltzmannMachine]) -> "DeepAutoencoder":
        if not rbms:
            raise ValueError("at least one RBM is required")
        layer_sizes = [rbms[0].visible_units] + [rbm.hidden_units for rbm in rbms]
        for previous, current in zip(rbms[:-1], rbms[1:]):
            if previous.hidden_units != current.visible_units:
                raise ValueError("RBM dimensions do not form a stack")

        model = cls(layer_sizes)
        with torch.no_grad():
            for layer, rbm in zip(model.encoder, rbms):
                layer.weight.copy_(rbm.weights.T.cpu())
                layer.bias.copy_(rbm.hidden_bias.cpu())
            for layer, rbm in zip(model.decoder, reversed(rbms)):
                layer.weight.copy_(rbm.weights.cpu())
                layer.bias.copy_(rbm.visible_bias.cpu())
        return model

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, layer in enumerate(self.encoder):
            hidden = layer(hidden)
            if index < len(self.encoder) - 1:
                hidden = torch.sigmoid(hidden)
        return hidden

    def decode_logits(self, codes: torch.Tensor) -> torch.Tensor:
        hidden = codes
        for index, layer in enumerate(self.decoder):
            hidden = layer(hidden)
            if index < len(self.decoder) - 1:
                hidden = torch.sigmoid(hidden)
        return hidden

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(codes))

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode_logits(self.encode(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(inputs))
