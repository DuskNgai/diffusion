from typing import Any

from diffusers.models import ModelMixin
from omegaconf import DictConfig
import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MLPWithEmbedding(ModelMixin):

    @configurable
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers

        self.input_layer = nn.Linear(in_features + 2, hidden_features)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(ResidualBlock(hidden_features))

        self.output_layer = nn.Linear(hidden_features, out_features)

        self.activation = activation()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "in_features": cfg.MODEL.IN_FEATURES,
            "hidden_features": cfg.MODEL.HIDDEN_FEATURES,
            "out_features": cfg.MODEL.OUT_FEATURES,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "activation": getattr(nn, cfg.MODEL.ACTIVATION),
        }

    def forward(
        self,
        noisy: torch.Tensor,
        scale: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        scale = scale.reshape(-1, 1).expand(noisy.shape[0], -1)
        sigma = sigma.reshape(-1, 1).expand(noisy.shape[0], -1)

        x = torch.cat([noisy, scale, sigma], dim=-1)

        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, features: int) -> None:
        super().__init__()

        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + x
