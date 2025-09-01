from abc import (
    ABCMeta,
    abstractmethod,
)

import torch
import torch.nn as nn

__all__ = ["DiffusionCriterion"]


class DiffusionCriterion(nn.Module, metaclass=ABCMeta):
    """
    The base class for diffusion model criterion.
    """

    def __init__(self, prediction_type: str) -> None:
        super().__init__()

        self.prediction_type = prediction_type

    @abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def adaptive_l2_loss(error: torch.Tensor, gamma=0.0, eps=1e-3) -> torch.Tensor:
    """
    Adaptive L2 loss:
        sg(w) * ||Δ||_2^2, where w = (||Δ||^2 + eps)^{gamma - 1}

    Args:
        `error`: Tensor of shape [B, ...]
        `gamma`: Power used in original ||Δ||^{2 * gamma} loss
        `eps`: Small constant for stability

    Returns:
        Scalar loss
    """
    error = error.flatten(1)
    loss = error.square().sum(dim=-1) # ||Δ||^2
    p = 1.0 - gamma
    w = 1.0 / (loss + eps).pow(p)
    return (w.detach() * loss).mean()
