from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY

from .base import DiffusionCriterion

__all__ = ["MeanFlowCriterion"]


def adaptive_l2_loss(error: torch.Tensor, gamma: float = 0.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Adaptive L2 loss:
        sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + eps)^{1 - gamma}

    Args:
        `error` (torch.Tensor): Tensor of shape [B, ...]
        `gamma` (float): Power used in original ||Δ||^{2 * gamma} loss
        `eps` (float): Small constant for stability

    Returns:
        Scalar loss
    """
    loss = error.square().flatten(1).sum(dim=-1) # ||Δ||^2
    w = 1.0 / (loss.detach() + eps).pow(1.0 - gamma)
    return (w * loss).mean()


@CRITERION_REGISTRY.register()
class MeanFlowCriterion(DiffusionCriterion):
    """
    Criterion for Mean Flow Diffusion model.
    """

    @configurable
    def __init__(self, prediction_type: str) -> None:
        super().__init__(prediction_type=prediction_type)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "prediction_type": cfg.MODULE.NOISE_SCHEDULER.PREDICTION_TYPE,
        }

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = adaptive_l2_loss(input - target)
        mse = (input - target).square().mean().detach()
        return {
            "loss": loss,
            "mse": mse,
        }
