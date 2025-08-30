from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY

from .base import DiffusionCriterion

__all__ = ["EDMCriterion"]


@CRITERION_REGISTRY.register()
class EDMCriterion(DiffusionCriterion):
    """
    Criterion for EDM Diffusion model.
    """

    @configurable
    def __init__(
        self,
        sigma_data: float,
        prediction_type: str,
    ) -> None:
        super().__init__(prediction_type=prediction_type)

        self.sigma_data = sigma_data

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "sigma_data": cfg.MODULE.NOISE_SCHEDULER.SIGMA_DATA,
            "prediction_type": cfg.MODULE.NOISE_SCHEDULER.PREDICTION_TYPE,
        }

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "sample":
            weight = ((scale * self.sigma_data) ** 2 + sigma ** 2) / ((sigma * self.sigma_data) ** 2)
            loss = (weight * (input - target).square()).mean()

        elif self.prediction_type == "epsilon":
            weight = ((scale * self.sigma_data) ** 2 + sigma ** 2) / ((scale * self.sigma_data) ** 2)
            loss = (weight * (input - target).square()).mean()

        elif self.prediction_type == "velocity":
            raise NotImplementedError

        else:
            raise KeyError(f"Unknown prediction type: {self.prediction_type}")

        return {
            "loss": loss
        }
