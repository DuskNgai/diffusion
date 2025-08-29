from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.criterion import CRITERION_REGISTRY

from .base import (
    adaptive_l2_loss,
    DiffusionCriterion,
)

__all__ = ["MeanFlowCriterion"]


@CRITERION_REGISTRY.register()
class MeanFlowCriterion(DiffusionCriterion):
    """
    Criterion for Mean Flow Diffusion model.
    """

    @configurable
    def __init__(
        self,
        prediction_type: str,
    ) -> None:
        super().__init__(prediction_type=prediction_type)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "prediction_type": cfg.MODULE.NOISE_SCHEDULER.PREDICTION_TYPE,
        }

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        d_output_d_curr_timestep: torch.Tensor,
        prev_timestep: torch.Tensor,
        curr_timestep: torch.Tensor,
    ) -> torch.Tensor:
        y = target - (curr_timestep - prev_timestep) * d_output_d_curr_timestep
        return adaptive_l2_loss(input - y.detach())
