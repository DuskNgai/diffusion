from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from sampler import SAMPLER_FORMULATION_TABLE
from .build import NOISE_SCHEDULER_REGISTRY
from .base import NoiseScheduler

__all__ = ["RectifiedFlowNoiseScheduler"]


@NOISE_SCHEDULER_REGISTRY.register()
class RectifiedFlowNoiseScheduler(NoiseScheduler):
    """
    Noise scheduler for Rectified Flow model.
    """
    @configurable
    def __init__(self,
        timestep_mean: float,
        timestep_std: float,
        sigma_data: float,
    ) -> None:
        super().__init__(
            sigma_data=sigma_data,
            scale_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["scale_fn"],
            scale_deriv_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["scale_deriv_fn"],
            sigma_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["sigma_fn"],
            sigma_deriv_fn=SAMPLER_FORMULATION_TABLE["Rectified Flow"]["sigma_deriv_fn"],
        )

        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "timestep_mean": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_MEAN,
            "timestep_std": cfg.MODEL.NOISE_SCHEDULER.TIMESTEP_STD,
            "sigma_data": cfg.MODEL.SIGMA_DATA,
        }

    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor | torch.LongTensor:
        timestep = torch.sigmoid(torch.randn(sample.shape[0], device=sample.device) * self.timestep_std + self.timestep_mean)
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep
