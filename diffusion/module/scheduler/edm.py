from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable

from .build import NOISE_SCHEDULER_REGISTRY
from sampler import (
    ContinuousTimeNoiseScheduler,
    SAMPLER_FORMULATION_TABLE,
)

__all__ = ["EDMNoiseScheduler"]


@NOISE_SCHEDULER_REGISTRY.register()
class EDMNoiseScheduler(ContinuousTimeNoiseScheduler):
    FORMULATION = SAMPLER_FORMULATION_TABLE["EDM"]

    @configurable
    def __init__(
        self,
        t_min: float = 0.002,
        t_max: float = 80.0,
        sigma_data: float = 1.0,
        prediction_type: str = "sample",
        algorithm_type: str = "ode",
        timestep_schedule: str = "linear_lognsr",
    ) -> None:
        super().__init__(
            t_min=t_min,
            t_max=t_max,
            sigma_data=sigma_data,
            scale_fn=self.FORMULATION["scale_fn"],
            scale_deriv_fn=self.FORMULATION["scale_deriv_fn"],
            sigma_fn=self.FORMULATION["sigma_fn"],
            sigma_deriv_fn=self.FORMULATION["sigma_deriv_fn"],
            nsr_inv_fn=self.FORMULATION["nsr_inv_fn"],
            prediction_type=prediction_type,
            algorithm_type=algorithm_type,
            timestep_schedule=timestep_schedule,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "sigma_data": cfg.MODULE.NOISE_SCHEDULER.SIGMA_DATA,
            "prediction_type": cfg.MODULE.NOISE_SCHEDULER.PREDICTION_TYPE,
        }

    def preprocess(self, noisy: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c_in = 1.0 / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        return c_in * noisy, scale, c_noise

    def postprocess(self, noisy: torch.Tensor, prediction: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == "sample":
            c_out = (sigma * self.config.sigma_data) / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = scale * self.config.sigma_data ** 2 / ((scale * self.config.sigma_data) ** 2 + sigma ** 2)
        elif self.config.prediction_type == "epsilon":
            c_out = (scale * self.config.sigma_data) / ((scale * self.config.sigma_data) ** 2 + sigma ** 2).sqrt()
            c_skip = sigma / ((scale * self.config.sigma_data) ** 2 + sigma ** 2)
        elif self.config.prediction_type == "velocity":
            raise NotImplementedError
        else:
            raise KeyError(f"Unknown prediction type: {self.config.prediction_type}")

        return c_skip * noisy + c_out * prediction
