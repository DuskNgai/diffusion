from typing import Any

from omegaconf import DictConfig

from coach_pl.configuration import configurable

from .build import NOISE_SCHEDULER_REGISTRY
from sampler import (
    ContinuousTimeNoiseScheduler,
    SAMPLER_FORMULATION_TABLE,
)

__all__ = ["RectifiedFlowNoiseScheduler"]


@NOISE_SCHEDULER_REGISTRY.register()
class RectifiedFlowNoiseScheduler(ContinuousTimeNoiseScheduler):
    FORMULATION = SAMPLER_FORMULATION_TABLE["Rectified Flow"]

    @configurable
    def __init__(
        self,
        t_min: float = 0.0001,
        t_max: float = 0.9999,
        sigma_data: float = 1.0,
        prediction_type: str = "velocity",
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
