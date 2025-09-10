from typing import Any

from omegaconf import DictConfig
import torch

from coach_pl.configuration import configurable
from coach_pl.criterion import build_criterion
from coach_pl.model import build_model
from coach_pl.module import MODULE_REGISTRY

from diffusion.module.scheduler import build_noise_scheduler
from diffusion.module.training.base import DiffusionTrainingModule
from sampler import ContinuousTimeNoiseScheduler

__all__ = ["EDMTrainingModule"]


@MODULE_REGISTRY.register()
class EDMTrainingModule(DiffusionTrainingModule):

    @configurable
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        noise_scheduler: ContinuousTimeNoiseScheduler,
        cfg: DictConfig,
        timestep_mean: float,
        timestep_std: float,
        num_classes: int,
    ) -> None:

        super().__init__(model, criterion, noise_scheduler, cfg)

        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        args = super().from_config(cfg)
        args.update({
            "timestep_mean": cfg.MODULE.TIMESTEP_MEAN,
            "timestep_std": cfg.MODULE.TIMESTEP_STD,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        })
        return args

    def forward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        # Sampling samples, noises, and timesteps
        sample, condition = batch
        if self.num_classes == 0:
            condition = None
        noise = torch.randn_like(sample)
        timestep = self.sample_timestep(sample)

        noisy, target, scale, sigma = self.noise_scheduler.add_noise(sample, noise, timestep)
        processed_noisy, processed_scale, processed_sigma = self.noise_scheduler.preprocess(noisy, scale, sigma)
        unprocessed_output = model(processed_noisy, processed_scale, processed_sigma, condition)
        output = self.noise_scheduler.postprocess(noisy, unprocessed_output, scale, sigma)
        loss = self.criterion(output, target, scale, sigma)
        return loss

    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Return:
            t \in (0, \infty)
        """
        timestep = torch.exp(torch.randn(sample.shape[0], device=sample.device) * self.timestep_std + self.timestep_mean)
        return timestep
