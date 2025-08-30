from functools import partial
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from coach_pl.module import MODULE_REGISTRY

from .rectified_flow import RectifiedFlowTrainingModule

__all__ = ["MeanFlowTrainingModule"]


@MODULE_REGISTRY.register()
class MeanFlowTrainingModule(RectifiedFlowTrainingModule):

    def forward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        # Sampling samples, noises, and timesteps
        sample, condition = batch
        if self.num_classes == 0:
            condition = None
        noise = torch.randn_like(sample)
        prev_timestep, curr_timestep = self.sample_timestep(sample)

        noisy, velocity, _, _ = self.noise_scheduler.add_noise(sample, noise, curr_timestep)
        with sdpa_kernel(backends=SDPBackend.MATH):
            output, d_output_d_curr_timestep = torch.func.jvp(
                partial(model, c=condition),
                (noisy, prev_timestep, curr_timestep),
                (velocity, torch.zeros_like(prev_timestep), torch.ones_like(curr_timestep)),
            )
        loss = self.criterion(output, velocity, d_output_d_curr_timestep, prev_timestep, curr_timestep)
        return loss

    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor:
        timestep_1 = self._sample_timestep(sample)
        timestep_2 = self._sample_timestep(sample)

        prev_timestep = torch.minimum(timestep_1, timestep_2)
        curr_timestep = torch.maximum(timestep_1, timestep_2)
        return prev_timestep, curr_timestep

    def _sample_timestep(self, sample: torch.Tensor) -> torch.Tensor:
        timestep = torch.sigmoid(torch.randn(sample.shape[0], device=sample.device) * self.timestep_std + self.timestep_mean)
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        return timestep
