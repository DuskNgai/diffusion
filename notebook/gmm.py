from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
import torch
import torch.nn as nn

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel(ModelMixin, ConfigMixin):
    """
    The Gaussian Mixture Model (GMM) is a probabilistic model that assumes all the data points are generated from a mixture of Gaussian distributions.
    Its score function is known in closed form and can be used to validate the implementation of the sampler.
    """

    def __init__(
        self,
        num_groups_per_model: int,
        num_gs_per_group: int,
        model_radius: float,
        group_radius: float,
        sigma: float,
    ) -> None:
        super().__init__()

        group_angles = torch.linspace(0, 2 * torch.pi, num_groups_per_model + 1)[:-1]
        group_offsets = torch.stack([torch.cos(group_angles), torch.sin(group_angles)], dim=1) * model_radius
        gs_angles = torch.linspace(0, 2 * torch.pi, num_gs_per_group + 1)[:-1]
        gs_offsets = torch.stack([torch.cos(gs_angles), torch.sin(gs_angles)], dim=1) * group_radius

        self.mu = nn.Parameter((group_offsets[:, None, :] + gs_offsets[None, :, :]).view(-1, 2), requires_grad=False)
        self.sigma = sigma

    def forward(self, x: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        The probability density function of the noisy samples generated from the Gaussian Mixture Model is given by:
        """
        x = x.unsqueeze(1) # Shape: [B, 1, 2]

        weight = torch.softmax(
            -0.5 * (scale * self.mu - x).square().sum(dim=-1) / ((self.sigma * scale) ** 2 + sigma ** 2), dim=-1
        ).unsqueeze(-1)
        score = (weight * (scale * self.mu - x)).sum(dim=1) / ((self.sigma * scale) ** 2 + sigma ** 2)
        return score


# Using GaussianMixtureModelScheduler = GaussianModelScheduler

# Using GaussianMixtureModelPipeline = GaussianModelPipeline
