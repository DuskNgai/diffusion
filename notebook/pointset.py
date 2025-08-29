from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1].as_posix()))

from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from diffusers.utils.torch_utils import randn_tensor
import torch
import torch.nn as nn

__all__ = ["PointSetModel"]


class PointSetModel(ModelMixin, ConfigMixin):
    """
    The Point Set Model is a probabilistic model that assumes all the data points are generated from a mixture of delta distributions.
    Its score function is known in closed form and can be used to validate the implementation of the sampler.
    """

    def __init__(self, num_samples: int, dim: int) -> None:
        super().__init__()

        self.mu = nn.Parameter(randn_tensor((num_samples, dim)), requires_grad=False)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        The probability density function of the noisy samples generated from the Gaussian Mixture Model is given by:
        """
        weight = torch.softmax(-((scale * self.mu - x.unsqueeze(-2)) / sigma).square().sum(dim=-1), dim=-1).unsqueeze(-1)
        score = ((weight * scale * self.mu).sum(dim=-2) - x) / (sigma ** 2)
        return score


# Using PointSetScheduler = GaussianModelScheduler

# Using PointSetPipeline = GaussianModelPipeline
