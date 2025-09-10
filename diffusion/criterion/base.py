from abc import (
    ABCMeta,
    abstractmethod,
)

import torch
import torch.nn as nn

__all__ = ["DiffusionCriterion"]


class DiffusionCriterion(nn.Module, metaclass=ABCMeta):
    """
    The base class for diffusion model criterion.
    """

    def __init__(self, prediction_type: str) -> None:
        super().__init__()

        self.prediction_type = prediction_type

    @abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
