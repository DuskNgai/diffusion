from typing import Any

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from rich import print
from timm.optim import optim_factory
from timm.scheduler import scheduler_factory
import torch

from coach_pl.configuration import configurable
from coach_pl.model import build_model
from coach_pl.module import MODULE_REGISTRY

__all__ = ["SamplingModule"]


@MODULE_REGISTRY.register()
class SamplingModule(LightningModule):

    @configurable
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()

        self.model = model

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
        }

    def test_step(self, batch, batch_idx):
        pass
