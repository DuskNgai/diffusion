from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, v2

from coach_pl.configuration import configurable
from coach_pl.datamodule import build_transform, DATASET_REGISTRY

__all__ = ["MNISTDataset"]


@DATASET_REGISTRY.register()
class MNISTDataset(MNIST):
    """
    A wrapper around torchvision.datasets.MNIST to provide a consistent interface.
    """

    @configurable
    def __init__(
        self,
        root: str,
        train: bool,
        transform: v2.Compose | Compose | None = None,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=True)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "root": Path(cfg.DATAMODULE.DATASET.ROOT),
            "train": stage == RunningStage.TRAINING,
            "transform": build_transform(cfg, stage),
        }

    @property
    def collate_fn(self) -> None:
        return None
