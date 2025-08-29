from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch
from torch.utils.data import Dataset

from coach_pl.configuration import configurable
from coach_pl.datamodule import DATASET_REGISTRY

__all__ = ["SpiralDataset"]


@DATASET_REGISTRY.register()
class SpiralDataset(Dataset):

    @configurable
    def __init__(
        self,
        b: float = 1.0,
        theta_range: tuple[float, float] = (0, 4 * torch.pi),
        num_points: int = 1024,
    ) -> None:

        theta = torch.linspace(theta_range[0], theta_range[1], num_points + 1)[:-1]
        x = b * theta * torch.cos(theta)
        y = b * theta * torch.sin(theta)

        self.points = torch.stack((x, y), dim=1)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "num_points": cfg.DATAMODULE.DATASET.NUM_POINTS,
        }

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.points[idx], 0

    @property
    def collate_fn(self) -> None:
        return None
