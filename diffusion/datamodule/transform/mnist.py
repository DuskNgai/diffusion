from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch
from torchvision.transforms import v2

from coach_pl.datamodule.transform import TRANSFORM_REGISTRY

__all__ = ["build_mnist_transform"]


@TRANSFORM_REGISTRY.register()
def build_mnist_transform(cfg: DictConfig, stage: RunningStage) -> v2.Compose:
    return v2.Compose([
        v2.Resize((32, 32)),                   # For DiT & UNet whose input is a multiple of 16.
        v2.Grayscale(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
