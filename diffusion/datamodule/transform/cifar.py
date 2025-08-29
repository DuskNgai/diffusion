from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torch
from torchvision.transforms import v2

from coach_pl.datamodule.transform import TRANSFORM_REGISTRY

__all__ = [
    "build_cifar_transform",
]


@TRANSFORM_REGISTRY.register()
def build_cifar_transform(cfg: DictConfig, stage: RunningStage) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
