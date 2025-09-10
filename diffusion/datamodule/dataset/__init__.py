from .cifar import CIFAR10Dataset
from .mnist import MNISTDataset
from .spiral import SpiralDataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
