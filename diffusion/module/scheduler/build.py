from fvcore.common.registry import Registry
from omegaconf import DictConfig

from sampler import ContinuousTimeNoiseScheduler

NOISE_SCHEDULER_REGISTRY = Registry("NOISE_SCHEDULER")
NOISE_SCHEDULER_REGISTRY.__doc__ = """
Registry for noise scheduler, which is a derived class of `ContinuousTimeNoiseScheduler`.
It adds noise to the input tensor based on the timestep.
"""


def build_noise_scheduler(cfg: DictConfig) -> ContinuousTimeNoiseScheduler:
    """
    Build the noise scheduler defined by `cfg.MODULE.NOISE_SCHEDULER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    noise_scheduler_name = cfg.MODULE.NOISE_SCHEDULER.NAME
    noise_scheduler = NOISE_SCHEDULER_REGISTRY.get(noise_scheduler_name)(cfg)
    return noise_scheduler
