from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch

from coach_pl.criterion import build_criterion
from coach_pl.model import build_model
from coach_pl.utils.logging import setup_logger

from diffusion.module.scheduler import build_noise_scheduler
from sampler import ContinuousTimeNoiseScheduler

logger = setup_logger(__name__, rank_zero_only=True)

__all__ = ["DiffusionTrainingModule"]


class DiffusionTrainingModule(LightningModule, metaclass=ABCMeta):
    """
    Training module for diffusion model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        noise_scheduler: ContinuousTimeNoiseScheduler,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(cfg)

        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model
        self.criterion = torch.compile(criterion) if cfg.MODULE.COMPILE else criterion
        self.noise_scheduler = noise_scheduler

        logger.info(self.model)

        self.batch_size = cfg.DATAMODULE.DATALOADER.TRAIN.BATCH_SIZE

        # Exponential moving average configuration
        self.ema_enabled = cfg.MODULE.EMA.ENABLED
        self.ema_base_decay = cfg.MODULE.EMA.BASE_DECAY
        self.ema_model = None
        if self.ema_enabled:
            self.ema_model = torch.optim.swa_utils.AveragedModel(self.model).eval()

        # Optimizer configuration
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.optimizer_name = cfg.MODULE.OPTIMIZER.NAME
        self.optimizer_params = {
            k.lower(): v
            for k, v in cfg.MODULE.OPTIMIZER.PARAMS.items()
        }

        # Scheduler configuration
        self.step_on_epochs = cfg.MODULE.SCHEDULER.STEP_ON_EPOCHS
        self.scheduler_name = cfg.MODULE.SCHEDULER.NAME
        self.scheduler_params = {
            k.lower(): v
            for k, v in cfg.MODULE.SCHEDULER.PARAMS.items()
        }
        self.scheduler_params["num_epochs"] = cfg.TRAINER.MAX_EPOCHS

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model": build_model(cfg),
            "criterion": build_criterion(cfg),
            "noise_scheduler": build_noise_scheduler(cfg),
            "cfg": cfg,
        }

    def configure_optimizers(self) -> Any:
        total_batch_size = self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        logger.info(
            f"Total training batch size ({total_batch_size}) = single batch size ({self.batch_size}) * accumulate ({self.trainer.accumulate_grad_batches}) * world size ({self.trainer.world_size})"
        )

        lr = self.base_lr * total_batch_size / 256
        logger.info(f"Learning rate ({lr:0.6g}) = base_lr ({self.base_lr:0.6g}) * total_batch_size ({total_batch_size}) / 256")

        if self.ema_enabled:
            ema_decay = 1 - ((1 - self.ema_base_decay) * total_batch_size / 1024)
            logger.info(
                f"EMA decay ({ema_decay:0.6g}) = 1 - ((1 - ema_base_decay ({self.ema_base_decay:0.6g})) * total_batch_size ({total_batch_size}) / 1024)"
            )
        else:
            ema_decay = 0.0

        hyperparameters = {
            "lr": lr,
            "ema_decay": ema_decay,
        }

        if self.ema_enabled:
            self.ema_model.multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(hyperparameters["ema_decay"])

        optimizer = create_optimizer_v2(
            model_or_params=self.model,
            opt=self.optimizer_name,
            lr=hyperparameters["lr"],
            **self.optimizer_params,
        )

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            sched=self.scheduler_name,
            step_on_epochs=self.step_on_epochs,
            updates_per_epoch=int(len(self.trainer.datamodule.train_dataloader()) / (self.trainer.accumulate_grad_batches * self.trainer.world_size)),
            **self.scheduler_params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if self.step_on_epochs else "step",
                "frequency": 1,
            }
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.step_on_epochs:
            scheduler.step(self.current_epoch, metric)
        else:
            scheduler.step_update(self.global_step, metric)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.forward(self.model, batch)
        self.log_dict({
            "train/" + k: v
            for k, v in loss.items()
        }, prog_bar=True, sync_dist=False, rank_zero_only=True)
        return loss

    def on_train_batch_end(self, outputs: torch.Tensor | dict[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if self.ema_enabled:
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self.forward(self.ema_model, batch)
        self.log({
            "val/" + k: v
            for k, v in loss.items()
        }, sync_dist=False, rank_zero_only=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss = self.forward(self.ema_model, batch)
        self.log({
            "test/" + k: v
            for k, v in loss.items()
        }, sync_dist=False, rank_zero_only=True)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        model = self.ema_model.module if self.ema_enabled else self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        checkpoint["model"] = model.state_dict()

    # --------------------------------------------------------------------------------

    @abstractmethod
    def forward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_timestep(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
