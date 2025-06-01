from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback

from loguru import logger

from .broker.base import Broker


class BrokerCallback(Callback):
    """..."""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger.info(f"Epoch {trainer.current_epoch} started")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger.info(f"Epoch {trainer.current_epoch} completed, sending model...")
        self.broker.set_model(pl_module)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        before = batch_idx / trainer.num_training_batches
        after = (batch_idx + 1) / trainer.num_training_batches
        if int(10 * before) < int(10 * after):
            logger.debug(f"{batch_idx}/{trainer.num_training_batches} ({before:.1%})")
