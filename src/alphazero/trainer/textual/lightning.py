import math
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from textual.app import App
from textual.message import Message


class TrainEvent(Message):
    def __init__(self, epoch: int, num_training_batches: int | None, batch: int) -> None:
        super().__init__()
        self.epoch = epoch
        self.num_training_batches = num_training_batches
        self.batch = batch


class LightningAdapter(Callback):
    def __init__(self, app: App) -> None:
        self.app = app

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._on_train_update(trainer, 0)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_train_update(trainer, batch_idx)

    def _on_train_update(self, trainer: L.Trainer, batch_idx: int) -> None:
        num_training_batches = None
        if trainer.num_training_batches != math.inf:
            num_training_batches = int(trainer.num_training_batches)
        self.app.post_message(TrainEvent(trainer.current_epoch, num_training_batches, batch_idx))
