from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback as LightningCallback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from rich.progress import Progress, TaskID

from .broker import Broker


class ModelUpdateCallback(LightningCallback):
    """..."""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.broker.set_model(pl_module)


class RichProgressCallback(LightningCallback):
    """..."""

    def __init__(self, progress: Progress):
        super().__init__()
        self.progress = progress
        self.task: TaskID | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        assert self.task is None
        self.task = self.progress.add_task(f"Epoch {trainer.current_epoch}", total=None)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        assert self.task is not None
        self.progress.remove_task(self.task)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        assert self.task is not None
        self.progress.update(
            self.task,
            description=f"Epoch {trainer.current_epoch}",
            completed=0,
            total=trainer.num_training_batches,
        )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pass

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert self.task is not None
        self.progress.update(self.task, advance=len(batch))
