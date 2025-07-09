from typing import Any, Callable

import lightning as L
from lightning.pytorch.callbacks import Callback as LightningCallback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from textual.app import App, ComposeResult
from textual.widgets import ProgressBar

from alphazero.data import Episode

from .broker import Broker, Callback as BrokerCallback


class TrainerApp(App):
    """..."""

    DEFAULT_CSS = """

    """

    work: Callable[[], None]

    def __init__(self) -> None:
        self.broker_callback = _BrokerCallback(self)
        self.lightning_callback = _LightningCallback(self)
        super().__init__()

    def compose(self) -> ComposeResult:
        # TODO better UI
        yield ProgressBar(id="training-progress")

    async def on_mount(self) -> None:
        self.run_worker(self.work, thread=True)


class _BrokerCallback(BrokerCallback):
    def __init__(self, app: TrainerApp) -> None:
        self.app = app

    def on_worker_start(self, broker: Broker, worker_id: str) -> None:
        pass

    def on_worker_end(self, broker: Broker, worker_id: str) -> None:
        pass

    def on_episode(self, broker: Broker, worker_id: str, episode: Episode) -> None:
        pass


class _LightningCallback(LightningCallback):
    def __init__(self, app: TrainerApp) -> None:
        self.app = app

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pass

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pass

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        async def handle():
            progress = self.app.query_one("#training-progress", ProgressBar)
            progress.update(progress=0, total=trainer.num_training_batches)

        self.app.call_from_thread(handle)

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
        async def handle():
            progress = self.app.query_one("#training-progress", ProgressBar)
            progress.advance()

        self.app.call_from_thread(handle)
