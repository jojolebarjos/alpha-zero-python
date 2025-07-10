from typing import Callable

from textual.app import App, ComposeResult
from textual.widgets import ProgressBar

from .broker import EpisodeEvent
from .lightning import TrainEvent


class TrainerApp(App):
    """..."""

    DEFAULT_CSS = """

    """

    work: Callable[[], None]

    def compose(self) -> ComposeResult:
        # TODO better UI
        yield ProgressBar(id="training-progress")

    async def on_mount(self) -> None:
        self.run_worker(self.work, thread=True)

    async def on_episode_event(self, event: EpisodeEvent) -> None:
        # TODO update worker statistics
        pass

    async def on_train_event(self, event: TrainEvent) -> None:
        progress = self.app.query_one("#training-progress", ProgressBar)
        # TODO set epoch number
        progress.update(progress=event.batch, total=event.num_training_batches)
