from textual.app import App, ComposeResult
from textual.widgets import ListView, ProgressBar

from .broker import EpisodeAdded, WorkerConnected, WorkerDisconnected
from .lightning import TrainStep
from .worker import WorkerListItem


class TrainerApp(App):
    """..."""

    DEFAULT_CSS = """

    """

    def compose(self) -> ComposeResult:
        yield ProgressBar(id="training-progress")
        yield ListView(id="worker-list")

    async def on_mount(self) -> None:
        self.workers.start_all()

    async def on_worker_connected(self, event: WorkerConnected) -> None:
        worker_list = self.query_one("#worker-list")
        worker_item = WorkerListItem(f"worker-{event.worker_id}", event.worker_id, event.worker_name)
        await worker_list.mount(worker_item)

    async def on_worker_disconnected(self, event: WorkerDisconnected) -> None:
        worker_list = self.query_one("#worker-list")
        worker_item = worker_list.get_child_by_id(f"worker-{event.worker_id}", WorkerListItem)
        await worker_item.remove()

    async def on_episode_added(self, event: EpisodeAdded) -> None:
        worker_list = self.query_one("#worker-list")
        worker_item = worker_list.get_child_by_id(f"worker-{event.worker_id}", WorkerListItem)
        worker_item.num_episodes += 1

    async def on_train_step(self, event: TrainStep) -> None:
        progress = self.query_one("#training-progress", ProgressBar)
        # TODO set epoch number
        progress.update(progress=event.batch, total=event.num_training_batches)
