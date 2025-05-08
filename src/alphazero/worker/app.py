from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widget import Widget
from textual.worker import get_current_worker

from alphazero.worker.remote import Remote
from alphazero.worker.sampler import Sampler


class WorkerApp(App):
    """Episode generator."""

    DEFAULT_CSS = """

    Grid {
        grid-size: 4;
    }

    BounceBoard {
        border: round white;
    }

    """

    def __init__(self, remote: Remote, widget_class: type[Widget], batch_size: int) -> None:
        self.remote = remote
        self.widget_class = widget_class
        self.batch_size = batch_size
        self.sampler = Sampler(remote, batch_size)
        super().__init__()

    def compose(self) -> ComposeResult:
        with Grid():
            for i in range(self.batch_size):
                board = self.widget_class(id=f"state-{i}", disabled=True)
                yield board

    async def on_mount(self) -> None:
        self.do_episodes()

    @work(thread=True)
    def do_episodes(self) -> None:
        worker = get_current_worker()
        while not worker.is_cancelled:
            self.call_from_thread(self.update_states)
            self.sampler.step()

    async def update_states(self) -> None:
        for i in range(self.batch_size):
            widget = self.get_widget_by_id(f"state-{i}")
            state = self.sampler.episodes[i].states[-1]
            widget.state = state


if __name__ == "__main__":
    from simulator.textual.connect import ConnectBoard
    from alphazero.worker.remote.dummy import DummyRemote

    remote = DummyRemote()
    app = WorkerApp(remote, ConnectBoard, batch_size=8)
    app.run()
