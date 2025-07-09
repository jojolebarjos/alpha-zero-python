from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widget import Widget
from textual.worker import get_current_worker

from alphazero.worker.remote import Remote
from alphazero.worker.sampler import Sampler

from .log import Log


class WorkerApp(App):
    """Episode generator."""

    DEFAULT_CSS = """

    #grid {
        grid-size: 4;

        * {
            border: round white;
        }
    }

    #log {
        dock: bottom;
        height: 8;
    }

    """

    def __init__(self, remote: Remote, widget_class: type[Widget], batch_size: int, num_steps: int) -> None:
        self.remote = remote
        self.widget_class = widget_class
        self.batch_size = batch_size
        self.sampler = Sampler(remote, batch_size, num_steps)
        super().__init__()

    def compose(self) -> ComposeResult:
        with Grid(id="grid"):
            for i in range(self.batch_size):
                board = self.widget_class(id=f"state-{i}", disabled=True)
                yield board
        yield Log(id="log")

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
            widget.state = state  # type: ignore
