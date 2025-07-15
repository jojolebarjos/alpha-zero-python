from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, ListItem


class WorkerListItem(ListItem):
    """..."""

    DEFAULT_CSS = """

    WorkerListItem {

    }

    """

    num_episodes = reactive[int](0)

    def __init__(self, id: str, worker_id: str, worker_name: str) -> None:
        super().__init__(id=id)
        self.worker_id = worker_id
        self.worker_name = worker_name

    def compose(self) -> ComposeResult:
        self.label = Label("...")
        yield self.label

    def watch_num_episodes(self, value: int) -> None:
        self.label.update(f"{self.worker_name} ({self.num_episodes})")
