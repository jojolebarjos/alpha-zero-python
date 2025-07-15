from textual.app import App
from textual.message import Message

from alphazero.data import Episode
from alphazero.trainer.broker import Broker, Callback


class WorkerConnected(Message):
    def __init__(self, worker_id: str, worker_name: str) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.worker_name = worker_name


class WorkerDisconnected(Message):
    def __init__(self, worker_id: str) -> None:
        super().__init__()
        self.worker_id = worker_id


class EpisodeAdded(Message):
    def __init__(self, worker_id: str, episode: Episode) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.episode = episode


class BrokerAdapter(Callback):
    def __init__(self, app: App) -> None:
        self.app = app

    def on_worker_start(self, broker: Broker, worker_id: str, worker_name: str) -> None:
        self.app.post_message(WorkerConnected(worker_id, worker_name))

    def on_worker_end(self, broker: Broker, worker_id: str) -> None:
        self.app.post_message(WorkerDisconnected(worker_id))

    def on_episode(self, broker: Broker, worker_id: str, episode: Episode) -> None:
        self.app.post_message(EpisodeAdded(worker_id, episode))
