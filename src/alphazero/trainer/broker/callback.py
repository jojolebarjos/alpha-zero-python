from alphazero.data import Episode

from .base import Broker


class Callback:
    """Broker callback."""

    def on_broker_start(self, broker: Broker) -> None:
        pass

    def on_broker_end(self, broker: Broker) -> None:
        pass

    def on_worker_start(self, broker: Broker, worker_id: str, worker_name: str) -> None:
        pass

    def on_worker_end(self, broker: Broker, worker_id: str) -> None:
        # TODO reason?
        pass

    def on_episode(self, broker: Broker, worker_id: str, episode: Episode) -> None:
        pass
