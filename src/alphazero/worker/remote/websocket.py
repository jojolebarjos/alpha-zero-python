import json
from threading import Thread
from typing import Self

from websockets import ConnectionClosed
from websockets.sync.client import connect, ClientConnection

from simulator.game.connect import Config

from alphazero.data import Episode

from .base import Remote


class WebsocketRemote(Remote):
    """..."""

    def __init__(self, uri: str):
        self.uri = uri
        self.config = None
        self.predictor = None
        self._connection: ClientConnection | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> Self:
        assert self._connection is None
        self._connection = connect(self.uri)
        try:
            payload = json.loads(self._connection.recv())
            assert payload["type"] == "config"
            # TODO get config class from server
            config_class = Config
            self.config = config_class.from_json(payload["data"])
            # payload = json.loads(self.connection.recv())
            # assert payload["type"] == "model"
            # TODO get model class from server
            # TODO make initial predictor

            self._thread = Thread(target=self._run)
            self._thread.start()
        except:
            self._connection.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._connection.close()
        self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                # TODO handle new model weights
                payload = json.loads(self._connection.recv())
                ...
        except ConnectionClosed:
            # TODO log this, at least, even if the worker can continue until `add_episode` fails?
            pass

    def add_episode(self, episode: Episode) -> None:
        # TODO maybe move this to background, to avoid blocking for too long?
        payload = {
            "type": "episode",
            "data": episode.to_json(),
        }
        self._connection.send(json.dumps(payload))
