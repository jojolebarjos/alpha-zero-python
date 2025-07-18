import base64
import json
import logging
from threading import Thread
from typing import Self

from websockets import ConnectionClosed
from websockets.sync.client import connect, ClientConnection

from simulator.game.connect import Config

from alphazero.data import Episode
from alphazero.model.connect import ConnectPredictor
from alphazero.random import Random
from alphazero.utility import from_torchscript

from .base import Remote


logger = logging.getLogger(__name__)


class WebsocketRemote(Remote):
    """Websocket-based remote."""

    def __init__(self, uri: str):
        self.uri = uri
        self.config = None
        self.predictor = Random()
        self._connection: ClientConnection | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> Self:
        assert self._connection is None
        logger.info(f"Connecting to {self.uri}...")
        self._connection = connect(self.uri)
        try:
            logger.info("Connected! Waiting for configuration...")
            payload = json.loads(self._connection.recv())
            assert payload["type"] == "config"
            # TODO get config class from server
            config_class = Config
            self.config = config_class.from_json(payload["data"])
            # TODO should we block until model is received?
            self._thread = Thread(target=self._run)
            self._thread.start()
        except:
            self._connection.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._connection is not None
        self._connection.close()
        assert self._thread is not None
        self._thread.join()

    def _run(self) -> None:
        assert self._connection is not None
        try:
            while True:
                # TODO handle new model weights
                payload = json.loads(self._connection.recv())

                if payload["type"] == "model":
                    model = from_torchscript(base64.b64decode(payload["data"]))
                    # TODO how do we get this one? Probably the class is also shared by the trainer
                    # TODO make sure it is on the proper device
                    self.predictor = ConnectPredictor(model)
                    logger.info("Received new model")
                    continue

                raise KeyError(payload["type"])

        except ConnectionClosed:
            logger.warning("Websocket connection closed, unable to exchange messages with remote!")

    def add_episode(self, episode: Episode) -> None:
        assert self._connection is not None
        payload = {
            "type": "episode",
            "data": episode.to_json(),
        }
        self._connection.send(json.dumps(payload))
