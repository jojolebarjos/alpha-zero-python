from base64 import b64encode
import json
from threading import Thread
from typing import Self

from websockets.sync.server import Server, ServerConnection, serve

import lightning as L

from alphazero.data import Config, Episode
from alphazero.trainer.buffer import Buffer
from alphazero.utility import to_torchscript

from .base import Broker


class WebsocketBroker(Broker):
    """..."""

    # TODO how do we bring the episodes/samples out of here? does the broker know/own the buffer? Do we have a callback?

    def __init__(
        self,
        config: Config,
        model: L.LightningModule,
        buffer: Buffer,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.buffer = buffer
        self.host = host
        self.port = port
        self._server: Server | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> Self:
        self._server = serve(self._handle, self.host, self.port)
        self._thread = Thread(target=self._run)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._server is not None
        self._server.shutdown()
        assert self._thread is not None
        self._thread.join()

    def _run(self) -> None:
        assert self._server is not None
        self._server.serve_forever()

    def _handle(self, connection: ServerConnection) -> None:
        payload = {
            "type": "config",
            # TODO class/name
            "data": self.config.to_json(),
        }
        connection.send(json.dumps(payload))

        last_model = None

        while True:
            # TODO report transfer times?

            if last_model is not self.model:
                last_model = self.model
                content = to_torchscript(last_model)
                payload = {
                    "type": "model",
                    # TODO class/name
                    "data": b64encode(content).decode("ascii"),
                }
                connection.send(json.dumps(payload))

            # TODO handle incoming episodes
            payload = json.loads(connection.recv())

            if payload["type"] == "episode":
                episode = Episode.from_json(payload["data"], self.config)
                self.buffer.add_episode(episode)
                continue

            raise KeyError(payload["type"])

    def set_model(self, model: L.LightningModule) -> None:
        self.model = model
