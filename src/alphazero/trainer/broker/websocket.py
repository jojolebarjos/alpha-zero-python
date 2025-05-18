import io
import json
from threading import Thread
from typing import Self

from websockets.sync.server import Server, ServerConnection, serve

import torch

import lightning as L

from alphazero.data import Config

from .base import Broker


class WebsocketBroker(Broker):
    """..."""

    def __init__(
        self,
        config: Config,
        model: L.LightningModule,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.model = model
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
        self._server.shutdown()
        self._thread.join()

    def _run(self) -> None:
        self._server.serve_forever()

    def _handle(self, connection: ServerConnection) -> None:
        payload = {
            "type": "config",
            # TODO class/name
            "data": self.config.to_json(),
        }
        connection.send(json.dumps(payload))

        content = to_torchscript(self.model)
        payload = {
            "type": "model",
            # TODO class/name
            "data": content,
        }
        connection.send(json.dumps(payload))

        while True:
            # TODO should also send new models here
            # TODO handle incoming episodes
            payload = json.loads(connection.recv())
            ...

    def set_model(self, model: L.LightningModule) -> None:
        raise NotImplementedError


def to_torchscript(lightning_model: L.LightningModule) -> bytes:
    script = lightning_model.to_torchscript()
    file = io.BytesIO()
    torch.jit.save(script, file)
    content = file.getvalue()
    return content
