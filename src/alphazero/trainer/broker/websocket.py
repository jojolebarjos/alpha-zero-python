from base64 import b64encode
import json
import logging
from threading import Thread
from typing import Self
from uuid import uuid4

from websockets.sync.server import Server, ServerConnection, serve

import lightning as L

from alphazero.data import Config, Episode
from alphazero.utility import to_torchscript

from .base import Broker
from .callback import Callback


logger = logging.getLogger(__name__)


class WebsocketBroker(Broker):
    """Websocket-based broker."""

    def __init__(
        self,
        config: Config,
        model: L.LightningModule,
        callback: Callback,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.callback = callback
        self.host = host
        self.port = port
        self._server: Server | None = None
        self._thread: Thread | None = None
        self._connections = set[ServerConnection]()

    def __enter__(self) -> Self:
        self._server = serve(self._handle, self.host, self.port)
        self._thread = Thread(target=self._run)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._server is not None
        logger.debug("Shutting down...")
        self._server.shutdown()
        for connection in self._connections:
            connection.close()
        assert self._thread is not None
        self._thread.join()

    def _run(self) -> None:
        assert self._server is not None
        logger.info("Websocket server is starting...")
        self.callback.on_broker_start(self)
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Websocket server closed")
            self.callback.on_broker_end(self)

    def _handle(self, connection: ServerConnection) -> None:
        worker_id = uuid4().hex
        host, port = connection.remote_address
        worker_name = f"{host}:{port}"
        logger.info(f"{worker_name} connected")
        self.callback.on_worker_start(self, worker_id, worker_name)
        self._connections.add(connection)

        try:
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

                payload = json.loads(connection.recv())

                if payload["type"] == "episode":
                    episode = Episode.from_json(payload["data"], self.config)
                    logger.info(f"{worker_name} new episode")
                    self.callback.on_episode(self, worker_id, episode)
                    continue

                raise KeyError(payload["type"])

        except BaseException as e:
            logger.exception(e)

        logger.info(f"{worker_name} disconnected")
        self._connections.remove(connection)
        self.callback.on_worker_end(self, worker_id)

    def set_model(self, model: L.LightningModule) -> None:
        self.model = model
