from typing import Self

import lightning as L


class Broker:
    """Worker broker.

    This is the main interface for the trainer server, as a way to communicate
    new model weights to remote workers.

    A typical broker notifies the trainer of new episodes using a callback
    mechanism.

    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def set_model(self, model: L.LightningModule) -> None:
        raise NotImplementedError
