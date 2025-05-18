from typing import Self

import lightning as L


class Broker:
    """..."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def set_model(self, model: L.LightningModule) -> None:
        raise NotImplementedError
