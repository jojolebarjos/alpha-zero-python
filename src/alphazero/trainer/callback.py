import lightning as L
from lightning.pytorch.callbacks import Callback

from .broker.base import Broker


class BrokerCallback(Callback):
    """..."""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.broker.set_model(pl_module)
