import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from alphazero.data import Config

from .broker import Broker
from .buffer import Buffer
from .callback import BrokerCallback
from .data_module import BufferDataModule


def train(
    folder: str,
    config: Config,
    model: L.LightningModule,
    transform,  # TODO type hint for transforms
    broker: Broker,
):
    """..."""

    # Everything will be saved in this folder
    folder = os.path.abspath(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    assert os.path.isdir(folder)

    buffer = Buffer(
        config,
        buffer_path=os.path.join(folder, "buffer.jsonl"),
        episode_path=os.path.join(folder, "episodes.jsonl"),
        max_length=50_000,
    )

    # TODO should wait until enough samples are there? maybe explicit warmup phase with Random model?
    # TODO where does the broker send the episodes? it probably needs to know the buffer...

    data_module = BufferDataModule(buffer, transform, batch_size=64)

    # Tensorboard will be the main logging strategy for deep learning related metrics
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.dirname(folder),
        name=None,
        version=os.path.basename(folder),
    )

    # TODO resume checkpoint, if any

    # Setup a never-ending training session, which reload a new dataset at each epoch
    # TODO could maybe switch to a pure infinite stream?
    trainer = L.Trainer(
        logger=tensorboard_logger,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        # enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(folder, "checkpoints"),
                filename="{epoch}",
                save_top_k=-1,
                every_n_epochs=20,
            ),
            BrokerCallback(broker),
        ],
    )

    # Let model train, use Lightning UI
    trainer.fit(
        model,
        datamodule=data_module,
        # TODO ckpt_path=last_checkpoint_path,
    )


def foo():
    from simulator.game.connect import Config as ConnectConfig

    from alphazero.model.connect import ConnectModel, transform
    from alphazero.trainer.broker.websocket import WebsocketBroker

    config = ConnectConfig(6, 7, 4)
    model = ConnectModel(
        height=6,
        width=7,
        model="simple",
        learning_rate=3e-4,
    )

    session_folder = "./sessions/foo/"

    broker = WebsocketBroker(config, model, host="0.0.0.0", port=8080)

    with broker:
        train(
            session_folder,
            config,
            model,
            transform,
            broker,
        )


foo()
