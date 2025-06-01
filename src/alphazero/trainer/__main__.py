import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from loguru import logger

from alphazero.data import Config

from .broker import Broker
from .buffer import Buffer
from .callback import BrokerCallback
from .data_module import BufferDataModule
from .initialization import sample_random_episodes


def train(
    session_folder: str,
    config: Config,
    model: L.LightningModule,
    buffer: Buffer,
    transform,  # TODO type hint for transforms
    broker: Broker,
):
    """..."""

    logger.info(f"Session saved in {session_folder}")

    # Thin wrapper around the sample buffer
    data_module = BufferDataModule(buffer, transform, batch_size=64, novelty=5)

    # Tensorboard will be the main logging strategy for deep learning related metrics
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.dirname(session_folder),
        name=None,
        version=os.path.basename(session_folder),
    )

    # TODO resume checkpoint, if any

    # Setup a never-ending training session, which reload a new dataset at each epoch
    # TODO could maybe switch to a pure infinite stream?
    trainer = L.Trainer(
        logger=tensorboard_logger,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(session_folder, "checkpoints"),
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
    # TODO add click CLI
    # TODO make the game/model arguments (i.e., need some form of registry)

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

    # Everything will be saved in this folder
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    assert os.path.isdir(session_folder)

    buffer = Buffer(
        config,
        buffer_path=os.path.join(session_folder, "buffer.jsonl"),
        episode_path=os.path.join(session_folder, "episodes.jsonl"),
        max_length=50_000,
    )

    try:
        sample_random_episodes(buffer, min_episodes=100)

        broker = WebsocketBroker(config, model, buffer, host="0.0.0.0", port=8080)

        with broker:
            train(session_folder, config, model, buffer, transform, broker)

    finally:
        buffer.save_buffer()


foo()
