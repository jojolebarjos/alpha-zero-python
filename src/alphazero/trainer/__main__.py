import logging
import os

import click

from rich.console import Console
from rich.progress import Progress

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from simulator.game.connect import Config as ConnectConfig

from alphazero.data import Episode
from alphazero.model.connect import ConnectModel, transform
from alphazero.utility import configure_logging

from .broker import Broker, Callback as BrokerCallback
from .broker.websocket import WebsocketBroker
from .buffer import Buffer
from .callback import ModelUpdateCallback, RichProgressCallback
from .data_module import BufferDataModule


logger = logging.getLogger(__name__)


@click.command()
@click.argument("folder")
@click.option("-h", "--host", default="0.0.0.0", help="Websocket server host")
@click.option("-p", "--port", default=8080, help="Websocket server port")
def run(folder: str, host: str, port: int) -> None:
    """..."""

    console = Console()

    if not os.path.exists(folder):
        os.makedirs(folder)
    assert os.path.isdir(folder)

    configure_logging(console, os.path.join(folder, "trainer.log"))

    # TODO make game and model configurable

    config = ConnectConfig(6, 7, 4)

    model = ConnectModel(
        height=6,
        width=7,
        model="simple",
        learning_rate=3e-4,
    )

    buffer = Buffer(
        config,
        buffer_path=os.path.join(folder, "buffer.jsonl"),
        episode_path=os.path.join(folder, "episodes.jsonl"),
        max_length=50_000,
    )

    with Progress(console=console) as progress:
        buffer_task = progress.add_task("Buffer", completed=len(buffer), total=buffer.max_length)

        class BrokerAdapter(BrokerCallback):
            def on_episode(self, broker: Broker, worker_id: str, episode: Episode) -> None:
                buffer.add_episode(episode)
                progress.update(buffer_task, completed=len(buffer))

        broker_adapter = BrokerAdapter()
        broker = WebsocketBroker(config, model, broker_adapter, host=host, port=port)

        tensorboard_logger = TensorBoardLogger(
            save_dir=os.path.dirname(folder),
            name=None,
            version=os.path.basename(folder),
        )

        trainer = L.Trainer(
            logger=tensorboard_logger,
            max_epochs=-1,
            reload_dataloaders_every_n_epochs=1,
            log_every_n_steps=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=os.path.join(folder, "checkpoints"),
                    save_weights_only=True,
                    save_top_k=-1,
                    every_n_epochs=20,
                ),
                ModelUpdateCallback(broker),
                RichProgressCallback(progress),
            ],
        )

        data_module = BufferDataModule(
            buffer,
            transform,
            batch_size=64,
            novelty=5,
        )

        checkpoint_path = os.path.join(folder, "latest.ckpt")
        last_checkpoint_path = None
        if os.path.exists(checkpoint_path):
            last_checkpoint_path = checkpoint_path
            logger.info("Resuming from previous checkpoint")

        try:
            with broker:
                trainer.fit(
                    model,
                    datamodule=data_module,
                    ckpt_path=last_checkpoint_path,
                )
        finally:
            buffer.save_buffer()
            trainer.save_checkpoint(checkpoint_path)


run()
