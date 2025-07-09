import os
import time

import click

from loguru import logger

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from simulator.game.connect import Config as ConnectConfig

from alphazero.model.connect import ConnectModel, transform

from textual.worker import get_current_worker

from .app import TrainerApp
from .broker.websocket import WebsocketBroker
from .buffer import Buffer
from .callback import ModelUpdateCallback
from .dataset import SampleDataset


@click.command()
@click.option("-h", "--host", default="0.0.0.0", help="Websocket server host")
@click.option("-p", "--port", default=8080, help="Websocket server port")
def run(host: str, port: int) -> None:
    """..."""

    # TODO folder from configuration
    session_folder = "./sessions/foo/"
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    assert os.path.isdir(session_folder)

    logger.remove()
    logger.add(os.path.join(session_folder, "trainer.log"), level="DEBUG")

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
        buffer_path=os.path.join(session_folder, "buffer.jsonl"),
        episode_path=os.path.join(session_folder, "episodes.jsonl"),
        max_length=50_000,
    )

    # TODO sample_random_episodes(buffer, min_episodes=100)

    app = TrainerApp()

    broker = WebsocketBroker(config, model, app.broker_callback, host=host, port=port)

    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.dirname(session_folder),
        name=None,
        version=os.path.basename(session_folder),
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
                dirpath=os.path.join(session_folder, "checkpoints"),
                filename="{epoch}",
                save_top_k=-1,
                every_n_epochs=20,
            ),
            ModelUpdateCallback(broker),
            app.lightning_callback,
        ],
    )

    class BufferDataModule(L.LightningDataModule):
        def __init__(self, buffer: Buffer, transform, batch_size: int, novelty: int = 5) -> None:
            super().__init__()
            self.buffer = buffer
            self.transform = transform
            self.batch_size = batch_size
            self.novelty = novelty
            self.last_num_episodes = None

        def train_dataloader(self) -> DataLoader:
            worker = get_current_worker()

            if self.last_num_episodes is not None:
                while self.buffer.num_episodes < self.last_num_episodes + self.novelty and not worker.is_cancelled:
                    time.sleep(0.1)
            self.last_num_episodes = self.buffer.num_episodes

            buffer.save_buffer()

            samples = self.buffer.get_samples()
            assert len(samples) > 0
            dataset = SampleDataset(samples, self.transform)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
            )

    data_module = BufferDataModule(
        buffer,
        transform,
        batch_size=64,
        novelty=5,
    )

    def work() -> None:
        # TODO maybe should wait here?
        trainer.fit(
            model,
            datamodule=data_module,
            # TODO ckpt_path=last_checkpoint_path,
        )

    app.work = work

    try:
        with broker:
            app.run()
    finally:
        trainer.should_stop = True
        buffer.save_buffer()


run()
