import time
from typing import Any, Callable

import numpy as np

from torch.utils.data import DataLoader

import lightning as L

from alphazero.data import Sample

from .buffer import Buffer
from .dataset import SampleDataset


class BufferDataModule(L.LightningDataModule):
    """Buffer-based data module.

    Each time a data loader is re-created, the latest buffer content is used.
    This operation waits until enough new samples have been added to the
    buffer.

    """

    def __init__(
        self,
        buffer: Buffer,
        transform: Callable[[Sample], Any],
        batch_size: int,
        novelty: int = 5,
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.transform = transform
        self.batch_size = batch_size
        self.novelty = novelty
        self.last_num_episodes = buffer.num_episodes

    def train_dataloader(self) -> DataLoader:
        assert self.trainer is not None

        while self.buffer.num_episodes < self.last_num_episodes + self.novelty and not self.trainer.should_stop:
            time.sleep(0.1)
        self.last_num_episodes = self.buffer.num_episodes

        self.buffer.save_buffer()

        samples = self.buffer.get_samples()
        assert len(samples) > 0

        tensorboard = self.trainer.logger.experiment  # type: ignore
        tensorboard.add_histogram(
            "policy_histogram",
            np.concatenate([sample.policy for sample in samples]),
            bins=64,
            global_step=self.trainer.global_step,
        )

        dataset = SampleDataset(samples, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
