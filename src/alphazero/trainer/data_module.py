import time

from torch.utils.data import DataLoader

import lightning as L

from .buffer import Buffer
from .dataset import SampleDataset


class BufferDataModule(L.LightningDataModule):
    def __init__(self, buffer: Buffer, transform, batch_size: int, novelty: int = 5) -> None:
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
        dataset = SampleDataset(samples, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
