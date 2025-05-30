from torch.utils.data import DataLoader

import lightning as L

from .buffer import Buffer
from .dataset import SampleDataset


class BufferDataModule(L.LightningDataModule):
    """..."""

    def __init__(self, buffer: Buffer, transform, batch_size: int) -> None:
        super().__init__()
        self.buffer = buffer
        self.transform = transform
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        samples = self.buffer.get_samples()
        assert len(samples) > 0
        dataset = SampleDataset(samples, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
