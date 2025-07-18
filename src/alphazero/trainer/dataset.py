from typing import Any, Callable

from torch.utils.data import Dataset

from alphazero.data import Sample


class SampleDataset(Dataset):
    """A collection of samples.

    A transformation function must be provided to convert the game state into
    tensors. This transform depends on the model.

    """

    def __init__(self, samples: list[Sample], transform: Callable[[Sample], Any]) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.transform(self.samples[index])
