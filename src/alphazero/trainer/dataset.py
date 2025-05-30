from torch.utils.data import Dataset

from alphazero.data import Sample


class SampleDataset(Dataset):
    """..."""

    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.transform(self.samples[index])
