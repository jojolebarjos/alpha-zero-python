import json
import os
import random
import threading

from alphazero.data import Config, Episode, Sample


class Buffer:
    """..."""

    def __init__(
        self,
        config: Config,
        buffer_path: str | None = None,
        episode_path: str | None = None,
        max_length: int = 50_000,
    ) -> None:
        self.config = config
        self.buffer_path = buffer_path
        self.episode_path = episode_path
        self.max_length = max_length
        self.num_episodes = 0
        self._lock = threading.RLock()
        self._samples: list[Sample] = []
        self._episode_file = None
        if self.episode_path is not None:
            self.num_episodes = _count_line_breaks(self.episode_path)
            self._episode_file = open(self.episode_path, "a", encoding="ascii")
        self.load_buffer()

    def load_buffer(self) -> None:
        with self._lock:
            samples = []
            if self.buffer_path is not None and os.path.exists(self.buffer_path):
                with open(self.buffer_path, "r", encoding="ascii") as file:
                    for line in file:
                        payload = json.loads(line)
                        sample = Sample.from_json(payload, self.config)
                        samples.append(sample)
            self._samples = samples
            # TODO probably reset some state

    def save_buffer(self) -> None:
        with self._lock:
            if self.buffer_path is not None:
                with open(self.buffer_path, "w", encoding="ascii") as file:
                    for sample in self._samples:
                        payload = sample.to_json()
                        line = json.dumps(payload)
                        file.write(line)
                        file.write("\n")

    def add_episode(self, episode: Episode) -> None:
        with self._lock:
            if self._episode_file is not None:
                payload = episode.to_json()
                line = json.dumps(payload)
                self._episode_file.write(line)
                self._episode_file.write("\n")
                self._episode_file.flush()
            self.num_episodes += 1
            new_samples = episode.to_samples()
            for new_sample in new_samples:
                count = len(self._samples)
                if count < self.max_length:
                    self._samples.append(new_sample)
                else:
                    index = random.randint(0, count - 1)
                    self._samples[index] = new_sample
            # TODO maybe auto-save buffer sometimes?

    def get_samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    # TODO thread-safe iterator, which re-shuffle at each epoch


def _count_line_breaks(path: str, chunk_size: int = 32000) -> int:
    count = 0
    if os.path.exists(path):
        with open(path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                count += chunk.count(b"\n")
    return count
