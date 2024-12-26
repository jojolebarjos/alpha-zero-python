import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import random

from torch.utils.data import DataLoader

import lightning as L

# TODO this class should be game independent
from simulator.game.connect import Action, State  # pyright: ignore[reportMissingModuleSource]

from alphazero.data import Turn
from alphazero.model.connect import TurnDataset


class Buffer(L.LightningDataModule):
    """..."""

    def __init__(self, batch_size: int, max_turns: int, path: str, executor: ThreadPoolExecutor) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_turns = max_turns
        self.path = path
        self.executor = executor

        self.turns = []
        self.num_episodes = 0

        with open(path, "r") as file:
            for line in file:
                episode_data = json.loads(line)
                episode = [Turn.from_dict(turn_data, State, Action) for turn_data in episode_data]
                self._collect_turns(episode)

        self.lock = asyncio.Lock()
        self.file = open(path, "a")

    async def _write_episode(self, episode: list[Turn]) -> None:
        episode_data = [turn.to_dict() for turn in episode]
        line = json.dumps(episode_data) + "\n"
        loop = asyncio.get_running_loop()
        async with self.lock:
            await loop.run_in_executor(self.executor, self.file.write, line)

    def _collect_turns(self, episode: list[Turn]) -> None:
        turns = [*self.turns, *episode]
        if len(turns) > self.max_turns:
            random.shuffle(turns)
            turns = turns[: self.max_turns]
        self.turns = turns
        self.num_episodes += 1

    async def add_episode(self, episode: list[Turn]) -> None:
        self._collect_turns(episode)
        await self._write_episode(episode)

    def train_dataloader(self) -> DataLoader:
        turns = list(self.turns)
        print(f"New data loader with {len(turns)} turns ({self.num_episodes} episodes so far)")
        assert len(turns) > 0
        # TODO this is the only component that is game-dependent, how to abstract this away?
        dataset = TurnDataset(turns)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
