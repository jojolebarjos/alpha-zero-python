import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from simulator.game.connect import Config, State

from alphazero.data import Prediction, Turn
from alphazero.predictor import RandomPredictor, BufferedPredictor, BatchedPredictor
from alphazero.sampler import sample_episode
from alphazero.search import SearchPredictor
from alphazero.model.connect import Model, ModelBatchedPredictor, TurnDataset, LitModel


executor = ThreadPoolExecutor()


class Buffer(L.LightningDataModule):
    """..."""

    def __init__(self, batch_size: int, max_turns: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_turns = max_turns
        self.turns = []
        self.num_episodes = 0

    def add_episode(self, episode: list[Turn]) -> None:
        turns = [*self.turns, *episode]
        if len(turns) > self.max_turns:
            random.shuffle(turns)
            turns = turns[:self.max_turns]
        self.turns = turns
        self.num_episodes += 1
    
    def train_dataloader(self) -> DataLoader:
        turns = list(self.turns)
        print(f"New data loader with {len(turns)} turns ({self.num_episodes} episodes so far)")
        assert len(turns) > 0
        dataset = TurnDataset(turns)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,)


class Bridge(Callback, BatchedPredictor):
    """..."""

    def __init__(self, executor: ThreadPoolExecutor, device: torch.device):
        self.batched_predictor = None
        self.executor = executor
        self.device = device
    
    def set_model(self, model: Model) -> None:
        model = deepcopy(model).eval().to(self.device)
        self.batched_predictor = ModelBatchedPredictor(model, self.executor, self.device)

    def on_train_epoch_end(self, trainer: L.Trainer, lightning_model: L.LightningModule) -> None:
        self.set_model(lightning_model.model)

    async def predict_many(self, states: list[State]) -> list[Prediction]:
        return await self.batched_predictor.predict_many(states)


async def main():

    loop = asyncio.get_running_loop()

    config = Config(6, 7, 4)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lightning_model = LitModel()

    buffer = Buffer(batch_size=64, max_turns=6400)

    bridge = Bridge(executor, device)
    bridge.set_model(lightning_model.model)

    predictor = BufferedPredictor(bridge)
    predictor = SearchPredictor(predictor, num_steps=100, c_puct=1.0)

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}",
                save_top_k=-1,
                every_n_epochs=1,
            ),
        ],
    )

    async def single_loop():
        while True:
            initial_state = config.sample_initial_state()
            episode = await sample_episode(initial_state, predictor, temperature=1.0)
            # TODO save episode to disk
            buffer.add_episode(episode)
    
    async def many_loops():
        await asyncio.gather(*[single_loop() for _ in range(64)])

    _ = loop.create_task(many_loops())

    print("Ready, waiting until a few episodes are completed...")
    while buffer.num_episodes < 32:
        await asyncio.sleep(1.0)

    await loop.run_in_executor(executor, lambda: trainer.fit(lightning_model, datamodule=buffer))
    

async def foo():
    config = Config(6, 7, 4)
    state = config.sample_initial_state()
    predictor = RandomPredictor()
    predictor = SearchPredictor(predictor, num_steps=1000, c_puct=1.0)
    for _ in range(10):
        _ = await sample_episode(state, predictor, temperature=1.0)


# pip install pyinstrument
# python -m pyinstrument -o out main.py

with executor:
    asyncio.run(main())
