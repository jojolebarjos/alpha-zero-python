import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
import shutil
import time

from tqdm import tqdm

import torch

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from simulator.game.connect import Config, State  # pyright: ignore[reportMissingModuleSource]

from alphazero.data import Prediction
from alphazero.buffer import Buffer
from alphazero.predictor import BufferedPredictor, BatchedPredictor, RandomPredictor
from alphazero.sampler import sample_episode
from alphazero.search import SearchPredictor
from alphazero.model.connect import Model, ModelBatchedPredictor, LitModel


executor = ThreadPoolExecutor()


def now():
    return datetime.now(timezone.utc)


class Bridge(Callback, BatchedPredictor):
    """..."""

    def __init__(self, executor: ThreadPoolExecutor, device: torch.device):
        self.batched_predictor = None
        self.executor = executor
        self.device = device
        self.epoch = None
        self.epoch_start = None

    def set_model(self, model: Model) -> None:
        model = deepcopy(model).eval().to(self.device)
        self.batched_predictor = ModelBatchedPredictor(model, self.executor, self.device)

    def on_train_epoch_start(self, trainer: L.Trainer, lightning_model: L.LightningModule) -> None:
        self.epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, lightning_model: L.LightningModule) -> None:
        # TODO should we log some episode-related metrics (e.g. number of turns, win rate, ...)?
        self.set_model(lightning_model.model)
        self.epoch = trainer.current_epoch
        # TODO should maybe make sure the new epoch will start with at least one new episode?

        # Naive delay, useful when training set is too small
        epoch_end = time.perf_counter()
        actual_duration = epoch_end - self.epoch_start
        min_duration = 30.0
        if actual_duration < min_duration:
            time.sleep(min_duration - actual_duration)

    async def predict_many(self, states: list[State]) -> list[Prediction]:
        return await self.batched_predictor.predict_many(states)


async def main():
    loop = asyncio.get_running_loop()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    sessions_folder = "./sessions/"
    name = "foo"
    if not os.path.isdir(sessions_folder + name):
        os.makedirs(sessions_folder + name)

    config = Config(6, 7, 4)

    lightning_model = LitModel(config)

    bridge = Bridge(executor, device)
    bridge.set_model(lightning_model.model)

    logger = TensorBoardLogger(
        save_dir=sessions_folder,
        name=None,
        version=name,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=logger,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(sessions_folder, name, "checkpoints"),
                filename="{epoch}",
                save_top_k=-1,
                every_n_epochs=20,
            ),
            bridge,
        ],
    )
    # TODO saturate at 300 episodes (epoch 45, i.e. after 20 minutes)
    buffer = Buffer(
        batch_size=64,
        max_turns=64 * 100,  # TODO adjust this!
        path=os.path.join(sessions_folder, name, "episode.jl"),
        executor=executor,
    )

    warmup = 50
    if buffer.num_episodes < warmup:
        print("Ready, waiting until a few episodes are completed...")
        predictor = RandomPredictor()
        predictor = SearchPredictor(predictor, num_steps=1000, c_puct=1.0)
        for _ in tqdm(range(warmup - buffer.num_episodes)):
            initial_state = config.sample_initial_state()
            episode = await sample_episode(initial_state, predictor, temperature=1.0)
            metadata = {
                "model_epoch": -1,
                "trainer_epoch": 0,
                "timestamp": now().isoformat(),
            }
            await buffer.add_episode(episode, metadata)

    predictor = BufferedPredictor(bridge)
    predictor = SearchPredictor(predictor, num_steps=1000, c_puct=1.0)

    async def single_loop():
        while True:
            initial_state = config.sample_initial_state()
            episode = await sample_episode(initial_state, predictor, temperature=1.0)
            metadata = {
                "model_epoch": bridge.epoch,
                "trainer_epoch": trainer.current_epoch,
                "timestamp": now().isoformat(),
            }
            await buffer.add_episode(episode, metadata)
        # TODO should print on error, as `gather` seems to postpone the failure

    async def many_loops():
        await asyncio.gather(*[single_loop() for _ in range(64)])

    _ = loop.create_task(many_loops())

    # if buffer.num_episodes < warmup:
    #     print("Ready, waiting until a few episodes are completed...")
    #     with tqdm(total=warmup) as progress:
    #         while buffer.num_episodes < warmup:
    #             if progress.n < buffer.num_episodes:
    #                 progress.update(buffer.num_episodes - progress.n)
    #             await asyncio.sleep(1.0)

    checkpoint_path = os.path.join(sessions_folder, name, "current.ckpt")

    last_checkpoint_path = None
    if os.path.exists(checkpoint_path):
        last_checkpoint_path = checkpoint_path

    start_time = time.perf_counter()
    start_num_episodes = buffer.num_episodes

    try:

        def train():
            trainer.fit(
                lightning_model,
                datamodule=buffer,
                ckpt_path=last_checkpoint_path,
            )

        await loop.run_in_executor(executor, train)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        trainer.should_stop = True

    end_time = time.perf_counter()
    end_num_episodes = buffer.num_episodes

    print(f"Training time: {end_time - start_time:0.02} seconds")
    print(f"Episodes generated (ignoring warmup): {end_num_episodes - start_num_episodes}")
    print(
        f"Time per episode (amortized): {(end_time - start_time) / (end_num_episodes - start_num_episodes):0.02f} seconds"
    )

    trainer.save_checkpoint(checkpoint_path)

    # best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    # if best_checkpoint_path:
    #    shutil.copyfile(best_checkpoint_path, checkpoint_path)


with executor:
    asyncio.run(main())
