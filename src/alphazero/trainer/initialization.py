from loguru import logger

from alphazero.trainer.buffer import Buffer
from alphazero.worker.remote.dummy import DummyRemote
from alphazero.worker.sampler import Sampler


def sample_random_episodes(buffer: Buffer, min_episodes: int, num_steps: int = 1000) -> None:
    if buffer.num_episodes < min_episodes:
        logger.info("Generating warm-start episodes using random predictor...")
        remote = DummyRemote()
        sampler = Sampler(remote, batch_size=1, num_steps=num_steps)
        while buffer.num_episodes < min_episodes:
            sampler.step()
            if remote.episodes:
                logger.debug(f"{buffer.num_episodes}/{min_episodes} ({buffer.num_episodes / min_episodes:.1%})")
                for episode in remote.episodes:
                    buffer.add_episode(episode)
                remote.episodes.clear()
