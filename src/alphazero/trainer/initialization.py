from tqdm import tqdm

from alphazero.trainer.buffer import Buffer
from alphazero.worker.remote.dummy import DummyRemote
from alphazero.worker.sampler import Sampler


def sample_random_episodes(buffer: Buffer, min_episodes: int) -> None:
    with tqdm(initial=buffer.num_episodes, total=min_episodes) as progress:
        remote = DummyRemote()
        sampler = Sampler(remote, batch_size=1)
        while buffer.num_episodes < min_episodes:
            sampler.step()
            for episode in remote.episodes:
                buffer.add_episode(episode)
                progress.update(1)
            remote.episodes.clear()
