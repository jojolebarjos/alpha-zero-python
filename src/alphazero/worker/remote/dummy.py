from simulator.game.connect import Config

from alphazero.data import Episode
from alphazero.random import Random

from .base import Remote


class DummyRemote(Remote):
    """..."""

    def __init__(self):
        self.config = Config(6, 7, 4)
        self.predictor = Random()
        self.episodes: list[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
