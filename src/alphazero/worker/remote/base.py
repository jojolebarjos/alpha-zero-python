from typing import Self

from alphazero.data import Config, Episode
from alphazero.predictor import Predictor


class Remote:
    """Remote training server.

    This is the main interface for an episode generator worker. It gives access to the
    game configuration and the latest prediction model. Workers can submit newly sampled
    episodes.

    """

    config: Config
    predictor: Predictor

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def add_episode(self, episode: Episode) -> None:
        raise NotImplementedError
