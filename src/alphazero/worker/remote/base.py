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

    def add_episode(self, episode: Episode) -> None:
        raise NotImplementedError
