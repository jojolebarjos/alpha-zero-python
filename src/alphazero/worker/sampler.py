import logging

import numpy as np

from alphazero.data import Episode
from alphazero.searcher import Searcher
from alphazero.worker.remote import Remote


logger = logging.getLogger(__name__)


class Sampler:
    """..."""

    def __init__(self, remote: Remote, batch_size: int, num_steps: int) -> None:
        self.remote = remote
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.episodes = [Episode([self.remote.config.sample_initial_state()], [], []) for _ in range(self.batch_size)]

    def step(self) -> None:
        """..."""

        predictor = self.remote.predictor
        # TODO should probably reuse part of the subtree
        predictor = Searcher(predictor, num_steps=self.num_steps, c_puct=4.0)

        states = [episode.states[-1] for episode in self.episodes]
        predictions = predictor.predict_many(states)

        for i in range(self.batch_size):
            prediction = predictions[i]

            # TODO temperature?
            action_index = np.random.choice(len(prediction.actions), p=prediction.policy)
            action = prediction.actions[action_index]

            state = action.sample_next_state()

            episode = self.episodes[i]
            episode.predictions.append(prediction)
            episode.actions.append(action_index)
            episode.states.append(state)

            if state.has_ended:
                logger.info(f"Episode generated, {len(episode.states)} states")
                self.remote.add_episode(episode)
                self.episodes[i] = Episode([self.remote.config.sample_initial_state()], [], [])
