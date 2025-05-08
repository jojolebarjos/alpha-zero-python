import numpy as np

from alphazero.data import Episode
from alphazero.searcher import Searcher
from alphazero.worker.remote.dummy import DummyRemote


# TODO load this from CLI argument, somehow

remote = DummyRemote()

config = remote.config
batch_size = 8


# TODO move this code into proper helper functions/classes

episodes = [Episode([config.sample_initial_state()], [], []) for _ in range(batch_size)]

while True:
    predictor = remote.predictor
    # TODO should probably reuse part of the subtree
    predictor = Searcher(predictor, num_steps=400, c_puct=4.0)

    states = [episode.states[-1] for episode in episodes]
    predictions = predictor.predict_many(states)

    for i in range(batch_size):
        prediction = predictions[i]
        # TODO temperature?
        action = np.random.choice(prediction.actions, p=prediction.policy)

        state = action.sample_next_state()

        episode = episodes[i]
        episode.predictions.append(prediction)
        episode.actions.append(action)
        episode.states.append(state)

        if state.has_ended:
            remote.add_episode(episode)
            episodes[i] = Episode([config.sample_initial_state()], [], [])
