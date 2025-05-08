import numpy as np

from alphazero.data import Episode
from alphazero.worker.remote.dummy import DummyRemote


# TODO load this from CLI argument, somehow

remote = DummyRemote()

config = remote.config
batch_size = 8
temperature = 1.0


# TODO move this code into proper helper functions/classes

episodes = [Episode([config.sample_initial_state()], [], []) for _ in range(batch_size)]

while True:
    predictor = remote.predictor

    states = [episode.states[-1] for episode in episodes]
    predictions = predictor.predict_many(states)

    for i in range(batch_size):
        prediction = predictions[i]
        e = np.exp(prediction.policy_logits / temperature)
        p = e / e.sum()
        action = np.random.choice(prediction.actions, p=p)

        state = action.sample_next_state()

        episode = episodes[i]
        episode.predictions.append(prediction)
        episode.actions.append(action)
        episode.states.append(state)

        if state.has_ended:
            remote.add_episode(episode)
            episodes[i] = Episode([config.sample_initial_state()], [], [])
