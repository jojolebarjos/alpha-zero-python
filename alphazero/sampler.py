import numpy as np

from .data import Action, State, Turn
from .predictor import Predictor


def sample_action(policy: dict[Action, float], temperature: float) -> Action:
    """..."""

    probabilities = np.array(list(policy.values()))
    if temperature != 1.0:
        # TODO handle 0 and infinity
        logits = np.log(probabilities)
        scaled_logits = np.exp(logits / temperature)
        probabilities = scaled_logits / scaled_logits.sum()
    return np.random.choice(list(policy.keys()), p=probabilities)


async def sample_episode(initial_state: State, predictor: Predictor, temperature: float) -> list[Turn]:
    """..."""

    # Play complete game
    state = initial_state
    turns = []
    while not state.has_ended:
        prediction = await predictor.predict(state)
        turn = Turn(state, prediction.policy, None)
        turns.append(turn)
        action = sample_action(prediction.policy, temperature)
        state = action.sample_next_state()

    # Backpropagate true reward
    value = state.reward
    for turn in turns:
        turn.value = value

    return turns
