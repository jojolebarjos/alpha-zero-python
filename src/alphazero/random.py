import numpy as np

from .data import State, Prediction
from .predictor import Predictor


class Random(Predictor):
    """Random prediction model.

    Actions are chosen uniformly. Players are assumed to have equal winning chances.

    """

    def predict_many(self, states: list[State]) -> list[Prediction]:
        predictions = []
        for state in states:
            actions = state.actions
            action_count = len(actions)
            policy = np.full(action_count, 1 / action_count, dtype=np.float32)
            value = np.zeros(state.config.num_players, dtype=np.float32)
            prediction = Prediction(actions, policy, value)
            predictions.append(prediction)
        return predictions
