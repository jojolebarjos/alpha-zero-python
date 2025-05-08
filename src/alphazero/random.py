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
            policy_logits = np.zeros(len(actions))
            value_logits = np.zeros(state.config.num_players)
            prediction = Prediction(actions, policy_logits, value_logits)
            predictions.append(prediction)
        return predictions
