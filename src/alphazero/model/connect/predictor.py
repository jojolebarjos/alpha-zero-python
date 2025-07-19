import numpy as np

import torch

from alphazero.data import Prediction, State
from alphazero.predictor import Predictor

from .transform import state_to_tensor


class ConnectPredictor(Predictor):
    """..."""

    def __init__(self, model) -> None:
        self.model = model
        self.device = next(self.model.parameters()).device

    def predict_many(self, states: list[State]) -> list[Prediction]:
        if not states:
            return []
        x = [state_to_tensor(state) for state in states]
        x = torch.stack(x, dim=0)
        x = x.to(self.device)
        with torch.no_grad():
            policy_logits, value_logits = self.model(x)
            policy = torch.softmax(policy_logits, dim=-1)
            value = torch.sigmoid(value_logits) * 2.0 - 1.0
        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
        predictions = []
        for i, state in enumerate(states):
            actions = state.actions
            policy_i = policy[i, [action.column for action in actions]]
            value_i = np.array([value[i], -value[i]])
            prediction = Prediction(actions, policy_i, value_i)
            predictions.append(prediction)
        return predictions
