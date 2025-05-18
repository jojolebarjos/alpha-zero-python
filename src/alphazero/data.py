from dataclasses import dataclass
from typing import Any, Self, TypeAlias

import numpy as np


# TODO use generics and the provided protocol instead
Config: TypeAlias = Any
State: TypeAlias = Any
Action: TypeAlias = Any


@dataclass
class Prediction:
    """Prediction output.

    The model outputs both policy and value logits. This will typically be normalized
    and used in tree search.

    """

    actions: list[Action]
    policy: np.ndarray
    value: np.ndarray

    def to_json(self) -> Any:
        return {
            "actions": [action.to_json() for action in self.actions],
            "policy": self.policy.tolist(),
            "value": self.value.tolist(),
        }

    @classmethod
    def from_json(cls, payload: Any, state: State) -> Self:
        actions = [state.Action.from_json(action_payload, state) for action_payload in payload["actions"]]
        policy = np.array(payload["policy"], dtype=np.float32)
        assert policy.shape == (len(actions),)
        value = np.array(payload["value"], dtype=np.float32)
        assert value.shape == (state.config.num_players,)
        return cls(actions, policy, value)


@dataclass
class Episode:
    """Self-play episode.

    A complete self-play game will collect states, predictions, and selected actions. As
    the last state (i.e., terminal state) has no action left, there is one element more.

    """

    states: list[State]
    predictions: list[Prediction]
    actions: list[int]

    def to_json(self) -> Any:
        return {
            "states": [state.to_json() for state in self.states],
            "predictions": [prediction.to_json() for prediction in self.predictions],
            "actions": self.actions,
        }

    @classmethod
    def from_json(cls, payload: Any, config: Config) -> Self:
        states = [config.State.from_json(state_payload, config) for state_payload in payload["states"]]
        assert len(states) > 0
        assert len(payload["predictions"]) == len(states) - 1
        predictions = [
            Prediction.from_json(prediction_payload, state)
            for prediction_payload, state in zip(payload["predictions"], states)
        ]
        assert len(predictions) == len(states) - 1
        actions = payload["actions"]
        assert len(actions) == len(states) - 1
        for prediction, action in zip(predictions, actions):
            assert 0 <= action < len(prediction.actions)
        return cls(states, predictions, actions)


@dataclass
class Sample:
    """Training sample.

    The policy head is trained to predict the policy posterior (i.e., the normalized
    visit-count vector after the search). In other words, we train using the improved
    policy.

    The value head is trained to predict the actual game outcome (i.e., the observed
    reward in that specific episode).

    """

    state: State
    policy: np.ndarray
    value: np.ndarray

    def to_json(self) -> Any:
        return {
            "state": self.state.to_json(),
            "policy": self.policy.tolist(),
            "value": self.value.tolist(),
        }

    @classmethod
    def from_json(cls, payload: Any, config: Config) -> Self:
        state = config.State.from_json(payload["state"], config)
        policy = np.array(payload["policy"], dtype=np.float32)
        assert policy.shape == (len(state.actions),)
        value = np.array(payload["value"], dtype=np.float32)
        assert value.shape == (config.num_players,)
        return cls(state, policy, value)
