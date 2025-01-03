from dataclasses import dataclass
from typing import Any, Self, TypeAlias

import numpy as np


# TODO protocol? how to properly introduce generics to tie states and actions?
Config: TypeAlias = Any
State: TypeAlias = Any
Action: TypeAlias = Any


@dataclass
class Prediction:
    policy: dict[Action, float]
    value: np.ndarray


@dataclass
class Turn:
    state: State
    policy: dict[Action, float]
    value: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        policy = []
        for action, probability in self.policy.items():
            data = action.to_json()
            policy.append([data, probability])
        return {
            "state": self.state.to_json(),
            "policy": policy,
            "value": self.value.tolist(),
        }

    @classmethod
    def from_dict(cls, data, config) -> Self:
        state = config.State.from_json(data["state"], config)
        policy = {}
        for action_data, probability in data["policy"]:
            action = state.Action.from_json(action_data, state)
            policy[action] = probability
        value = np.array(data["value"])
        return cls(state, policy, value)


# TODO episode class
