from dataclasses import dataclass
from typing import Any, Self, TypeAlias

import numpy as np


# TODO protocol? how to properly introduce generics to tie states and actions?
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
            del data["state"]
            policy.append([data, probability])
        return {
            "state": self.state.to_json(),  # TODO delete config key?
            "policy": policy,
            "value": self.value.tolist(),
        }

    @classmethod
    def from_dict(cls, data, state_cls, action_cls) -> Self:
        state = state_cls.from_json(data["state"])
        policy = {}
        for action_data, probability in data["policy"]:
            action_data = {"state": data["state"], **action_data}
            # TODO `from_json` should probably reuse an existing state!
            action = action_cls.from_json(action_data)
            policy[action] = probability
        value = np.array(data["value"])
        return cls(state, policy, value)
