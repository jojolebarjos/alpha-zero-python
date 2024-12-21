from dataclasses import dataclass
from typing import Any, TypeAlias

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

    # TODO from_json, which needs to know State and Action classes
