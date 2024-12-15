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
