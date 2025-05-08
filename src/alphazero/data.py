from dataclasses import dataclass
from typing import Any, TypeAlias

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


@dataclass
class Episode:
    """Self-play episode.

    A complete self-play game will collect states, predictions, and selected actions. As
    the last state (i.e., terminal state) has no action left, there is one element more.

    """

    states: list[State]
    predictions: list[Prediction]
    actions: list[int]


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
