import numpy as np

import torch

from simulator.game.connect import State

from alphazero.data import Sample


DEPTH = 3


def state_to_tensor(state: State) -> torch.Tensor:
    """..."""

    config = state.config
    player = state.player
    assert player >= 0
    grid = state.grid
    data = np.zeros((DEPTH, config.height, config.width), dtype=np.float32)

    # Channel 0: own pieces
    data[0] = grid == player

    # Channel 1: opponent pieces
    data[1] = grid == (1 - player)

    # Channel 2: playable locations
    data[2, 0] = grid[0] < 0
    data[2, 1:] = (grid[1:] < 0) & (grid[:-1] >= 0)

    return torch.tensor(data)


def transform(sample: Sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = state_to_tensor(sample.state)
    y_policy = np.zeros(sample.state.config.width, dtype=np.float32)
    for action, probability in zip(sample.actions, sample.policy):
        y_policy[action.column] = probability
    y_policy = torch.from_numpy(y_policy)
    y_value = torch.tensor(sample.value[0])
    return x, y_policy, y_value
