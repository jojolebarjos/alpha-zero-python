import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L

from simulator.game.connect import State

from ..data import Prediction, Turn
from ..predictor import BatchedPredictor


DEPTH = 3


def states_to_tensor(states: list[State]) -> torch.Tensor:
    """..."""

    count = len(states)
    config = states[0].config
    data = np.zeros((count, DEPTH, config.height, config.width), dtype=np.float32)
    for i, state in enumerate(states):
        player = state.player
        assert player >= 0
        grid = state.grid

        # Channel 0: own pieces
        data[i, 0] = grid == player

        # Channel 1: opponent pieces
        data[i, 1] = grid == (1 - player)

        # Channel 2: playable locations
        data[i, 2, 0] = grid[0] < 0
        data[i, 2, 1:] = (grid[1:] < 0) & (grid[:-1] >= 0)
    
    return torch.tensor(data)


class TurnDataset(Dataset):
    """..."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = turns
    
    def __len__(self):
        return len(self.turns)
    
    def __getitem__(self, index):
        turn = self.turns[index]
        state = turn.state
        config = state.config
        x = states_to_tensor([state])[0]
        policy = torch.zeros(config.width, dtype=torch.float32)
        for action, probability in turn.policy.items():
            policy[action.column] = probability
        value = torch.tensor(turn.value[0], dtype=torch.float32)
        return x, policy, value


#@torch.compile
class Model(nn.Module):
    """..."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(DEPTH, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.dropout = nn.Dropout1d(0.5)
        self.policy_head = nn.Conv1d(64, 1, kernel_size=1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        _, _, height, width = x.shape

        # Stacked convolutions, to capture local neighborhood
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Project vertically, to get an embedding per column
        x = F.max_pool2d(x, (height, 1))
        x = x.squeeze(2)
        x = self.dropout(x)

        # Apply policy head on each column
        policy_logits = self.policy_head(x).squeeze(1)

        # Aggregate column embeddings, and apply value head for board-wide value
        x = F.max_pool1d(x, width).squeeze(2)
        value_logits = self.value_head(x).squeeze(1)

        return policy_logits, value_logits


class LitModel(L.LightningModule):
    """..."""

    def __init__(self):
        super().__init__()
        self.model = Model()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, target_policy, target_value = batch
        policy_logits, value_logits = self.model(x)
        policy_loss = F.cross_entropy(policy_logits, target_policy)
        value = torch.tanh(value_logits)
        value_loss = nn.functional.mse_loss(value, target_value)
        loss = policy_loss + value_loss
        self.log("train_policy_loss", policy_loss)
        self.log("train_value_loss", value_loss)
        return loss
    
    # TODO add validation?


class ModelBatchedPredictor(BatchedPredictor):
    """..."""

    def __init__(self, model: Model, executor: ThreadPoolExecutor, device: torch.device) -> None:
        self.model = model
        self.executor = executor
        self.device = device
    
    async def predict_many(self, states: list[State]) -> list[Prediction]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_many, states)
    
    def _predict_many(self, states: list[State]) -> list[Prediction]:
        # TODO should probably limit batch size? implement as wrapper?

        # Apply neural network
        with torch.no_grad():
            x = states_to_tensor(states)
            x = x.to(self.device)
            policy_logits, value_logits = self.model(x)
            policy_logits = policy_logits.cpu().numpy()
            value_logits = value_logits.cpu().numpy()
        
        # Repack output as prediction objects
        values = np.tanh(value_logits)
        predictions = []
        for i, state in enumerate(states):
            actions = state.actions
            columns = np.array([action.column for action in actions])
            weights = np.exp(policy_logits[i, columns])
            probabilities = weights / weights.sum()
            policy = dict(zip(actions, probabilities))
            value = np.array([values[i], -values[i]])
            prediction = Prediction(policy, value)
            predictions.append(prediction)
        return predictions
