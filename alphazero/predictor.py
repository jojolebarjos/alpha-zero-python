import asyncio

import numpy as np

from .data import Prediction, State


class Predictor:
    """..."""

    async def predict(self, state: State) -> Prediction:
        raise NotImplementedError


class RandomPredictor(Predictor):
    """..."""

    async def predict(self, state: State) -> Prediction:
        actions = state.actions
        policy = {action: 1 / len(actions) for action in actions}
        value = np.zeros(state.config.num_players)
        return Prediction(policy, value)


class BatchedPredictor:
    """..."""

    async def predict_many(self, states: list[State]) -> list[Prediction]:
        raise NotImplementedError


class BufferedPredictor(Predictor):
    """..."""

    def __init__(self, batched_predictor: BatchedPredictor) -> None:
        self.batched_predictor = batched_predictor
        self._states: list[State] | None = None
        self._task: asyncio.Task[list[Prediction]] | None = None
    
    async def predict(self, state: State) -> Prediction:
        if self._task is None:
            index = 0
            self._states = [state]
            self._task = asyncio.create_task(self._execute())
        else:
            index = len(self._states)
            self._states.append(state)
        predictions = await self._task
        prediction = predictions[index]
        return prediction

    async def _execute(self) -> None:
        states = self._states
        self._states = None
        self._task = None
        predictions = await self.batched_predictor.predict_many(states)
        return predictions
