from alphazero.data import Prediction, State


class Predictor:
    """Prediction model."""

    def predict_many(self, states: list[State]) -> list[Prediction]:
        """Perform batched prediction.

        Many implementations, typically relying on neural networks, benefit from batched
        inference.

        """

        raise NotImplementedError

    def predict(self, state: State) -> Prediction:
        """Perform a single prediction."""

        [prediction] = self.predict_many([state])
        return prediction
