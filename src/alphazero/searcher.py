from alphazero.data import State, Prediction
from alphazero.predictor import Predictor
from alphazero.search import Search


class Searcher(Predictor):
    """Search-augmented model.

    This helper wraps a predictor and apply graph search for a fixed number of
    iterations.

    """

    def __init__(self, predictor: Predictor, num_steps: int, c_puct: float) -> None:
        self.predictor = predictor
        self.num_steps = num_steps
        self.c_puct = c_puct

    def predict_many(self, states: list[State]) -> list[Prediction]:
        searches = [Search(state, self.c_puct) for state in states]
        for _ in range(self.num_steps):
            selected_searches = [search for search in searches if search.select()]
            selected_states = [search.state for search in selected_searches]
            selected_predictions = self.predictor.predict_many(selected_states)
            for search, prediction in zip(selected_searches, selected_predictions):
                search.feed(prediction)
        predictions = [search.prediction for search in searches]
        return predictions
