from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Pretty

from simulator.game.connect import Config
from simulator.textual.connect import ConnectBoard

from alphazero.data import Prediction, State
from alphazero.predictor import Predictor


# TODO make the app more game-agnostic


class PlayApp(App):
    """Play against the agent."""

    def __init__(self, config: Config, predictor: Predictor) -> None:
        self.config = config
        self.predictor = predictor
        super().__init__()

    def compose(self) -> ComposeResult:
        state = self.config.sample_initial_state()
        board = ConnectBoard(state)
        yield board
        yield Pretty(None, id="policy")

    async def on_connect_board_reset(self, event: ConnectBoard.Reset) -> None:
        state = self.config.sample_initial_state()
        event.board.state = state
        self._play(event.board, state)

    async def on_connect_board_selected(self, event: ConnectBoard.Selected) -> None:
        state = event.action.sample_next_state()
        event.board.state = state
        self._play(event.board, state)

    @work(thread=True)
    def _play(self, board: ConnectBoard, state: State) -> None:
        if state.has_ended:
            prediction = None
        else:
            [prediction] = predictor.predict_many([state])
        self.call_from_thread(self._update, board, state, prediction)

    async def _update(self, board: ConnectBoard, state: State, prediction: Prediction | None = None) -> None:
        board.state = state
        self.get_child_by_id("policy").update(prediction)


if __name__ == "__main__":
    from alphazero.random import Random
    from alphazero.searcher import Searcher

    config = Config(6, 7, 4)
    predictor = Random()
    predictor = Searcher(predictor, num_steps=400, c_puct=4.0)
    app = PlayApp(config, predictor)
    app.run()
