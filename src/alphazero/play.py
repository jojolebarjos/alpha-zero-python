import os
import re

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Pretty

import click

from simulator.game.connect import Config
from simulator.textual.connect import ConnectBoard

from alphazero.data import Prediction, State
from alphazero.model.connect import ConnectModel, ConnectPredictor
from alphazero.predictor import Predictor
from alphazero.random import Random
from alphazero.searcher import Searcher


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
        self.get_child_by_id("policy").update("...")  # type: ignore
        self._play(event.board, state)

    @work(thread=True)
    def _play(self, board: ConnectBoard, state: State) -> None:
        if state.has_ended:
            prediction = None
        else:
            [prediction] = self.predictor.predict_many([state])
        self.call_from_thread(self._update, board, state, prediction)

    async def _update(self, board: ConnectBoard, state: State, prediction: Prediction | None = None) -> None:
        board.state = state
        self.get_child_by_id("policy").update(prediction)  # type: ignore


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        subpath = os.path.join(path, "checkpoints")
        if os.path.isdir(subpath):
            path = subpath
        entries = []
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                match = re.fullmatch(r"epoch=(\d+)(?:-v(\d+))?\.ckpt", name)
                assert match is not None
                epoch = int(match.group(1))
                version = int(match.group(2) or 0)
                entry = epoch, version, name
                entries.append(entry)
        entries.sort()
        _, _, latest_name = entries[-1]
        path = os.path.join(path, latest_name)
    return path


@click.command()
@click.option("-p", "--path", help="Checkpoint path")
@click.option("-n", "--num-steps", default=0, help="Search iterations")
def run(path: str | None, num_steps: int) -> None:
    """..."""

    if path is None:
        config = Config(6, 7, 4)
        predictor = Random()
    else:
        path = resolve_checkpoint_path(path)
        model_class = ConnectModel
        model = model_class.load_from_checkpoint(path)
        config = Config(model.hparams["height"], model.hparams["width"], 4)
        predictor_class = ConnectPredictor
        predictor = predictor_class(model)

    if num_steps > 0:
        predictor = Searcher(predictor, num_steps=num_steps, c_puct=4.0)

    app = PlayApp(config, predictor)
    app.run()


if __name__ == "__main__":
    run()
