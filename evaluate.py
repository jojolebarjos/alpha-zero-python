import asyncio
from concurrent.futures import ThreadPoolExecutor

from textual.app import App, ComposeResult
from textual.widgets import Pretty

import torch

from simulator.game.connect import Config, State
from simulator.textual.connect import ConnectBoard

from alphazero.model.connect import LitModel, ModelBatchedPredictor
from alphazero.predictor import RandomPredictor, BufferedPredictor
from alphazero.search import SearchPredictor
from alphazero.sampler import sample_action


executor = ThreadPoolExecutor()

config = Config(6, 7, 4)
state = config.sample_initial_state()

# predictor = RandomPredictor()

device = torch.device("cpu")
path = "./sessions/foo/checkpoints/epoch=1559.ckpt"
lightning_model = LitModel.load_from_checkpoint(path, config=config).to(device)
batched_predictor = ModelBatchedPredictor(lightning_model.model, executor, device)
predictor = BufferedPredictor(batched_predictor)

# predictor = SearchPredictor(predictor, num_steps=1000, c_puct=1.0)


class AgentApp(App):
    """Play against the agent."""

    def compose(self) -> ComposeResult:
        board = ConnectBoard()
        board.state = config.sample_initial_state()
        yield board
        yield Pretty(None, id="policy")

    async def on_connect_board_reset(self, event: ConnectBoard.Reset) -> None:
        await self._play(event.board, config.sample_initial_state())

    async def on_connect_board_selected(self, event: ConnectBoard.Selected) -> None:
        await self._play(event.board, event.action.sample_next_state())

    async def _play(self, board: ConnectBoard, state: State) -> None:
        board.state = state
        if state.has_ended:
            prediction = None
        else:
            prediction = await predictor.predict(state)
        self.get_child_by_id("policy").update(prediction)


with executor:
    app = AgentApp()
    app.run()
