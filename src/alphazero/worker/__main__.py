import click

from loguru import logger

from simulator.textual.connect import ConnectBoard

from alphazero.worker.remote.dummy import DummyRemote
from alphazero.worker.remote.websocket import WebsocketRemote
from alphazero.worker.sampler import Sampler
from alphazero.worker.textual import WorkerApp


@click.command()
@click.option("-u", "--uri", help="Host name")
@click.option("-b", "--batch-size", default=8, help="Number of games in parallel.")
@click.option("-n", "--num-steps", default=400, help="Search iterations")
@click.option("-s", "--show", is_flag=True, help="Show generated games")
def run(uri: str | None, batch_size: int, num_steps: int, show: bool) -> None:
    """..."""

    # TODO setup loguru to have a disk log?

    if uri is None:
        logger.warning("No URI specified, generated episodes will be lost!")
        remote = DummyRemote()
    else:
        remote = WebsocketRemote(uri)

    with remote:
        if show:
            # TODO properly handle widget class
            # TODO loguru display in app?
            app = WorkerApp(remote, ConnectBoard, batch_size=batch_size, num_steps=num_steps)
            app.run()
        else:
            sampler = Sampler(remote, batch_size=batch_size, num_steps=num_steps)
            from tqdm import tqdm

            for i in tqdm(range(10)):
                sampler.step()


run()
