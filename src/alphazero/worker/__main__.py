import logging

import click

from rich.console import Console

from simulator.textual.connect import ConnectBoard

from alphazero.utility import configure_logging
from alphazero.worker.remote.dummy import DummyRemote
from alphazero.worker.remote.websocket import WebsocketRemote
from alphazero.worker.sampler import Sampler
from alphazero.worker.textual import WorkerApp


logger = logging.getLogger(__name__)


@click.command()
@click.option("-u", "--uri", help="Host name")
@click.option("-b", "--batch-size", default=8, help="Number of games in parallel.")
@click.option("-n", "--num-steps", default=400, help="Search iterations")
@click.option("-s", "--show", is_flag=True, help="Show generated games")
def run(uri: str | None, batch_size: int, num_steps: int, show: bool) -> None:
    """..."""

    console = Console()
    configure_logging(console)

    if uri is None:
        logger.warning("No URI specified, generated episodes will be lost!")
        remote = DummyRemote()
    else:
        remote = WebsocketRemote(uri)

    with remote:
        if show:
            # TODO properly handle widget class
            app = WorkerApp(remote, ConnectBoard, batch_size=batch_size, num_steps=num_steps)
            app.run()
        else:
            sampler = Sampler(remote, batch_size=batch_size, num_steps=num_steps)
            while True:
                sampler.step()


run()
