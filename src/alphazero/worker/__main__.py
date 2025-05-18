import click

from simulator.textual.connect import ConnectBoard

from alphazero.worker.app import WorkerApp
from alphazero.worker.remote.websocket import WebsocketRemote
from alphazero.worker.sampler import Sampler


@click.command()
@click.option("-u", "--uri", help="Host name")
@click.option("-b", "--batch-size", default=8, help="Number of games in parallel.")
@click.option("-s", "--show", is_flag=True, help="Show generated games")
def run(uri: str, batch_size: int, show: bool) -> None:
    """..."""

    # TODO maybe have some dummy mode?

    with WebsocketRemote(uri) as remote:
        if show:
            # TODO properly handle widget class
            app = WorkerApp(remote, ConnectBoard, batch_size=batch_size)
            app.run()
        else:
            sampler = Sampler(remote, batch_size=batch_size)
            while True:
                sampler.step()


run()
