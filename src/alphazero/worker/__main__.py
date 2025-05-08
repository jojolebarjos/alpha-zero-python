from alphazero.worker.remote.dummy import DummyRemote
from alphazero.worker.sampler import Sampler


# TODO load this from CLI argument, somehow

remote = DummyRemote()

sampler = Sampler(remote, batch_size=8)
while True:
    sampler.step()
