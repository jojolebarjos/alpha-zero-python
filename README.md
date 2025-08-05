# AlphaZero

...

Run with interactive game interface, without any training server (i.e., episodes are discarded), for debugging purpose:

```sh
python -m alphazero.worker --batch-size 4 --show
```

Run with interactive game interface, to see generated games live:

```sh
python -m alphazero.worker --uri ws://0.0.0.0:8080/ --batch-size 4 --show
```

Run without without interface, for efficiency:

```sh
python -m alphazero.worker --uri ws://0.0.0.0:8080/ --batch-size 64
```

Try to play against a trained model:

```sh
python -m alphazero.play --path sessions/foo --num-steps 100
```

Train model:

```sh
python -m alphazero.trainer sessions/foo
```

Show training logs:

```sh
tensorboard --logdir sessions
```

Build optimized MCGS:

```sh
C_INCLUDE_PATH=$(python -c 'import numpy; print(numpy.get_include())') cythonize -a -i src/alphazero/search/cython.pyx
```
