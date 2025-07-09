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

Try to play against a trained model:

```sh
python -m alphazero.play --path sessions/foo --num-steps 100
```

Train model:

```sh
python -m alphazero.trainer
```
