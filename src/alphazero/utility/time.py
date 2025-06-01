from time import perf_counter_ns


def format_duration(duration: float) -> str:
    if duration < 1e-6:
        return f"{duration * 1e9:.01f}ns"
    if duration < 1e-3:
        return f"{duration * 1e6:.01f}us"
    if duration < 1.0:
        return f"{duration * 1e3:.01f}ms"
    return f"{duration:.01f}s"


class Tic:
    def __init__(self) -> None:
        self.start_ns = perf_counter_ns()

    def toc(self) -> float:
        end_ns = perf_counter_ns()
        return (end_ns - self.start_ns) * 1e-9


class Rate:
    """..."""

    def __init__(self, smoothing: float = 0.3) -> None:
        self.smoothing = smoothing

    # https://github.com/tqdm/tqdm/blob/master/tqdm/std.py#L214
