import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(console: Console, path: str | None = None) -> None:
    """..."""

    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()
            logger.propagate = True

    logging.root.handlers.clear()

    logging.root.setLevel(logging.DEBUG)

    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
    )
    console_formatter = Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logging.root.addHandler(console_handler)

    if path is not None:
        file_handler = RotatingFileHandler(
            path,
            maxBytes=64 * 1024 * 1024,
            encoding="utf-8",
            backupCount=5,
        )
        file_formatter = Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

    logging.captureWarnings(True)
