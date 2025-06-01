from logging import Handler, LogRecord

from rich.logging import RichHandler

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog

from loguru import logger


# Hack from https://github.com/Textualize/textual/discussions/3568


class _Console:
    def __init__(self, rich_log: RichLog) -> None:
        self.rich_log = rich_log
        self.file = False

    def print(self, content):
        # TODO thread-safe?
        self.rich_log.write(content)


class Log(Widget):
    """..."""

    def compose(self) -> ComposeResult:
        yield RichLog()

    async def on_mount(self) -> None:
        rich_log = self.get_child_by_type(RichLog)
        console = _Console(rich_log)
        handler = RichHandler(
            console=console,  # type: ignore
            rich_tracebacks=True,
        )
        logger.remove()
        self._logger_id = logger.add(handler)

    async def on_unmount(self) -> None:
        logger.remove(self._logger_id)
