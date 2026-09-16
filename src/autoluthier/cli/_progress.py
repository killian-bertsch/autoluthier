"""Binds the pipeline's structured events (`pipeline.events`) to a Rich terminal progress bar.

Deliberately reads only `StepProgress`/`StepCompleted`'s `fraction`, never step counts or
kinds: the CLI's ``run`` command spans one reporter across load, the stage chain, and export
(see `cli.run`), and this sink has no need to know that composition — the reporter's own
monotonic overall fraction is already the right thing to show.
"""

from __future__ import annotations

from types import TracebackType

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from autoluthier.pipeline.events import (
    Event,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
)


class RichProgressSink:
    """An `EventSink` driving one Rich bar per instrument run.

    Reusable across a multi-instrument ``--scan`` batch: each `RunStarted` opens a new task and
    each `RunCompleted`/`RunFailed` closes it, so the terminal shows one live bar per instrument
    rather than an ever-growing stack of finished ones.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Build the sink.

        Args:
            console: Rich console to render into; `None` uses Rich's default.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> RichProgressSink:
        """Start rendering the progress display."""
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop rendering the progress display."""
        self._progress.stop()

    def __call__(self, event: Event) -> None:
        """Update the current bar in response to one pipeline event.

        Args:
            event: The event to render.
        """
        if isinstance(event, RunStarted):
            self._task_id = self._progress.add_task(event.instrument, total=1.0)
        elif isinstance(event, StepProgress | StepCompleted):
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=event.fraction)
        elif isinstance(event, RunCompleted):
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=1.0)
                self._task_id = None
        elif isinstance(event, RunFailed):
            self._task_id = None
