"""Structured progress events for a pipeline run, and the reporter that emits them.

A run is reported as a flat sequence of **steps**: one ``load`` step, then one step per
segment of the stage chain (see ``pipeline/graph.py::partition_chain``). Steps — not
individual stages — are the reporting unit because a run of consecutive per-sample stages is
executed as *one* pass over the samples: a worker applies the whole run to a sample before the
next sample is touched, so there is no instant at which "the trim stage" is what's running.
Reporting per stage would mean either lying about that or making a separate parallel pass per
stage, which would multiply the audio sent to and from worker processes by the number of
stages. Each step therefore names the ``stage_ids`` it covers.

`StepProgress.fraction` is overall run progress in ``[0, 1]``, weighting every step equally:
``(completed_steps + within_step) / total_steps``. That is monotonic by construction — the
step index only rises and within-step completion only rises — and `ProgressReporter` additionally
clamps it against the highest value already emitted, because the consumers are a CLI progress
bar (step 8) and an SSE stream feeding a browser (step 9), and a bar that jumps backwards reads
as a bug even when the underlying numbers are defensible.

Events are plain frozen dataclasses with an explicit `Event.payload` rather than
``dataclasses.asdict``: the payload *is* the SSE wire contract, so its shape is written out
once here instead of being an accident of field naming.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from autosampler.config.schema import JSONValue

EventPayload = dict[str, JSONValue]
"""JSON-serializable form of an event, as sent over SSE in step 9."""

StepKind = Literal["load", "stages", "barrier"]
"""What a step does: read+slice the source audio, run per-sample stages, or run a set-level one."""


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for pipeline events; `kind` is the discriminator on the wire."""

    kind: ClassVar[str] = "event"

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event, including its `kind`.

        Raises:
            NotImplementedError: always — subclasses define their own payload.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    """A run began; `total_steps` is fixed for the whole run and drives the progress bar."""

    kind: ClassVar[str] = "run_started"

    instrument: str
    total_steps: int

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {"event": self.kind, "instrument": self.instrument, "total_steps": self.total_steps}


@dataclass(frozen=True, slots=True)
class StepStarted(Event):
    """One step began. `total_units` is the number of samples it will process."""

    kind: ClassVar[str] = "step_started"

    index: int
    label: str
    step_kind: StepKind
    stage_ids: tuple[str, ...]
    total_units: int

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {
            "event": self.kind,
            "index": self.index,
            "label": self.label,
            "step_kind": self.step_kind,
            "stage_ids": list(self.stage_ids),
            "total_units": self.total_units,
        }


@dataclass(frozen=True, slots=True)
class StepProgress(Event):
    """Progress within a step. `fraction` is overall run progress, never decreasing."""

    kind: ClassVar[str] = "step_progress"

    index: int
    label: str
    completed_units: int
    total_units: int
    fraction: float

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {
            "event": self.kind,
            "index": self.index,
            "label": self.label,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "fraction": self.fraction,
        }


@dataclass(frozen=True, slots=True)
class StepCompleted(Event):
    """One step finished, with its own wall time."""

    kind: ClassVar[str] = "step_completed"

    index: int
    label: str
    duration_s: float
    fraction: float

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {
            "event": self.kind,
            "index": self.index,
            "label": self.label,
            "duration_s": self.duration_s,
            "fraction": self.fraction,
        }


@dataclass(frozen=True, slots=True)
class RunCompleted(Event):
    """A run finished successfully.

    `dynamic_range_db` is present only when the chain ran ``normalize`` in ``velocity`` mode;
    it becomes the SFZ ``amp_velcurve_1`` opcode in export (step 7).
    """

    kind: ClassVar[str] = "run_completed"

    instrument: str
    sample_count: int
    duration_s: float
    dynamic_range_db: float | None = None

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {
            "event": self.kind,
            "instrument": self.instrument,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "dynamic_range_db": self.dynamic_range_db,
        }


@dataclass(frozen=True, slots=True)
class RunFailed(Event):
    """A run raised. The exception is re-raised after this event is emitted, never swallowed."""

    kind: ClassVar[str] = "run_failed"

    instrument: str
    message: str
    error_type: str

    def payload(self) -> EventPayload:
        """Return the JSON-serializable form of this event."""
        return {
            "event": self.kind,
            "instrument": self.instrument,
            "message": self.message,
            "error_type": self.error_type,
        }


EventSink = Callable[[Event], None]
"""Where events go: a Rich progress bar (step 8), an SSE queue (step 9), or a list in tests."""


@dataclass(slots=True)
class StepHandle:
    """Live handle to the step currently running; advance it as samples complete."""

    reporter: ProgressReporter
    index: int
    label: str
    total_units: int
    completed_units: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def advance(self, units: int = 1) -> None:
        """Record `units` more finished samples and emit a `StepProgress`.

        Args:
            units: How many samples just finished (a chunk's worth in the parallel path,
                one at a time in the serial path).
        """
        self.completed_units = min(self.total_units, self.completed_units + units)
        self.reporter.emit(
            StepProgress(
                index=self.index,
                label=self.label,
                completed_units=self.completed_units,
                total_units=self.total_units,
                fraction=self.reporter.fraction_for(self._within()),
            )
        )

    def complete(self) -> None:
        """Mark the step finished, emitting `StepCompleted` and closing it out."""
        self.completed_units = self.total_units
        self.reporter.finish_step(self)

    def _within(self) -> float:
        """Fraction of this step's own work that is done, in ``[0, 1]``."""
        if self.total_units <= 0:
            return 1.0
        return self.completed_units / self.total_units


class ProgressReporter:
    """Numbers the steps of a run and emits events to a sink; the sink may be absent.

    The reporter owns the step counter, so callers never pass step indices in. That matters for
    `pipeline/preview.py`, which runs two chains back to back (before and after the barrier) and
    needs them numbered as one continuous run.
    """

    def __init__(self, sink: EventSink | None = None, *, total_steps: int = 1) -> None:
        """Initialize the reporter.

        Args:
            sink: Where to send events; ``None`` discards them (the default for a library
                caller that doesn't care about progress).
            total_steps: Steps the whole run will report, used for the overall fraction.
        """
        self._sink = sink
        self._total_steps = max(1, total_steps)
        self._next_index = 0
        self._completed_steps = 0
        self._last_fraction = 0.0

    @property
    def total_steps(self) -> int:
        """Number of steps this run will report."""
        return self._total_steps

    def emit(self, event: Event) -> None:
        """Send `event` to the sink, if there is one.

        Args:
            event: The event to emit.
        """
        if self._sink is not None:
            self._sink(event)

    def fraction_for(self, within_step: float) -> float:
        """Overall run progress, clamped to never fall below the highest value already emitted.

        Args:
            within_step: Fraction of the current step's work that is done.

        Returns:
            Overall progress in ``[0, 1]``.
        """
        raw = (self._completed_steps + within_step) / self._total_steps
        self._last_fraction = min(1.0, max(self._last_fraction, raw))
        return self._last_fraction

    def run_started(self, instrument: str) -> None:
        """Emit `RunStarted`.

        Args:
            instrument: Instrument name being processed.
        """
        self.emit(RunStarted(instrument=instrument, total_steps=self._total_steps))

    def begin_step(
        self,
        label: str,
        *,
        step_kind: StepKind,
        stage_ids: Sequence[str] = (),
        total_units: int = 1,
    ) -> StepHandle:
        """Emit `StepStarted` and return the handle to advance as its samples finish.

        Args:
            label: Short human-readable name, e.g. ``"dc -> trim"``.
            step_kind: Which kind of step this is.
            stage_ids: Stage ids this step covers, in execution order.
            total_units: Samples this step will process.

        Returns:
            The live `StepHandle`.
        """
        index = self._next_index
        self._next_index += 1
        self.emit(
            StepStarted(
                index=index,
                label=label,
                step_kind=step_kind,
                stage_ids=tuple(stage_ids),
                total_units=total_units,
            )
        )
        return StepHandle(reporter=self, index=index, label=label, total_units=total_units)

    def finish_step(self, handle: StepHandle) -> None:
        """Close out `handle`'s step and emit `StepCompleted`.

        Args:
            handle: The step being completed; call `StepHandle.complete` rather than this.
        """
        self._completed_steps = min(self._total_steps, self._completed_steps + 1)
        self.emit(
            StepCompleted(
                index=handle.index,
                label=handle.label,
                duration_s=time.perf_counter() - handle.started_at,
                fraction=self.fraction_for(0.0),
            )
        )

    def run_completed(
        self,
        instrument: str,
        *,
        sample_count: int,
        duration_s: float,
        dynamic_range_db: float | None,
    ) -> None:
        """Emit `RunCompleted`.

        Args:
            instrument: Instrument name.
            sample_count: Total samples processed (sustain + release).
            duration_s: Wall time for the whole run.
            dynamic_range_db: Velocity-mode dynamic range, or ``None``.
        """
        self.emit(
            RunCompleted(
                instrument=instrument,
                sample_count=sample_count,
                duration_s=duration_s,
                dynamic_range_db=dynamic_range_db,
            )
        )

    def run_failed(self, instrument: str, error: BaseException) -> None:
        """Emit `RunFailed`; the caller re-raises.

        Args:
            instrument: Instrument name.
            error: The exception that ended the run.
        """
        self.emit(
            RunFailed(
                instrument=instrument, message=str(error), error_type=type(error).__name__
            )
        )
