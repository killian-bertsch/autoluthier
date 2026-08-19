"""Runs the chain over a small subset of samples, on the same code path as a full render.

A preview exists so a parameter can be dragged and *heard* before committing to a full render,
which only means anything if what you hear is what the render will produce. That is easy for
per-sample stages and genuinely hard for one thing: ``normalize`` is a barrier. Its gain comes
from a group statistic — the mean level of every note in a velocity layer (lufs/rms mode) or
every layer of a note (velocity mode) — so a sample's final level is not a property of that
sample alone. Measure four samples in isolation and you get four different gains than the render
will apply.

Hence two modes, and the difference between them is exactly that:

- ``"exact"`` (default) processes the **whole** set through everything up to and including the
  last barrier, then only the previewed samples through the per-sample stages after it. The
  previewed samples come out bit-identical to a full run. The cost is one full pass over the
  pre-barrier stages — normally the cheap half of the chain, since loop detection sits after
  normalize in the default order.
- ``"subset"`` processes only the previewed samples through the entire chain. It is as cheap as
  a preview can be, and its barrier statistics are computed over the subset, so levels are
  approximate. Useful while dragging; not something to judge final level by.

With no barrier in the chain the two are the same thing, and `run_preview` takes the cheap path.

Preview never mutates the caller's `InstrumentAudio`: it works on copies, so the server (step 9)
can hold one loaded instrument in memory and preview against it repeatedly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from autosampler.domain.models import InstrumentAudio, SampleSet
from autosampler.domain.notes import select_evenly, select_velocities
from autosampler.pipeline.events import EventSink, ProgressReporter
from autosampler.pipeline.executor import run_chain
from autosampler.pipeline.graph import ChainStep, partition_chain, split_at_last_barrier

PreviewMode = Literal["exact", "subset"]
"""``"exact"`` reproduces a full run bit-for-bit; ``"subset"`` is cheaper and approximates it."""

DEFAULT_PREVIEW_NOTES = 2
DEFAULT_PREVIEW_VELOCITIES = 2


@dataclass(frozen=True, slots=True)
class PreviewSelection:
    """The ``notes x velocities`` grid of samples to preview."""

    notes: tuple[int, ...]
    velocities: tuple[int, ...]

    def contains(self, note: int, velocity: int) -> bool:
        """Whether this selection includes a given sample.

        Args:
            note: MIDI note.
            velocity: MIDI velocity.

        Returns:
            ``True`` if the sample is in the previewed grid.
        """
        return note in self.notes and velocity in self.velocities


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """The processed previewed samples, plus the barrier value that shaped them."""

    audio: InstrumentAudio
    dynamic_range_db: float | None
    mode: PreviewMode


def select_preview(
    audio: InstrumentAudio,
    *,
    note_count: int = DEFAULT_PREVIEW_NOTES,
    velocity_count: int = DEFAULT_PREVIEW_VELOCITIES,
) -> PreviewSelection:
    """Pick an evenly spread grid of samples to preview.

    Uses the same selectors as output thinning (`domain.notes`), so a preview of one velocity
    layer gives the loudest one and a preview of one note gives the middle of the keyboard —
    V1's tie-breaks, in one implementation.

    Args:
        audio: The loaded instrument.
        note_count: How many notes to preview.
        velocity_count: How many velocity layers to preview.

    Returns:
        The selection.
    """
    notes = select_evenly(audio.sustain.notes(), note_count)
    velocities = select_velocities(audio.sustain.velocities(), velocity_count)
    return PreviewSelection(notes=tuple(notes), velocities=tuple(velocities))


def _copy_all(audio: InstrumentAudio) -> InstrumentAudio:
    """Deep-copy a whole instrument, so processing it can't touch the caller's samples."""
    return InstrumentAudio(
        sustain=SampleSet([sample.copy() for sample in audio.sustain]),
        release=SampleSet([sample.copy() for sample in audio.release]),
    )


def _copy_subset(audio: InstrumentAudio, selection: PreviewSelection) -> InstrumentAudio:
    """Deep-copy only the selected samples."""
    return InstrumentAudio(
        sustain=SampleSet(
            [s.copy() for s in audio.sustain if selection.contains(s.note, s.velocity)]
        ),
        release=SampleSet(
            [s.copy() for s in audio.release if selection.contains(s.note, s.velocity)]
        ),
    )


def _restrict(audio: InstrumentAudio, selection: PreviewSelection) -> InstrumentAudio:
    """View the selected samples of an already-copied instrument, without copying again."""
    return InstrumentAudio(
        sustain=SampleSet(
            [s for s in audio.sustain if selection.contains(s.note, s.velocity)]
        ),
        release=SampleSet(
            [s for s in audio.release if selection.contains(s.note, s.velocity)]
        ),
    )


def run_preview(
    audio: InstrumentAudio,
    chain: Sequence[ChainStep],
    selection: PreviewSelection,
    *,
    mode: PreviewMode = "exact",
    workers: int | None = 1,
    sink: EventSink | None = None,
) -> PreviewResult:
    """Process the selected samples through `chain` and return them.

    Args:
        audio: The loaded instrument; never mutated.
        chain: The chain from `pipeline.graph.build_chain`.
        selection: Which samples to preview.
        mode: ``"exact"`` (bit-identical to a full run for these samples) or ``"subset"``
            (cheaper, barrier statistics computed over the subset only).
        workers: Worker processes; defaults to 1, since a preview subset is usually far too
            small to repay pool startup. ``"exact"`` mode's full pre-barrier pass is the case
            where raising this is worth it.
        sink: Where progress events go; ``None`` discards them.

    Returns:
        The `PreviewResult` holding the processed subset.
    """
    head, tail = split_at_last_barrier(chain)
    if mode == "subset" or not head:
        # No barrier means nothing is cross-sample, so the subset path *is* exact here.
        work = _copy_subset(audio, selection)
        reporter = ProgressReporter(sink, total_steps=len(partition_chain(chain)))
        result = run_chain(work, chain, workers=workers, reporter=reporter)
        return PreviewResult(
            audio=work,
            dynamic_range_db=result.dynamic_range_db,
            mode="exact" if not head else "subset",
        )

    total_steps = len(partition_chain(head)) + len(partition_chain(tail))
    reporter = ProgressReporter(sink, total_steps=total_steps)
    full = _copy_all(audio)
    head_result = run_chain(full, head, workers=workers, reporter=reporter)
    subset = _restrict(full, selection)
    run_chain(subset, tail, workers=workers, reporter=reporter)
    return PreviewResult(
        audio=subset, dynamic_range_db=head_result.dynamic_range_db, mode="exact"
    )
