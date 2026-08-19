"""Loads an instrument and runs its chain: parallel across samples, serial at barriers.

Two kinds of concurrency, matched to two kinds of work:

- **Threads for IO.** ``sustain`` and ``release`` are read concurrently, because `soundfile`
  releases the GIL inside libsndfile, so two decodes genuinely overlap.
- **Processes for DSP.** NumPy only partly releases the GIL — plenty of what these stages do
  (indexing, small-array arithmetic, the per-sample Python around each vectorized call) holds
  it — so threads would serialize the expensive part. `ProcessPoolExecutor` sidesteps that at
  the cost of pickling audio across the boundary, which is why the unit of work is a *chunk of
  samples through a whole segment* rather than one stage over one sample: the audio crosses the
  boundary twice per segment, not twice per stage.

Determinism is a hard requirement, not a nice-to-have: the parallel result must be bit-identical
to the serial one. It is, because every per-sample step is a pure function of that sample's own
audio and params, chunking never changes the operations applied to a sample, and results are
written back by index so sample order is preserved. Barriers run serially in the parent process,
where group statistics are computed over the samples in their original order.

Workers write back through `SampleUpdate` rather than returning `Sample` objects: `SampleSet`
holds a ``(note, velocity)`` index of `Sample` *identities*, so replacing a list entry with a
new object would leave that index pointing at a stale one. Updates are applied onto the
existing objects instead.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from autosampler.config.schema import ProjectConfig
from autosampler.domain.models import InstrumentAudio, Sample, SampleSet
from autosampler.domain.notes import recorded_notes, velocity_list
from autosampler.dsp.normalize import apply_normalize
from autosampler.io.reader import find_audio_file, read_audio
from autosampler.io.slicer import slice_notes_velocities
from autosampler.io.writer import resample
from autosampler.pipeline.events import EventSink, ProgressReporter, StepHandle
from autosampler.pipeline.graph import (
    BarrierSegment,
    ChainStep,
    PerSampleStep,
    PreparedSteps,
    SampleSegment,
    Segment,
    build_chain,
    partition_chain,
    targets_release,
    targets_sustain,
)

_CHUNKS_PER_WORKER = 4
"""Chunks to aim for per worker: enough to even out uneven per-sample cost, few enough that
each task still amortizes its pickling overhead over many samples."""

_MIN_CHUNKS = 16
"""Floor on the chunk count regardless of worker count, so a serial run still reports progress
more than once per segment. A chunk is one progress event, so this is also the reporting
granularity."""


class PipelineError(RuntimeError):
    """Raised when an instrument cannot be loaded or its run cannot proceed."""


@dataclass(frozen=True, slots=True)
class SampleUpdate:
    """What a per-sample worker is allowed to change on a `Sample`.

    This is the parallel write-back contract. If per-sample DSP ever gains another mutable
    output on `Sample`, it has to be added here too, or that output will silently be dropped on
    the parallel path while working fine on the serial one.
    """

    index: int
    audio: NDArray[np.float32]
    loop_start: int | None
    loop_end: int | None
    loop_crossfade_frames: int


@dataclass(frozen=True, slots=True)
class ChainResult:
    """Outcome of running a chain: the one value a barrier stage produces.

    `dynamic_range_db` is set only by ``normalize`` in ``velocity`` mode, and becomes the SFZ
    ``amp_velcurve_1`` opcode in export (step 7).
    """

    dynamic_range_db: float | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything a full run produces, ready for export (step 7)."""

    instrument_name: str
    audio: InstrumentAudio
    dynamic_range_db: float | None
    duration_s: float


def _apply_update(sample: Sample, update: SampleUpdate) -> None:
    """Copy a worker's results onto the parent's `Sample`, preserving object identity."""
    sample.audio = update.audio
    sample.loop_start = update.loop_start
    sample.loop_end = update.loop_end
    sample.loop_crossfade_frames = update.loop_crossfade_frames


def _process_chunk(
    steps: tuple[PerSampleStep, ...], items: list[tuple[int, Sample]]
) -> list[SampleUpdate]:
    """Apply `steps` to every sample in `items` and return the write-back updates.

    Module-level and picklable so it can be the `ProcessPoolExecutor` task. It runs in the
    parent process too, on the serial path, so both paths share one code path.

    Args:
        steps: The per-sample steps of one segment, in execution order.
        items: ``(index, sample)`` pairs; the index is into the owning `SampleSet.samples`.

    Returns:
        One `SampleUpdate` per item, in the same order.
    """
    prepared = PreparedSteps(steps)
    updates: list[SampleUpdate] = []
    for index, sample in items:
        prepared.apply(sample)
        updates.append(
            SampleUpdate(
                index=index,
                audio=sample.audio,
                loop_start=sample.loop_start,
                loop_end=sample.loop_end,
                loop_crossfade_frames=sample.loop_crossfade_frames,
            )
        )
    return updates


def resolve_workers(workers: int | None, sample_count: int) -> int:
    """Decide how many worker processes to use.

    Args:
        workers: Requested worker count; ``None`` means one per CPU.
        sample_count: Samples about to be processed — more workers than samples is pointless.

    Returns:
        A worker count of at least 1; 1 means "run in this process, no pool".
    """
    requested = workers if workers is not None else (os.cpu_count() or 1)
    return max(1, min(requested, max(1, sample_count)))


def _chunk_size(sample_count: int, workers: int) -> int:
    """Samples per task: enough chunks to balance the load, few enough to amortize pickling."""
    target_chunks = max(workers * _CHUNKS_PER_WORKER, _MIN_CHUNKS)
    return max(1, -(-sample_count // target_chunks))


def _chunks(samples: list[Sample], size: int) -> list[list[tuple[int, Sample]]]:
    """Split `samples` into ``(index, sample)`` chunks of at most `size` entries."""
    indexed = list(enumerate(samples))
    return [indexed[start : start + size] for start in range(0, len(indexed), size)]


def _run_group(
    steps: tuple[PerSampleStep, ...],
    samples: list[Sample],
    *,
    pool: Executor | None,
    workers: int,
    handle: StepHandle,
) -> None:
    """Apply `steps` to `samples`, in this process or across `pool`, reporting progress."""
    if not steps or not samples:
        return
    chunks = _chunks(samples, _chunk_size(len(samples), workers))
    if pool is None:
        # Same chunks, same `_process_chunk`, in this process: one code path for both, which
        # is part of why the parallel result is bit-identical to the serial one.
        for chunk in chunks:
            _process_chunk(steps, chunk)
            handle.advance(len(chunk))
        return

    futures = {pool.submit(_process_chunk, steps, chunk): len(chunk) for chunk in chunks}
    for future in as_completed(futures):
        for update in future.result():
            _apply_update(samples[update.index], update)
        handle.advance(futures[future])


def _run_sample_segment(
    segment: SampleSegment,
    audio: InstrumentAudio,
    *,
    pool: Executor | None,
    workers: int,
    reporter: ProgressReporter,
) -> None:
    """Run one parallel segment over whichever sample sets its steps target."""
    sustain_steps = tuple(step for step in segment.steps if targets_sustain(step))
    release_steps = tuple(step for step in segment.steps if targets_release(step))
    total_units = len(audio.sustain) * bool(sustain_steps) + len(audio.release) * bool(
        release_steps
    )
    handle = reporter.begin_step(
        segment.label,
        step_kind="stages",
        stage_ids=segment.stage_ids,
        total_units=total_units,
    )
    _run_group(
        sustain_steps, audio.sustain.samples, pool=pool, workers=workers, handle=handle
    )
    _run_group(
        release_steps, audio.release.samples, pool=pool, workers=workers, handle=handle
    )
    handle.complete()


def _run_barrier_segment(
    segment: BarrierSegment, audio: InstrumentAudio, *, reporter: ProgressReporter
) -> float | None:
    """Run one barrier segment in this process, over the whole sample set."""
    total_units = len(audio.sustain) + len(audio.release)
    handle = reporter.begin_step(
        segment.label,
        step_kind="barrier",
        stage_ids=segment.stage_ids,
        total_units=total_units,
    )
    result = apply_normalize(audio.sustain, audio.release, segment.step.params)
    handle.complete()
    return None if result is None else result.dynamic_range_db


def run_chain(
    audio: InstrumentAudio,
    chain: Sequence[ChainStep],
    *,
    workers: int | None = None,
    reporter: ProgressReporter | None = None,
) -> ChainResult:
    """Run `chain` over `audio` in place, parallelizing per-sample segments.

    Args:
        audio: The loaded instrument; every sample is mutated in place.
        chain: The chain from `pipeline.graph.build_chain`.
        workers: Worker processes for per-sample work; ``None`` means one per CPU, ``1`` runs
            everything in this process with no pool (the right choice for small sets, where
            pool startup would dominate).
        reporter: Progress reporter; ``None`` discards progress. Pass one in to number a
            preview's two chain runs as a single continuous run.

    Returns:
        The `ChainResult` for this run.
    """
    segments = partition_chain(chain)
    active_reporter = (
        reporter if reporter is not None else ProgressReporter(total_steps=len(segments))
    )
    sample_count = len(audio.sustain) + len(audio.release)
    worker_count = resolve_workers(workers, sample_count)
    needs_pool = worker_count > 1 and any(isinstance(seg, SampleSegment) for seg in segments)

    dynamic_range_db: float | None = None
    pool: ProcessPoolExecutor | None = None
    try:
        if needs_pool:
            pool = ProcessPoolExecutor(max_workers=worker_count)
        for segment in segments:
            dynamic_range_db = _run_segment(
                segment,
                audio,
                pool=pool,
                workers=worker_count,
                reporter=active_reporter,
                dynamic_range_db=dynamic_range_db,
            )
    finally:
        if pool is not None:
            # cancel_futures matters only on the failure path: a run that has already raised
            # should not block while the rest of the batch renders.
            pool.shutdown(cancel_futures=True)
    return ChainResult(dynamic_range_db=dynamic_range_db)


def _run_segment(
    segment: Segment,
    audio: InstrumentAudio,
    *,
    pool: Executor | None,
    workers: int,
    reporter: ProgressReporter,
    dynamic_range_db: float | None,
) -> float | None:
    """Run one segment, returning the dynamic range known so far."""
    if isinstance(segment, BarrierSegment):
        measured = _run_barrier_segment(segment, audio, reporter=reporter)
        return dynamic_range_db if measured is None else measured
    _run_sample_segment(segment, audio, pool=pool, workers=workers, reporter=reporter)
    return dynamic_range_db


def _resample_set(samples: SampleSet, target_rate: int) -> None:
    """Resample every sample in `samples` to `target_rate`, in place."""
    for sample in samples:
        if sample.sample_rate != target_rate:
            sample.audio = resample(sample.audio, sample.sample_rate, target_rate)
            sample.sample_rate = target_rate


def load_instrument_audio(
    folder: Path, config: ProjectConfig, *, reporter: ProgressReporter | None = None
) -> InstrumentAudio:
    """Read and slice one instrument folder into per-``(note, velocity)`` samples.

    ``sustain.wav``/``.flac`` is required; ``release`` is optional and yields an empty release
    set when absent. Both files are read concurrently on threads.

    Resampling to ``output.sample_rate`` happens **here**, before any DSP, so the whole chain —
    loop detection included — runs at the rate that will actually be written. Resampling at
    write time instead would leave loop points expressed in source-rate frames, and rescaling
    them afterwards would land them off the zero crossings detection chose them for, putting a
    step back into the seam the crossfade exists to remove.

    Args:
        folder: The instrument folder holding the source render and ``project.toml``.
        config: The project configuration; its ``recording`` section must match the render.
        reporter: Progress reporter; ``None`` discards progress.

    Returns:
        The sliced, un-processed `InstrumentAudio`.

    Raises:
        PipelineError: if no sustain render exists, or if the release render's sample rate
            differs from the sustain render's.
    """
    active_reporter = reporter if reporter is not None else ProgressReporter()
    sustain_path = find_audio_file(folder, "sustain")
    if sustain_path is None:
        raise PipelineError(
            f"no sustain render in {folder}: expected sustain.wav or sustain.flac"
        )
    release_path = find_audio_file(folder, "release")

    handle = active_reporter.begin_step(
        "load", step_kind="load", total_units=1 if release_path is None else 2
    )
    with ThreadPoolExecutor(max_workers=2) as threads:
        sustain_future = threads.submit(read_audio, sustain_path)
        release_future = (
            threads.submit(read_audio, release_path) if release_path is not None else None
        )
        sustain_audio, sustain_rate = sustain_future.result()
        handle.advance()
        release_audio, release_rate = (
            release_future.result() if release_future is not None else (None, sustain_rate)
        )
        if release_future is not None:
            handle.advance()

    if release_audio is not None and release_rate != sustain_rate:
        raise PipelineError(
            f"sample rate mismatch: {sustain_path.name} is {sustain_rate} Hz but "
            f"{release_path.name if release_path else 'release'} is {release_rate} Hz — "
            "both renders must come from the same session"
        )

    recording = config.recording
    notes = recorded_notes(
        recording.start_note, recording.end_note, recording.semitone_interval
    )
    velocities = velocity_list(recording.velocity_layers)
    sustain_set = slice_notes_velocities(
        sustain_audio,
        sustain_rate,
        notes,
        velocities,
        hold_time=recording.hold_time,
        release_time=recording.release_time,
        extract_release=False,
        min_note=config.selection.min_note,
        max_note=config.selection.max_note,
        collapse_to_mono=config.output.collapse_to_mono,
    )
    if release_audio is None:
        release_set = SampleSet([])
    else:
        release_set = slice_notes_velocities(
            release_audio,
            release_rate,
            notes,
            velocities,
            hold_time=(
                recording.hold_time
                if recording.release_hold_time is None
                else recording.release_hold_time
            ),
            release_time=(
                recording.release_time
                if recording.release_release_time is None
                else recording.release_release_time
            ),
            extract_release=True,
            min_note=config.selection.min_note,
            max_note=config.selection.max_note,
            collapse_to_mono=config.output.collapse_to_mono,
        )

    if config.output.sample_rate is not None:
        _resample_set(sustain_set, config.output.sample_rate)
        _resample_set(release_set, config.output.sample_rate)

    handle.complete()
    return InstrumentAudio(sustain=sustain_set, release=release_set)


def run_instrument(
    folder: Path,
    config: ProjectConfig,
    *,
    workers: int | None = None,
    sink: EventSink | None = None,
) -> RunResult:
    """Load one instrument folder and run its whole chain, reporting progress to `sink`.

    The chain is built (and its params validated) before any audio is read, so a malformed
    project fails immediately rather than after a long load.

    ``instrument_name`` falls back to the folder name but is otherwise honored — V1 parsed the
    field and then unconditionally overwrote it with the folder name (bug 11).

    Args:
        folder: The instrument folder.
        config: The project configuration.
        workers: Worker processes for per-sample work; ``None`` means one per CPU.
        sink: Where progress events go; ``None`` discards them.

    Returns:
        The `RunResult`, ready for export.

    Raises:
        ChainConfigError: if the stage chain is invalid.
        PipelineError: if the instrument cannot be loaded.
    """
    chain = build_chain(config)
    reporter = ProgressReporter(sink, total_steps=1 + len(partition_chain(chain)))
    name = config.instrument_name or folder.name
    reporter.run_started(name)
    started = perf_counter()
    try:
        audio = load_instrument_audio(folder, config, reporter=reporter)
        result = run_chain(audio, chain, workers=workers, reporter=reporter)
    except Exception as error:
        reporter.run_failed(name, error)
        raise
    duration = perf_counter() - started
    reporter.run_completed(
        name,
        sample_count=len(audio.sustain) + len(audio.release),
        duration_s=duration,
        dynamic_range_db=result.dynamic_range_db,
    )
    return RunResult(
        instrument_name=name,
        audio=audio,
        dynamic_range_db=result.dynamic_range_db,
        duration_s=duration,
    )
