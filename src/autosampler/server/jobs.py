"""Runs one instrument's full chain plus export as a background job, tracked by id.

Mirrors `cli.run._run_and_export`: `pipeline.executor` cannot import `export.writer` (the
latter already imports the former for `RunResult`), so the load+chain+export composition has to
live at whichever thin-client layer is doing the composing. The CLI has its own copy of this
~20-line composition for exactly that reason; this is the server's.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from time import perf_counter

from autosampler.config.schema import ProjectConfig
from autosampler.export.writer import ExportResult, export_instrument
from autosampler.pipeline.events import Event, ProgressReporter
from autosampler.pipeline.executor import RunResult, load_instrument_audio, run_chain
from autosampler.pipeline.graph import build_chain, partition_chain
from autosampler.server.sse import EventBroadcaster
from autosampler.server.state import JobRecord


def new_job_id() -> str:
    """Return a fresh, opaque job id.

    Returns:
        A UUID4 hex string.
    """
    return uuid.uuid4().hex


def _run_and_export(
    folder: Path,
    config: ProjectConfig,
    output_dir: Path,
    job: JobRecord,
    broadcaster: EventBroadcaster,
    *,
    workers: int | None,
) -> ExportResult:
    """Load, process, and export one instrument, recording and broadcasting every event."""

    def sink(event: Event) -> None:
        job.events.append(event.payload())
        broadcaster.publish(job.id, event)

    chain = build_chain(config)
    reporter = ProgressReporter(sink, total_steps=1 + len(partition_chain(chain)) + 1)
    name = config.instrument_name or folder.name
    reporter.run_started(name)
    started = perf_counter()
    audio = load_instrument_audio(folder, config, reporter=reporter)
    chain_result = run_chain(audio, chain, workers=workers, reporter=reporter)
    run_result = RunResult(
        instrument_name=name,
        audio=audio,
        dynamic_range_db=chain_result.dynamic_range_db,
        duration_s=perf_counter() - started,
    )
    handle = reporter.begin_step("export", step_kind="export", total_units=1)
    export_result = export_instrument(run_result, config, output_dir)
    handle.complete()
    reporter.run_completed(
        name,
        sample_count=len(audio.sustain) + len(audio.release),
        duration_s=perf_counter() - started,
        dynamic_range_db=chain_result.dynamic_range_db,
    )
    return export_result


def run_job(
    broadcaster: EventBroadcaster,
    job: JobRecord,
    folder: Path,
    config: ProjectConfig,
    output_dir: Path,
    *,
    workers: int | None = None,
) -> None:
    """Run one full instrument job to completion, updating `job` in place.

    Runs synchronously — call this from a background thread (see `routes/jobs.py`), never
    directly from an async route handler, since it blocks on CPU-bound DSP for as long as a
    full render takes.

    Args:
        broadcaster: Where progress events are published.
        job: The job record to update; the caller has already registered it in server state.
        folder: The instrument folder.
        config: The project configuration to run.
        output_dir: Directory the instrument's output folder is created inside.
        workers: Worker processes for per-sample work; ``None`` means one per CPU.
    """
    try:
        result = _run_and_export(folder, config, output_dir, job, broadcaster, workers=workers)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        return
    job.status = "completed"
    job.output_dir = str(result.output_dir)
