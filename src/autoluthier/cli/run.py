"""``autoluthier run`` — load, process, and export one or more instrument folders.

Spans **one** `ProgressReporter` across load, the stage chain, *and* export — unlike
`pipeline.executor.run_instrument`, which stops reporting once its chain finishes (that
function's own contract: it hands back a `RunResult` for a caller to export separately, and
`pipeline.executor` cannot import `export.writer` without a circular import, since
`export.writer` already imports `pipeline.executor` for `RunResult`). `_run_and_export` below
is where that composition belongs instead: the CLI layer, which is explicitly the place
`pipeline` and `export` get composed together.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from rich.console import Console

from autoluthier.cli._progress import RichProgressSink
from autoluthier.config.schema import ProjectConfig
from autoluthier.config.toml_io import ProjectConfigError, load_project
from autoluthier.config.workspace import discover_projects
from autoluthier.export.writer import ExportError, ExportResult, export_instrument
from autoluthier.pipeline.events import EventSink, ProgressReporter
from autoluthier.pipeline.executor import (
    PipelineError,
    RunResult,
    load_instrument_audio,
    run_chain,
)
from autoluthier.pipeline.graph import ChainConfigError, build_chain, partition_chain

console = Console()


def _run_and_export(
    folder: Path,
    config: ProjectConfig,
    output_dir: Path,
    *,
    workers: int | None,
    sink: EventSink | None,
) -> ExportResult:
    """Load, process, and export one instrument, reporting progress as a single run.

    Args:
        folder: The instrument folder holding the source render and ``project.toml``.
        config: The already-loaded project configuration.
        output_dir: Directory the instrument's output folder is created inside.
        workers: Worker processes for per-sample work; `None` means one per CPU.
        sink: Where progress events go; `None` discards them.

    Returns:
        The `ExportResult` naming what was written.

    Raises:
        ChainConfigError: if the stage chain is invalid.
        PipelineError: if the instrument cannot be loaded.
        ExportError: if selection leaves no sustain samples.
    """
    chain = build_chain(config)
    reporter = ProgressReporter(sink, total_steps=1 + len(partition_chain(chain)) + 1)
    name = config.instrument_name or folder.name
    reporter.run_started(name)
    started = perf_counter()
    try:
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
    except Exception as error:
        reporter.run_failed(name, error)
        raise
    reporter.run_completed(
        name,
        sample_count=len(audio.sustain) + len(audio.release),
        duration_s=perf_counter() - started,
        dynamic_range_db=chain_result.dynamic_range_db,
    )
    return export_result


def _process_one(folder: Path, output_dir: Path, *, workers: int | None) -> bool:
    """Run and export one instrument folder, printing a summary or a warning.

    Errors never propagate out of this function — per-instrument failures are reported and
    skipped, matching V1's ``main.py`` behavior across a batch of instruments.

    Args:
        folder: The instrument folder to process.
        output_dir: Directory the instrument's output folder is created inside.
        workers: Worker processes for per-sample work; `None` means one per CPU.

    Returns:
        `True` if the instrument was processed successfully.
    """
    try:
        config = load_project(folder)
    except ProjectConfigError as exc:
        console.print(f"[yellow]Skipping '{folder.name}'[/] — {exc}")
        return False

    try:
        with RichProgressSink(console=console) as sink:
            result = _run_and_export(folder, config, output_dir, workers=workers, sink=sink)
    except (ChainConfigError, PipelineError, ExportError) as exc:
        console.print(f"[yellow]Skipping '{folder.name}'[/] — {exc}")
        return False
    except Exception:
        console.print(f"[yellow]Skipping '{folder.name}'[/] — unexpected error")
        console.print_exception()
        return False

    console.print(
        f"[green]{result.instrument_name}[/]: {result.audio_files} file(s) written "
        f"({result.sustain_regions} sustain region(s), {result.release_regions} release "
        f"region(s)) -> {result.output_dir}"
    )
    return True


def run_command(
    target: Annotated[
        Path, typer.Argument(help="Instrument folder, or (with --scan) its parent directory.")
    ],
    *,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory processed output is written to.")
    ] = Path("output"),
    instrument: Annotated[
        str | None,
        typer.Option("--instrument", help="With --scan, process only this subfolder."),
    ] = None,
    scan: Annotated[
        bool, typer.Option("--scan", help="Treat target as a parent of instrument folders.")
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option("--workers", help="Worker processes for per-sample DSP; default one per CPU."),
    ] = None,
) -> None:
    """Process one instrument, or (with --scan) every instrument folder under target.

    Exits non-zero only if every instrument attempted failed, matching V1: a partial batch
    failure is reported per instrument but does not fail the whole run.
    """
    if instrument is not None and not scan:
        console.print("[red]--instrument requires --scan[/]")
        raise typer.Exit(code=1)

    if scan:
        folders = [target / instrument] if instrument is not None else discover_projects(target)
        if instrument is not None and not folders[0].is_dir():
            console.print(f"[red]Instrument folder not found: {folders[0]}[/]")
            raise typer.Exit(code=1)
    else:
        folders = [target]

    if not folders:
        console.print(f"[red]No project.toml found under {target}[/]")
        raise typer.Exit(code=1)

    console.print(f"Found {len(folders)} instrument(s) to process.")
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = sum(_process_one(folder, output_dir, workers=workers) for folder in folders)
    console.print(f"Done. {successes}/{len(folders)} instrument(s) processed successfully.")
    if successes == 0:
        raise typer.Exit(code=1)
