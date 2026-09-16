"""``autoluthier preview`` — render a small subset of an instrument, for a quick listen.

The CLI has no browser to drag a control in, so "preview" here means the closest useful
equivalent: run the configured chain over an evenly spread grid of samples and write them to
disk, on the exact same `pipeline.preview.run_preview` code path the browser (step 11) will use
for live playback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from autoluthier.config.toml_io import ProjectConfigError, load_project
from autoluthier.domain.notes import midi_to_note_name
from autoluthier.io.writer import write_audio
from autoluthier.pipeline.executor import PipelineError, load_instrument_audio
from autoluthier.pipeline.graph import ChainConfigError, build_chain
from autoluthier.pipeline.preview import (
    DEFAULT_PREVIEW_NOTES,
    DEFAULT_PREVIEW_VELOCITIES,
    PreviewMode,
    select_preview,
)
from autoluthier.pipeline.preview import run_preview as run_preview_chain

console = Console()
_VALID_MODES = ("exact", "subset")


def preview_command(
    folder: Annotated[Path, typer.Argument(help="Instrument folder containing project.toml.")],
    *,
    notes: Annotated[
        int, typer.Option("--notes", help="How many notes to preview.")
    ] = DEFAULT_PREVIEW_NOTES,
    velocities: Annotated[
        int, typer.Option("--velocities", help="How many velocity layers to preview.")
    ] = DEFAULT_PREVIEW_VELOCITIES,
    mode: Annotated[
        str,
        typer.Option(help=f"Preview mode: one of {_VALID_MODES}."),
    ] = "exact",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output dir for previewed samples; default: <folder>/preview."),
    ] = None,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
) -> None:
    """Process an evenly spread subset of samples and write them to disk to listen to."""
    if mode not in _VALID_MODES:
        console.print(f"[red]--mode must be one of {_VALID_MODES}, got {mode!r}[/]")
        raise typer.Exit(code=1)
    preview_mode: PreviewMode = "exact" if mode == "exact" else "subset"

    try:
        config = load_project(folder)
        chain = build_chain(config)
        audio = load_instrument_audio(folder, config)
    except (ProjectConfigError, ChainConfigError, PipelineError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    selection = select_preview(audio, note_count=notes, velocity_count=velocities)
    result = run_preview_chain(audio, chain, selection, mode=preview_mode, workers=workers)

    out_dir = out if out is not None else folder / "preview"
    extension = config.output.container
    written = 0
    for sample in result.audio.sustain:
        path = out_dir / f"{midi_to_note_name(sample.note)}_v{sample.velocity}.{extension}"
        write_audio(path, sample.audio, sample.sample_rate, config.output)
        written += 1
    for sample in result.audio.release:
        path = (
            out_dir / f"{midi_to_note_name(sample.note)}_v{sample.velocity}_rel.{extension}"
        )
        write_audio(path, sample.audio, sample.sample_rate, config.output)
        written += 1

    console.print(
        f"[green]Wrote {written} file(s)[/] to {out_dir} "
        f"(mode={result.mode}, {len(selection.notes)} note(s) x "
        f"{len(selection.velocities)} velocity layer(s))"
    )
