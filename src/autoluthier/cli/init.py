"""``autoluthier init`` — scaffold a project.toml for a folder that already has a render."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from autoluthier.config.schema import ProjectConfig, RecordingConfig, SelectionConfig
from autoluthier.config.toml_io import PROJECT_FILENAME, save_project

console = Console()


def init_command(
    folder: Annotated[Path, typer.Argument(help="Instrument folder to scaffold.")],
    *,
    velocity_layers: Annotated[
        int, typer.Option("--velocity-layers", "-X", help="Velocity layers recorded.")
    ],
    semitone_interval: Annotated[
        int, typer.Option("--semitone-interval", "-N", help="Semitone step between notes.")
    ],
    hold_time: Annotated[
        float, typer.Option("--hold-time", "-H", help="Seconds each note is held.")
    ],
    release_time: Annotated[
        float, typer.Option("--release-time", "-R", help="Silence after note-off.")
    ],
    start_note: Annotated[int, typer.Option(help="Lowest recorded MIDI note.")] = 21,
    end_note: Annotated[int, typer.Option(help="Highest recorded MIDI note.")] = 108,
    instrument_name: Annotated[
        str | None, typer.Option("--name", help="Overrides the instrument name in output.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing project.toml.")
    ] = False,
) -> None:
    """Write a project.toml describing an existing render's recording settings.

    Use this when ``sustain.wav``/``.flac`` already exists (from `autoluthier concat` or an
    external recording) and only needs a project.toml to be processed. For a brand-new session
    that also needs a MIDI file, use ``autoluthier midi`` instead.
    """
    project_path = folder / PROJECT_FILENAME
    if project_path.exists() and not force:
        console.print(f"[red]{project_path} already exists[/] — pass --force to overwrite")
        raise typer.Exit(code=1)

    try:
        config = ProjectConfig(
            instrument_name=instrument_name,
            recording=RecordingConfig(
                velocity_layers=velocity_layers,
                semitone_interval=semitone_interval,
                hold_time=hold_time,
                release_time=release_time,
                start_note=start_note,
                end_note=end_note,
            ),
            selection=SelectionConfig(
                min_note=start_note, max_note=end_note, velocity_layers_out=velocity_layers
            ),
        )
    except ValidationError as exc:
        console.print(f"[red]Invalid settings:[/] {exc}")
        raise typer.Exit(code=1) from None

    written = save_project(config, folder)
    console.print(f"[green]Wrote {written}[/]")
