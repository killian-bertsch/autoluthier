"""``autoluthier midi`` — generate an autoluthier MIDI session and its project.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from autoluthier.helpers.midi_session import describe_session, generate_midi_session

console = Console()


def midi_command(
    *,
    velocity_layers: Annotated[
        int, typer.Option("--velocity-layers", "-X", help="Number of velocity layers to record.")
    ],
    semitone_interval: Annotated[
        int,
        typer.Option(
            "--semitone-interval", "-N", help="Semitone step between notes (1 = every key)."
        ),
    ],
    hold_time: Annotated[
        float, typer.Option("--hold-time", "-H", help="Seconds each note is held.")
    ],
    release_time: Annotated[
        float, typer.Option("--release-time", "-R", help="Silence after note-off.")
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output", "-o", help="Output stem: writes <output>.mid and <output>/project.toml."
        ),
    ],
    start_note: Annotated[int, typer.Option(help="Lowest MIDI note to record.")] = 21,
    end_note: Annotated[int, typer.Option(help="Highest MIDI note to record.")] = 108,
    out_dir: Annotated[
        Path | None, typer.Option(help="Directory for project.toml; default: ./<output>/")
    ] = None,
) -> None:
    """Write a MIDI autoluthier session plus a matching project.toml.

    Next steps after this: load the ``.mid`` file into a DAW/plugin, render it to
    ``sustain.wav``, save it next to ``project.toml``, then run ``autoluthier run``.
    """
    try:
        result = generate_midi_session(
            velocity_layers,
            semitone_interval,
            hold_time,
            release_time,
            output,
            start_note=start_note,
            end_note=end_note,
            out_dir=out_dir,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    console.print(describe_session(result))
    console.print(
        f"\nNext: render {result.midi_path} to "
        f"{result.project_path.parent / 'sustain.wav'}, then run 'autoluthier run "
        f"{result.project_path.parent}'."
    )
