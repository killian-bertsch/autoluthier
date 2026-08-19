"""``autosampler validate`` — check a project.toml without processing any audio."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from autosampler.config.toml_io import ProjectConfigError, load_project
from autosampler.pipeline.graph import ChainConfigError, build_chain

console = Console()


def validate_command(
    folder: Annotated[Path, typer.Argument(help="Instrument folder containing project.toml.")],
) -> None:
    """Load project.toml and validate its stage chain, without reading any audio.

    Catches exactly what a real run would fail on before it reads a single file: a malformed
    project.toml (unknown key, wrong type, a failed cross-field check) or an invalid/duplicated
    stage chain entry.
    """
    try:
        config = load_project(folder)
    except ProjectConfigError as exc:
        console.print(f"[red]Invalid project:[/] {exc}")
        raise typer.Exit(code=1) from None

    try:
        chain = build_chain(config)
    except ChainConfigError as exc:
        console.print(f"[red]Invalid stage chain:[/] {exc}")
        raise typer.Exit(code=1) from None

    name = config.instrument_name or folder.name
    console.print(f"[green]OK[/] — '{name}': {len(chain)} stage(s) in the chain.")
