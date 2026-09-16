"""``autoluthier concat`` — concatenate numbered layer recordings into per-instrument renders."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from autoluthier.helpers.concat_layers import (
    DEFAULT_LAYER_NUMBERS,
    TARGET_DB_DEFAULT,
    concat_layers,
)

console = Console()
_DEFAULT_LAYERS_OPTION = ",".join(str(n) for n in DEFAULT_LAYER_NUMBERS)


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def concat_command(
    input_dir: Annotated[Path, typer.Argument(help="Directory holding the numbered layer files.")],
    output_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory instrument subfolders go in; default: input_dir."),
    ] = None,
    *,
    stereo: Annotated[
        list[str] | None,
        typer.Option("--stereo", help="Instrument name to keep stereo; repeatable."),
    ] = None,
    target_db: Annotated[
        float, typer.Option("--target-db", help="Peak level in dBFS to normalize to.")
    ] = TARGET_DB_DEFAULT,
    layers: Annotated[
        str, typer.Option("--layers", help="Comma-separated sustain layer numbers, in order.")
    ] = _DEFAULT_LAYERS_OPTION,
) -> None:
    """Concatenate and peak-normalize every instrument's numbered layers under input_dir.

    Input files: ``{n} <Name>.flac`` (or ``.wav``) for sustain layer n, ``R <Name>.flac`` for
    the release tail. Output: ``<output_dir>/<Name>/sustain.flac`` and ``release.flac``.
    """
    reports = concat_layers(
        input_dir,
        output_dir if output_dir is not None else input_dir,
        stereo_instruments=frozenset(stereo or ()),
        target_db=target_db,
        layer_numbers=_parse_layers(layers),
    )
    if not reports:
        console.print(f"[yellow]No instruments found under {input_dir}[/]")
        raise typer.Exit(code=1)

    for report in reports:
        if report.error is not None:
            console.print(f"  [red]{report.instrument}[/]: {report.error}")
            continue
        channels = "stereo" if report.stereo else "mono"
        console.print(
            f"  [green]{report.instrument}[/] ({channels}, {report.layers_found} layer(s)"
            f"{f', missing {report.layers_missing}' if report.layers_missing else ''}): "
            f"{report.sustain_path}"
            + (f", {report.release_path}" if report.release_path else " (no release)")
        )

    succeeded = sum(report.error is None for report in reports)
    console.print(f"Done. {succeeded}/{len(reports)} instrument(s) converted.")
    if succeeded == 0:
        raise typer.Exit(code=1)
