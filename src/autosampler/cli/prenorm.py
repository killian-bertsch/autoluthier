"""``autosampler prenorm`` — peak-normalize raw source renders before slicing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from autosampler.helpers.prenormalize import TARGET_DB_DEFAULT, prenormalize_sources

console = Console()


def prenorm_command(
    source_dir: Annotated[
        Path, typer.Argument(help="Parent directory holding one subfolder per instrument.")
    ],
    *,
    target_db: Annotated[
        float, typer.Option("--target-db", help="Desired peak level in dBFS.")
    ] = TARGET_DB_DEFAULT,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would change without writing.")
    ] = False,
) -> None:
    """Peak-normalize every sustain/release render under source_dir to target_db, in place."""
    reports = prenormalize_sources(source_dir, target_db=target_db, dry_run=dry_run)
    if not reports:
        console.print(f"[yellow]No sustain/release renders found under {source_dir}[/]")
        raise typer.Exit(code=1)

    for report in reports:
        if report.peak_db is None:
            console.print(f"  [dim]SKIP[/] {report.path} (silent file)")
            continue
        verb = "would apply" if not report.applied else "applied"
        console.print(
            f"  {report.path}  peak {report.peak_db:+.2f} dBFS  ->  "
            f"gain {report.gain_db:+.2f} dB  ({verb})"
        )

    applied = sum(report.applied for report in reports)
    console.print(f"Done. {applied}/{len(reports)} file(s) normalized.")
