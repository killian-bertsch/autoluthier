"""``autosampler ui`` — serve the browser frontend.

Not implemented yet: the FastAPI server and static frontend arrive in steps 9-10. The command
is registered now so the CLI's command surface matches the plan and ``autosampler --help``
already documents where this is going; running it explains that and exits cleanly rather than
failing with an import error for a server module that doesn't exist yet.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

console = Console()


def ui_command(
    *,
    host: Annotated[str, typer.Option(help="Interface to bind the server to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8000,
    dev: Annotated[
        bool, typer.Option("--dev", help="Enable asset live-reload for frontend development.")
    ] = False,
) -> None:
    """Serve the browser UI and open it — not available until step 9 (Server) lands."""
    del host, port, dev  # accepted now so the flag surface is stable once the server lands
    console.print(
        "[yellow]autosampler ui[/] is not available yet — the FastAPI server (step 9) and "
        "frontend (step 10) haven't landed. Use 'run', 'preview', 'validate', 'midi', "
        "'prenorm', or 'concat' in the meantime."
    )
    raise typer.Exit(code=1)
