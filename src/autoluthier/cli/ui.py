"""``autoluthier ui`` — serve the browser frontend and API."""

from __future__ import annotations

import webbrowser
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from autoluthier.server.app import FRONTEND_DIR, create_app

console = Console()


def ui_command(
    *,
    host: Annotated[str, typer.Option(help="Interface to bind the server to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8000,
    dev: Annotated[
        bool, typer.Option("--dev", help="Enable asset live-reload for frontend development.")
    ] = False,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the URL in the default browser.")
    ] = True,
) -> None:
    """Serve the browser UI and API, opening it in the default browser.

    Falls back to the API's own ``/docs`` page instead of a 404 if ``frontend/`` is ever
    missing (a from-source checkout that skipped it, say) — everything under ``/api`` and
    ``/events`` works either way.
    """
    del dev  # accepted now for a stable flag surface; live-reload lands with step 11's dev loop
    url = f"http://{host}:{port}"
    has_frontend = (FRONTEND_DIR / "index.html").is_file()
    console.print(f"[green]Serving autoluthier[/] at {url}")
    if not has_frontend:
        console.print("[yellow]No frontend build found[/] — opening the API docs instead.")
    if open_browser:
        webbrowser.open(f"{url}/" if has_frontend else f"{url}/docs")
    uvicorn.run(create_app(), host=host, port=port)
