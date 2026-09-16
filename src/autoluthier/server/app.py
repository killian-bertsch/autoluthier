"""FastAPI app factory: wires state, routes, error handling, and the frontend mount.

`create_app` takes an explicit `workspace_path` rather than always reading
`config.workspace.default_workspace_path` so tests (and, later, a ``--workspace`` CLI flag) can
point the server at an isolated index instead of a real user's ``~/.autoluthier``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from autoluthier.config.workspace import default_workspace_path
from autoluthier.server.errors import install_error_handlers
from autoluthier.server.routes import (
    analysis,
    events,
    helpers,
    jobs,
    loop,
    matrix,
    preview,
    project,
    samples,
    schema,
    sfz,
    workspace,
)
from autoluthier.server.sse import EventBroadcaster
from autoluthier.server.state import AppState
from autoluthier.server.static import mount_frontend

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
"""Repo-root ``frontend/`` — three levels up from ``src/autoluthier/server/app.py``."""

_ROUTERS = (
    schema,
    workspace,
    project,
    samples,
    loop,
    matrix,
    analysis,
    sfz,
    preview,
    jobs,
    events,
    helpers,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bind the broadcaster to the now-running event loop for the server's lifetime."""
    app.state.broadcaster.bind_loop(asyncio.get_running_loop())
    yield


def create_app(*, workspace_path: Path | None = None) -> FastAPI:
    """Build the FastAPI app: state, routers, error handlers, and the frontend mount.

    Args:
        workspace_path: Where the workspace index file lives; ``None`` uses the default
            per-user location. Tests pass a temp path so a test run never touches a real
            user's index.

    Returns:
        The configured `FastAPI` app, not yet running.
    """
    app = FastAPI(title="autoluthier", version="2.0.0", lifespan=_lifespan)
    app.state.autoluthier = AppState(workspace_path or default_workspace_path())
    app.state.broadcaster = EventBroadcaster()

    install_error_handlers(app)
    for router_module in _ROUTERS:
        app.include_router(router_module.router)

    mount_frontend(app, FRONTEND_DIR)
    return app
