"""``GET/POST/DELETE /api/workspace`` — the app-level list of known instrument projects."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autoluthier.config.workspace import WorkspaceEntry, load_workspace, save_workspace
from autoluthier.server.state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


class AddProjectRequest(BaseModel):
    """Body of ``POST /api/workspace``."""

    path: str
    name: str | None = None


@router.get("/api/workspace")
def list_workspace(request: Request) -> list[WorkspaceEntry]:
    """Return every project tracked by the workspace index.

    Returns:
        The tracked entries, in index order.
    """
    return load_workspace(_state(request).workspace_path).entries


@router.post("/api/workspace")
def add_to_workspace(body: AddProjectRequest, request: Request) -> list[WorkspaceEntry]:
    """Add (or rename) a tracked project.

    Args:
        body: The project path, and optional display name, to add.
        request: The current request, used to reach server state.

    Returns:
        The updated list of tracked entries.
    """
    state = _state(request)
    index = load_workspace(state.workspace_path)
    index.add(body.path, body.name)
    save_workspace(index, state.workspace_path)
    return index.entries


@router.delete("/api/workspace")
def remove_from_workspace(path: str, request: Request) -> list[WorkspaceEntry]:
    """Remove a tracked project by path.

    Args:
        path: The project path to remove.
        request: The current request, used to reach server state.

    Returns:
        The updated list of tracked entries.
    """
    state = _state(request)
    index = load_workspace(state.workspace_path)
    index.remove(path)
    save_workspace(index, state.workspace_path)
    return index.entries
