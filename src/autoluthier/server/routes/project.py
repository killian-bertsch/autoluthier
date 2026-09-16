"""``GET/POST/PUT /api/project`` — open, inspect, and save the current instrument project."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autoluthier.config.schema import ProjectConfig
from autoluthier.config.toml_io import load_project as load_project_toml
from autoluthier.config.toml_io import save_project
from autoluthier.config.workspace import load_workspace, save_workspace
from autoluthier.pipeline.executor import load_instrument_audio
from autoluthier.server.state import AppState, LoadedInstrument

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


class OpenProjectRequest(BaseModel):
    """Body of ``POST /api/project``."""

    folder: str


class ProjectSummary(BaseModel):
    """Shape returned by ``GET``/``POST``/``PUT /api/project``."""

    folder: str
    config: ProjectConfig
    sustain_count: int
    release_count: int
    notes: list[int]
    velocities: list[int]


def _summary(loaded: LoadedInstrument) -> ProjectSummary:
    """Build the response shape for one loaded instrument."""
    return ProjectSummary(
        folder=str(loaded.folder),
        config=loaded.config,
        sustain_count=len(loaded.audio.sustain),
        release_count=len(loaded.audio.release),
        notes=loaded.audio.sustain.notes(),
        velocities=loaded.audio.sustain.velocities(),
    )


def _open(state: AppState, folder: Path) -> LoadedInstrument:
    """Load `folder`'s project.toml and raw audio, set it as current, and track it.

    Clears any previous preview: it was measured against whichever instrument was open before,
    and would silently misrepresent the new one.
    """
    config = load_project_toml(folder)
    audio = load_instrument_audio(folder, config)
    loaded = LoadedInstrument(folder=folder, config=config, audio=audio)
    state.current = loaded
    state.preview = None
    index = load_workspace(state.workspace_path)
    index.add(folder)
    save_workspace(index, state.workspace_path)
    return loaded


@router.get("/api/project")
def get_project(request: Request) -> ProjectSummary:
    """Return the currently open instrument.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if no instrument is open.
    """
    return _summary(_state(request).require_current())


@router.post("/api/project")
def open_project(body: OpenProjectRequest, request: Request) -> ProjectSummary:
    """Open an instrument folder: load its project.toml and slice its source audio.

    Raises:
        autoluthier.config.toml_io.ProjectConfigError: mapped to 400 on a missing or
            invalid project.toml.
        autoluthier.pipeline.executor.PipelineError: mapped to 400 if the recording can't
            be loaded.
    """
    return _summary(_open(_state(request), Path(body.folder)))


@router.put("/api/project")
def save_and_reload(config: ProjectConfig, request: Request) -> ProjectSummary:
    """Validate and save `config` over the open instrument's project.toml, then reload it.

    Reloading (rather than patching the session in place) is what keeps the response truthful
    when ``recording``/``selection``/``output.collapse_to_mono`` changed, since any of those can
    move where each sample's audio actually comes from.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if no instrument is open.
        autoluthier.pipeline.executor.PipelineError: mapped to 400 if the reload fails.
    """
    state = _state(request)
    folder = state.require_current().folder
    save_project(config, folder)
    return _summary(_open(state, folder))
