"""Workspace-level indexing of instrument projects.

Each instrument keeps its own ``project.toml`` (see `.toml_io`); the workspace index is a
separate, app-level list of known project folders so the UI can offer a "recent projects"
view without rescanning the filesystem on every launch.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autosampler.config.toml_io import PROJECT_FILENAME, TomlParseError, read_toml, write_toml


class WorkspaceIndexError(ValueError):
    """Raised when a workspace index file is missing or fails schema validation."""


class WorkspaceEntry(BaseModel):
    """One instrument project tracked by the workspace index."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    name: str


class WorkspaceIndex(BaseModel):
    """Ordered list of instrument projects known to the app."""

    model_config = ConfigDict(extra="forbid", strict=True)

    entries: list[WorkspaceEntry] = Field(default_factory=list)

    def add(self, path: str | Path, name: str | None = None) -> None:
        """Add a project, or update its name if the (resolved) path is already tracked.

        Args:
            path: Path to the instrument project folder.
            name: Display name; defaults to the folder name.
        """
        resolved = str(Path(path).resolve())
        display_name = name if name is not None else Path(path).name
        for entry in self.entries:
            if entry.path == resolved:
                entry.name = display_name
                return
        self.entries.append(WorkspaceEntry(path=resolved, name=display_name))

    def remove(self, path: str | Path) -> None:
        """Remove a tracked project by path, if present.

        Args:
            path: Path to the instrument project folder.
        """
        resolved = str(Path(path).resolve())
        self.entries = [entry for entry in self.entries if entry.path != resolved]


def default_workspace_path() -> Path:
    """Return the default location for the app-level workspace index file.

    ``~/.autosampler/workspace.toml`` — outside any project folder, since the index tracks
    projects across the whole machine, not just one instrument. The server (step 9) uses this
    unless a caller (tests, in particular) supplies its own path.

    Returns:
        The default workspace index path.
    """
    return Path.home() / ".autosampler" / "workspace.toml"


def discover_projects(root: str | Path) -> list[Path]:
    """Find every immediate subdirectory of `root` that contains a project.toml.

    Mirrors V1's folder-scanning `find_instruments()`, but keyed on `project.toml`
    presence rather than a metadata file plus sustain audio pair.

    Args:
        root: Parent directory to scan.

    Returns:
        Matching subdirectories, sorted by name. Empty if `root` does not exist.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / PROJECT_FILENAME).is_file()
    )


def load_workspace(path: str | Path) -> WorkspaceIndex:
    """Load a workspace index file.

    Args:
        path: Path to the workspace index ``.toml`` file.

    Returns:
        The validated ``WorkspaceIndex``, or an empty one if the file does not exist.

    Raises:
        WorkspaceIndexError: if the file exists but is not valid TOML or fails validation.
    """
    path = Path(path)
    if not path.exists():
        return WorkspaceIndex()
    try:
        raw = read_toml(path)
    except TomlParseError as exc:
        raise WorkspaceIndexError(str(exc)) from exc
    try:
        return WorkspaceIndex.model_validate(raw)
    except ValidationError as exc:
        raise WorkspaceIndexError(f"Invalid workspace index in {path}:\n{exc}") from exc


def save_workspace(index: WorkspaceIndex, path: str | Path) -> Path:
    """Serialize a `WorkspaceIndex` to a TOML file.

    Args:
        index: The workspace index to write.
        path: Destination ``.toml`` file path.

    Returns:
        The path written to.
    """
    data = index.model_dump(mode="json", exclude_none=True)
    return write_toml(data, path)
