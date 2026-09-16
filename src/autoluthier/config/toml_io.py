"""Generic TOML read/write plus typed load/save for ``project.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import ValidationError

from autoluthier.config.schema import ProjectConfig

PROJECT_FILENAME = "project.toml"


class TomlParseError(ValueError):
    """Raised when a file is not valid TOML."""


class ProjectConfigError(ValueError):
    """Raised when a project.toml file is missing or fails schema validation."""


def read_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file into a plain dict.

    Args:
        path: Path to a ``.toml`` file.

    Returns:
        The parsed TOML content.

    Raises:
        TomlParseError: if the file does not exist or is not valid TOML.
    """
    path = Path(path)
    if not path.exists():
        raise TomlParseError(f"No such file: {path}")
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise TomlParseError(f"TOML parse error in {path}: {exc}") from exc


def write_toml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a plain dict to a TOML file, creating parent directories as needed.

    Args:
        data: JSON-compatible data to serialize. TOML has no null, so any ``None`` values
            must already be excluded by the caller.
        path: Destination ``.toml`` file path.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(data, f)
    return path


def load_project(path: str | Path) -> ProjectConfig:
    """Load and validate a project.toml file.

    Args:
        path: Path to a project.toml file, or a directory containing one.

    Returns:
        The validated ``ProjectConfig``.

    Raises:
        ProjectConfigError: if the file is missing, not valid TOML, or fails schema
            validation (unknown keys, wrong types, or a failed cross-field check).
    """
    path = Path(path)
    if path.is_dir():
        path = path / PROJECT_FILENAME
    try:
        raw = read_toml(path)
    except TomlParseError as exc:
        raise ProjectConfigError(str(exc)) from exc
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ProjectConfigError(f"Invalid project config in {path}:\n{exc}") from exc


def save_project(config: ProjectConfig, path: str | Path) -> Path:
    """Serialize a ``ProjectConfig`` to project.toml.

    Args:
        config: The project configuration to write.
        path: Path to write to, or a directory to write ``project.toml`` into.

    Returns:
        The path written to.
    """
    path = Path(path)
    if path.is_dir() or path.suffix.lower() != ".toml":
        path = path / PROJECT_FILENAME
    data = config.model_dump(mode="json", exclude_none=True)
    return write_toml(data, path)
