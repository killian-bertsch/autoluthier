"""Tests for autoluthier.config.workspace."""

from pathlib import Path

import pytest

from autoluthier.config.workspace import (
    WorkspaceIndex,
    WorkspaceIndexError,
    discover_projects,
    load_workspace,
    save_workspace,
)


def test_discover_projects_finds_only_dirs_with_project_toml(tmp_path: Path) -> None:
    (tmp_path / "keybass").mkdir()
    (tmp_path / "keybass" / "project.toml").write_text("", encoding="utf-8")
    (tmp_path / "no_project").mkdir()
    (tmp_path / "not_a_dir.toml").write_text("", encoding="utf-8")

    found = discover_projects(tmp_path)
    assert found == [tmp_path / "keybass"]


def test_discover_projects_missing_root_returns_empty() -> None:
    assert discover_projects("/no/such/directory") == []


def test_workspace_add_dedupes_by_resolved_path(tmp_path: Path) -> None:
    index = WorkspaceIndex()
    index.add(tmp_path, name="First Name")
    index.add(tmp_path, name="Renamed")
    assert len(index.entries) == 1
    assert index.entries[0].name == "Renamed"


def test_workspace_add_defaults_name_to_folder_name(tmp_path: Path) -> None:
    project_dir = tmp_path / "keybass"
    project_dir.mkdir()
    index = WorkspaceIndex()
    index.add(project_dir)
    assert index.entries[0].name == "keybass"


def test_workspace_remove(tmp_path: Path) -> None:
    index = WorkspaceIndex()
    index.add(tmp_path)
    index.remove(tmp_path)
    assert index.entries == []


def test_load_workspace_missing_file_returns_empty(tmp_path: Path) -> None:
    index = load_workspace(tmp_path / "workspace.toml")
    assert index.entries == []


def test_save_then_load_workspace_round_trips(tmp_path: Path) -> None:
    index = WorkspaceIndex()
    index.add(tmp_path / "a", name="A")
    index.add(tmp_path / "b", name="B")
    path = tmp_path / "workspace.toml"
    save_workspace(index, path)

    loaded = load_workspace(path)
    assert loaded == index


def test_load_workspace_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "workspace.toml"
    path.write_text('typo_field = "oops"\n', encoding="utf-8")
    with pytest.raises(WorkspaceIndexError):
        load_workspace(path)
