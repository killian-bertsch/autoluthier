"""Tests for autosampler.config.toml_io."""

from pathlib import Path

import pytest

from autosampler.config.schema import ProjectConfig, RecordingConfig, SelectionConfig
from autosampler.config.toml_io import (
    ProjectConfigError,
    load_project,
    read_toml,
    save_project,
    write_toml,
)


def _sample_config() -> ProjectConfig:
    return ProjectConfig(
        instrument_name="keybass",
        recording=RecordingConfig(
            velocity_layers=18, semitone_interval=1, hold_time=10.0, release_time=0.5
        ),
        selection=SelectionConfig(velocity_layers_out=4, velocity_map="0-63:1, 64-127:5"),
    )


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    original = _sample_config()
    written = save_project(original, tmp_path)
    assert written == tmp_path / "project.toml"

    loaded = load_project(tmp_path)
    assert loaded == original


def test_save_project_omits_none_fields_from_toml(tmp_path: Path) -> None:
    config = _sample_config()
    path = save_project(config, tmp_path)
    raw = read_toml(path)
    assert "release_hold_time" not in raw["recording"]


def test_load_project_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigError):
        load_project(tmp_path / "does_not_exist")


def test_load_project_malformed_toml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "project.toml"
    bad.write_text("this is not [valid toml", encoding="utf-8")
    with pytest.raises(ProjectConfigError):
        load_project(tmp_path)


def test_load_project_unknown_key_raises(tmp_path: Path) -> None:
    path = save_project(_sample_config(), tmp_path)
    raw = read_toml(path)
    raw["typo_field"] = "oops"
    write_toml(raw, path)
    with pytest.raises(ProjectConfigError):
        load_project(tmp_path)


def test_load_project_bool_for_int_field_raises(tmp_path: Path) -> None:
    """V1 bug 4, exercised end-to-end through a hand-written TOML file."""
    path = tmp_path / "project.toml"
    path.write_text(
        """
        [recording]
        velocity_layers = true
        semitone_interval = 1
        hold_time = 10.0
        release_time = 0.5

        [selection]
        velocity_layers_out = 1
        """,
        encoding="utf-8",
    )
    with pytest.raises(ProjectConfigError):
        load_project(tmp_path)
