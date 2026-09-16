"""Tests for the `autoluthier init` command."""

from __future__ import annotations

from pathlib import Path

from autoluthier.cli.app import app
from autoluthier.config.toml_io import load_project
from tests.cli.conftest import runner


def test_init_writes_project_toml(tmp_path: Path) -> None:
    folder = tmp_path / "keybass"
    folder.mkdir()

    result = runner.invoke(
        app,
        [
            "init",
            str(folder),
            "-X",
            "4",
            "-N",
            "1",
            "-H",
            "10.0",
            "-R",
            "0.5",
            "--start-note",
            "21",
            "--end-note",
            "108",
        ],
    )

    assert result.exit_code == 0, result.output
    config = load_project(folder)
    assert config.recording.velocity_layers == 4
    assert config.selection.velocity_layers_out == 4


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    folder = tmp_path / "keybass"
    folder.mkdir()
    args = [str(folder), "-X", "1", "-N", "1", "-H", "1.0", "-R", "0.1"]

    first = runner.invoke(app, ["init", *args])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["init", *args])
    assert second.exit_code == 1
    assert "--force" in second.output

    third = runner.invoke(app, ["init", *args, "--force"])
    assert third.exit_code == 0, third.output


def test_init_rejects_invalid_settings(tmp_path: Path) -> None:
    folder = tmp_path / "keybass"
    folder.mkdir()
    result = runner.invoke(
        app, ["init", str(folder), "-X", "0", "-N", "1", "-H", "1.0", "-R", "0.1"]
    )
    assert result.exit_code == 1
    assert "Invalid settings" in result.output
