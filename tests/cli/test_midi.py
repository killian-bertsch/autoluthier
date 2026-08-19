"""Tests for the `autosampler midi` command."""

from __future__ import annotations

from pathlib import Path

from autosampler.cli.app import app
from autosampler.config.toml_io import load_project
from tests.cli.conftest import runner


def test_midi_writes_mid_and_project(tmp_path: Path) -> None:
    stem = str(tmp_path / "keybass")
    result = runner.invoke(
        app,
        [
            "midi",
            "-X",
            "2",
            "-N",
            "1",
            "-H",
            "0.5",
            "-R",
            "0.1",
            "-o",
            stem,
            "--start-note",
            "60",
            "--end-note",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    assert Path(f"{stem}.mid").is_file()
    config = load_project(Path(stem))
    assert config.recording.velocity_layers == 2
    assert config.recording.start_note == 60
    assert config.recording.end_note == 64
    assert "Next:" in result.output


def test_midi_rejects_inverted_note_range(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "midi",
            "-X",
            "1",
            "-N",
            "1",
            "-H",
            "0.5",
            "-R",
            "0.1",
            "-o",
            str(tmp_path / "x"),
            "--start-note",
            "70",
            "--end-note",
            "60",
        ],
    )
    assert result.exit_code == 1
