"""Tests for autoluthier.helpers.midi_session."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoluthier.config.toml_io import load_project
from autoluthier.domain.notes import recorded_notes, velocity_list
from autoluthier.helpers.midi_session import describe_session, generate_midi_session


def test_writes_midi_and_project_toml(tmp_path: Path) -> None:
    stem = str(tmp_path / "keybass")
    result = generate_midi_session(
        4, 2, 1.0, 0.5, stem, start_note=60, end_note=72, out_dir=tmp_path / "keybass"
    )

    assert result.midi_path == Path(f"{stem}.mid")
    assert result.midi_path.is_file()
    assert result.project_path.is_file()
    assert result.notes == recorded_notes(60, 72, 2)
    assert result.velocities == velocity_list(4)
    assert result.total_events == len(result.notes) * len(result.velocities)
    assert result.duration_s == pytest.approx(result.total_events * 1.5)


def test_project_toml_matches_recording_settings(tmp_path: Path) -> None:
    stem = str(tmp_path / "session")
    result = generate_midi_session(3, 1, 0.8, 0.2, stem, start_note=48, end_note=52)

    config = load_project(result.project_path)
    assert config.recording.velocity_layers == 3
    assert config.recording.semitone_interval == 1
    assert config.recording.hold_time == pytest.approx(0.8)
    assert config.recording.release_time == pytest.approx(0.2)
    assert config.recording.start_note == 48
    assert config.recording.end_note == 52
    assert config.selection.min_note == 48
    assert config.selection.max_note == 52
    assert config.selection.velocity_layers_out == 3


def test_out_dir_defaults_to_output_stem_folder(tmp_path: Path) -> None:
    stem = str(tmp_path / "mystem")
    result = generate_midi_session(1, 1, 0.1, 0.1, stem, start_note=60, end_note=61)
    assert result.project_path.parent == Path(stem)


def test_rejects_inverted_note_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_note"):
        generate_midi_session(1, 1, 0.1, 0.1, str(tmp_path / "x"), start_note=70, end_note=60)


def test_describe_session_mentions_paths_and_counts(tmp_path: Path) -> None:
    result = generate_midi_session(2, 1, 0.1, 0.1, str(tmp_path / "x"), start_note=60, end_note=61)
    text = describe_session(result)
    assert str(result.midi_path) in text
    assert str(result.project_path) in text
    assert "Events:" in text
