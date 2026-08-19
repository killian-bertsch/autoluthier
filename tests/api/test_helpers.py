"""Tests for POST /api/helpers/{midi-session,prenormalize,concat}."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient


def test_midi_session_writes_files_and_offers_download(
    client: TestClient, tmp_path: Path
) -> None:
    out_dir = tmp_path / "keybass"
    response = client.post(
        "/api/helpers/midi-session",
        json={
            "velocity_layers": 2,
            "semitone_interval": 12,
            "hold_time": 0.1,
            "release_time": 0.05,
            "output": str(tmp_path / "keybass"),
            "start_note": 60,
            "end_note": 72,
            "out_dir": str(out_dir),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert Path(body["midi_path"]).is_file()
    assert Path(body["project_path"]).is_file()
    assert body["velocities"] == [1, 127]

    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "audio/midi"


def test_midi_session_invalid_note_range_is_400(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/helpers/midi-session",
        json={
            "velocity_layers": 1,
            "semitone_interval": 1,
            "hold_time": 0.1,
            "release_time": 0.05,
            "output": str(tmp_path / "keybass"),
            "start_note": 80,
            "end_note": 60,
        },
    )
    assert response.status_code == 400


def test_download_missing_midi_is_404(client: TestClient, tmp_path: Path) -> None:
    response = client.get(
        "/api/helpers/midi-session/download", params={"path": str(tmp_path / "nope.mid")}
    )
    assert response.status_code == 404


def test_prenormalize_dry_run_reports_without_writing(
    client: TestClient, tmp_path: Path
) -> None:
    instrument_dir = tmp_path / "keybass"
    instrument_dir.mkdir()
    path = instrument_dir / "sustain.wav"
    sf.write(str(path), [0.1, -0.1, 0.2, -0.2], 8000)
    original_bytes = path.read_bytes()

    response = client.post(
        "/api/helpers/prenormalize",
        json={"source_dir": str(tmp_path), "target_db": -6.0, "dry_run": True},
    )
    assert response.status_code == 200
    reports = response.json()
    assert len(reports) == 1
    assert reports[0]["applied"] is False
    assert path.read_bytes() == original_bytes


def test_concat_reports_missing_instrument(client: TestClient, tmp_path: Path) -> None:
    input_dir = tmp_path / "layers"
    input_dir.mkdir()
    response = client.post("/api/helpers/concat", json={"input_dir": str(input_dir)})
    assert response.status_code == 200
    assert response.json() == []
