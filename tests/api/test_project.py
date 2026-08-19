"""Tests for GET/POST/PUT /api/project."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_get_project_before_open_is_404(client: TestClient) -> None:
    response = client.get("/api/project")
    assert response.status_code == 404


def test_open_project_returns_summary(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    response = client.get("/api/project")
    assert response.status_code == 200
    body = response.json()
    assert body["folder"] == str(api_instrument.folder)
    assert body["sustain_count"] == len(api_instrument.notes) * len(api_instrument.velocities)
    assert body["notes"] == api_instrument.notes
    assert body["velocities"] == api_instrument.velocities


def test_open_missing_project_toml_is_400(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/api/project", json={"folder": str(tmp_path / "nope")})
    assert response.status_code == 400


def test_open_project_adds_to_workspace(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    entries = client.get("/api/workspace").json()
    assert any(entry["path"] == str(api_instrument.folder.resolve()) for entry in entries)


def test_save_and_reload_persists_changes(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    config = client.get("/api/project").json()["config"]
    config["output"]["ampeg_release"] = 1.25

    response = client.put("/api/project", json=config)
    assert response.status_code == 200, response.text
    assert response.json()["config"]["output"]["ampeg_release"] == 1.25

    reread = client.get("/api/project").json()
    assert reread["config"]["output"]["ampeg_release"] == 1.25


def test_save_invalid_config_is_422(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    config = client.get("/api/project").json()["config"]
    config["output"]["unknown_field"] = "oops"

    response = client.put("/api/project", json=config)
    assert response.status_code == 422
