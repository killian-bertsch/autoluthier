"""Tests for GET/POST/DELETE /api/workspace."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_empty_workspace_lists_nothing(client: TestClient) -> None:
    response = client.get("/api/workspace")
    assert response.status_code == 200
    assert response.json() == []


def test_add_then_list_then_remove(client: TestClient, tmp_path: Path) -> None:
    project_dir = tmp_path / "keybass"
    project_dir.mkdir()

    added = client.post("/api/workspace", json={"path": str(project_dir)})
    assert added.status_code == 200
    entries = added.json()
    assert len(entries) == 1
    assert entries[0]["name"] == "keybass"

    listed = client.get("/api/workspace")
    assert listed.json() == entries

    removed = client.delete("/api/workspace", params={"path": str(project_dir)})
    assert removed.status_code == 200
    assert removed.json() == []


def test_add_with_custom_name(client: TestClient, tmp_path: Path) -> None:
    project_dir = tmp_path / "keybass"
    project_dir.mkdir()
    response = client.post(
        "/api/workspace", json={"path": str(project_dir), "name": "My Keybass"}
    )
    assert response.json()[0]["name"] == "My Keybass"
