"""Tests for GET /api/schema."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_schema_matches_project_config(client: TestClient) -> None:
    response = client.get("/api/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "ProjectConfig"
    assert "recording" in body["properties"]
    assert "stages" in body["properties"]
