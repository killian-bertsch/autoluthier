"""Tests for GET /api/schema."""

from __future__ import annotations

from fastapi.testclient import TestClient

from autoluthier.config.schema import default_stage_chain


def test_schema_matches_project_config(client: TestClient) -> None:
    response = client.get("/api/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "ProjectConfig"
    assert "recording" in body["properties"]
    assert "stages" in body["properties"]


def test_stage_schemas_cover_every_default_chain_id(client: TestClient) -> None:
    response = client.get("/api/schema/stages")
    assert response.status_code == 200
    body = response.json()
    for stage in default_stage_chain():
        assert stage.id in body, f"missing schema for stage id={stage.id!r}"
        assert "properties" in body[stage.id]


def test_stage_schema_carries_ui_hints(client: TestClient) -> None:
    response = client.get("/api/schema/stages")
    body = response.json()
    trim_props = body["trim"]["properties"]
    assert trim_props["pre_trim_ms"]["unit"] == "ms"
