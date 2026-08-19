"""Tests for GET /api/sfz/preview."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_sfz_preview_before_open_is_404(client: TestClient) -> None:
    assert client.get("/api/sfz/preview").status_code == 404


def test_sfz_preview_renders_sustain_regions(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    response = client.get("/api/sfz/preview")
    assert response.status_code == 200
    body = response.json()
    assert "<region>" in body["sustain_sfz"]
    assert body["release_sfz"] is not None
    assert "<region>" in body["release_sfz"]


def test_sfz_preview_never_mutates_loaded_audio(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    before = client.get("/api/samples").json()
    client.get("/api/sfz/preview")
    after = client.get("/api/samples").json()
    assert before == after
