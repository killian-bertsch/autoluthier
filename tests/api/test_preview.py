"""Tests for POST /api/preview."""

from __future__ import annotations

from io import BytesIO

import soundfile as sf
from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_preview_before_open_is_404(client: TestClient) -> None:
    assert client.post("/api/preview", json={}).status_code == 404


def test_preview_returns_requested_grid_size(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    response = client.post(
        "/api/preview", json={"note_count": 1, "velocity_count": 1, "mode": "subset"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "subset"
    sustain_ids = [s for s in body["samples"] if s["kind"] == "sustain"]
    assert len(sustain_ids) == 1


def test_preview_audio_fetchable_afterward(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    body = client.post(
        "/api/preview", json={"note_count": 1, "velocity_count": 1, "mode": "subset"}
    ).json()
    sample_id = body["samples"][0]["id"]

    response = client.get(f"/api/audio/{sample_id}", params={"source": "preview"})
    assert response.status_code == 200
    data, _ = sf.read(BytesIO(response.content))
    assert data.size > 0


def test_preview_default_mode_is_exact(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    body = client.post("/api/preview", json={}).json()
    assert body["mode"] == "exact"
