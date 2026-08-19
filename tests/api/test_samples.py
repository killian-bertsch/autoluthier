"""Tests for GET /api/samples, /api/peaks/{id}, and /api/audio/{id}."""

from __future__ import annotations

from io import BytesIO

import soundfile as sf
from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_list_samples_before_open_is_404(client: TestClient) -> None:
    assert client.get("/api/samples").status_code == 404


def test_list_samples_covers_sustain_and_release(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    n = len(api_instrument.notes) * len(api_instrument.velocities)
    assert sum(s["kind"] == "sustain" for s in samples) == n
    assert sum(s["kind"] == "release" for s in samples) == n
    first = samples[0]
    assert first["id"] == f"sustain-{first['note']}-{first['velocity']}"
    assert first["has_loop"] is False


def test_peaks_returns_three_zoom_levels(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    sample_id = samples[0]["id"]

    response = client.get(f"/api/peaks/{sample_id}")
    assert response.status_code == 200
    levels = response.json()
    assert len(levels) == 3
    assert all("mins" in level and "maxs" in level for level in levels)


def test_peaks_unknown_id_is_404(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    assert client.get("/api/peaks/sustain-9-9").status_code == 404


def test_audio_returns_playable_bytes(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    sample_id = samples[0]["id"]

    response = client.get(f"/api/audio/{sample_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/flac"
    data, rate = sf.read(BytesIO(response.content))
    assert rate > 0
    assert data.size > 0


def test_audio_unknown_id_is_404(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    assert client.get("/api/audio/sustain-9-9").status_code == 404


def test_audio_preview_source_without_preview_is_404(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    sample_id = samples[0]["id"]
    response = client.get(f"/api/audio/{sample_id}", params={"source": "preview"})
    assert response.status_code == 404
