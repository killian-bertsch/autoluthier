"""Tests for GET /api/zero-crossings/{id}."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_zero_crossings_before_open_is_404(client: TestClient) -> None:
    assert client.get("/api/zero-crossings/sustain-60-1").status_code == 404


def test_zero_crossings_unknown_id_is_404(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    assert client.get("/api/zero-crossings/sustain-9-9").status_code == 404


def test_zero_crossings_returns_ascending_frame_indices(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    sample_id = next(s["id"] for s in samples if s["kind"] == "sustain")

    response = client.get(f"/api/zero-crossings/{sample_id}")
    assert response.status_code == 200
    indices = response.json()
    assert isinstance(indices, list)
    assert indices == sorted(indices)
    assert all(isinstance(i, int) for i in indices)


def test_zero_crossings_preview_source_without_preview_is_404(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    samples = client.get("/api/samples").json()
    sample_id = samples[0]["id"]
    response = client.get(f"/api/zero-crossings/{sample_id}", params={"source": "preview"})
    assert response.status_code == 404
