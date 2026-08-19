"""Tests for GET /api/analysis."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_analysis_before_open_is_404(client: TestClient) -> None:
    assert client.get("/api/analysis").status_code == 404


def test_analysis_covers_every_release_sample(
    client: TestClient, api_instrument: Instrument
) -> None:
    open_instrument(client, api_instrument)
    entries = client.get("/api/analysis").json()
    n = len(api_instrument.notes) * len(api_instrument.velocities)
    assert len(entries) == n
    for entry in entries:
        assert 1.0 <= entry["rt_decay_db_per_s"] <= 24.0
