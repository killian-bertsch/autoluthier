"""Tests for GET /api/matrix."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument


def test_matrix_before_open_is_404(client: TestClient) -> None:
    assert client.get("/api/matrix").status_code == 404


def test_matrix_has_one_row_per_sample(client: TestClient, api_instrument: Instrument) -> None:
    open_instrument(client, api_instrument)
    rows = client.get("/api/matrix").json()
    n = len(api_instrument.notes) * len(api_instrument.velocities)
    assert len(rows) == 2 * n
    row = rows[0]
    assert {"kind", "note", "velocity", "n_frames", "duration_s", "peak_db", "rms_db"} <= row.keys()
    assert row["rms_db"] <= row["peak_db"]
