"""Tests for POST /api/jobs and GET /api/jobs/{id}."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.conftest import open_instrument
from tests.pipeline.conftest import Instrument

_POLL_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 0.05


def _wait_for_completion(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] != "running":
            return status
        time.sleep(_POLL_INTERVAL_S)
    raise AssertionError(f"job {job_id} did not finish within {_POLL_TIMEOUT_S}s")


def test_start_job_before_open_is_404(client: TestClient) -> None:
    assert client.post("/api/jobs", json={}).status_code == 404


def test_unknown_job_id_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_output_before_completion_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist/output").status_code == 404


def test_job_runs_and_writes_output(
    client: TestClient, api_instrument: Instrument, tmp_path: Path
) -> None:
    open_instrument(client, api_instrument)
    output_dir = tmp_path / "output"

    started = client.post("/api/jobs", json={"output_dir": str(output_dir), "workers": 1})
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    status = _wait_for_completion(client, job_id)
    assert status["status"] == "completed", status
    assert status["output_dir"] is not None
    assert status["events"], "expected at least one recorded progress event"

    out = output_dir / api_instrument.folder.name
    assert (out / f"{api_instrument.folder.name}_sustain.sfz").is_file()
    assert list((out / "samples").glob("*.flac"))

    output = client.get(f"/api/jobs/{job_id}/output")
    assert output.status_code == 200
    body = output.json()
    assert body["sustain_sfz"] == (out / f"{api_instrument.folder.name}_sustain.sfz").read_text()
