"""Shared fixtures for API tests: a FastAPI TestClient plus a real instrument folder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoluthier.config.toml_io import save_project
from autoluthier.server.app import create_app
from tests.pipeline.conftest import Instrument, write_instrument


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A `TestClient` over a fresh app, its workspace index isolated to `tmp_path`."""
    app = create_app(workspace_path=tmp_path / "workspace.toml")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_instrument(tmp_path: Path) -> Instrument:
    """A small mono instrument folder with a project.toml-ready config."""
    return write_instrument(tmp_path / "keybass")


def open_instrument(client: TestClient, instrument: Instrument) -> None:
    """Write `instrument`'s project.toml and open it via ``POST /api/project``."""
    save_project(instrument.config, instrument.folder)
    response = client.post("/api/project", json={"folder": str(instrument.folder)})
    assert response.status_code == 200, response.text
