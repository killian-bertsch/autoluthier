"""Shared fixtures for CLI tests: a real instrument folder plus a Typer CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.pipeline.conftest import Instrument, write_instrument

runner = CliRunner()


@pytest.fixture
def cli_instrument(tmp_path: Path) -> Instrument:
    """A small mono instrument folder with a project.toml-ready config."""
    return write_instrument(tmp_path / "keybass")
