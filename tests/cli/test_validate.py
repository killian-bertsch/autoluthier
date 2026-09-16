"""Tests for the `autoluthier validate` command."""

from __future__ import annotations

from pathlib import Path

from autoluthier.cli.app import app
from autoluthier.config.schema import StageConfig
from autoluthier.config.toml_io import save_project
from tests.cli.conftest import runner
from tests.pipeline.conftest import Instrument


def test_validate_ok_project(cli_instrument: Instrument) -> None:
    save_project(cli_instrument.config, cli_instrument.folder)
    result = runner.invoke(app, ["validate", str(cli_instrument.folder)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_missing_project_toml(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    result = runner.invoke(app, ["validate", str(folder)])
    assert result.exit_code == 1
    assert "Invalid project" in result.output


def test_validate_reports_bad_stage_chain(cli_instrument: Instrument) -> None:
    config = cli_instrument.config.model_copy(deep=True)
    config.stages.append(StageConfig(id="normalize"))  # normalize is not idempotent, must be unique
    save_project(config, cli_instrument.folder)

    result = runner.invoke(app, ["validate", str(cli_instrument.folder)])
    assert result.exit_code == 1
    assert "Invalid stage chain" in result.output
