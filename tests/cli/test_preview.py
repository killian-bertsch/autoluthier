"""Tests for the `autoluthier preview` command."""

from __future__ import annotations

from autoluthier.cli.app import app
from autoluthier.config.toml_io import save_project
from tests.cli.conftest import runner
from tests.pipeline.conftest import Instrument


def test_preview_writes_selected_samples(cli_instrument: Instrument) -> None:
    save_project(cli_instrument.config, cli_instrument.folder)
    out_dir = cli_instrument.folder / "preview"

    result = runner.invoke(
        app,
        [
            "preview",
            str(cli_instrument.folder),
            "--notes",
            "1",
            "--velocities",
            "1",
            "--out",
            str(out_dir),
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    written = list(out_dir.glob(f"*.{cli_instrument.config.output.container}"))
    assert written
    assert "Wrote" in result.output


def test_preview_rejects_unknown_mode(cli_instrument: Instrument) -> None:
    save_project(cli_instrument.config, cli_instrument.folder)
    result = runner.invoke(app, ["preview", str(cli_instrument.folder), "--mode", "bogus"])
    assert result.exit_code == 1
    assert "--mode" in result.output
