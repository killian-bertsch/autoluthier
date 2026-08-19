"""Tests for the `autosampler run` command."""

from __future__ import annotations

from pathlib import Path

from autosampler.cli.app import app
from autosampler.config.toml_io import save_project
from tests.cli.conftest import runner
from tests.pipeline.conftest import Instrument, write_instrument


def _init_project(instrument: Instrument) -> None:
    save_project(instrument.config, instrument.folder)


def test_run_single_instrument_writes_sfz_and_samples(cli_instrument: Instrument) -> None:
    _init_project(cli_instrument)
    output_dir = cli_instrument.folder.parent / "output"

    result = runner.invoke(
        app, ["run", str(cli_instrument.folder), "--output-dir", str(output_dir), "--workers", "1"]
    )

    assert result.exit_code == 0, result.output
    out = output_dir / cli_instrument.folder.name
    assert (out / f"{cli_instrument.folder.name}_sustain.sfz").is_file()
    assert list((out / "samples").glob("*.flac"))
    assert "processed successfully" in result.output


def test_run_scan_processes_every_instrument(tmp_path: Path) -> None:
    parent = tmp_path / "instruments"
    one = write_instrument(parent / "one")
    two = write_instrument(parent / "two")
    _init_project(one)
    _init_project(two)
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app, ["run", str(parent), "--scan", "--output-dir", str(output_dir), "--workers", "1"]
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "one").is_dir()
    assert (output_dir / "two").is_dir()
    assert "2/2 instrument(s)" in result.output


def test_run_scan_with_instrument_flag_processes_only_that_one(tmp_path: Path) -> None:
    parent = tmp_path / "instruments"
    one = write_instrument(parent / "one")
    two = write_instrument(parent / "two")
    _init_project(one)
    _init_project(two)
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "run",
            str(parent),
            "--scan",
            "--instrument",
            "one",
            "--output-dir",
            str(output_dir),
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "one").is_dir()
    assert not (output_dir / "two").exists()


def test_run_instrument_without_scan_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path), "--instrument", "one"])
    assert result.exit_code == 1
    assert "--scan" in result.output


def test_run_scan_skips_malformed_project_and_continues(tmp_path: Path) -> None:
    parent = tmp_path / "instruments"
    good = write_instrument(parent / "good")
    _init_project(good)
    bad_dir = parent / "bad"
    bad_dir.mkdir()
    (bad_dir / "project.toml").write_text("this = 'is not a valid project'\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app, ["run", str(parent), "--scan", "--output-dir", str(output_dir), "--workers", "1"]
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "good").is_dir()
    assert "1/2 instrument(s)" in result.output


def test_run_missing_project_toml_fails(tmp_path: Path) -> None:
    folder = tmp_path / "no_project"
    folder.mkdir()
    result = runner.invoke(app, ["run", str(folder)])
    assert result.exit_code == 1
