"""Tests for the `autosampler prenorm` command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from autosampler.cli.app import app
from tests.cli.conftest import runner

SAMPLE_RATE = 8000


def _write_tone(path: Path, peak: float) -> None:
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    audio = (peak * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE)


def test_prenorm_applies_gain(tmp_path: Path) -> None:
    path = tmp_path / "keybass" / "sustain.wav"
    _write_tone(path, 0.5)

    result = runner.invoke(app, ["prenorm", str(tmp_path), "--target-db", "-3.0"])

    assert result.exit_code == 0, result.output
    assert "1/1 file(s) normalized" in result.output


def test_prenorm_dry_run_leaves_files_untouched(tmp_path: Path) -> None:
    path = tmp_path / "keybass" / "sustain.wav"
    _write_tone(path, 0.5)
    before = path.read_bytes()

    result = runner.invoke(app, ["prenorm", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before


def test_prenorm_no_files_found_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["prenorm", str(tmp_path)])
    assert result.exit_code == 1
