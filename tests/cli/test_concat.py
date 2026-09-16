"""Tests for the `autoluthier concat` command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from autoluthier.cli.app import app
from tests.cli.conftest import runner

SAMPLE_RATE = 8000


def _tone(n_frames: int, amp: float, *, channels: int = 1) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float64) / SAMPLE_RATE
    mono = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono, mono], axis=1)


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def test_concat_converts_instruments(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    _write(input_dir / "2 Piano.flac", _tone(SAMPLE_RATE, 0.15))
    output_dir = tmp_path / "out"

    result = runner.invoke(app, ["concat", str(input_dir), str(output_dir)])

    assert result.exit_code == 0, result.output
    assert (output_dir / "Piano" / "sustain.flac").is_file()
    assert "1/1 instrument(s) converted" in result.output


def test_concat_defaults_output_to_input_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))

    result = runner.invoke(app, ["concat", str(input_dir)])

    assert result.exit_code == 0, result.output
    assert (input_dir / "Piano" / "sustain.flac").is_file()


def test_concat_stereo_flag_keeps_channels(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Guitar.flac", _tone(SAMPLE_RATE, 0.1, channels=2))
    output_dir = tmp_path / "out"

    result = runner.invoke(app, ["concat", str(input_dir), str(output_dir), "--stereo", "Guitar"])

    assert result.exit_code == 0, result.output
    audio, _ = sf.read(str(output_dir / "Guitar" / "sustain.flac"))
    assert audio.ndim == 2


def test_concat_no_instruments_found_fails(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    result = runner.invoke(app, ["concat", str(input_dir)])
    assert result.exit_code == 1
