"""Tests for autosampler.helpers.concat_layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autosampler.helpers.concat_layers import ConcatError, concat_instrument, concat_layers

SAMPLE_RATE = 8000


def _tone(n_frames: int, amp: float, *, channels: int = 1) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float64) / SAMPLE_RATE
    mono = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if channels == 1:
        return mono
    return np.stack([mono, mono * 0.5], axis=1)


def _write(path: Path, audio: np.ndarray, *, fmt: str = "FLAC", subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, format=fmt, subtype=subtype)


def test_concat_instrument_concatenates_layers_in_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    _write(input_dir / "2 Piano.flac", _tone(SAMPLE_RATE, 0.2))
    output_dir = tmp_path / "out"

    report = concat_instrument(input_dir, output_dir, "Piano", layer_numbers=(1, 2, 3))

    assert report.error is None
    assert report.layers_found == 2
    assert report.layers_missing == [3]
    assert report.stereo is False
    assert report.sustain_path is not None
    audio, sr = sf.read(str(report.sustain_path))
    assert sr == SAMPLE_RATE
    assert len(audio) == 2 * SAMPLE_RATE


def test_concat_instrument_downmixes_to_mono_by_default(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Guitar.wav", _tone(SAMPLE_RATE, 0.3, channels=2), fmt="WAV")
    output_dir = tmp_path / "out"

    report = concat_instrument(input_dir, output_dir, "Guitar")

    assert report.stereo is False
    audio, _ = sf.read(str(report.sustain_path))
    assert audio.ndim == 1


def test_concat_instrument_keeps_stereo_when_named(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Guitar.wav", _tone(SAMPLE_RATE, 0.3, channels=2), fmt="WAV")
    output_dir = tmp_path / "out"

    report = concat_instrument(
        input_dir, output_dir, "Guitar", stereo_instruments=frozenset({"Guitar"})
    )

    assert report.stereo is True
    audio, _ = sf.read(str(report.sustain_path))
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_stereo_instruments_is_exact_membership_not_substring(tmp_path: Path) -> None:
    """Bug 12 regression: a plain frozenset cannot suffer V1's quoted-comma corruption."""
    input_dir = tmp_path / "in"
    _write(input_dir / "1 ag.wav", _tone(SAMPLE_RATE, 0.3, channels=2), fmt="WAV")
    output_dir = tmp_path / "out"

    report = concat_instrument(
        input_dir, output_dir, "ag", stereo_instruments=frozenset({"ag", "wg", "pearl"})
    )

    assert report.stereo is True
    audio, _ = sf.read(str(report.sustain_path))
    assert audio.ndim == 2


def test_concat_instrument_writes_release_when_present(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    _write(input_dir / "R Piano.flac", _tone(SAMPLE_RATE, 0.05))
    output_dir = tmp_path / "out"

    report = concat_instrument(input_dir, output_dir, "Piano")

    assert report.release_path is not None
    assert report.release_path.is_file()


def test_concat_instrument_no_release_leaves_release_path_none(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    output_dir = tmp_path / "out"

    report = concat_instrument(input_dir, output_dir, "Piano")

    assert report.release_path is None


def test_concat_instrument_peak_normalizes_to_target(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    output_dir = tmp_path / "out"

    report = concat_instrument(input_dir, output_dir, "Piano", target_db=-6.0)

    audio, _ = sf.read(str(report.sustain_path))
    peak_db = 20.0 * np.log10(np.max(np.abs(audio)))
    assert peak_db == pytest.approx(-6.0, abs=0.05)


def test_concat_instrument_raises_on_sample_rate_mismatch(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    sf.write(
        str(input_dir / "2 Piano.flac"), _tone(4000, 0.1), 4000, format="FLAC", subtype="PCM_16"
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ConcatError, match="sample rate"):
        concat_instrument(input_dir, output_dir, "Piano")


def test_concat_instrument_raises_when_no_layers_found(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(ConcatError, match="no sustain layers"):
        concat_instrument(input_dir, tmp_path / "out", "Ghost")


def test_concat_layers_discovers_all_instruments(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Piano.flac", _tone(SAMPLE_RATE, 0.1))
    _write(input_dir / "1 Guitar.wav", _tone(SAMPLE_RATE, 0.2), fmt="WAV")
    output_dir = tmp_path / "out"

    reports = concat_layers(input_dir, output_dir)

    assert {r.instrument for r in reports} == {"Piano", "Guitar"}
    assert all(r.error is None for r in reports)


def test_concat_layers_continues_past_a_failing_instrument(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _write(input_dir / "1 Good.flac", _tone(SAMPLE_RATE, 0.1))
    _write(input_dir / "1 Bad.flac", _tone(SAMPLE_RATE, 0.1))
    sf.write(
        str(input_dir / "2 Bad.flac"), _tone(4000, 0.1), 4000, format="FLAC", subtype="PCM_16"
    )
    output_dir = tmp_path / "out"

    reports = concat_layers(input_dir, output_dir)
    by_name = {r.instrument: r for r in reports}

    assert by_name["Good"].error is None
    assert by_name["Bad"].error is not None
    assert by_name["Bad"].sustain_path is None
