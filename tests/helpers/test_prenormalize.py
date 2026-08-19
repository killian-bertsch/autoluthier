"""Tests for autosampler.helpers.prenormalize."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autosampler.helpers.prenormalize import (
    find_source_files,
    normalize_file,
    prenormalize_sources,
)

SAMPLE_RATE = 8000


def _write_tone(path: Path, peak: float, *, subtype: str = "PCM_16", fmt: str = "WAV") -> None:
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    audio = (peak * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, format=fmt, subtype=subtype)


def test_find_source_files_discovers_sustain_and_release(tmp_path: Path) -> None:
    _write_tone(tmp_path / "keybass" / "sustain.wav", 0.5)
    _write_tone(tmp_path / "keybass" / "release.wav", 0.2)
    _write_tone(tmp_path / "pearl" / "sustain.flac", 0.3, fmt="FLAC", subtype="PCM_24")

    found = find_source_files(tmp_path)
    names = [(p.parent.name, p.name) for p in found]
    assert names == [
        ("keybass", "sustain.wav"),
        ("keybass", "release.wav"),
        ("pearl", "sustain.flac"),
    ]


def test_find_source_files_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert find_source_files(tmp_path / "nope") == []


def test_normalize_file_measures_and_applies_gain(tmp_path: Path) -> None:
    path = tmp_path / "sustain.wav"
    _write_tone(path, 0.5)

    report = normalize_file(path, target_db=-6.0, dry_run=False)

    assert report.peak_db == pytest.approx(20.0 * math.log10(0.5), abs=0.1)
    assert report.applied is True
    new_peak = float(np.max(np.abs(sf.read(str(path))[0])))
    assert 20.0 * math.log10(new_peak) == pytest.approx(-6.0, abs=0.05)


def test_normalize_file_dry_run_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "sustain.wav"
    _write_tone(path, 0.5)
    original_bytes = path.read_bytes()

    report = normalize_file(path, target_db=-6.0, dry_run=True)

    assert report.applied is False
    assert report.gain_db is not None
    assert path.read_bytes() == original_bytes


def test_normalize_file_silent_file_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    report = normalize_file(path, target_db=-6.0, dry_run=False)

    assert report.peak_db is None
    assert report.applied is False


def test_normalize_file_preserves_container_and_subtype(tmp_path: Path) -> None:
    path = tmp_path / "sustain.flac"
    _write_tone(path, 0.5, fmt="FLAC", subtype="PCM_24")

    normalize_file(path, target_db=-6.0, dry_run=False)

    info = sf.info(str(path))
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"


def test_prenormalize_sources_processes_every_file(tmp_path: Path) -> None:
    _write_tone(tmp_path / "keybass" / "sustain.wav", 0.5)
    _write_tone(tmp_path / "keybass" / "release.wav", 0.25)

    reports = prenormalize_sources(tmp_path, target_db=-3.0)

    assert len(reports) == 2
    assert all(report.applied for report in reports)
