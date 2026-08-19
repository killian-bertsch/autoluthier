"""Tests for autosampler.analysis.matrix."""

from __future__ import annotations

import numpy as np
import pytest

from autosampler.analysis.matrix import compute_matrix
from autosampler.domain.models import InstrumentAudio, Sample, SampleSet

SAMPLE_RATE = 22050


def _tone(amplitude: float, n_frames: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def test_peak_and_rms_match_a_known_sine() -> None:
    sample = Sample(note=60, velocity=127, audio=_tone(0.5), sample_rate=SAMPLE_RATE)
    audio = InstrumentAudio(sustain=SampleSet([sample]), release=SampleSet([]))
    entries = compute_matrix(audio)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "sustain"
    assert entry.note == 60
    assert entry.velocity == 127
    assert entry.n_frames == SAMPLE_RATE
    assert entry.duration_s == pytest.approx(1.0)
    assert entry.peak_db == pytest.approx(20.0 * np.log10(0.5), abs=0.01)
    expected_rms_db = 20.0 * np.log10(0.5 / np.sqrt(2.0))
    assert entry.rms_db == pytest.approx(expected_rms_db, abs=0.01)


def test_silence_is_floored_not_infinite() -> None:
    silence = np.zeros(1000, dtype=np.float32)
    audio = InstrumentAudio(
        sustain=SampleSet([Sample(note=60, velocity=1, audio=silence, sample_rate=SAMPLE_RATE)]),
        release=SampleSet([]),
    )
    entry = compute_matrix(audio)[0]
    assert np.isfinite(entry.peak_db)
    assert np.isfinite(entry.rms_db)


def test_sustain_then_release_order() -> None:
    audio = InstrumentAudio(
        sustain=SampleSet(
            [Sample(note=60, velocity=127, audio=_tone(0.1, 100), sample_rate=SAMPLE_RATE)]
        ),
        release=SampleSet(
            [Sample(note=60, velocity=127, audio=_tone(0.1, 100), sample_rate=SAMPLE_RATE)]
        ),
    )
    kinds = [entry.kind for entry in compute_matrix(audio)]
    assert kinds == ["sustain", "release"]


def test_stereo_reduces_to_mono_for_levels() -> None:
    mono = _tone(0.4, 500)
    stereo = np.column_stack([mono, -mono])
    audio = InstrumentAudio(
        sustain=SampleSet([Sample(note=60, velocity=127, audio=stereo, sample_rate=SAMPLE_RATE)]),
        release=SampleSet([]),
    )
    entry = compute_matrix(audio)[0]
    # Left and right cancel exactly, so the mono-reduced signal is silence.
    assert entry.peak_db < -150.0
