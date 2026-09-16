"""Tests for autoluthier.io.peaks."""

from __future__ import annotations

import numpy as np
import pytest

from autoluthier.io.peaks import compute_peak_level, compute_peaks


def test_bin_count_clamped_to_frame_count() -> None:
    audio = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    level = compute_peak_level(audio, bin_count=1000)
    assert level.mins.shape == (1, 5)
    assert level.maxs.shape == (1, 5)


def test_reduces_mono_audio_to_requested_bin_count() -> None:
    # Four bins of constant value 0, 1, 2, 3, so each bin's min/max is exactly that value.
    audio = np.repeat(np.arange(4, dtype=np.float32), 100)
    level = compute_peak_level(audio, bin_count=4)
    assert level.samples_per_bin == 100
    assert level.mins.tolist() == [[0.0, 1.0, 2.0, 3.0]]
    assert level.maxs.tolist() == [[0.0, 1.0, 2.0, 3.0]]


def test_min_max_bracket_a_sine_within_each_bin() -> None:
    t = np.linspace(0.0, 1.0, 4000, dtype=np.float64)
    audio = np.sin(2.0 * np.pi * 37.0 * t).astype(np.float32)
    level = compute_peak_level(audio, bin_count=40)
    assert np.all(level.mins <= level.maxs)
    assert level.maxs.max() == pytest.approx(1.0, abs=0.05)
    assert level.mins.min() == pytest.approx(-1.0, abs=0.05)


def test_stereo_audio_keeps_channels_independent() -> None:
    left = np.full(100, 0.5, dtype=np.float32)
    right = np.full(100, -0.25, dtype=np.float32)
    audio = np.column_stack([left, right])
    level = compute_peak_level(audio, bin_count=1)
    assert level.mins.tolist() == [[0.5], [-0.25]]
    assert level.maxs.tolist() == [[0.5], [-0.25]]


def test_trailing_partial_bin_is_padded_not_dropped() -> None:
    # 10 frames into 3 bins: ceil(10/3) = 4 samples/bin, last bin padded by repeating frame 9.
    audio = np.arange(10, dtype=np.float32)
    level = compute_peak_level(audio, bin_count=3)
    assert level.samples_per_bin == 4
    assert level.maxs.tolist() == [[3.0, 7.0, 9.0]]


def test_empty_audio_returns_empty_level() -> None:
    level = compute_peak_level(np.zeros(0, dtype=np.float32), bin_count=100)
    assert level.mins.shape == (1, 0)
    assert level.maxs.shape == (1, 0)


def test_compute_peaks_returns_one_level_per_bin_count() -> None:
    audio = np.linspace(-1.0, 1.0, 10_000, dtype=np.float32)
    levels = compute_peaks(audio, bin_counts=(10, 100, 1000))
    assert [level.mins.shape[1] for level in levels] == [10, 100, 1000]


def test_compute_peaks_default_bin_counts() -> None:
    audio = np.zeros(50_000, dtype=np.float32)
    levels = compute_peaks(audio)
    assert len(levels) == 3
