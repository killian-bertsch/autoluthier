"""`estimate_rt_decay` against decays whose dB/s rate is known in closed form."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from autoluthier.analysis.metrics import (
    RT_DECAY_DEFAULT,
    RT_DECAY_MAX,
    RT_DECAY_MIN,
    estimate_rt_decay,
)

SAMPLE_RATE = 22050
_DB_PER_NEPER = 20.0 * np.log10(np.e)
"""An envelope of exp(-k*t) loses 20*log10(e)*k dB per second: 8.6859*k."""


def _decaying_tone(
    rate_per_second: float, duration_s: float = 1.0, *, channels: int = 1
) -> NDArray[np.float32]:
    t = np.arange(int(duration_s * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    mono = 0.8 * np.exp(-rate_per_second * t) * np.sin(2.0 * np.pi * 220.0 * t)
    if channels == 1:
        return mono.astype(np.float32)
    return np.column_stack([mono] * channels).astype(np.float32)


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0])
def test_measured_rate_matches_the_closed_form(rate: float) -> None:
    expected = _DB_PER_NEPER * rate
    assert estimate_rt_decay(_decaying_tone(rate), SAMPLE_RATE) == pytest.approx(
        expected, abs=0.15
    )


def test_a_faster_decay_than_the_clamp_reports_the_ceiling() -> None:
    assert estimate_rt_decay(_decaying_tone(10.0), SAMPLE_RATE) == RT_DECAY_MAX


def test_a_steady_tone_reports_the_floor_rather_than_zero() -> None:
    """A non-decaying sample fits a flat (or rising) line; rt_decay is never below 1 dB/s."""
    assert estimate_rt_decay(_decaying_tone(0.0), SAMPLE_RATE) == RT_DECAY_MIN


def test_a_rising_sample_also_reports_the_floor() -> None:
    assert estimate_rt_decay(_decaying_tone(-2.0), SAMPLE_RATE) == RT_DECAY_MIN


def test_a_sample_too_short_for_two_windows_uses_the_default() -> None:
    short = _decaying_tone(1.0, duration_s=0.03)
    assert estimate_rt_decay(short, SAMPLE_RATE) == RT_DECAY_DEFAULT


def test_a_constant_gain_does_not_move_the_measurement() -> None:
    """A slope in dB is gain-invariant, which is why normalize's placement cannot affect it."""
    audio = _decaying_tone(1.5)
    assert estimate_rt_decay(audio * 0.05, SAMPLE_RATE) == pytest.approx(
        estimate_rt_decay(audio, SAMPLE_RATE), abs=1e-6
    )


def test_stereo_is_measured_on_the_channel_mean() -> None:
    mono = _decaying_tone(1.5)
    stereo = _decaying_tone(1.5, channels=2)
    assert estimate_rt_decay(stereo, SAMPLE_RATE) == pytest.approx(
        estimate_rt_decay(mono, SAMPLE_RATE), abs=1e-6
    )


def test_silence_reports_the_floor_instead_of_producing_nan() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert estimate_rt_decay(silence, SAMPLE_RATE) == RT_DECAY_MIN
