"""Tests for dsp.limiter: true-peak detection and the ceiling guarantee."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.signal import resample_poly

from autosampler.dsp.base import StageContext
from autosampler.dsp.limiter import LimiterParams, LimiterStage

SR = 44_100


def _true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    """Independent true-peak measurement, freshly computed here rather than reusing internals.

    Uses the same 4x oversampling ITU-R BS.1770 (and the stage) uses; a much higher oversample
    factor would reveal FIR ringing on a synthetic worst-case signal like a full-scale square
    wave that no 4x-based true-peak detector (this one included) claims to catch, so isn't a
    fair comparison here.
    """
    oversampled = resample_poly(audio.astype(np.float64), oversample, 1, axis=0)
    return float(np.max(np.abs(oversampled)))


def _clipping_hostile_signal(n: int = 4_410) -> np.ndarray:
    """A near-Nyquist tone plus random spikes: rich in inter-sample (true) peaks."""
    rng = np.random.default_rng(0)
    t = np.arange(n) / SR
    tone = 0.95 * np.sin(2 * np.pi * (SR / 2.0 - 200.0) * t)
    spikes = np.zeros(n)
    spike_positions = rng.integers(0, n, size=n // 50)
    spikes[spike_positions] = rng.uniform(0.5, 1.0, size=spike_positions.shape[0])
    return (tone + spikes).astype(np.float32)


class TestLimiterParams:
    def test_defaults(self) -> None:
        params = LimiterParams()
        assert params.ceiling_db == -0.3

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            LimiterParams(bogus=1)  # type: ignore[call-arg]

    def test_rejects_positive_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            LimiterParams(ceiling_db=0.1)


class TestLimiterStage:
    def test_never_exceeds_ceiling_on_hostile_signal(self) -> None:
        audio = _clipping_hostile_signal()
        params = LimiterParams(ceiling_db=-1.0, lookahead_ms=5.0, release_ms=50.0)
        stage = LimiterStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))

        ceiling_linear = 10.0 ** (-1.0 / 20.0)
        measured_true_peak = _true_peak(result)
        assert measured_true_peak <= ceiling_linear + 1e-4

    def test_never_exceeds_ceiling_on_full_scale_square_wave(self) -> None:
        """A square wave has the worst-case ratio of sample peak to true peak."""
        n = 2_000
        audio = np.where(np.arange(n) % 40 < 20, 1.0, -1.0).astype(np.float32)
        params = LimiterParams(ceiling_db=-0.5, lookahead_ms=3.0, release_ms=30.0)
        stage = LimiterStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))

        ceiling_linear = 10.0 ** (-0.5 / 20.0)
        assert _true_peak(result) <= ceiling_linear + 1e-4

    def test_signal_already_under_ceiling_is_left_alone(self) -> None:
        audio = np.full(1000, 0.1, dtype=np.float32)
        params = LimiterParams(ceiling_db=-1.0)
        stage = LimiterStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_silence_stays_silent(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        stage = LimiterStage(LimiterParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_array_equal(result, audio)

    def test_empty_input_returned_unchanged(self) -> None:
        stage = LimiterStage(LimiterParams())
        result = stage.apply(np.zeros(0, dtype=np.float32), StageContext(sample_rate=SR))
        assert result.shape[0] == 0

    def test_stereo_shape_preserved_and_ceiling_respected(self) -> None:
        left = _clipping_hostile_signal(2_000)
        right = _clipping_hostile_signal(2_000) * -1.0
        audio = np.stack([left, right], axis=1)
        params = LimiterParams(ceiling_db=-1.0)
        stage = LimiterStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))

        assert result.shape == audio.shape
        ceiling_linear = 10.0 ** (-1.0 / 20.0)
        assert _true_peak(result[:, 0]) <= ceiling_linear + 1e-4
        assert _true_peak(result[:, 1]) <= ceiling_linear + 1e-4

    def test_output_dtype_is_float32(self) -> None:
        audio = _clipping_hostile_signal(500)
        stage = LimiterStage(LimiterParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.dtype == np.float32
