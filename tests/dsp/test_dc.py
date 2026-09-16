"""Tests for dsp.dc: DC offset removal."""

from __future__ import annotations

import numpy as np

from autoluthier.dsp.base import StageContext
from autoluthier.dsp.dc import DcRemoveParams, DcRemoveStage

SR = 44_100
_TOLERANCE = 1e-6


def _sine(freq: float, amplitude: float, n: int, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestDcRemoveStage:
    def test_mono_offset_removed(self) -> None:
        audio = _sine(100.0, 0.5, 4410) + 0.3
        stage = DcRemoveStage(DcRemoveParams())
        result = stage.apply(audio.astype(np.float32), StageContext(sample_rate=SR))
        assert abs(float(result.mean())) < _TOLERANCE

    def test_stereo_offset_removed_per_channel(self) -> None:
        left = _sine(100.0, 0.5, 4410) + 0.3
        right = _sine(150.0, 0.2, 4410) - 0.1
        audio = np.stack([left, right], axis=1).astype(np.float32)
        stage = DcRemoveStage(DcRemoveParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert abs(float(result[:, 0].mean())) < _TOLERANCE
        assert abs(float(result[:, 1].mean())) < _TOLERANCE

    def test_already_centered_signal_unchanged_shape(self) -> None:
        audio = _sine(100.0, 0.5, 4410)
        stage = DcRemoveStage(DcRemoveParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape == audio.shape

    def test_output_dtype_is_float32(self) -> None:
        audio = _sine(100.0, 0.5, 4410) + 0.3
        stage = DcRemoveStage(DcRemoveParams())
        result = stage.apply(audio.astype(np.float32), StageContext(sample_rate=SR))
        assert result.dtype == np.float32

    def test_zero_signal_stays_zero(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        stage = DcRemoveStage(DcRemoveParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_array_equal(result, audio)
