"""Tests for dsp.trim: TrimStage, clamped so pre_trim_ms can never empty the buffer (bug 5)."""

from __future__ import annotations

import numpy as np

from autoluthier.dsp.base import StageContext
from autoluthier.dsp.trim import TrimParams, TrimStage

SR = 44_100


class TestTrimStage:
    def test_zero_trim_is_noop(self) -> None:
        audio = np.arange(1000, dtype=np.float32)
        stage = TrimStage(TrimParams(pre_trim_ms=0.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_array_equal(result, audio)

    def test_partial_trim_cuts_correct_frame_count(self) -> None:
        audio = np.arange(SR, dtype=np.float32)  # 1 second
        stage = TrimStage(TrimParams(pre_trim_ms=10.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        expected_cut = round(0.010 * SR)
        assert result.shape[0] == audio.shape[0] - expected_cut
        np.testing.assert_array_equal(result, audio[expected_cut:])

    def test_trim_longer_than_sample_clamps_to_one_frame(self) -> None:
        """Bug 5: pre_trim_ms past the sample's end used to produce an empty array."""
        audio = np.arange(100, dtype=np.float32)
        stage = TrimStage(TrimParams(pre_trim_ms=1_000.0))  # far longer than the sample
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape[0] == 1
        assert result[0] == audio[-1]

    def test_trim_exactly_sample_duration_clamps_to_one_frame(self) -> None:
        n = 100
        audio = np.arange(n, dtype=np.float32)
        exact_ms = n / SR * 1000.0
        stage = TrimStage(TrimParams(pre_trim_ms=exact_ms))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape[0] >= 1

    def test_empty_input_returned_unchanged(self) -> None:
        audio = np.zeros(0, dtype=np.float32)
        stage = TrimStage(TrimParams(pre_trim_ms=10.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape[0] == 0

    def test_stereo_trim_preserves_channel_count(self) -> None:
        audio = np.zeros((SR, 2), dtype=np.float32)
        stage = TrimStage(TrimParams(pre_trim_ms=10.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape[1] == 2
        assert result.shape[0] == SR - round(0.010 * SR)
