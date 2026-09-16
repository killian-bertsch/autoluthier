"""Tests for dsp.transient: attack/sustain split, optional saturation."""

from __future__ import annotations

import numpy as np

from autoluthier.dsp.base import StageContext
from autoluthier.dsp.transient import TransientParams, TransientStage

SR = 44_100


def _click_then_tone(n: int = 4_410) -> np.ndarray:
    """A sharp attack followed by a steady tone — a clear transient/sustain split."""
    audio = np.full(n, 0.2, dtype=np.float32)
    audio[:50] = 0.9  # transient spike at the start
    return audio


class TestTransientStage:
    def test_empty_input_returned_unchanged(self) -> None:
        stage = TransientStage(TransientParams())
        result = stage.apply(np.zeros(0, dtype=np.float32), StageContext(sample_rate=SR))
        assert result.shape[0] == 0

    def test_boosted_attack_raises_early_peak_above_body(self) -> None:
        audio = _click_then_tone()
        params = TransientParams(attack_db=6.0, sustain_db=0.0, speed_ms=2.0)
        stage = TransientStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert float(np.max(np.abs(result[:50]))) > float(np.max(np.abs(audio[:50])))

    def test_zero_gain_is_near_identity(self) -> None:
        audio = _click_then_tone()
        params = TransientParams(attack_db=0.0, sustain_db=0.0, speed_ms=10.0)
        stage = TransientStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_allclose(result, audio, atol=1e-5)

    def test_saturate_off_can_exceed_unity(self) -> None:
        audio = np.full(1000, 0.9, dtype=np.float32)
        params = TransientParams(attack_db=12.0, sustain_db=12.0, speed_ms=5.0, saturate=False)
        stage = TransientStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert float(np.max(np.abs(result))) > 1.0

    def test_saturate_on_soft_clips_to_unity(self) -> None:
        audio = np.full(1000, 0.9, dtype=np.float32)
        params = TransientParams(attack_db=12.0, sustain_db=12.0, speed_ms=5.0, saturate=True)
        stage = TransientStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6

    def test_stereo_shape_preserved(self) -> None:
        audio = np.stack([_click_then_tone(1000), _click_then_tone(1000)], axis=1)
        stage = TransientStage(TransientParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape == audio.shape

    def test_output_dtype_is_float32(self) -> None:
        audio = _click_then_tone()
        stage = TransientStage(TransientParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.dtype == np.float32
