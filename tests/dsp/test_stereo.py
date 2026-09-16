"""Tests for dsp.stereo: mid-side width control."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from autoluthier.dsp.base import StageContext
from autoluthier.dsp.stereo import StereoParams, StereoStage

SR = 44_100


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1).astype(np.float32)


class TestStereoParams:
    def test_default_width_is_unity(self) -> None:
        assert StereoParams().width == 1.0

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            StereoParams(bogus=1)  # type: ignore[call-arg]

    def test_rejects_negative_width(self) -> None:
        with pytest.raises(ValidationError):
            StereoParams(width=-0.1)

    def test_rejects_width_above_two(self) -> None:
        with pytest.raises(ValidationError):
            StereoParams(width=2.1)


class TestStereoStage:
    def test_mono_passes_through_unchanged(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        stage = StereoStage(StereoParams(width=0.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_array_equal(result, audio)

    def test_unity_width_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        audio = _stereo(rng.uniform(-0.5, 0.5, 100), rng.uniform(-0.5, 0.5, 100))
        stage = StereoStage(StereoParams(width=1.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_zero_width_collapses_to_mono_on_both_channels(self) -> None:
        left = np.array([1.0, 0.5, -0.2], dtype=np.float32)
        right = np.array([-1.0, 0.5, 0.2], dtype=np.float32)
        audio = _stereo(left, right)
        stage = StereoStage(StereoParams(width=0.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        mid = (left.astype(np.float64) + right.astype(np.float64)) / 2.0
        np.testing.assert_allclose(result[:, 0], mid, atol=1e-6)
        np.testing.assert_allclose(result[:, 1], mid, atol=1e-6)

    def test_double_width_doubles_the_side_signal(self) -> None:
        left = np.array([1.0, 0.0], dtype=np.float32)
        right = np.array([0.0, 0.0], dtype=np.float32)
        audio = _stereo(left, right)
        stage = StereoStage(StereoParams(width=2.0))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        # mid=0.5, side=0.5 -> width=2 doubles side to 1.0 -> L=1.5, R=-0.5
        assert result[0, 0] == pytest.approx(1.5, abs=1e-6)
        assert result[0, 1] == pytest.approx(-0.5, abs=1e-6)

    def test_mid_preserved_regardless_of_width(self) -> None:
        rng = np.random.default_rng(1)
        left = rng.uniform(-0.5, 0.5, 200)
        right = rng.uniform(-0.5, 0.5, 200)
        audio = _stereo(left, right)
        for width in (0.0, 0.5, 1.0, 1.5, 2.0):
            result = StereoStage(StereoParams(width=width)).apply(
                audio, StageContext(sample_rate=SR)
            )
            mid_after = (result[:, 0].astype(np.float64) + result[:, 1].astype(np.float64)) / 2.0
            mid_before = (left + right) / 2.0
            np.testing.assert_allclose(mid_after, mid_before, atol=1e-5)

    def test_more_than_two_channels_raises(self) -> None:
        audio = np.zeros((10, 3), dtype=np.float32)
        stage = StereoStage(StereoParams())
        with pytest.raises(ValueError, match="mono or 2-channel"):
            stage.apply(audio, StageContext(sample_rate=SR))

    def test_output_dtype_is_float32(self) -> None:
        audio = _stereo(np.array([0.1]), np.array([0.2]))
        stage = StereoStage(StereoParams(width=1.5))
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.dtype == np.float32
