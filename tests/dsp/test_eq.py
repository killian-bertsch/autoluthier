"""Tests for dsp.eq: biquad magnitude response at known frequencies, via sosfreqz."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.signal import sosfreqz

from autosampler.dsp.base import StageContext
from autosampler.dsp.eq import EqParams, EqStage, _biquad_sos

SR = 48_000


def _magnitude_db(sos: np.ndarray, freq_hz: float, sample_rate: int) -> float:
    w = 2.0 * np.pi * freq_hz / sample_rate
    _, h = sosfreqz(sos, worN=[w])
    return float(20.0 * np.log10(np.abs(h[0])))


class TestBiquadMagnitudeResponse:
    def test_lowpass_passes_dc_attenuates_above_corner(self) -> None:
        sos = _biquad_sos("lowpass", freq_hz=1_000.0, q=0.7071, gain_db=0.0, sample_rate=SR)
        assert _magnitude_db(sos, 20.0, SR) == pytest.approx(0.0, abs=0.5)
        assert _magnitude_db(sos, 10_000.0, SR) < -20.0

    def test_highpass_attenuates_dc_passes_above_corner(self) -> None:
        sos = _biquad_sos("highpass", freq_hz=1_000.0, q=0.7071, gain_db=0.0, sample_rate=SR)
        assert _magnitude_db(sos, 20.0, SR) < -20.0
        assert _magnitude_db(sos, 10_000.0, SR) == pytest.approx(0.0, abs=0.5)

    def test_lowpass_corner_is_near_minus_3db(self) -> None:
        sos = _biquad_sos("lowpass", freq_hz=1_000.0, q=0.7071, gain_db=0.0, sample_rate=SR)
        assert _magnitude_db(sos, 1_000.0, SR) == pytest.approx(-3.0, abs=0.5)

    def test_peaking_boosts_at_center_leaves_far_bands_flat(self) -> None:
        sos = _biquad_sos("peaking", freq_hz=1_000.0, q=1.0, gain_db=6.0, sample_rate=SR)
        assert _magnitude_db(sos, 1_000.0, SR) == pytest.approx(6.0, abs=0.2)
        assert _magnitude_db(sos, 20.0, SR) == pytest.approx(0.0, abs=0.5)
        assert _magnitude_db(sos, SR / 2 - 100, SR) == pytest.approx(0.0, abs=0.5)

    def test_peaking_cut(self) -> None:
        sos = _biquad_sos("peaking", freq_hz=1_000.0, q=1.0, gain_db=-6.0, sample_rate=SR)
        assert _magnitude_db(sos, 1_000.0, SR) == pytest.approx(-6.0, abs=0.2)

    def test_low_shelf_boosts_dc_leaves_highs_flat(self) -> None:
        sos = _biquad_sos("low_shelf", freq_hz=200.0, q=0.7071, gain_db=6.0, sample_rate=SR)
        assert _magnitude_db(sos, 20.0, SR) == pytest.approx(6.0, abs=0.3)
        assert _magnitude_db(sos, 20_000.0, SR) == pytest.approx(0.0, abs=0.3)

    def test_high_shelf_boosts_highs_leaves_dc_flat(self) -> None:
        sos = _biquad_sos("high_shelf", freq_hz=5_000.0, q=0.7071, gain_db=6.0, sample_rate=SR)
        assert _magnitude_db(sos, 20.0, SR) == pytest.approx(0.0, abs=0.3)
        assert _magnitude_db(sos, 20_000.0, SR) == pytest.approx(6.0, abs=0.3)

    def test_freq_at_nyquist_raises(self) -> None:
        with pytest.raises(ValueError, match="Nyquist"):
            _biquad_sos("lowpass", freq_hz=SR / 2.0, q=0.7071, gain_db=0.0, sample_rate=SR)

    def test_freq_above_nyquist_raises(self) -> None:
        with pytest.raises(ValueError, match="Nyquist"):
            _biquad_sos("lowpass", freq_hz=SR, q=0.7071, gain_db=0.0, sample_rate=SR)


class TestEqParams:
    def test_defaults(self) -> None:
        params = EqParams()
        assert params.filter_type == "peaking"

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            EqParams(bogus=1)  # type: ignore[call-arg]

    def test_rejects_invalid_filter_type(self) -> None:
        with pytest.raises(ValidationError):
            EqParams(filter_type="bandpass")  # type: ignore[arg-type]


class TestEqStage:
    def test_applies_filter_to_signal(self) -> None:
        n = 4_800
        t = np.arange(n) / SR
        audio = (0.5 * np.sin(2 * np.pi * 5_000.0 * t)).astype(np.float32)
        params = EqParams(filter_type="lowpass", freq_hz=500.0, q=0.7071)
        stage = EqStage(params)
        result = stage.apply(audio, StageContext(sample_rate=SR))
        # A 5kHz tone through a 500Hz lowpass should end up much quieter.
        assert float(np.std(result)) < float(np.std(audio)) * 0.1

    def test_empty_input_returned_unchanged(self) -> None:
        stage = EqStage(EqParams())
        result = stage.apply(np.zeros(0, dtype=np.float32), StageContext(sample_rate=SR))
        assert result.shape[0] == 0

    def test_stereo_shape_preserved(self) -> None:
        audio = np.random.default_rng(0).uniform(-0.5, 0.5, (1000, 2)).astype(np.float32)
        stage = EqStage(EqParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.shape == audio.shape

    def test_output_dtype_is_float32(self) -> None:
        audio = np.random.default_rng(0).uniform(-0.5, 0.5, 1000).astype(np.float32)
        stage = EqStage(EqParams())
        result = stage.apply(audio, StageContext(sample_rate=SR))
        assert result.dtype == np.float32

    def test_caches_sos_per_sample_rate(self) -> None:
        stage = EqStage(EqParams())
        audio = np.zeros(10, dtype=np.float32)
        stage.apply(audio, StageContext(sample_rate=SR))
        stage.apply(audio, StageContext(sample_rate=SR))
        assert len(stage._sos_cache) == 1
        stage.apply(audio, StageContext(sample_rate=22_050))
        assert len(stage._sos_cache) == 2
