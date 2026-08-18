"""Tests for dsp.normalize: lufs/rms grouping, velocity-mode gain curve, bugs 1-3."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from autosampler.domain.models import Sample, SampleSet
from autosampler.dsp.normalize import (
    NormalizeParams,
    _meter_for,
    apply_normalize,
    apply_velocity_mode,
    normalize_by_velocity_layer,
)

SR = 44_100


def _sine_sample(
    note: int, velocity: int, freq: float, amplitude: float, n: int = 4_410
) -> Sample:
    t = np.arange(n) / SR
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return Sample(note=note, velocity=velocity, audio=audio, sample_rate=SR)


def _flat_sample(note: int, velocity: int, amplitude: float, n: int = 1_000) -> Sample:
    audio = np.full(n, amplitude, dtype=np.float32)
    return Sample(note=note, velocity=velocity, audio=audio, sample_rate=SR)


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms) if rms > 0.0 else -100.0


class TestMeterCache:
    def test_same_sample_rate_returns_same_meter(self) -> None:
        assert _meter_for(48_000) is _meter_for(48_000)

    def test_different_sample_rates_return_different_meters(self) -> None:
        assert _meter_for(22_050) is not _meter_for(96_000)


class TestNormalizeParams:
    def test_defaults(self) -> None:
        params = NormalizeParams()
        assert params.mode == "lufs"
        assert params.target_db == -18.0
        assert params.peak_ceiling_db == -1.0

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            NormalizeParams(bogus=1)  # type: ignore[call-arg]

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            NormalizeParams(mode="bogus")  # type: ignore[arg-type]


class TestNormalizeByVelocityLayer:
    def test_known_rms_sine_normalizes_to_target(self) -> None:
        """A sine of known RMS, normalized in rms mode, lands on the target within tolerance."""
        sample = _sine_sample(note=60, velocity=100, freq=100.0, amplitude=0.1)
        samples = SampleSet([sample])
        normalize_by_velocity_layer(samples, "rms", target_db=-6.0, peak_ceiling_db=0.0)

        result = samples.get(60, 100)
        assert result is not None
        assert _rms_db(result.audio) == pytest.approx(-6.0, abs=0.05)

    def test_groups_by_velocity_not_note(self) -> None:
        """Two different notes sharing a velocity land in the same normalization group."""
        quiet = _sine_sample(note=60, velocity=100, freq=100.0, amplitude=0.05)
        loud = _sine_sample(note=72, velocity=100, freq=100.0, amplitude=0.5)
        samples = SampleSet([quiet, loud])
        normalize_by_velocity_layer(samples, "rms", target_db=-6.0, peak_ceiling_db=0.0)

        # Group-mean normalization preserves the relative gap between the two notes.
        r_quiet = samples.get(60, 100)
        r_loud = samples.get(72, 100)
        assert r_quiet is not None
        assert r_loud is not None
        assert _rms_db(r_loud.audio) - _rms_db(r_quiet.audio) == pytest.approx(20.0, abs=0.1)

    def test_peak_ceiling_only_attenuates(self) -> None:
        """A group already under the ceiling is left untouched (never boosted)."""
        sample = _flat_sample(note=60, velocity=100, amplitude=0.1)
        samples = SampleSet([sample])
        normalize_by_velocity_layer(samples, "rms", target_db=-60.0, peak_ceiling_db=-1.0)
        result = samples.get(60, 100)
        assert result is not None
        # Target -60dB RMS would call for heavy attenuation, not boosting past the ceiling.
        assert float(np.max(np.abs(result.audio))) <= 10 ** (-1.0 / 20.0) + 1e-6

    def test_silent_group_left_unscaled(self) -> None:
        sample = _flat_sample(note=60, velocity=100, amplitude=0.0)
        samples = SampleSet([sample])
        normalize_by_velocity_layer(samples, "rms", target_db=-6.0, peak_ceiling_db=-1.0)
        result = samples.get(60, 100)
        assert result is not None
        assert np.all(result.audio == 0.0)

    def test_empty_set_is_noop(self) -> None:
        samples = SampleSet([])
        normalize_by_velocity_layer(samples, "rms", target_db=-6.0, peak_ceiling_db=-1.0)
        assert len(samples) == 0


class TestApplyVelocityMode:
    def test_range_matches_closed_form_on_synthetic_ramp(self) -> None:
        """The measured range must equal `clip(slope * 127, 0, 60)` from the two endpoints."""
        vel_min, vel_max = 1, 127
        peak_min_db, peak_max_db = -30.0, 0.0
        amp_min = 10 ** (peak_min_db / 20.0)
        amp_max = 10 ** (peak_max_db / 20.0)
        amp_mid = 10 ** (-15.0 / 20.0)

        sustain = SampleSet(
            [
                _flat_sample(note=60, velocity=vel_min, amplitude=amp_min),
                _flat_sample(note=60, velocity=64, amplitude=amp_mid),
                _flat_sample(note=60, velocity=vel_max, amplitude=amp_max),
            ]
        )
        release = SampleSet([])

        result = apply_velocity_mode(
            sustain, release, peak_ceiling_db=0.0, release_target_db=-18.0
        )

        slope = (peak_max_db - peak_min_db) / (vel_max - vel_min)
        expected_range_db = float(np.clip(slope * 127.0, 0.0, 60.0))
        assert result.dynamic_range_db == pytest.approx(expected_range_db, abs=1e-6)

    def test_endpoints_align_to_the_same_peak(self) -> None:
        """velocity=127 keeps its raw peak (0 dB gain); velocity=1 is boosted to match it."""
        amp_min, amp_max = 0.05, 0.9
        sustain = SampleSet(
            [
                _flat_sample(note=60, velocity=1, amplitude=amp_min),
                _flat_sample(note=60, velocity=127, amplitude=amp_max),
            ]
        )
        release = SampleSet([])
        apply_velocity_mode(sustain, release, peak_ceiling_db=0.0, release_target_db=-18.0)

        top = sustain.get(60, 127)
        bottom = sustain.get(60, 1)
        assert top is not None
        assert bottom is not None
        assert float(np.max(np.abs(top.audio))) == pytest.approx(amp_max, abs=1e-6)
        assert float(np.max(np.abs(bottom.audio))) == pytest.approx(amp_max, abs=1e-3)

    def test_bug1_single_layer_note_excluded_from_average(self) -> None:
        """A single-velocity-layer note used to contribute a 0.0 that dragged the mean down."""
        vel_min, vel_max = 1, 127
        peak_min_db, peak_max_db = -20.0, 0.0
        amp_min = 10 ** (peak_min_db / 20.0)
        amp_max = 10 ** (peak_max_db / 20.0)

        sustain = SampleSet(
            [
                _flat_sample(note=60, velocity=vel_min, amplitude=amp_min),
                _flat_sample(note=60, velocity=vel_max, amplitude=amp_max),
                _flat_sample(note=72, velocity=100, amplitude=0.3),  # single layer: degenerate
            ]
        )
        release = SampleSet([])
        result = apply_velocity_mode(
            sustain, release, peak_ceiling_db=0.0, release_target_db=-18.0
        )

        slope = (peak_max_db - peak_min_db) / (vel_max - vel_min)
        expected_range_db = float(np.clip(slope * 127.0, 0.0, 60.0))
        # If bug 1 were present, averaging in the degenerate note's 0.0 would halve this.
        assert result.dynamic_range_db == pytest.approx(expected_range_db, abs=1e-6)

    def test_bug2_release_uses_configured_target_not_hardcoded(self) -> None:
        release_sample = _sine_sample(note=60, velocity=100, freq=100.0, amplitude=0.1)
        sustain = SampleSet([_flat_sample(note=60, velocity=100, amplitude=0.3)])
        release = SampleSet([release_sample])

        apply_velocity_mode(sustain, release, peak_ceiling_db=0.0, release_target_db=-9.0)

        result = release.get(60, 100)
        assert result is not None
        assert _rms_db(result.audio) == pytest.approx(-9.0, abs=0.1)

    def test_bug3_ceiling_applied_per_note_not_globally(self) -> None:
        """A note whose loudest layer is under the ceiling is untouched by another note's.

        V1 applied one shared scale factor across all of sustain instead.
        """
        peak_ceiling_db = -3.0
        target_linear = 10 ** (peak_ceiling_db / 20.0)

        loud_note = SampleSet(
            [
                _flat_sample(note=60, velocity=1, amplitude=0.05),
                _flat_sample(note=60, velocity=127, amplitude=0.99),  # exceeds the ceiling
            ]
        )
        quiet_note = [
            _flat_sample(note=72, velocity=1, amplitude=0.05),
            _flat_sample(note=72, velocity=127, amplitude=0.5),  # already under the ceiling
        ]
        sustain = SampleSet(list(loud_note) + quiet_note)
        release = SampleSet([])

        apply_velocity_mode(
            sustain, release, peak_ceiling_db=peak_ceiling_db, release_target_db=-18.0
        )

        loud_top = sustain.get(60, 127)
        quiet_top = sustain.get(72, 127)
        assert loud_top is not None
        assert quiet_top is not None
        # note 60 got ceiling-limited...
        assert float(np.max(np.abs(loud_top.audio))) == pytest.approx(target_linear, abs=1e-6)
        # ...but note 72 was never touched by it (gain at v=127 is always 0 dB).
        assert float(np.max(np.abs(quiet_top.audio))) == pytest.approx(0.5, abs=1e-6)

    def test_no_layers_returns_zero_range(self) -> None:
        result = apply_velocity_mode(
            SampleSet([]), SampleSet([]), peak_ceiling_db=0.0, release_target_db=-18.0
        )
        assert result.dynamic_range_db == 0.0


class TestApplyNormalize:
    def test_velocity_mode_returns_result(self) -> None:
        sustain = SampleSet(
            [
                _flat_sample(note=60, velocity=1, amplitude=0.05),
                _flat_sample(note=60, velocity=127, amplitude=0.9),
            ]
        )
        release = SampleSet([])
        result = apply_normalize(sustain, release, NormalizeParams(mode="velocity"))
        assert result is not None

    def test_rms_mode_returns_none(self) -> None:
        sustain = SampleSet([_flat_sample(note=60, velocity=100, amplitude=0.1)])
        release = SampleSet([])
        result = apply_normalize(sustain, release, NormalizeParams(mode="rms"))
        assert result is None

    def test_lufs_mode_returns_none(self) -> None:
        sustain = SampleSet([_sine_sample(note=60, velocity=100, freq=100.0, amplitude=0.1)])
        release = SampleSet([])
        result = apply_normalize(sustain, release, NormalizeParams(mode="lufs"))
        assert result is None
