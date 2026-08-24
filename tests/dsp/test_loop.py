"""Tests for dsp.loop: zero-crossing search, the 0.65/0.35 score, bug 7, baked crossfade."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from autosampler.config.schema import CrossfadeConfig
from autosampler.domain.models import Sample, SampleSet
from autosampler.domain.units import ms_to_frames
from autosampler.dsp.loop import (
    LoopParams,
    LoopPoints,
    apply_loop,
    bake_crossfade,
    crossfade_frames,
    fade_curves,
    find_loop_points,
    upward_zero_crossings,
    zero_crossings,
)

SR = 44_100


def _sine(freq: float, n: int, amplitude: float = 0.5) -> NDArray[np.float32]:
    t = np.arange(n) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _decaying_sine(freq: float, n: int, decay_db: float = -24.0) -> NDArray[np.float32]:
    t = np.arange(n) / SR
    envelope = 10.0 ** (decay_db / 20.0 * (np.arange(n) / n))
    return (0.5 * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _naive_upward_crossings(mono: NDArray[np.float64], start: int, stop: int) -> list[int]:
    """V1's per-frame Python scan, kept as the reference for the vectorized version."""
    region = mono[start:stop]
    return [start + i for i in range(1, len(region)) if region[i - 1] < 0.0 and region[i] >= 0.0]


def _rms(audio: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _v1_region_end_frac(crossfade_ms: float, n: int, sample_rate: int) -> float:
    """V1's crossfade-coupled window end — the formula that caused bug 7."""
    crossfade_samples = int(crossfade_ms / 1000.0 * sample_rate)
    tail_frac = max(crossfade_samples / n, 0.03)
    return min(1.0 - tail_frac, 0.97)


class TestLoopParams:
    def test_defaults_match_v1_constants(self) -> None:
        params = LoopParams()
        assert params.region_start_frac == 0.85
        assert params.region_end_frac == 0.97
        assert params.min_loop_ms == 50.0

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            LoopParams(bogus=1)  # type: ignore[call-arg]

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(ValidationError, match="must be below"):
            LoopParams(region_start_frac=0.9, region_end_frac=0.5)

    def test_rejects_equal_window_bounds(self) -> None:
        with pytest.raises(ValidationError, match="must be below"):
            LoopParams(region_start_frac=0.9, region_end_frac=0.9)

    def test_rejects_out_of_range_fraction(self) -> None:
        with pytest.raises(ValidationError):
            LoopParams(region_start_frac=1.0)

    def test_has_no_crossfade_field(self) -> None:
        """Crossfade length lives once, in CrossfadeConfig — never duplicated here."""
        assert "loop_crossfade_ms" not in LoopParams.model_fields
        assert not any("crossfade" in name for name in LoopParams.model_fields)


class TestZeroCrossings:
    """`zero_crossings` vs. `upward_zero_crossings` over the whole sample.

    It must not disagree with the function the loop finder itself uses.
    """

    def test_mono_matches_full_range_upward_crossings(self) -> None:
        audio = _sine(220.0, 2000)
        expected = upward_zero_crossings(audio.astype(np.float64), 0, audio.size)
        assert zero_crossings(audio).tolist() == expected.tolist()

    def test_stereo_matches_channel_mean(self) -> None:
        left = _sine(220.0, 2000)
        right = _sine(220.0, 2000, amplitude=0.3)
        stereo = np.stack([left, right], axis=1)
        mono = stereo.astype(np.float64).mean(axis=1)
        expected = upward_zero_crossings(mono, 0, mono.size)
        assert zero_crossings(stereo).tolist() == expected.tolist()

    def test_returns_ascending_indices(self) -> None:
        audio = _sine(440.0, 5000)
        indices = zero_crossings(audio)
        assert indices.tolist() == sorted(indices.tolist())
        assert indices.size > 0


class TestUpwardZeroCrossings:
    def test_hand_checked_array(self) -> None:
        mono = np.array([-1.0, -0.5, 0.5, 1.0, -0.5, 0.25], dtype=np.float64)
        assert upward_zero_crossings(mono, 0, mono.size).tolist() == [2, 5]

    def test_exact_zero_counts_as_upward(self) -> None:
        mono = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        assert upward_zero_crossings(mono, 0, mono.size).tolist() == [1]

    def test_downward_crossing_ignored(self) -> None:
        mono = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float64)
        assert upward_zero_crossings(mono, 0, mono.size).size == 0

    def test_matches_naive_reference_on_noise(self) -> None:
        rng = np.random.default_rng(20250819)
        mono = rng.standard_normal(50_000)
        vectorized = upward_zero_crossings(mono, 10_000, 40_000).tolist()
        assert vectorized == _naive_upward_crossings(mono, 10_000, 40_000)

    def test_matches_naive_reference_on_sine(self) -> None:
        mono = _sine(220.0, 44_100).astype(np.float64)
        vectorized = upward_zero_crossings(mono, 37_485, 42_777).tolist()
        assert vectorized == _naive_upward_crossings(mono, 37_485, 42_777)

    def test_region_shorter_than_two_frames_is_empty(self) -> None:
        mono = np.array([-1.0, 1.0], dtype=np.float64)
        assert upward_zero_crossings(mono, 1, 2).size == 0


class TestFindLoopPoints:
    def test_points_land_on_upward_zero_crossings(self) -> None:
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        mono = audio.astype(np.float64)
        for idx in (points.start, points.end):
            assert mono[idx - 1] < 0.0
            assert mono[idx] >= 0.0

    def test_points_lie_inside_the_search_window(self) -> None:
        audio = _sine(100.0, SR)
        params = LoopParams()
        points = find_loop_points(audio, SR, params)
        assert points is not None
        assert int(SR * params.region_start_frac) <= points.start
        assert points.end < int(SR * params.region_end_frac)

    def test_respects_min_loop_ms(self) -> None:
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams(min_loop_ms=50.0))
        assert points is not None
        assert points.end - points.start >= 0.050 * SR

    def test_none_when_sample_too_short_for_min_gap(self) -> None:
        assert find_loop_points(_sine(100.0, 1_000), SR, LoopParams()) is None

    def test_none_on_silence(self) -> None:
        assert find_loop_points(np.zeros(SR, dtype=np.float32), SR, LoopParams()) is None

    def test_none_on_empty_buffer(self) -> None:
        assert find_loop_points(np.zeros(0, dtype=np.float32), SR, LoopParams()) is None

    def test_stereo_analyzed_as_channel_mean(self) -> None:
        left = _sine(100.0, SR)
        right = _sine(150.0, SR)
        stereo = np.stack([left, right], axis=1)
        points = find_loop_points(stereo, SR, LoopParams())
        assert points is not None
        mono = stereo.astype(np.float64).mean(axis=1)
        assert mono[points.start - 1] < 0.0
        assert mono[points.start] >= 0.0

    def test_prefers_amplitude_matched_pair_over_mismatched_one(self) -> None:
        """A tail whose level collapses mid-window shouldn't win on amplitude similarity."""
        n = SR
        audio = _sine(100.0, n)
        # Attenuate the last third of the search window so its crossings are far quieter.
        audio[int(n * 0.93) :] *= np.float32(0.02)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        assert points.end < int(n * 0.93)


class TestCrossfadeFrames:
    def test_exact_when_it_fits(self) -> None:
        points = LoopPoints(start=30_000, end=40_000)
        assert crossfade_frames(points, 20.0, SR, headroom_frames=points.start) == 882

    def test_zero_ms_gives_zero_frames(self) -> None:
        points = LoopPoints(start=30_000, end=40_000)
        assert crossfade_frames(points, 0.0, SR, headroom_frames=points.start) == 0

    def test_clamped_to_loop_length(self) -> None:
        points = LoopPoints(start=30_000, end=32_205)
        assert crossfade_frames(points, 500.0, SR, headroom_frames=points.start) == 2_205

    def test_clamped_to_headroom_before_loop_start(self) -> None:
        """Baked mode's headroom: the material it fades with sits before loop_start."""
        points = LoopPoints(start=100, end=50_000)
        assert crossfade_frames(points, 500.0, SR, headroom_frames=points.start) == 100

    def test_clamped_to_headroom_after_loop_end(self) -> None:
        """SFZ mode's headroom: the sampler fades with the tail after loop_end."""
        points = LoopPoints(start=30_000, end=40_000)
        n_frames = 40_200
        assert (
            crossfade_frames(
                points, 500.0, SR, headroom_frames=n_frames - points.end
            )
            == 200
        )

    def test_negative_headroom_gives_zero_not_a_negative_length(self) -> None:
        points = LoopPoints(start=0, end=10_000)
        assert crossfade_frames(points, 20.0, SR, headroom_frames=-5) == 0


class TestFadeCurves:
    def test_linear_sums_to_unity(self) -> None:
        """The level-preserving property for correlated material: fade_in + fade_out == 1."""
        fade_in, fade_out = fade_curves(882, "linear")
        np.testing.assert_allclose(fade_in + fade_out, 1.0, atol=1e-12)

    def test_equal_power_squares_sum_to_unity(self) -> None:
        """The power-preserving property for decorrelated material: sin^2 + cos^2 == 1."""
        fade_in, fade_out = fade_curves(882, "equal_power")
        np.testing.assert_allclose(fade_in**2 + fade_out**2, 1.0, atol=1e-12)

    @pytest.mark.parametrize("shape", ["linear", "equal_power"])
    def test_reaches_full_crossover_on_the_final_frame(self, shape: str) -> None:
        """The endpoint that makes the loop wrap sample-continuous."""
        fade_in, fade_out = fade_curves(882, shape)  # type: ignore[arg-type]
        assert fade_in[-1] == pytest.approx(1.0, abs=1e-12)
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("shape", ["linear", "equal_power"])
    def test_monotonic(self, shape: str) -> None:
        fade_in, fade_out = fade_curves(882, shape)  # type: ignore[arg-type]
        assert np.all(np.diff(fade_in) > 0.0)
        assert np.all(np.diff(fade_out) < 0.0)

    def test_default_shape_is_linear(self) -> None:
        """Correlated summands are the normal case here, so linear is the config default."""
        assert CrossfadeConfig().loop_crossfade_shape == "linear"


class TestCorrelatedMaterialLevel:
    """Why linear is the default: the loop finder selects for correlation on purpose."""

    def _segments(self) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        frames = crossfade_frames(points, 20.0, SR, headroom_frames=points.start)
        tail = audio[points.end - frames : points.end].astype(np.float64)
        pre = audio[points.start - frames : points.start].astype(np.float64)
        return tail, pre, frames

    def test_the_two_crossfaded_segments_are_fully_correlated(self) -> None:
        tail, pre, _ = self._segments()
        correlation = float(
            np.dot(tail, pre) / (np.linalg.norm(tail) * np.linalg.norm(pre))
        )
        assert correlation == pytest.approx(1.0, abs=1e-4)

    def test_linear_holds_the_level_while_equal_power_lifts_it(self) -> None:
        tail, pre, frames = self._segments()
        deviations = {}
        for shape in ("linear", "equal_power"):
            fade_in, fade_out = fade_curves(frames, shape)  # type: ignore[arg-type]
            mixed = tail * fade_out + pre * fade_in
            deviations[shape] = 20.0 * np.log10(
                float(np.max(np.abs(mixed))) / float(np.max(np.abs(pre)))
            )

        assert deviations["linear"] == pytest.approx(0.0, abs=0.1)
        assert deviations["equal_power"] > 2.5  # approaches +3 dB, the sqrt(2) sum


class TestBakeCrossfade:
    def test_last_loop_frame_equals_frame_before_loop_start(self) -> None:
        """The crossover completes exactly on the loop's final frame — the seam guarantee."""
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)
        assert frames > 0
        assert out[points.end - 1] == pytest.approx(audio[points.start - 1], abs=1e-6)

    def test_start_of_fade_is_still_essentially_the_original_tail(self) -> None:
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)
        first = points.end - frames
        assert out[first] == pytest.approx(audio[first], abs=2e-3)

    def test_only_the_fade_region_is_modified(self) -> None:
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)
        first = points.end - frames
        np.testing.assert_array_equal(out[:first], audio[:first])
        np.testing.assert_array_equal(out[points.end :], audio[points.end :])

    def test_zero_length_crossfade_leaves_audio_untouched(self) -> None:
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=0.0, sample_rate=SR)
        assert frames == 0
        np.testing.assert_array_equal(out, audio)

    def test_stereo_channels_faded_independently(self) -> None:
        left = _decaying_sine(100.0, SR)
        right = _decaying_sine(150.0, SR)
        stereo = np.stack([left, right], axis=1)
        points = find_loop_points(stereo, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(stereo, points, crossfade_ms=20.0, sample_rate=SR)
        assert frames > 0
        assert out.shape == stereo.shape
        for channel in (0, 1):
            assert out[points.end - 1, channel] == pytest.approx(
                stereo[points.start - 1, channel], abs=1e-6
            )

    @pytest.mark.parametrize("shape", ["linear", "equal_power"])
    def test_both_shapes_complete_the_crossover(self, shape: str) -> None:
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(
            audio,
            points,
            crossfade_ms=20.0,
            sample_rate=SR,
            shape=shape,  # type: ignore[arg-type]
        )
        assert frames > 0
        assert out[points.end - 1] == pytest.approx(audio[points.start - 1], abs=1e-6)

    def test_output_stays_float32(self) -> None:
        audio = _decaying_sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, _ = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)
        assert out.dtype == np.float32


class TestSeamContinuity:
    def test_pure_sine_loops_with_no_discontinuity(self) -> None:
        """Two concatenated loop passes have no frame-to-frame step larger than the sine's own."""
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, _ = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)

        loop = out[points.start : points.end].astype(np.float64)
        played = np.concatenate([loop, loop])
        interior_max_step = float(np.max(np.abs(np.diff(loop))))
        assert float(np.max(np.abs(np.diff(played)))) <= interior_max_step * 1.05

    def test_decaying_sine_level_matches_across_the_seam(self) -> None:
        """A decaying tail is quieter than the loop start until the crossfade equalizes them."""
        audio = _decaying_sine(100.0, SR, decay_db=-24.0)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        window = 441

        raw_ratio = _rms(audio[points.end - window : points.end]) / _rms(
            audio[points.start : points.start + window]
        )
        assert raw_ratio < 0.95  # the discontinuity the baked crossfade exists to remove

        out, _ = bake_crossfade(audio, points, crossfade_ms=20.0, sample_rate=SR)
        baked_ratio = _rms(out[points.end - window : points.end]) / _rms(
            out[points.start : points.start + window]
        )
        assert baked_ratio == pytest.approx(1.0, abs=0.02)


class TestBug7LongCrossfade:
    def test_v1_window_formula_would_have_collapsed(self) -> None:
        """Documents the bug: V1's window end drops below its window start past 15%."""
        params = LoopParams()
        v1_end = _v1_region_end_frac(crossfade_ms=300.0, n=SR, sample_rate=SR)
        assert v1_end < params.region_start_frac

    def test_crossfade_past_fifteen_percent_still_finds_a_loop(self) -> None:
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=300.0, sample_rate=SR)
        assert frames > 0
        assert out[points.end - 1] == pytest.approx(audio[points.start - 1], abs=1e-6)

    def test_crossfade_at_half_the_sample_length_still_yields_a_valid_loop(self) -> None:
        n = SR
        audio = _sine(100.0, n)
        crossfade_ms = 1000.0 * (n // 2) / SR

        points = find_loop_points(audio, SR, LoopParams())
        assert points is not None
        out, frames = bake_crossfade(audio, points, crossfade_ms=crossfade_ms, sample_rate=SR)

        assert frames == points.end - points.start  # clamped to the loop, not disabled
        assert out[points.end - 1] == pytest.approx(audio[points.start - 1], abs=1e-6)

    def test_search_window_is_independent_of_crossfade_length(self) -> None:
        """The whole point of the fix: detection no longer sees the crossfade at all."""
        audio = _sine(100.0, SR)
        points = find_loop_points(audio, SR, LoopParams())
        for crossfade_ms in (0.0, 20.0, 300.0, 500.0):
            _, frames = bake_crossfade(
                audio, points, crossfade_ms=crossfade_ms, sample_rate=SR
            )
            assert find_loop_points(audio, SR, LoopParams()) == points
            assert frames <= points.end - points.start


class TestApplyLoop:
    def test_sets_loop_points_and_bakes_audio(self) -> None:
        sample = Sample(
            note=60, velocity=100, audio=_decaying_sine(100.0, SR), sample_rate=SR
        )
        original = sample.audio.copy()
        apply_loop(SampleSet([sample]), LoopParams(), crossfade_ms=20.0)

        assert sample.loop_start is not None
        assert sample.loop_end is not None
        assert sample.loop_end > sample.loop_start
        assert not np.array_equal(sample.audio, original)
        assert sample.audio[sample.loop_end - 1] == pytest.approx(
            original[sample.loop_start - 1], abs=1e-6
        )

    def test_undetectable_sample_keeps_none_and_untouched_audio(self) -> None:
        sample = Sample(
            note=60, velocity=100, audio=np.zeros(SR, dtype=np.float32), sample_rate=SR
        )
        original = sample.audio.copy()
        apply_loop(SampleSet([sample]), LoopParams(), crossfade_ms=20.0)

        assert sample.loop_start is None
        assert sample.loop_end is None
        np.testing.assert_array_equal(sample.audio, original)

    def test_processes_every_sample_in_the_set(self) -> None:
        samples = [
            Sample(note=note, velocity=100, audio=_sine(100.0, SR), sample_rate=SR)
            for note in (60, 62, 64)
        ]
        apply_loop(SampleSet(samples), LoopParams(), crossfade_ms=20.0)
        assert all(s.loop_start is not None and s.loop_end is not None for s in samples)

    def test_empty_set_is_a_no_op(self) -> None:
        apply_loop(SampleSet([]), LoopParams(), crossfade_ms=20.0)

    def test_crossfade_shape_is_threaded_through(self) -> None:
        """`apply_loop` must honor the shape, not silently bake the default curve."""
        baked = {}
        for shape in ("linear", "equal_power"):
            sample = Sample(note=60, velocity=100, audio=_sine(100.0, SR), sample_rate=SR)
            apply_loop(
                SampleSet([sample]),
                LoopParams(),
                crossfade_ms=20.0,
                crossfade_shape=shape,  # type: ignore[arg-type]
            )
            assert sample.loop_end is not None
            baked[shape] = sample.audio[sample.loop_end - 441 : sample.loop_end].copy()

        assert not np.allclose(baked["linear"], baked["equal_power"], atol=1e-4)

    def test_default_shape_holds_the_level_through_the_loop(self) -> None:
        """End-to-end: no level surge anywhere inside the baked loop of a steady tone."""
        sample = Sample(note=60, velocity=100, audio=_sine(100.0, SR), sample_rate=SR)
        apply_loop(SampleSet([sample]), LoopParams(), crossfade_ms=20.0)

        assert sample.loop_start is not None
        assert sample.loop_end is not None
        loop = sample.audio[sample.loop_start : sample.loop_end]
        assert float(np.max(np.abs(loop))) == pytest.approx(0.5, abs=0.005)


class TestSfzCrossfadeMode:
    """`sfz` mode hands the fade to the sampler: audio untouched, a length for the opcode."""

    def _sample(self) -> Sample:
        return Sample(note=60, velocity=100, audio=_sine(100.0, SR), sample_rate=SR)

    def test_default_mode_is_baked(self) -> None:
        assert CrossfadeConfig().loop_crossfade_mode == "baked"

    def test_audio_is_left_bit_identical(self) -> None:
        sample = self._sample()
        original = sample.audio.copy()
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=20.0, crossfade_mode="sfz"
        )
        np.testing.assert_array_equal(sample.audio, original)

    def test_loop_points_still_detected(self) -> None:
        sample = self._sample()
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=20.0, crossfade_mode="sfz"
        )
        assert sample.loop_start is not None
        assert sample.loop_end is not None

    def test_records_the_requested_length_when_the_tail_allows_it(self) -> None:
        sample = self._sample()
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=20.0, crossfade_mode="sfz"
        )
        assert sample.loop_crossfade_frames == 882  # 20 ms at 44.1 kHz, honored exactly

    def test_clamped_to_the_loop_length(self) -> None:
        """A fade longer than the loop makes no sense in either mode."""
        sample = self._sample()
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=5_000.0, crossfade_mode="sfz"
        )
        assert sample.loop_start is not None
        assert sample.loop_end is not None
        assert sample.loop_crossfade_frames == sample.loop_end - sample.loop_start

    def test_clamped_to_the_tail_never_claiming_more_than_exists(self) -> None:
        """The opcode must never promise a longer fade than the audio behind loop_end.

        A long loop reaching close to the end of the sample makes the tail the tighter bound
        rather than the loop length — the situation V1 tried to prevent by shrinking the search
        window, and the one where a naively-emitted opcode would over-claim.
        """
        sample = self._sample()
        params = LoopParams(region_start_frac=0.5, region_end_frac=1.0, min_loop_ms=400.0)
        apply_loop(
            SampleSet([sample]), params, crossfade_ms=5_000.0, crossfade_mode="sfz"
        )
        assert sample.loop_start is not None
        assert sample.loop_end is not None
        tail = sample.n_frames - sample.loop_end
        assert tail < sample.loop_end - sample.loop_start  # tail is the tighter bound
        assert sample.loop_crossfade_frames == tail
        assert sample.loop_crossfade_frames > 0

    def test_over_long_request_still_yields_a_valid_loop(self) -> None:
        """Bug 7 must not return through the sfz path: clamp the fade, keep the loop."""
        sample = self._sample()
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=5_000.0, crossfade_mode="sfz"
        )
        assert sample.loop_start is not None
        assert sample.loop_end is not None
        assert sample.loop_end - sample.loop_start >= 0.050 * SR

    def test_detection_identical_to_baked_mode(self) -> None:
        """Mode changes what happens to the fade, never where the loop is."""
        baked, delegated = self._sample(), self._sample()
        apply_loop(SampleSet([baked]), LoopParams(), crossfade_ms=20.0)
        apply_loop(
            SampleSet([delegated]), LoopParams(), crossfade_ms=20.0, crossfade_mode="sfz"
        )
        assert (baked.loop_start, baked.loop_end) == (
            delegated.loop_start,
            delegated.loop_end,
        )

    def test_default_region_end_honors_a_three_percent_crossfade(self) -> None:
        """Documents the headroom the 0.97 default guarantees: 3% of the sample, exactly."""
        n = 10 * SR
        sample = Sample(note=60, velocity=100, audio=_sine(100.0, n), sample_rate=SR)
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=300.0, crossfade_mode="sfz"
        )
        assert sample.loop_crossfade_frames == ms_to_frames(300.0, SR)

    def test_baked_mode_records_the_frames_it_baked(self) -> None:
        sample = self._sample()
        apply_loop(SampleSet([sample]), LoopParams(), crossfade_ms=20.0)
        assert sample.loop_crossfade_frames == 882

    def test_undetectable_sample_records_no_crossfade(self) -> None:
        sample = Sample(
            note=60, velocity=100, audio=np.zeros(SR, dtype=np.float32), sample_rate=SR
        )
        apply_loop(
            SampleSet([sample]), LoopParams(), crossfade_ms=20.0, crossfade_mode="sfz"
        )
        assert sample.loop_crossfade_frames == 0

    def test_shape_is_irrelevant_in_sfz_mode(self) -> None:
        """The curve is the sampler's choice there, so it must not change our output."""
        results = []
        for shape in ("linear", "equal_power"):
            sample = self._sample()
            apply_loop(
                SampleSet([sample]),
                LoopParams(),
                crossfade_ms=20.0,
                crossfade_mode="sfz",
                crossfade_shape=shape,  # type: ignore[arg-type]
            )
            results.append((sample.audio.copy(), sample.loop_crossfade_frames))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
