"""Tests for domain.units: sample-rate-relative duration/frame conversions."""

from __future__ import annotations

from autosampler.domain.units import frames_to_seconds, ms_to_frames, seconds_to_frames


class TestSecondsToFrames:
    def test_matches_v1_rounding(self) -> None:
        assert seconds_to_frames(1.0, 44_100) == 44_100
        assert seconds_to_frames(0.5, 22_050) == 11_025

    def test_rounds_to_nearest(self) -> None:
        assert seconds_to_frames(0.00001, 44_100) == round(0.00001 * 44_100)


class TestMsToFrames:
    def test_matches_seconds_equivalent(self) -> None:
        assert ms_to_frames(20.0, 48_000) == seconds_to_frames(0.02, 48_000)

    def test_zero(self) -> None:
        assert ms_to_frames(0.0, 44_100) == 0


class TestFramesToSeconds:
    def test_round_trip(self) -> None:
        sample_rate = 44_100
        frames = seconds_to_frames(2.5, sample_rate)
        assert abs(frames_to_seconds(frames, sample_rate) - 2.5) < 1e-6
