"""Tests for domain.models: Sample, SampleSet, InstrumentAudio."""

from __future__ import annotations

import numpy as np

from autosampler.domain.models import InstrumentAudio, Sample, SampleSet


def _sample(note: int, velocity: int, n_frames: int = 4, n_channels: int = 1) -> Sample:
    shape = (n_frames,) if n_channels == 1 else (n_frames, n_channels)
    audio = np.zeros(shape, dtype=np.float32)
    return Sample(note=note, velocity=velocity, audio=audio, sample_rate=44_100)


class TestSample:
    def test_n_frames_mono(self) -> None:
        assert _sample(60, 127, n_frames=100).n_frames == 100

    def test_n_channels_mono(self) -> None:
        assert _sample(60, 127).n_channels == 1

    def test_n_channels_stereo(self) -> None:
        assert _sample(60, 127, n_channels=2).n_channels == 2

    def test_loop_points_default_none(self) -> None:
        s = _sample(60, 127)
        assert s.loop_start is None
        assert s.loop_end is None


class TestSampleSet:
    def test_get_hit(self) -> None:
        s = _sample(60, 127)
        sample_set = SampleSet([s])
        assert sample_set.get(60, 127) is s

    def test_get_miss_returns_none(self) -> None:
        sample_set = SampleSet([_sample(60, 127)])
        assert sample_set.get(61, 127) is None

    def test_notes_sorted_unique(self) -> None:
        sample_set = SampleSet([_sample(72, 1), _sample(60, 1), _sample(60, 127)])
        assert sample_set.notes() == [60, 72]

    def test_velocities_sorted_unique(self) -> None:
        sample_set = SampleSet([_sample(60, 127), _sample(60, 1), _sample(72, 1)])
        assert sample_set.velocities() == [1, 127]

    def test_len(self) -> None:
        sample_set = SampleSet([_sample(60, 1), _sample(60, 127)])
        assert len(sample_set) == 2

    def test_iter_preserves_order(self) -> None:
        samples = [_sample(60, 1), _sample(72, 1), _sample(84, 1)]
        sample_set = SampleSet(samples)
        assert list(sample_set) == samples

    def test_empty(self) -> None:
        sample_set = SampleSet([])
        assert len(sample_set) == 0
        assert sample_set.notes() == []
        assert sample_set.get(60, 1) is None


class TestInstrumentAudio:
    def test_holds_sustain_and_release(self) -> None:
        sustain = SampleSet([_sample(60, 127)])
        release = SampleSet([])
        audio = InstrumentAudio(sustain=sustain, release=release)
        assert audio.sustain is sustain
        assert audio.release is release
