"""Tests for io.slicer: offsets against a synthesized marker render, stereo zero-fill."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from autosampler.io.slicer import slice_notes_velocities

SR = 22_050


def _marker_render(
    notes: list[int],
    velocities: list[int],
    hold_time: float,
    release_time: float,
    *,
    extract_release: bool = False,
    n_channels: int = 1,
    n_events: int | None = None,
) -> NDArray[np.float32]:
    """Build a synthetic render where every event slot holds a unique DC marker value.

    Mirrors V1's ``test_phase2.py::make_long_wav`` so slice offsets can be checked against
    a known-good marker rather than just shape.
    """
    frames_hold = round(hold_time * SR)
    frames_release = round(release_time * SR)
    event_duration = hold_time + release_time
    seg_len = frames_release if extract_release else frames_hold
    frames_per_event = round(event_duration * SR)

    total_events = n_events if n_events is not None else len(notes) * len(velocities)
    total_frames = total_events * frames_per_event
    shape = (total_frames,) if n_channels == 1 else (total_frames, n_channels)
    audio = np.zeros(shape, dtype=np.float32)

    event_index = 0
    for note in notes:
        for velocity in velocities:
            marker = (note * 1000 + velocity) / 1_000_000.0
            event_start = round(event_index * event_duration * SR)
            if n_channels == 1:
                audio[event_start : event_start + seg_len] = marker
            else:
                audio[event_start : event_start + seg_len, :] = marker
            event_index += 1

    return audio


class TestSliceOffsets:
    def test_correct_note_velocity_count(self) -> None:
        notes, velocities = [60, 72], [1, 127]
        audio = _marker_render(notes, velocities, hold_time=1.0, release_time=0.5)
        result = slice_notes_velocities(
            audio, SR, notes, velocities, hold_time=1.0, release_time=0.5,
            extract_release=False, min_note=60, max_note=72, collapse_to_mono=False,
        )
        assert len(result) == 4

    def test_min_max_note_filtering(self) -> None:
        notes, velocities = [60, 64, 68, 72], [127]
        audio = _marker_render(notes, velocities, hold_time=0.5, release_time=0.25)
        result = slice_notes_velocities(
            audio, SR, notes, velocities, hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=64, max_note=68, collapse_to_mono=False,
        )
        assert result.notes() == [64, 68]

    def test_sustain_slice_duration(self) -> None:
        hold = 2.0
        audio = _marker_render([60], [127], hold_time=hold, release_time=0.5)
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=hold, release_time=0.5,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.n_frames == round(hold * SR)

    def test_release_slice_duration(self) -> None:
        release = 1.5
        audio = _marker_render(
            [60], [127], hold_time=2.0, release_time=release, extract_release=True
        )
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=2.0, release_time=release,
            extract_release=True, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.n_frames == round(release * SR)

    def test_marker_offsets_match_expected_note_velocity(self) -> None:
        """Every extracted slice carries the marker for its own (note, velocity).

        Not a neighbor's — the direct check that slicing math lands on the right offsets.
        """
        notes, velocities = [60, 72, 84], [1, 64, 127]
        audio = _marker_render(notes, velocities, hold_time=0.3, release_time=0.1)
        result = slice_notes_velocities(
            audio, SR, notes, velocities, hold_time=0.3, release_time=0.1,
            extract_release=False, min_note=60, max_note=84, collapse_to_mono=False,
        )
        for note in notes:
            for velocity in velocities:
                sample = result.get(note, velocity)
                assert sample is not None
                expected_marker = (note * 1000 + velocity) / 1_000_000.0
                assert np.allclose(sample.audio, expected_marker)

    def test_event_index_advances_for_out_of_range_notes(self) -> None:
        """Notes outside [min_note, max_note] still consume an event slot.

        So later notes' offsets are unaffected by the filtering (no onset detection to
        compensate).
        """
        notes, velocities = [60, 64, 68], [127]
        audio = _marker_render(notes, velocities, hold_time=0.4, release_time=0.1)
        result = slice_notes_velocities(
            audio, SR, notes, velocities, hold_time=0.4, release_time=0.1,
            extract_release=False, min_note=68, max_note=68, collapse_to_mono=False,
        )
        sample = result.get(68, 127)
        assert sample is not None
        expected_marker = (68 * 1000 + 127) / 1_000_000.0
        assert np.allclose(sample.audio, expected_marker)

    def test_collapse_to_mono(self) -> None:
        audio = _marker_render([60], [127], hold_time=0.5, release_time=0.25, n_channels=2)
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=True,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.n_channels == 1

    def test_stereo_preserved_when_not_collapsing(self) -> None:
        audio = _marker_render([60], [127], hold_time=0.5, release_time=0.25, n_channels=2)
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.n_channels == 2

    def test_sample_rate_stored(self) -> None:
        audio = _marker_render([60], [127], hold_time=0.5, release_time=0.25)
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.sample_rate == SR

    def test_dtype_is_float32(self) -> None:
        audio = _marker_render([60], [127], hold_time=0.5, release_time=0.25)
        result = slice_notes_velocities(
            audio, SR, [60], [127], hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.audio.dtype == np.float32


class TestTruncatedRenderZeroFill:
    """V1 bug 6: a truncated render's out-of-bounds slice was always mono zeros.

    Even for stereo instruments — every fallback here must match the source's channel count.
    """

    def test_truncated_mono_render_pads_mono(self) -> None:
        notes, velocities = [60, 72], [127]
        full = _marker_render(notes, velocities, hold_time=0.5, release_time=0.25)
        truncated = full[: len(full) // 2]
        result = slice_notes_velocities(
            truncated, SR, notes, velocities, hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=72, collapse_to_mono=False,
        )
        sample = result.get(72, 127)
        assert sample is not None
        assert sample.n_channels == 1
        assert sample.n_frames == round(0.5 * SR)
        assert np.all(sample.audio == 0.0)

    def test_truncated_stereo_render_pads_stereo(self) -> None:
        notes, velocities = [60, 72], [127]
        full = _marker_render(
            notes, velocities, hold_time=0.5, release_time=0.25, n_channels=2
        )
        truncated = full[: len(full) // 2]
        result = slice_notes_velocities(
            truncated, SR, notes, velocities, hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=72, collapse_to_mono=False,
        )
        sample = result.get(72, 127)
        assert sample is not None
        assert sample.n_channels == 2
        assert sample.audio.shape == (round(0.5 * SR), 2)
        assert np.all(sample.audio == 0.0)

    def test_partially_truncated_slice_pads_with_correct_channel_count(self) -> None:
        """A slice that starts in-bounds but runs past EOF pads, not zero-fills wholesale.

        The pad must still match the source's channel count.
        """
        notes, velocities = [60], [127]
        full = _marker_render(
            notes, velocities, hold_time=0.5, release_time=0.25, n_channels=2
        )
        truncated = full[: round(0.5 * SR) - 100]
        result = slice_notes_velocities(
            truncated, SR, notes, velocities, hold_time=0.5, release_time=0.25,
            extract_release=False, min_note=60, max_note=60, collapse_to_mono=False,
        )
        sample = result.get(60, 127)
        assert sample is not None
        assert sample.audio.shape == (round(0.5 * SR), 2)
        assert np.all(sample.audio[-100:] == 0.0)
