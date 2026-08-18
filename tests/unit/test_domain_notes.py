"""Tests for domain.notes: velocity list, recorded-note grid, MIDI<->note-name."""

from __future__ import annotations

import pytest

from autosampler.domain.notes import (
    midi_to_note_name,
    note_name_to_midi,
    recorded_notes,
    velocity_list,
)


class TestVelocityList:
    """Exact velocity lists, verified against V1's ``_compute_velocities`` output."""

    def test_single_layer(self) -> None:
        assert velocity_list(1) == [127]

    def test_two_layers(self) -> None:
        assert velocity_list(2) == [1, 127]

    def test_four_layers(self) -> None:
        assert velocity_list(4) == [1, 43, 85, 127]

    def test_eighteen_layers(self) -> None:
        assert velocity_list(18) == [
            1, 8, 16, 23, 31, 38, 45, 53, 60, 68, 75, 83, 90, 97, 105, 112, 120, 127,
        ]

    @pytest.mark.parametrize("layer_count", [1, 2, 3, 4, 5, 8, 18, 32])
    def test_endpoints_and_monotonic(self, layer_count: int) -> None:
        velocities = velocity_list(layer_count)
        assert len(velocities) == layer_count
        assert velocities[0] == (1 if layer_count > 1 else 127)
        assert velocities[-1] == 127
        assert velocities == sorted(velocities)
        assert all(1 <= v <= 127 for v in velocities)


class TestRecordedNotes:
    def test_every_semitone(self) -> None:
        assert recorded_notes(60, 72, 1) == list(range(60, 73))

    def test_interval_3(self) -> None:
        assert recorded_notes(60, 72, 3) == [60, 63, 66, 69, 72]

    def test_single_note(self) -> None:
        assert recorded_notes(60, 60, 1) == [60]

    def test_every_octave(self) -> None:
        notes = recorded_notes(21, 108, 12)
        assert 21 in notes
        assert all((n - 21) % 12 == 0 for n in notes)


class TestNoteNameConversion:
    @pytest.mark.parametrize(
        ("midi", "name"),
        [(60, "C4"), (69, "A4"), (21, "A0"), (108, "C8"), (61, "C#4"), (0, "C-1")],
    )
    def test_midi_to_note_name(self, midi: int, name: str) -> None:
        assert midi_to_note_name(midi) == name

    @pytest.mark.parametrize("midi", [0, 21, 60, 61, 69, 108, 127])
    def test_round_trip(self, midi: int) -> None:
        assert note_name_to_midi(midi_to_note_name(midi)) == midi

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid note name"):
            note_name_to_midi("H4")
