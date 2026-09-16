"""Tests for domain.notes: velocity list, note grid, subset selection, MIDI<->note-name."""

from __future__ import annotations

import pytest

from autoluthier.domain.notes import (
    midi_to_note_name,
    note_name_to_midi,
    recorded_notes,
    select_evenly,
    select_notes,
    select_velocities,
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


class TestSelectEvenly:
    """Bug 8: V1 deduplicated the chosen indices through a set, which could return fewer."""

    @pytest.mark.parametrize("n", range(1, 41))
    def test_every_count_returns_exactly_that_many(self, n: int) -> None:
        items = list(range(n))
        for count in range(1, n + 1):
            assert len(select_evenly(items, count)) == count

    def test_first_and_last_are_always_kept(self) -> None:
        items = list(range(21, 109))
        for count in range(2, 20):
            chosen = select_evenly(items, count)
            assert chosen[0] == 21
            assert chosen[-1] == 108

    def test_result_keeps_the_input_order(self) -> None:
        chosen = select_evenly(list(range(100)), 7)
        assert chosen == sorted(chosen)

    def test_one_item_is_the_middle(self) -> None:
        assert select_evenly([1, 2, 3, 4, 5], 1) == [3]

    def test_count_above_the_input_size_returns_everything(self) -> None:
        assert select_evenly([1, 2, 3], 10) == [1, 2, 3]

    def test_empty_input(self) -> None:
        assert select_evenly([], 3) == []

    def test_evenly_spaced_selection(self) -> None:
        assert select_evenly(list(range(9)), 3) == [0, 4, 8]


class TestSelectNotes:
    def test_full_percentage_keeps_everything(self) -> None:
        notes = recorded_notes(21, 108, 1)
        assert select_notes(notes, 100.0) == notes

    def test_half_keeps_half_spanning_the_range(self) -> None:
        notes = recorded_notes(21, 108, 1)
        chosen = select_notes(notes, 50.0)
        assert len(chosen) == round(0.5 * len(notes))
        assert (chosen[0], chosen[-1]) == (21, 108)

    def test_a_target_of_one_is_the_middle_note(self) -> None:
        """V1's tie-break, preserved: one note means the middle of the keyboard."""
        notes = [60, 62, 64, 66, 68]
        assert select_notes(notes, 1.0) == [64]

    def test_empty_input(self) -> None:
        assert select_notes([], 50.0) == []


class TestSelectVelocities:
    def test_one_layer_is_the_loudest(self) -> None:
        """V1's tie-break, preserved: one layer means the loudest, not the middle."""
        assert select_velocities([1, 43, 85, 127], 1) == [127]

    def test_count_above_the_input_size_returns_everything(self) -> None:
        assert select_velocities([1, 127], 5) == [1, 127]

    def test_evenly_spaced_layers_span_the_range(self) -> None:
        velocities = velocity_list(18)
        chosen = select_velocities(velocities, 6)
        assert len(chosen) == 6
        assert (chosen[0], chosen[-1]) == (1, 127)

    def test_empty_input(self) -> None:
        assert select_velocities([], 3) == []
