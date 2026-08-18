"""The one place velocity lists, recorded-note grids, and MIDI<->note-name live.

V1 duplicated the velocity-list formula between ``generate_session.py`` and
``src/input_module.py``, with a comment warning the two had to be kept in sync by hand. V2
has exactly one implementation, imported everywhere it's needed.
"""

from __future__ import annotations

import re

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NOTE_NAME_TO_PITCH_CLASS = {name: pitch_class for pitch_class, name in enumerate(_NOTE_NAMES)}
_NOTE_NAME_RE = re.compile(r"^([A-G]#?)(-?\d+)$")


def velocity_list(layer_count: int) -> list[int]:
    """Return ``layer_count`` MIDI velocities from 1 to 127 inclusive, evenly spaced.

    Preserved verbatim from V1: ``x==1 -> [127]``; else ``v_0=1``, ``v_{x-1}=127``,
    ``v_i = round(1 + i*126/(x-1))`` (which for ``x==2`` already reduces to ``[1, 127]``).

    Args:
        layer_count: Number of velocity layers recorded (``x``, >= 1).

    Returns:
        ``layer_count`` MIDI velocity values, ascending.
    """
    if layer_count == 1:
        return [127]
    step = 126 / (layer_count - 1)
    return [round(1 + i * step) for i in range(layer_count)]


def recorded_notes(start_note: int, end_note: int, semitone_interval: int) -> list[int]:
    """Return the MIDI notes recorded between ``start_note`` and ``end_note``.

    Args:
        start_note: Lowest recorded MIDI note.
        end_note: Highest recorded MIDI note.
        semitone_interval: Semitone step between recorded notes.

    Returns:
        MIDI notes from ``start_note`` to ``end_note`` inclusive, stepped by
        ``semitone_interval``.
    """
    return list(range(start_note, end_note + 1, semitone_interval))


def midi_to_note_name(midi_note: int) -> str:
    """Convert a MIDI note number to a note name, e.g. ``60 -> "C4"``.

    Args:
        midi_note: MIDI note number (0-127).

    Returns:
        The note name using ``C#``-style sharps.
    """
    octave = (midi_note // 12) - 1
    return f"{_NOTE_NAMES[midi_note % 12]}{octave}"


def note_name_to_midi(name: str) -> int:
    """Convert a note name back to a MIDI note number, e.g. ``"C4" -> 60``.

    Args:
        name: A note name using ``C#``-style sharps, e.g. ``"C4"``, ``"A#3"``.

    Returns:
        The MIDI note number.

    Raises:
        ValueError: if ``name`` is not a well-formed note name.
    """
    match = _NOTE_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"'{name}' is not a valid note name, e.g. 'C4' or 'A#3'")
    pitch_name, octave_str = match.groups()
    octave = int(octave_str)
    return _NOTE_NAME_TO_PITCH_CLASS[pitch_name] + (octave + 1) * 12
