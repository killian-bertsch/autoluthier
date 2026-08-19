"""The one place velocity lists, note grids, subset selection, and MIDI<->note-name live.

V1 duplicated the velocity-list formula between ``generate_session.py`` and
``src/input_module.py``, with a comment warning the two had to be kept in sync by hand. V2
has exactly one implementation, imported everywhere it's needed.

The even-spacing selectors (`select_evenly`, `select_notes`, `select_velocities`) are here for
the same reason: `pipeline.preview` picks which samples to preview and `export.zones` (step 7)
thins the output set, and both want the same rule with V1's tie-breaks.
`select_velocities_segmented` (``velocity_map``-driven thinning) belongs beside them and arrives
with export, which is its only caller.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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


def select_evenly[T](items: Sequence[T], count: int) -> list[T]:
    """Return exactly `count` evenly spaced entries from `items`, first and last included.

    The index list is built as a plain list, not a set. V1 wrote
    ``{round(i * (n - 1) / (count - 1)) for i in range(count)}`` and then sorted it (bug 8),
    which silently returned fewer entries than asked for if any two indices collided. They
    cannot collide here — with ``count < n`` the spacing ``(n - 1) / (count - 1)`` is strictly
    greater than 1, so the rounded indices strictly increase — but the set was hiding that
    guarantee rather than relying on it, and a caller asking for `count` layers and getting
    `count - 1` had no way to notice.

    Args:
        items: Ordered items to choose from.
        count: How many to keep; values above ``len(items)`` return everything.

    Returns:
        `count` items in their original order (or all of them, if fewer exist). A `count` of
        1 or less returns the middle item, matching V1's ``select_notes`` tie-break.
    """
    n = len(items)
    if n == 0:
        return []
    if count >= n:
        return list(items)
    if count <= 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (count - 1))] for i in range(count)]


def select_notes(notes: Sequence[int], note_percentage: float) -> list[int]:
    """Return an evenly spaced subset of `notes` covering `note_percentage` percent of them.

    Preserved from V1: the target count is ``max(1, round(pct / 100 * n))``, and a target of 1
    yields the *middle* note rather than the lowest.

    Args:
        notes: Recorded MIDI notes, ascending.
        note_percentage: Percentage of `notes` to keep, in ``(0, 100]``.

    Returns:
        The selected MIDI notes, ascending.
    """
    if not notes:
        return []
    target = max(1, round(note_percentage / 100.0 * len(notes)))
    return select_evenly(notes, target)


def select_velocities(velocities: Sequence[int], count: int) -> list[int]:
    """Return an evenly spaced subset of `velocities` of size `count`.

    Preserved from V1: a `count` of 1 yields the *highest* velocity, not the middle one — the
    loudest layer is the one that covers the full range on its own.

    Args:
        velocities: Recorded MIDI velocities, ascending.
        count: How many layers to keep; values above ``len(velocities)`` return everything.

    Returns:
        The selected MIDI velocities, ascending.
    """
    if not velocities:
        return []
    if count <= 1:
        return [velocities[-1]]
    return select_evenly(velocities, count)
