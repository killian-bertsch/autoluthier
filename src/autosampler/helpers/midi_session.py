"""Generates an autosampler MIDI session plus a matching ``project.toml`` skeleton.

Ports V1's ``generate_session.py``. V1 duplicated the velocity-list formula here and in
``src/input_module.py`` with a comment begging the two be kept in sync by hand — this module
imports `domain.notes.velocity_list` instead, so there is exactly one implementation to drift
out of sync with the slicer that later reads the resulting render.

The written ``project.toml`` is deliberately minimal: `RecordingConfig` fields that must match
the MIDI file exactly, plus the required `SelectionConfig` fields defaulted to something sane
(the full note range, `velocity_layers` unthinned). Everything else keeps `ProjectConfig`'s own
defaults, so a field added to the schema later shows up here for free rather than needing this
generator to be kept in sync with it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from midiutil import MIDIFile

from autosampler.config.schema import ProjectConfig, RecordingConfig, SelectionConfig
from autosampler.config.toml_io import save_project
from autosampler.domain.notes import midi_to_note_name, recorded_notes, velocity_list

BPM = 120.0
"""Fixed tempo for the generated session; only relative note timing matters, not the tempo."""


@dataclass(frozen=True, slots=True)
class MidiSessionResult:
    """What `generate_midi_session` wrote and a summary of the session it describes."""

    midi_path: Path
    project_path: Path
    notes: list[int]
    velocities: list[int]
    total_events: int
    duration_s: float


def _add_events(
    midi: MIDIFile, notes: list[int], velocities: list[int], *, hold_time: float, spacing: float
) -> None:
    """Write one note-on/note-off event per ``(note, velocity)``, note-outer, back to back."""
    beats_per_second = BPM / 60.0
    hold_beats = hold_time * beats_per_second
    spacing_beats = spacing * beats_per_second
    beat = 0.0
    for note in notes:
        for velocity in velocities:
            midi.addNote(
                track=0, channel=0, pitch=note, time=beat, duration=hold_beats, volume=velocity
            )
            beat += spacing_beats


def generate_midi_session(
    velocity_layers: int,
    semitone_interval: int,
    hold_time: float,
    release_time: float,
    output_stem: str,
    *,
    start_note: int = 21,
    end_note: int = 108,
    out_dir: Path | None = None,
) -> MidiSessionResult:
    """Write ``<output_stem>.mid`` and a matching ``project.toml``.

    Args:
        velocity_layers: Number of velocity layers to record (``recording.velocity_layers``).
        semitone_interval: Semitone step between recorded notes.
        hold_time: Seconds each note is held.
        release_time: Silence after note-off, before the next event.
        output_stem: Base name; the MIDI file is written to ``<output_stem>.mid``.
        start_note: Lowest MIDI note to record.
        end_note: Highest MIDI note to record.
        out_dir: Directory `project.toml` is written into; defaults to a folder named after
            `output_stem` in the current directory, matching where the render will later live.

    Returns:
        The `MidiSessionResult` describing what was written.

    Raises:
        ValueError: if `start_note` is not strictly below `end_note`.
    """
    if start_note >= end_note:
        raise ValueError(f"start_note ({start_note}) must be < end_note ({end_note})")

    notes = recorded_notes(start_note, end_note, semitone_interval)
    velocities = velocity_list(velocity_layers)
    total_events = len(notes) * len(velocities)
    duration_s = total_events * (hold_time + release_time)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, BPM)
    midi.addTrackName(0, 0, "Autosample")
    _add_events(midi, notes, velocities, hold_time=hold_time, spacing=hold_time + release_time)

    midi_path = Path(f"{output_stem}.mid")
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with midi_path.open("wb") as f:
        midi.writeFile(f)

    project_dir = out_dir if out_dir is not None else Path(output_stem)
    config = ProjectConfig(
        recording=RecordingConfig(
            velocity_layers=velocity_layers,
            semitone_interval=semitone_interval,
            hold_time=hold_time,
            release_time=release_time,
            start_note=start_note,
            end_note=end_note,
        ),
        selection=SelectionConfig(
            min_note=start_note, max_note=end_note, velocity_layers_out=velocity_layers
        ),
    )
    project_path = save_project(config, project_dir)

    return MidiSessionResult(
        midi_path=midi_path,
        project_path=project_path,
        notes=notes,
        velocities=velocities,
        total_events=total_events,
        duration_s=duration_s,
    )


def describe_session(result: MidiSessionResult) -> str:
    """Render a human-readable summary of a generated session, for CLI/UI display.

    Args:
        result: The result of `generate_midi_session`.

    Returns:
        A multi-line summary.
    """
    first, last = result.notes[0], result.notes[-1]
    return (
        f"MIDI:    {result.midi_path}\n"
        f"Project: {result.project_path}\n"
        f"Notes:   {len(result.notes)} ({midi_to_note_name(first)}-{midi_to_note_name(last)})\n"
        f"Layers:  {len(result.velocities)} {result.velocities}\n"
        f"Events:  {result.total_events} ({result.duration_s:.1f}s / "
        f"{result.duration_s / 60:.1f} min)"
    )
