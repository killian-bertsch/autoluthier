"""A synthesized instrument folder: a real render on disk, sliced by real arithmetic.

The render is built with the same event arithmetic `io.slicer` uses to take it apart
(``event_start = round(event_index * (hold + release) * sr)``, note-outer/velocity-inner), so a
misalignment in either one shows up as a test failure rather than as quietly shifted audio.

Each event carries a sine whose amplitude rises with velocity *and* with note, which is what
makes the pipeline tests meaningful: velocity-mode normalize has a real slope to measure, and
lufs/rms grouping has real variation across the notes in a velocity layer — the variation that
makes a subset preview differ from a full run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from autoluthier.config.schema import ProjectConfig
from autoluthier.domain.notes import recorded_notes, velocity_list

SAMPLE_RATE = 8000
HOLD_TIME = 1.0
RELEASE_TIME = 0.5
START_NOTE = 60
END_NOTE = 62
VELOCITY_LAYERS = 2
DC_OFFSET = 0.02


def event_frames(sample_rate: int = SAMPLE_RATE) -> tuple[int, int]:
    """Return ``(hold_frames, release_frames)`` for the fixture's timing."""
    return round(HOLD_TIME * sample_rate), round(RELEASE_TIME * sample_rate)


def _tone(
    n_frames: int, freq: float, amp: float, sample_rate: int, *, decay: float, dc: float
) -> NDArray[np.float64]:
    t = np.arange(n_frames, dtype=np.float64) / sample_rate
    return amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * freq * t) + dc


def _amp(note: int, velocity: int) -> float:
    """Amplitude rising with both velocity and note, so group statistics actually vary."""
    return (0.1 + 0.6 * velocity / 127.0) * (1.0 + 0.2 * (note - START_NOTE))


def build_render(
    notes: list[int],
    velocities: list[int],
    *,
    sample_rate: int = SAMPLE_RATE,
    hold_time: float = HOLD_TIME,
    release_time: float = RELEASE_TIME,
    stereo: bool = False,
    dc: float = DC_OFFSET,
    body_scale: float = 1.0,
) -> NDArray[np.float32]:
    """Build a full autoluthier render: one event per ``(note, velocity)``, note-outer."""
    hold_frames = round(hold_time * sample_rate)
    release_frames = round(release_time * sample_rate)
    n_events = len(notes) * len(velocities)
    total = round((n_events - 1) * (hold_time + release_time) * sample_rate)
    total += hold_frames + release_frames
    channels = 2 if stereo else 1
    render = np.zeros((total, channels), dtype=np.float64)

    event_index = 0
    for note in notes:
        for velocity in velocities:
            start = round(event_index * (hold_time + release_time) * sample_rate)
            freq = 100.0 + 20.0 * (note - START_NOTE)
            amp = _amp(note, velocity) * body_scale
            body = _tone(hold_frames, freq, amp, sample_rate, decay=0.3, dc=dc)
            tail = _tone(release_frames, freq, amp * 0.3, sample_rate, decay=3.0, dc=dc)
            render[start : start + hold_frames, 0] = body
            render[start + hold_frames : start + hold_frames + release_frames, 0] = tail
            if channels == 2:
                # A different level and a slight phase offset, so mid/side has real content.
                render[start : start + hold_frames, 1] = 0.8 * np.roll(body, 3)
                render[start + hold_frames : start + hold_frames + release_frames, 1] = (
                    0.8 * tail
                )
            event_index += 1

    if channels == 1:
        return render[:, 0].astype(np.float32)
    return render.astype(np.float32)


@dataclass(frozen=True)
class Instrument:
    """A synthesized instrument folder plus the config that describes it."""

    folder: Path
    config: ProjectConfig
    notes: list[int]
    velocities: list[int]


def make_config(**overrides: Any) -> ProjectConfig:
    """Build a `ProjectConfig` matching the fixture render, with section overrides applied."""
    data: dict[str, Any] = {
        "recording": {
            "velocity_layers": VELOCITY_LAYERS,
            "semitone_interval": 1,
            "hold_time": HOLD_TIME,
            "release_time": RELEASE_TIME,
            "start_note": START_NOTE,
            "end_note": END_NOTE,
        },
        "selection": {
            "min_note": START_NOTE,
            "max_note": END_NOTE,
            "velocity_layers_out": VELOCITY_LAYERS,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return ProjectConfig.model_validate(data)


def write_instrument(
    folder: Path,
    *,
    with_release: bool = True,
    stereo: bool = False,
    sample_rate: int = SAMPLE_RATE,
    release_sample_rate: int | None = None,
    with_sustain: bool = True,
    **config_overrides: Any,
) -> Instrument:
    """Write ``sustain``/``release`` renders into `folder` and return the matching config."""
    folder.mkdir(parents=True, exist_ok=True)
    config = make_config(**config_overrides)
    notes = recorded_notes(
        config.recording.start_note,
        config.recording.end_note,
        config.recording.semitone_interval,
    )
    velocities = velocity_list(config.recording.velocity_layers)

    if with_sustain:
        sustain = build_render(notes, velocities, sample_rate=sample_rate, stereo=stereo)
        sf.write(str(folder / "sustain.wav"), sustain, sample_rate, subtype="FLOAT")
    if with_release:
        rate = release_sample_rate or sample_rate
        release = build_render(
            notes,
            velocities,
            sample_rate=rate,
            stereo=stereo,
            body_scale=0.5,
            hold_time=config.recording.release_hold_time or HOLD_TIME,
            release_time=(
                RELEASE_TIME
                if config.recording.release_release_time is None
                else config.recording.release_release_time
            ),
        )
        sf.write(str(folder / "release.wav"), release, rate, subtype="FLOAT")
    return Instrument(folder=folder, config=config, notes=notes, velocities=velocities)


@pytest.fixture
def instrument(tmp_path: Path) -> Instrument:
    """A mono instrument folder with both renders and a default config."""
    return write_instrument(tmp_path / "keybass")
