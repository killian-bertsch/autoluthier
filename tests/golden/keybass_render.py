"""Deterministic source render for the golden instrument, and the constants describing it.

The golden SFZ files beside this module were produced by **V1** from a render built by this
exact code, so the render has to be reproducible rather than checked in: a few kilobytes of
NumPy beats a multi-megabyte WAV in the repo, and it keeps the recording parameters readable
next to the config they must agree with.

Reproducibility was verified by generating the render under V1's interpreter and V2's and
comparing the sample arrays, which were bit-identical.

The content is chosen to make the comparison meaningful rather than merely pass:

- Three harmonics with unrelated phase offsets, so the loop finder's upward zero crossings and
  correlation scoring have real structure to choose between instead of the single obvious
  period a bare sine would give it.
- A slow body decay, so the 85%-97% search window is not stationary.
- A DC offset, so the ``dc`` stage has something to remove.
- Amplitude rising with both note and velocity, so normalize's per-velocity grouping produces
  a different gain per group.
- A release tail decaying at ~8.7 dB/s, comfortably inside the ``rt_decay`` clamp of
  ``[1, 24]``, so that opcode measures something instead of pinning at a bound.

To regenerate the golden files, see ``test_sfz_golden.py``'s module docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

SAMPLE_RATE = 22050
HOLD_TIME = 1.0
RELEASE_TIME = 0.5
START_NOTE = 48
END_NOTE = 60
SEMITONE_INTERVAL = 2
VELOCITY_LAYERS = 6
INSTRUMENT_NAME = "keybass"

_DC_OFFSET = 0.01
_BODY_DECAY = 0.25
_TAIL_DECAY = 1.0
_TAIL_LEVEL = 0.3
_RELEASE_BODY_SCALE = 0.45
_HARMONICS = ((1.0, 1.0, 0.0), (2.0, 0.35, 0.7), (3.0, 0.15, 1.9))
"""``(multiple, level, phase)`` per harmonic; the phases are deliberately unrelated."""


def _notes() -> list[int]:
    return list(range(START_NOTE, END_NOTE + 1, SEMITONE_INTERVAL))


def _velocities() -> list[int]:
    step = 126 / (VELOCITY_LAYERS - 1)
    return [round(1 + i * step) for i in range(VELOCITY_LAYERS)]


def _amplitude(note: int, velocity: int) -> float:
    return (0.08 + 0.5 * velocity / 127.0) * (1.0 + 0.05 * (note - START_NOTE))


def _tone(n_frames: int, freq: float, amplitude: float, decay: float) -> NDArray[np.float64]:
    t = np.arange(n_frames, dtype=np.float64) / SAMPLE_RATE
    wave = sum(
        level * np.sin(2.0 * np.pi * multiple * freq * t + phase)
        for multiple, level, phase in _HARMONICS
    )
    return amplitude * np.exp(-decay * t) * np.asarray(wave) / 1.5 + _DC_OFFSET


def build_render(body_scale: float) -> NDArray[np.float32]:
    """Build one full autoluthier render, note-outer and velocity-inner.

    Args:
        body_scale: Overall level of the held portion; the release pass is quieter than the
            sustain pass, as a real release-tail recording would be.

    Returns:
        The mono float32 render.
    """
    hold_frames = round(HOLD_TIME * SAMPLE_RATE)
    release_frames = round(RELEASE_TIME * SAMPLE_RATE)
    notes, velocities = _notes(), _velocities()
    n_events = len(notes) * len(velocities)
    total = round((n_events - 1) * (HOLD_TIME + RELEASE_TIME) * SAMPLE_RATE)
    render = np.zeros(total + hold_frames + release_frames, dtype=np.float64)

    event_index = 0
    for note in notes:
        for velocity in velocities:
            start = round(event_index * (HOLD_TIME + RELEASE_TIME) * SAMPLE_RATE)
            freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
            amplitude = _amplitude(note, velocity) * body_scale
            render[start : start + hold_frames] = _tone(
                hold_frames, freq, amplitude, _BODY_DECAY
            )
            render[start + hold_frames : start + hold_frames + release_frames] = _tone(
                release_frames, freq, amplitude * _TAIL_LEVEL, _TAIL_DECAY
            )
            event_index += 1
    return render.astype(np.float32)


def write_golden_instrument(folder: Path) -> Path:
    """Write the golden instrument's ``sustain``/``release`` renders into `folder`.

    Args:
        folder: Directory to create and write into.

    Returns:
        The folder written to.
    """
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "sustain.wav"), build_render(1.0), SAMPLE_RATE, subtype="FLOAT")
    sf.write(
        str(folder / "release.wav"),
        build_render(_RELEASE_BODY_SCALE),
        SAMPLE_RATE,
        subtype="FLOAT",
    )
    return folder
