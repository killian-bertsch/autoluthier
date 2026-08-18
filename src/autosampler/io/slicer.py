"""Pure-arithmetic slicing of a long autosampler render into per-(note, velocity) samples.

No onset detection: ``event_index * (hold_time + release_time) * sample_rate`` locates every
event, exactly as V1. ``event_index`` advances even for notes outside ``[min_note, max_note]``
so offsets stay correct for the notes that *are* kept.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from autosampler.domain.models import Sample, SampleSet
from autosampler.domain.units import seconds_to_frames

_STEREO_CHANNELS = 2


def _to_mono(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mix a ``(n_frames, n_channels)`` array down to ``(n_frames,)``; mono is unchanged."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1, dtype=np.float32)


def _zero_fill(slice_len: int, template: NDArray[np.float32]) -> NDArray[np.float32]:
    """A ``slice_len``-frame zero array with the same channel layout as ``template``.

    Fixes V1 bug 6, where a truncated render's out-of-bounds slice was always filled with a
    *mono* zero array, silently dropping a channel on stereo instruments.
    """
    if template.ndim == 1:
        return np.zeros(slice_len, dtype=np.float32)
    return np.zeros((slice_len, template.shape[1]), dtype=np.float32)


def _pad_to_length(chunk: NDArray[np.float32], slice_len: int) -> NDArray[np.float32]:
    """Zero-pad ``chunk`` at the end, along the frame axis, up to ``slice_len`` frames."""
    missing = slice_len - chunk.shape[0]
    if chunk.ndim == 1:
        return np.pad(chunk, (0, missing))
    return np.pad(chunk, ((0, missing), (0, 0)))


def slice_notes_velocities(
    audio: NDArray[np.float32],
    sample_rate: int,
    notes: list[int],
    velocities: list[int],
    *,
    hold_time: float,
    release_time: float,
    extract_release: bool,
    min_note: int,
    max_note: int,
    collapse_to_mono: bool,
) -> SampleSet:
    """Slice ``audio`` into one `Sample` per ``(note, velocity)`` pair in ``[min_note, max_note]``.

    Args:
        audio: Full render, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        sample_rate: Sample rate of ``audio`` in Hz.
        notes: Ordered MIDI notes that were recorded (note-outer iteration order).
        velocities: Ordered MIDI velocities that were recorded (velocity-inner).
        hold_time: Seconds each note was held (sustain portion of the event).
        release_time: Seconds of silence/tail captured after note-off.
        extract_release: If ``True``, extract the post-note-off tail; if ``False``, extract
            the sustain (during note-on) portion.
        min_note: Only slice notes ``>= min_note``.
        max_note: Only slice notes ``<= max_note``.
        collapse_to_mono: If ``True``, mix each slice down to mono.

    Returns:
        A `SampleSet` with one `Sample` per kept ``(note, velocity)`` pair.
    """
    frames_hold = seconds_to_frames(hold_time, sample_rate)
    frames_release = seconds_to_frames(release_time, sample_rate)
    total_frames = audio.shape[0]
    event_duration = hold_time + release_time

    slice_offset = frames_hold if extract_release else 0
    slice_len = frames_release if extract_release else frames_hold

    samples: list[Sample] = []
    event_index = 0
    for note in notes:
        for velocity in velocities:
            event_start = seconds_to_frames(event_index * event_duration, sample_rate)
            slice_start = event_start + slice_offset
            slice_end = slice_start + slice_len

            if min_note <= note <= max_note:
                if slice_start >= total_frames:
                    chunk = _zero_fill(max(slice_len, 1), audio)
                else:
                    actual_end = min(slice_end, total_frames)
                    chunk = audio[slice_start:actual_end]
                    if chunk.shape[0] < slice_len:
                        chunk = _pad_to_length(chunk, slice_len)

                if collapse_to_mono:
                    chunk = _to_mono(chunk)
                elif chunk.ndim == _STEREO_CHANNELS and chunk.shape[1] == 1:
                    chunk = chunk[:, 0]

                samples.append(
                    Sample(note=note, velocity=velocity, audio=chunk, sample_rate=sample_rate)
                )

            event_index += 1

    return SampleSet(samples)
