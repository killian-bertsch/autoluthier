"""Loudness normalization: grouped lufs/rms matching, or a per-note velocity gain curve.

Ports V1's ``NormalizeProcessor`` (``src/processors/normalize.py``), minus DC removal (that's
the separate always-on ``dc`` stage now) and fixing three bugs:

- **Bug 1**: velocity mode's dynamic-range average used to include a synthetic ``0.0`` for
  every single-layer or silent note, dragging the mean down. Those notes are now excluded from
  the average entirely instead of counted as zero.
- **Bug 2**: velocity mode's release-sample pass hardcoded a ``-18.0`` dB RMS target,
  ignoring whatever the project actually configured. It now uses ``params.target_db``.
- **Bug 3**: the peak ceiling was applied per-group in lufs/rms mode but globally across all
  sustain samples in velocity mode. Both modes now apply the ceiling within their own natural
  grouping — per velocity layer for lufs/rms, per note for velocity mode (matching the group
  each mode already normalizes together) — instead of per-group in one and global in the other.

This module operates on whole `SampleSet`s rather than one buffer at a time, so it
deliberately does not implement the `dsp.base.Stage` protocol; see that module's docstring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyloudnorm as pyln
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autoluthier.config.hints import ui_hint
from autoluthier.domain.models import Sample, SampleSet

_SILENCE_FLOOR_DB = -100.0
_MIN_MEASURABLE_PEAK_DB = -99.0
_MAX_VELOCITY = 127.0
_MAX_VELOCITY_RANGE_DB = 60.0

_meter_cache: dict[int, pyln.Meter] = {}


def _meter_for(sample_rate: int) -> pyln.Meter:
    """Return a cached `pyln.Meter` for `sample_rate` (construction re-parses K-weighting)."""
    meter = _meter_cache.get(sample_rate)
    if meter is None:
        meter = pyln.Meter(sample_rate)
        _meter_cache[sample_rate] = meter
    return meter


class NormalizeParams(BaseModel):
    """Parameters for the ``normalize`` DSP stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["lufs", "rms", "velocity"] = Field(
        default="lufs",
        json_schema_extra=ui_hint(group="Normalize", order=0, help_text="Normalization mode."),
    )
    target_db: float = Field(
        default=-18.0,
        ge=-60.0,
        le=0.0,
        json_schema_extra=ui_hint(
            group="Normalize",
            order=1,
            unit="dB",
            help_text="Target LUFS (lufs mode) or RMS dBFS (rms mode); unused in velocity mode.",
        ),
    )
    peak_ceiling_db: float = Field(
        default=-1.0,
        le=0.0,
        json_schema_extra=ui_hint(
            group="Normalize",
            order=2,
            unit="dB",
            help_text="Hard peak ceiling after normalization; only ever attenuates.",
        ),
    )


@dataclass(frozen=True, slots=True)
class VelocityModeResult:
    """Outcome of `apply_velocity_mode`: the measured dynamic range.

    `dynamic_range_db` becomes the SFZ `amp_velcurve_1` opcode later, in export (step 7).
    """

    dynamic_range_db: float


def _rms_db(audio: NDArray[np.float32]) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms) if rms > 0.0 else _SILENCE_FLOOR_DB


def _peak(audio: NDArray[np.float32]) -> float:
    return float(np.max(np.abs(audio.astype(np.float64))))


def _peak_db(audio: NDArray[np.float32]) -> float:
    peak = _peak(audio)
    return 20.0 * np.log10(peak) if peak > 0.0 else _SILENCE_FLOOR_DB


def _measure_level(
    audio: NDArray[np.float32], sample_rate: int, mode: Literal["lufs", "rms"]
) -> float:
    """Signal level in dB; lufs mode falls back to RMS if pyloudnorm can't measure it."""
    if mode == "lufs":
        arr = audio.astype(np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        try:
            loudness = _meter_for(sample_rate).integrated_loudness(arr)
        except (ValueError, Warning):
            loudness = float("-inf")
        if np.isfinite(loudness):
            return float(loudness)
    return _rms_db(audio)


def _scale(samples: list[Sample], factor: float) -> None:
    for s in samples:
        s.audio = (s.audio.astype(np.float64) * factor).astype(np.float32)


def _apply_peak_ceiling(samples: list[Sample], peak_ceiling_db: float) -> None:
    """Scale `samples` down (never up) so the loudest one sits at `peak_ceiling_db`."""
    if not samples:
        return
    group_peak = max(_peak(s.audio) for s in samples)
    if group_peak == 0.0:
        return
    target_linear = 10.0 ** (peak_ceiling_db / 20.0)
    scale = target_linear / group_peak
    if scale < 1.0:
        _scale(samples, scale)


def _normalize_group(
    samples: list[Sample],
    sample_rate: int,
    mode: Literal["lufs", "rms"],
    target_db: float,
    peak_ceiling_db: float,
) -> None:
    """Normalize one group (same velocity layer) in place: mean-level match, then ceiling."""
    if not samples:
        return
    levels = [_measure_level(s.audio, sample_rate, mode) for s in samples]
    mean_level = float(np.mean(levels))
    if np.isfinite(mean_level) and mean_level > _SILENCE_FLOOR_DB:
        scale = 10.0 ** ((target_db - mean_level) / 20.0)
        _scale(samples, scale)
    _apply_peak_ceiling(samples, peak_ceiling_db)


def normalize_by_velocity_layer(
    samples: SampleSet, mode: Literal["lufs", "rms"], target_db: float, peak_ceiling_db: float
) -> None:
    """Group `samples` by velocity, matching each group's mean level to `target_db` in place.

    Args:
        samples: Samples to normalize (mutated in place).
        mode: `"lufs"` (with RMS fallback) or `"rms"`.
        target_db: Target level in dB for each velocity layer's group mean.
        peak_ceiling_db: Hard ceiling applied per velocity-layer group afterward.
    """
    by_velocity: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_velocity[s.velocity].append(s)
    for group in by_velocity.values():
        _normalize_group(group, group[0].sample_rate, mode, target_db, peak_ceiling_db)


def apply_velocity_mode(
    sustain: SampleSet, release: SampleSet, peak_ceiling_db: float, release_target_db: float
) -> VelocityModeResult:
    """Per-note linear gain curve so all velocity layers peak alike; see module docstring.

    Args:
        sustain: Sustain samples (mutated in place).
        release: Release samples, normalized by RMS per velocity layer (mutated in place).
        peak_ceiling_db: Hard ceiling, applied per note for sustain and per velocity layer for
            release.
        release_target_db: RMS target for the release pass (bug 2: no longer hardcoded).

    Returns:
        The average measured dynamic range across notes with a real range to measure.
    """
    by_note: dict[int, list[Sample]] = defaultdict(list)
    for s in sustain:
        by_note[s.note].append(s)

    range_values: list[float] = []

    for note_samples in by_note.values():
        velocities = [s.velocity for s in note_samples]
        peaks_db = [_peak_db(s.audio) for s in note_samples]

        min_idx = int(np.argmin(velocities))
        max_idx = int(np.argmax(velocities))
        vel_min, vel_max = velocities[min_idx], velocities[max_idx]
        peak_min, peak_max = peaks_db[min_idx], peaks_db[max_idx]

        degenerate = (
            vel_max <= vel_min
            or peak_min <= _MIN_MEASURABLE_PEAK_DB
            or peak_max <= _MIN_MEASURABLE_PEAK_DB
        )
        if degenerate:
            continue  # bug 1: excluded, not counted as a 0.0 dragging the average down

        slope = (peak_max - peak_min) / (vel_max - vel_min)
        range_db = float(np.clip(slope * _MAX_VELOCITY, 0.0, _MAX_VELOCITY_RANGE_DB))
        range_values.append(range_db)

        for s, v in zip(note_samples, velocities, strict=True):
            gain_db = range_db * (_MAX_VELOCITY - v) / _MAX_VELOCITY
            scale = 10.0 ** (gain_db / 20.0)
            s.audio = (s.audio.astype(np.float64) * scale).astype(np.float32)

        # bug 3: ceiling applied per note — this mode's own normalization group — not
        # globally across every sustain sample regardless of note.
        _apply_peak_ceiling(note_samples, peak_ceiling_db)

    if release:
        normalize_by_velocity_layer(release, "rms", release_target_db, peak_ceiling_db)

    dynamic_range_db = float(np.mean(range_values)) if range_values else 0.0
    return VelocityModeResult(dynamic_range_db=dynamic_range_db)


def apply_normalize(
    sustain: SampleSet, release: SampleSet, params: NormalizeParams
) -> VelocityModeResult | None:
    """Apply the configured normalize mode to `sustain`/`release` in place.

    Args:
        sustain: Sustain samples (mutated in place).
        release: Release samples (mutated in place).
        params: Validated `NormalizeParams`.

    Returns:
        A `VelocityModeResult` in velocity mode; `None` for lufs/rms, which have no
        equivalent computed value.
    """
    if params.mode == "velocity":
        return apply_velocity_mode(sustain, release, params.peak_ceiling_db, params.target_db)
    normalize_by_velocity_layer(sustain, params.mode, params.target_db, params.peak_ceiling_db)
    normalize_by_velocity_layer(release, params.mode, params.target_db, params.peak_ceiling_db)
    return None
