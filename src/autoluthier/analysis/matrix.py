"""Per-sample summary metrics for the sample matrix view (step 11) and its ``/api/matrix`` route.

Peak and RMS are read off whatever audio a `Sample` currently holds — raw immediately after
load, processed after a chain has run — rather than being tied to one point in the pipeline,
since both the matrix view and the route want "whatever's loaded right now."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from autoluthier.domain.models import InstrumentAudio, Sample

SampleKind = Literal["sustain", "release"]

_SILENCE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    """One sample's summary metrics."""

    kind: SampleKind
    note: int
    velocity: int
    n_frames: int
    duration_s: float
    peak_db: float
    rms_db: float


def _levels(audio: NDArray[np.float32]) -> tuple[float, float]:
    """Return ``(peak_db, rms_db)`` for `audio`, floored so silence doesn't produce ``-inf``."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if mono.size == 0:
        return 20.0 * np.log10(_SILENCE_FLOOR), 20.0 * np.log10(_SILENCE_FLOOR)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    peak_db = 20.0 * np.log10(max(peak, _SILENCE_FLOOR))
    rms_db = 20.0 * np.log10(max(rms, _SILENCE_FLOOR))
    return peak_db, rms_db


def _entry(kind: SampleKind, sample: Sample) -> MatrixEntry:
    """Build one `MatrixEntry` from a loaded `Sample`."""
    peak_db, rms_db = _levels(sample.audio)
    return MatrixEntry(
        kind=kind,
        note=sample.note,
        velocity=sample.velocity,
        n_frames=sample.n_frames,
        duration_s=sample.n_frames / sample.sample_rate,
        peak_db=peak_db,
        rms_db=rms_db,
    )


def compute_matrix(audio: InstrumentAudio) -> list[MatrixEntry]:
    """Compute per-sample metrics for every sustain and release sample.

    Args:
        audio: The loaded instrument.

    Returns:
        One `MatrixEntry` per sample, sustain first, in each set's own order.
    """
    return [_entry("sustain", sample) for sample in audio.sustain] + [
        _entry("release", sample) for sample in audio.release
    ]
