"""``GET /api/matrix`` — per-sample peak/RMS/length metrics for the sample matrix view."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autosampler.analysis.matrix import MatrixEntry, SampleKind, compute_matrix
from autosampler.server.state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autosampler
    return state


class MatrixRow(BaseModel):
    """One ``GET /api/matrix`` row."""

    kind: SampleKind
    note: int
    velocity: int
    n_frames: int
    duration_s: float
    peak_db: float
    rms_db: float


def _row(entry: MatrixEntry) -> MatrixRow:
    """Convert one `MatrixEntry` into its wire form."""
    return MatrixRow(
        kind=entry.kind,
        note=entry.note,
        velocity=entry.velocity,
        n_frames=entry.n_frames,
        duration_s=entry.duration_s,
        peak_db=entry.peak_db,
        rms_db=entry.rms_db,
    )


@router.get("/api/matrix")
def get_matrix(request: Request) -> list[MatrixRow]:
    """Return per-sample metrics for the currently open instrument's raw audio.

    Raises:
        autosampler.server.state.SessionError: mapped to 404 if no instrument is open.

    Returns:
        One row per sample, sustain then release.
    """
    loaded = _state(request).require_current()
    return [_row(entry) for entry in compute_matrix(loaded.audio)]
