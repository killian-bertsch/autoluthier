"""``GET /api/analysis`` — per-release-sample decay-time estimates for the analysis view."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autoluthier.analysis.metrics import estimate_rt_decay
from autoluthier.server.state import AppState, sample_id

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


class RtDecayEntry(BaseModel):
    """One release sample's estimated decay rate."""

    id: str
    note: int
    velocity: int
    rt_decay_db_per_s: float


@router.get("/api/analysis")
def get_analysis(request: Request) -> list[RtDecayEntry]:
    """Return decay-rate estimates for every release sample in the open instrument.

    `estimate_rt_decay` is invariant to gain (see `analysis.metrics`), so this is meaningful on
    the raw audio without running the chain first.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if no instrument is open.

    Returns:
        One entry per release sample.
    """
    loaded = _state(request).require_current()
    return [
        RtDecayEntry(
            id=sample_id("release", sample.note, sample.velocity),
            note=sample.note,
            velocity=sample.velocity,
            rt_decay_db_per_s=estimate_rt_decay(sample.audio, sample.sample_rate),
        )
        for sample in loaded.audio.release
    ]
