"""``POST /api/preview`` — process an evenly spread subset of samples for a quick listen."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autosampler.config.schema import ProjectConfig
from autosampler.domain.models import SampleSet
from autosampler.domain.notes import midi_to_note_name
from autosampler.pipeline.graph import build_chain
from autosampler.pipeline.preview import (
    DEFAULT_PREVIEW_NOTES,
    DEFAULT_PREVIEW_VELOCITIES,
    PreviewMode,
    select_preview,
)
from autosampler.pipeline.preview import run_preview as run_preview_chain
from autosampler.server.state import AppState, SampleKind, sample_id

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autosampler
    return state


class RunPreviewRequest(BaseModel):
    """Body of ``POST /api/preview``.

    ``config``, if supplied, is used to build the chain *instead of* the open project's saved
    config — it is never written to disk or stored on the session. This is what lets the UI hear
    an in-progress, unsaved edit (a dragged DSP parameter, a loop override) without a Save
    round-trip first. Selection (which samples get previewed) still comes from the loaded audio,
    so edits to ``recording``/``selection``/``output`` that would change how audio is loaded or
    sliced are not reflected here — only a reload (Save, which reopens the project) picks those
    up.
    """

    note_count: int = DEFAULT_PREVIEW_NOTES
    velocity_count: int = DEFAULT_PREVIEW_VELOCITIES
    mode: PreviewMode = "exact"
    config: ProjectConfig | None = None


class PreviewSampleSummary(BaseModel):
    """One previewed sample, ready to fetch via ``GET /api/audio/{id}?source=preview``."""

    id: str
    kind: SampleKind
    note: int
    note_name: str
    velocity: int


class RunPreviewResponse(BaseModel):
    """What ``POST /api/preview`` returns."""

    mode: PreviewMode
    dynamic_range_db: float | None
    samples: list[PreviewSampleSummary]


def _summaries(kind: SampleKind, samples: SampleSet) -> list[PreviewSampleSummary]:
    """Build one `PreviewSampleSummary` per sample in `samples`."""
    return [
        PreviewSampleSummary(
            id=sample_id(kind, sample.note, sample.velocity),
            kind=kind,
            note=sample.note,
            note_name=midi_to_note_name(sample.note),
            velocity=sample.velocity,
        )
        for sample in samples
    ]


@router.post("/api/preview")
def run_preview_route(body: RunPreviewRequest, request: Request) -> RunPreviewResponse:
    """Process an evenly spread subset of the open instrument and hold it for playback.

    The result replaces any previous preview and is fetched per-sample via
    ``GET /api/audio/{id}?source=preview`` or ``GET /api/peaks/{id}?source=preview``.

    Raises:
        autosampler.server.state.SessionError: mapped to 404 if no instrument is open.
        autosampler.pipeline.graph.ChainConfigError: mapped to 400 on an invalid stage chain.

    Returns:
        The preview mode actually used, its measured dynamic range, and the previewed samples.
    """
    state = _state(request)
    loaded = state.require_current()
    chain = build_chain(body.config if body.config is not None else loaded.config)
    selection = select_preview(
        loaded.audio, note_count=body.note_count, velocity_count=body.velocity_count
    )
    result = run_preview_chain(loaded.audio, chain, selection, mode=body.mode)
    state.preview = result.audio
    samples = _summaries("sustain", result.audio.sustain) + _summaries(
        "release", result.audio.release
    )
    return RunPreviewResponse(
        mode=result.mode, dynamic_range_db=result.dynamic_range_db, samples=samples
    )
