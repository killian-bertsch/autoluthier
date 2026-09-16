"""``GET /api/sfz/preview`` — run the full chain and show the SFZ text a real run would write."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autoluthier.export.writer import plan_export
from autoluthier.pipeline.executor import RunResult, run_chain
from autoluthier.pipeline.graph import build_chain
from autoluthier.pipeline.preview import copy_instrument_audio
from autoluthier.server.state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


class SfzPreview(BaseModel):
    """The two SFZ documents a real run would currently write."""

    sustain_sfz: str
    release_sfz: str | None
    measured_dynamic_range_db: float | None


@router.get("/api/sfz/preview")
def get_sfz_preview(request: Request) -> SfzPreview:
    """Run the configured chain over a copy of the loaded audio and show the resulting SFZ.

    Never mutates the session's loaded audio — the same guarantee `pipeline.preview` gives,
    applied here to a full-set run instead of a subset, so this is exactly what `POST /api/jobs`
    would write, without writing it.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if no instrument is open.
        autoluthier.pipeline.graph.ChainConfigError: mapped to 400 on an invalid stage chain.
        autoluthier.export.writer.ExportError: mapped to 400 if selection leaves nothing to
            export.

    Returns:
        The rendered sustain/release SFZ text and the measured dynamic range, if any.
    """
    loaded = _state(request).require_current()
    chain = build_chain(loaded.config)
    audio = copy_instrument_audio(loaded.audio)
    chain_result = run_chain(audio, chain)
    result = RunResult(
        instrument_name=loaded.config.instrument_name or loaded.folder.name,
        audio=audio,
        dynamic_range_db=chain_result.dynamic_range_db,
        duration_s=0.0,
    )
    plan = plan_export(result, loaded.config)
    return SfzPreview(
        sustain_sfz=plan.sustain.render(),
        release_sfz=None if plan.release is None else plan.release.render(),
        measured_dynamic_range_db=plan.measured_dynamic_range_db,
    )
