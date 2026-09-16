"""``GET /api/zero-crossings/{id}`` — upward zero crossings, for the loop editor's snap toggle."""

from __future__ import annotations

from fastapi import APIRouter, Request

from autoluthier.dsp.loop import zero_crossings
from autoluthier.server.state import AppState, AudioSource, parse_sample_id, resolve_sample

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


@router.get("/api/zero-crossings/{sample_id_str}")
def get_zero_crossings(
    sample_id_str: str, request: Request, source: AudioSource = "raw"
) -> list[int]:
    """Return every upward zero crossing in one sample.

    The waveform view's snap-to-zero-crossing toggle uses this rather than scanning the
    decoded audio in JavaScript, so a dragged handle snaps to the exact crossing definition
    `dsp.loop.find_loop_points` itself searches over.

    Args:
        sample_id_str: A sample id from `GET /api/samples` or a preview listing.
        request: The current request, used to reach server state.
        source: ``"raw"`` reads the loaded instrument; ``"preview"`` reads the last
            `POST /api/preview` result.

    Returns:
        Ascending frame indices of every upward zero crossing in the sample.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if the requested source isn't
            loaded.
        autoluthier.server.state.SampleNotFoundError: mapped to 404 if the id doesn't match
            a loaded sample.
    """
    kind, note, velocity = parse_sample_id(sample_id_str)
    sample = resolve_sample(_state(request), kind, note, velocity, source)
    return [int(i) for i in zero_crossings(sample.audio)]
