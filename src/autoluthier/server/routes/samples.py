"""``GET /api/samples``, ``/api/peaks/{id}``, ``/api/audio/{id}`` — inspect and hear samples."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from autoluthier.domain.models import SampleSet
from autoluthier.domain.notes import midi_to_note_name
from autoluthier.io.peaks import DEFAULT_BIN_COUNTS, PeakLevel, compute_peaks
from autoluthier.io.writer import encode_audio
from autoluthier.server.state import (
    AppState,
    AudioSource,
    SampleKind,
    parse_sample_id,
    resolve_sample,
    sample_id,
)

router = APIRouter()

_MEDIA_TYPE_BY_CONTAINER = {"flac": "audio/flac", "wav": "audio/wav"}


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autoluthier
    return state


class SampleSummary(BaseModel):
    """One row of ``GET /api/samples``."""

    id: str
    kind: SampleKind
    note: int
    note_name: str
    velocity: int
    n_frames: int
    duration_s: float
    sample_rate: int
    n_channels: int
    has_loop: bool


def _summaries_for(kind: SampleKind, samples: SampleSet) -> list[SampleSummary]:
    """Build one `SampleSummary` per sample in `samples`."""
    return [
        SampleSummary(
            id=sample_id(kind, sample.note, sample.velocity),
            kind=kind,
            note=sample.note,
            note_name=midi_to_note_name(sample.note),
            velocity=sample.velocity,
            n_frames=sample.n_frames,
            duration_s=sample.n_frames / sample.sample_rate,
            sample_rate=sample.sample_rate,
            n_channels=sample.n_channels,
            has_loop=sample.loop_start is not None,
        )
        for sample in samples
    ]


@router.get("/api/samples")
def list_samples(request: Request) -> list[SampleSummary]:
    """List every raw sample in the currently open instrument, sustain then release.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if no instrument is open.
    """
    loaded = _state(request).require_current()
    return _summaries_for("sustain", loaded.audio.sustain) + _summaries_for(
        "release", loaded.audio.release
    )


class PeakLevelPayload(BaseModel):
    """One zoom level of ``GET /api/peaks/{id}``."""

    samples_per_bin: int
    mins: list[list[float]]
    """Shape ``(n_channels, n_bins)``."""
    maxs: list[list[float]]
    """Shape ``(n_channels, n_bins)``."""


def _peak_payload(level: PeakLevel) -> PeakLevelPayload:
    """Convert one `PeakLevel`'s NumPy arrays into a JSON-serializable payload."""
    return PeakLevelPayload(
        samples_per_bin=level.samples_per_bin,
        mins=level.mins.tolist(),
        maxs=level.maxs.tolist(),
    )


@router.get("/api/peaks/{sample_id_str}")
def get_peaks(
    sample_id_str: str, request: Request, source: AudioSource = "raw"
) -> list[PeakLevelPayload]:
    """Return precomputed min/max peak levels for one sample, at a few zoom levels.

    Args:
        sample_id_str: A sample id from `GET /api/samples` or a preview listing.
        request: The current request, used to reach server state.
        source: ``"raw"`` reads the loaded instrument; ``"preview"`` reads the last
            `POST /api/preview` result.

    Returns:
        One `PeakLevelPayload` per zoom level, coarsest first.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if the requested source isn't
            loaded.
        autoluthier.server.state.SampleNotFoundError: mapped to 404 if the id doesn't match
            a loaded sample.
    """
    kind, note, velocity = parse_sample_id(sample_id_str)
    sample = resolve_sample(_state(request), kind, note, velocity, source)
    return [_peak_payload(level) for level in compute_peaks(sample.audio, DEFAULT_BIN_COUNTS)]


@router.get("/api/audio/{sample_id_str}")
def get_audio(sample_id_str: str, request: Request, source: AudioSource = "raw") -> Response:
    """Return one sample's audio, encoded per the project's output settings.

    Args:
        sample_id_str: A sample id from `GET /api/samples` or a preview listing.
        request: The current request, used to reach server state.
        source: ``"raw"`` reads the loaded instrument; ``"preview"`` reads the last
            `POST /api/preview` result.

    Returns:
        The encoded audio bytes, with a matching ``audio/*`` media type.

    Raises:
        autoluthier.server.state.SessionError: mapped to 404 if the requested source isn't
            loaded.
        autoluthier.server.state.SampleNotFoundError: mapped to 404 if the id doesn't match
            a loaded sample.
    """
    state = _state(request)
    kind, note, velocity = parse_sample_id(sample_id_str)
    sample = resolve_sample(state, kind, note, velocity, source)
    output = state.require_current().config.output
    data = encode_audio(sample.audio, sample.sample_rate, output)
    return Response(content=data, media_type=_MEDIA_TYPE_BY_CONTAINER[output.container])
