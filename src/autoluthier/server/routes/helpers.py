"""``POST /api/helpers/{midi-session,prenormalize,concat}`` — web routes over the ported helpers.

Each route calls the exact function its CLI command calls (`cli.midi`, `cli.prenorm`,
`cli.concat`), so the browser has full parity with the CLI — the web-first decision means none
of these should be reachable only from a terminal.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from autoluthier.helpers.concat_layers import DEFAULT_LAYER_NUMBERS, concat_layers
from autoluthier.helpers.concat_layers import TARGET_DB_DEFAULT as CONCAT_TARGET_DB_DEFAULT
from autoluthier.helpers.midi_session import MidiSessionResult, generate_midi_session
from autoluthier.helpers.prenormalize import TARGET_DB_DEFAULT as PRENORM_TARGET_DB_DEFAULT
from autoluthier.helpers.prenormalize import prenormalize_sources

router = APIRouter()


class MidiSessionRequest(BaseModel):
    """Body of ``POST /api/helpers/midi-session``."""

    velocity_layers: int
    semitone_interval: int
    hold_time: float
    release_time: float
    output: str
    start_note: int = 21
    end_note: int = 108
    out_dir: str | None = None


class MidiSessionResponse(BaseModel):
    """What ``POST /api/helpers/midi-session`` returns."""

    midi_path: str
    project_path: str
    download_url: str
    notes: list[int]
    velocities: list[int]
    total_events: int
    duration_s: float


def _midi_response(result: MidiSessionResult) -> MidiSessionResponse:
    """Convert a `MidiSessionResult` into its wire form, adding the `.mid` download link."""
    return MidiSessionResponse(
        midi_path=str(result.midi_path),
        project_path=str(result.project_path),
        download_url=f"/api/helpers/midi-session/download?path={result.midi_path}",
        notes=result.notes,
        velocities=result.velocities,
        total_events=result.total_events,
        duration_s=result.duration_s,
    )


@router.post("/api/helpers/midi-session")
def midi_session(body: MidiSessionRequest) -> MidiSessionResponse:
    """Generate an autoluthier MIDI session plus its project.toml.

    Raises:
        HTTPException: 400 if `start_note` is not strictly below `end_note`.

    Returns:
        Paths to what was written, plus a download link for the ``.mid`` file — the one step
        that stays manual: playing it into a DAW/plugin and rendering it.
    """
    try:
        result = generate_midi_session(
            body.velocity_layers,
            body.semitone_interval,
            body.hold_time,
            body.release_time,
            body.output,
            start_note=body.start_note,
            end_note=body.end_note,
            out_dir=None if body.out_dir is None else Path(body.out_dir),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _midi_response(result)


@router.get("/api/helpers/midi-session/download")
def download_midi_session(path: str) -> FileResponse:
    """Download a ``.mid`` file previously written by ``POST /api/helpers/midi-session``.

    Args:
        path: The ``.mid`` file path, as returned in `download_url`.

    Raises:
        HTTPException: 404 if `path` doesn't exist or isn't a ``.mid`` file.

    Returns:
        The file, as an attachment.
    """
    midi_path = Path(path)
    if midi_path.suffix.lower() != ".mid" or not midi_path.is_file():
        raise HTTPException(status_code=404, detail=f"no .mid file at {path!r}")
    return FileResponse(midi_path, media_type="audio/midi", filename=midi_path.name)


class PrenormalizeRequest(BaseModel):
    """Body of ``POST /api/helpers/prenormalize``."""

    source_dir: str
    target_db: float = PRENORM_TARGET_DB_DEFAULT
    dry_run: bool = False


class PrenormalizeReportPayload(BaseModel):
    """One file's prenormalize outcome."""

    path: str
    peak_db: float | None
    gain_db: float | None
    applied: bool


@router.post("/api/helpers/prenormalize")
def prenormalize(body: PrenormalizeRequest) -> list[PrenormalizeReportPayload]:
    """Peak-normalize every sustain/release render under `source_dir` to `target_db`.

    Returns:
        One report per file found; empty if `source_dir` holds no renders.
    """
    reports = prenormalize_sources(
        Path(body.source_dir), target_db=body.target_db, dry_run=body.dry_run
    )
    return [
        PrenormalizeReportPayload(
            path=str(report.path),
            peak_db=report.peak_db,
            gain_db=report.gain_db,
            applied=report.applied,
        )
        for report in reports
    ]


class ConcatRequest(BaseModel):
    """Body of ``POST /api/helpers/concat``."""

    input_dir: str
    output_dir: str | None = None
    stereo_instruments: list[str] = []
    target_db: float = CONCAT_TARGET_DB_DEFAULT
    layer_numbers: list[int] = list(DEFAULT_LAYER_NUMBERS)


class ConcatReportPayload(BaseModel):
    """One instrument's concat outcome."""

    instrument: str
    sustain_path: str | None
    release_path: str | None
    layers_found: int
    layers_missing: list[int]
    stereo: bool
    error: str | None


@router.post("/api/helpers/concat")
def concat(body: ConcatRequest) -> list[ConcatReportPayload]:
    """Concatenate and peak-normalize every instrument's numbered layers under `input_dir`.

    Returns:
        One report per discovered instrument, successful or not; empty if none were found.
    """
    reports = concat_layers(
        Path(body.input_dir),
        Path(body.output_dir) if body.output_dir is not None else Path(body.input_dir),
        stereo_instruments=frozenset(body.stereo_instruments),
        target_db=body.target_db,
        layer_numbers=tuple(body.layer_numbers),
    )
    return [
        ConcatReportPayload(
            instrument=report.instrument,
            sustain_path=None if report.sustain_path is None else str(report.sustain_path),
            release_path=None if report.release_path is None else str(report.release_path),
            layers_found=report.layers_found,
            layers_missing=report.layers_missing,
            stereo=report.stereo,
            error=report.error,
        )
        for report in reports
    ]
