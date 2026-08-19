"""In-memory server state: the one instrument currently open, and running jobs.

A single mutable session rather than one per client matches how the tool is actually used —
one person, one browser tab, one instrument open at a time (CLAUDE.md's "local web app"
decision). Preview results live alongside the raw load so `/api/audio/{id}` and
`/api/peaks/{id}` can serve either without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from autosampler.config.schema import ProjectConfig
from autosampler.domain.models import InstrumentAudio, Sample, SampleSet
from autosampler.pipeline.events import EventPayload

SampleKind = Literal["sustain", "release"]
AudioSource = Literal["raw", "preview"]

_SAMPLE_ID_PARTS = 3
"""A sample id is ``{kind}-{note}-{velocity}``: exactly three hyphen-separated parts."""


class SessionError(RuntimeError):
    """Raised when an operation needs an open instrument (or a preview) and there isn't one."""


class SampleNotFoundError(KeyError):
    """Raised when a sample id doesn't resolve to a loaded sample."""


@dataclass(frozen=True, slots=True)
class LoadedInstrument:
    """One project, opened: its config and its raw (unprocessed) sliced audio."""

    folder: Path
    config: ProjectConfig
    audio: InstrumentAudio


@dataclass(slots=True)
class JobRecord:
    """Status of one background run job (see `server/jobs.py`)."""

    id: str
    status: Literal["running", "completed", "failed"] = "running"
    events: list[EventPayload] = field(default_factory=list)
    error: str | None = None
    output_dir: str | None = None


class AppState:
    """Server-wide mutable state: the open instrument, the last preview, and job history."""

    def __init__(self, workspace_path: Path) -> None:
        """Initialize empty state.

        Args:
            workspace_path: Where the workspace index file lives.
        """
        self.workspace_path = workspace_path
        self.current: LoadedInstrument | None = None
        self.preview: InstrumentAudio | None = None
        self.jobs: dict[str, JobRecord] = {}

    def require_current(self) -> LoadedInstrument:
        """Return the currently open instrument.

        Returns:
            The `LoadedInstrument`.

        Raises:
            SessionError: if no instrument is open.
        """
        if self.current is None:
            raise SessionError("no instrument is open — POST /api/project first")
        return self.current


def sample_id(kind: SampleKind, note: int, velocity: int) -> str:
    """Build the id ``/api/peaks``, ``/api/audio``, and matrix rows address a sample by.

    Args:
        kind: Whether this is a sustain or release sample.
        note: MIDI note.
        velocity: MIDI velocity.

    Returns:
        The opaque id string, e.g. ``"sustain-60-127"``.
    """
    return f"{kind}-{note}-{velocity}"


def parse_sample_id(raw: str) -> tuple[SampleKind, int, int]:
    """Parse a sample id back into its ``(kind, note, velocity)``.

    Args:
        raw: An id built by `sample_id`.

    Returns:
        The parsed ``(kind, note, velocity)``.

    Raises:
        SampleNotFoundError: if `raw` isn't a well-formed id.
    """
    parts = raw.split("-")
    if len(parts) != _SAMPLE_ID_PARTS or parts[0] not in ("sustain", "release"):
        raise SampleNotFoundError(f"malformed sample id: {raw!r}")
    kind_str, note_str, velocity_str = parts
    kind: SampleKind = "sustain" if kind_str == "sustain" else "release"
    try:
        return kind, int(note_str), int(velocity_str)
    except ValueError as exc:
        raise SampleNotFoundError(f"malformed sample id: {raw!r}") from exc


def _sample_set(audio: InstrumentAudio, kind: SampleKind) -> SampleSet:
    """Return `audio`'s sustain or release set, whichever `kind` names."""
    return audio.sustain if kind == "sustain" else audio.release


def resolve_sample(
    state: AppState, kind: SampleKind, note: int, velocity: int, source: AudioSource
) -> Sample:
    """Look up one sample by kind/note/velocity, from the raw load or the last preview.

    Args:
        state: Server state.
        kind: Sustain or release.
        note: MIDI note.
        velocity: MIDI velocity.
        source: ``"raw"`` reads the loaded instrument; ``"preview"`` reads the last
            `POST /api/preview` result.

    Returns:
        The matching `Sample`.

    Raises:
        SessionError: if no instrument (or, for ``"preview"``, no preview) is loaded.
        SampleNotFoundError: if no sample matches.
    """
    audio = state.require_current().audio if source == "raw" else state.preview
    if audio is None:
        raise SessionError("no preview has been run yet — POST /api/preview first")
    sample = _sample_set(audio, kind).get(note, velocity)
    if sample is None:
        raise SampleNotFoundError(f"no {kind} sample at note={note} velocity={velocity}")
    return sample
