"""Pydantic v2 config schema for one instrument project (``project.toml``).

Every model here uses ``model_config = ConfigDict(extra="forbid", strict=True)``: `extra`
turns a typo'd field name into a validation error instead of V1's silent ignore-and-default
(bug 9), and `strict` stops TOML's ``true``/``false`` from being quietly coerced into ``1``/``0``
for an int field, since bool is a Python int subclass (bug 4).

Field metadata (unit, step, group, ...) is attached via ``json_schema_extra=ui_hint(...)`` so
``ProjectConfig.model_json_schema()`` alone is enough to drive the frontend's parameter forms
and the CLI later — a field defined once here appears in TOML, validation, and the UI.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from autosampler.config.hints import ui_hint

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
"""A JSON-compatible value: what TOML/JSON Schema can actually represent."""

SampleFormat = Literal["pcm16", "pcm24", "pcm32", "float32"]
ContainerFormat = Literal["flac", "wav"]
LoopCrossfadeMode = Literal["baked", "sfz"]
LoopCrossfadeShape = Literal["linear", "equal_power"]

_MIN_MIDI_VELOCITY = 0
_MAX_MIDI_VELOCITY = 127


def parse_velocity_map(raw: str) -> list[tuple[int, int, int]]:
    """Parse a ``velocity_map`` string into ``(lo, hi, n)`` segments sorted by ``lo``.

    Format: ``"lo-hi:n, lo-hi:n, ..."``, e.g. ``"0-63:1, 64-127:5"`` — sparse low end, dense
    high end. ``lo``/``hi`` are inclusive MIDI velocities (0-127); ``n`` is the number of
    recorded layers to keep from that range.

    Args:
        raw: The raw ``velocity_map`` string.

    Returns:
        Parsed segments sorted by ``lo``.

    Raises:
        ValueError: on malformed syntax or out-of-range values.
    """
    segments: list[tuple[int, int, int]] = []
    for raw_part in raw.split(","):
        part = raw_part.strip()
        if not part:
            continue
        range_part, sep, n_str = part.partition(":")
        if not sep:
            raise ValueError(f"velocity_map segment '{part}' is missing ':n' — expected 'lo-hi:n'")
        lo_str, sep2, hi_str = range_part.partition("-")
        if not sep2:
            raise ValueError(f"velocity_map segment '{part}' is missing '-' — expected 'lo-hi:n'")
        try:
            lo, hi, n = int(lo_str.strip()), int(hi_str.strip()), int(n_str.strip())
        except ValueError as exc:
            raise ValueError(f"velocity_map segment '{part}' has non-integer lo/hi/n") from exc
        if not (_MIN_MIDI_VELOCITY <= lo <= _MAX_MIDI_VELOCITY):
            raise ValueError(f"velocity_map segment '{part}': lo/hi must be within 0-127")
        if not (_MIN_MIDI_VELOCITY <= hi <= _MAX_MIDI_VELOCITY):
            raise ValueError(f"velocity_map segment '{part}': lo/hi must be within 0-127")
        if lo > hi:
            raise ValueError(f"velocity_map segment '{part}': lo must be <= hi")
        if n < 1:
            raise ValueError(f"velocity_map segment '{part}': layer count must be >= 1")
        segments.append((lo, hi, n))
    if not segments:
        raise ValueError("velocity_map must contain at least one segment")
    segments.sort(key=lambda segment: segment[0])
    return segments


class RecordingConfig(BaseModel):
    """How the source sustain/release audio was recorded.

    Must exactly match the autosampler MIDI session used to render ``sustain.wav`` (and
    ``release.wav``, if recorded) — slicing is pure arithmetic off these values, with no
    onset detection, so any drift here misaligns every sample.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    velocity_layers: int = Field(
        ge=1,
        json_schema_extra=ui_hint(
            group="Recording", order=0, help_text="Number of velocity layers recorded."
        ),
    )
    semitone_interval: int = Field(
        ge=1,
        json_schema_extra=ui_hint(
            group="Recording",
            order=1,
            unit="semitones",
            help_text="Semitone step between recorded notes.",
        ),
    )
    hold_time: float = Field(
        gt=0,
        json_schema_extra=ui_hint(
            group="Recording",
            order=2,
            unit="s",
            step=0.1,
            help_text="Seconds each note was held in sustain.wav.",
        ),
    )
    release_time: float = Field(
        ge=0,
        json_schema_extra=ui_hint(
            group="Recording",
            order=3,
            unit="s",
            step=0.1,
            help_text="Silence after note-off in sustain.wav.",
        ),
    )
    start_note: int = Field(
        default=21,
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Recording", order=4, help_text="Lowest recorded MIDI note."
        ),
    )
    end_note: int = Field(
        default=108,
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Recording", order=5, help_text="Highest recorded MIDI note."
        ),
    )
    release_hold_time: float | None = Field(
        default=None,
        gt=0,
        json_schema_extra=ui_hint(
            group="Recording",
            order=6,
            unit="s",
            help_text="Overrides hold_time for release.wav, if it used different timing.",
        ),
    )
    release_release_time: float | None = Field(
        default=None,
        ge=0,
        json_schema_extra=ui_hint(
            group="Recording",
            order=7,
            unit="s",
            help_text="Overrides release_time for release.wav.",
        ),
    )

    @model_validator(mode="after")
    def _check_note_range(self) -> Self:
        """Ensure ``start_note <= end_note``."""
        if self.start_note > self.end_note:
            raise ValueError("start_note must be <= end_note")
        return self


class SelectionConfig(BaseModel):
    """Filters and thins the recorded ``(note, velocity)`` set for the output SFZ."""

    model_config = ConfigDict(extra="forbid", strict=True)

    min_note: int = Field(
        default=21,
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Selection", order=0, help_text="Lowest MIDI note to include in output."
        ),
    )
    max_note: int = Field(
        default=108,
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Selection", order=1, help_text="Highest MIDI note to include in output."
        ),
    )
    note_percentage: float = Field(
        default=100.0,
        ge=1.0,
        le=100.0,
        json_schema_extra=ui_hint(
            group="Selection",
            order=2,
            unit="%",
            help_text="Percentage of in-range notes to keep, evenly spaced.",
        ),
    )
    velocity_layers_out: int = Field(
        ge=1,
        json_schema_extra=ui_hint(
            group="Selection",
            order=3,
            help_text="Velocity layers to keep in the output (<= recording.velocity_layers).",
        ),
    )
    velocity_map: str | None = Field(
        default=None,
        json_schema_extra=ui_hint(
            group="Selection",
            order=4,
            help_text=(
                "Optional per-range layer counts, e.g. '0-63:1, 64-127:5', "
                "overriding velocity_layers_out."
            ),
        ),
    )

    @model_validator(mode="after")
    def _check_note_range(self) -> Self:
        """Ensure ``min_note <= max_note``."""
        if self.min_note > self.max_note:
            raise ValueError("min_note must be <= max_note")
        return self

    @field_validator("velocity_map")
    @classmethod
    def _validate_velocity_map(cls, value: str | None) -> str | None:
        """Reject a malformed ``velocity_map`` early; the raw string is kept as-is."""
        if value is not None:
            parse_velocity_map(value)
        return value

    def parsed_velocity_map(self) -> list[tuple[int, int, int]] | None:
        """Return the parsed ``(lo, hi, n)`` segments, or ``None`` if unset."""
        if self.velocity_map is None:
            return None
        return parse_velocity_map(self.velocity_map)


class CrossfadeConfig(BaseModel):
    """Overlap behavior between adjacent velocity zones and baked sample loops.

    Everything about the *loop* crossfade lives here (length, mode, and curve); everything about
    loop *detection* lives in ``dsp.loop.LoopParams``. That split is why `LoopParams` has no
    crossfade fields of its own — each field is defined exactly once.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    crossfade_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        json_schema_extra=ui_hint(
            group="Crossfade",
            order=0,
            unit="%",
            help_text="Percent of each velocity zone that overlaps its neighbors (xfin/xfout).",
        ),
    )
    loop_crossfade_ms: float = Field(
        default=20.0,
        ge=0.0,
        json_schema_extra=ui_hint(
            group="Crossfade",
            order=1,
            unit="ms",
            step=1.0,
            help_text="Baked crossfade length at each detected loop point.",
        ),
    )
    loop_crossfade_mode: LoopCrossfadeMode = Field(
        default="baked",
        json_schema_extra=ui_hint(
            group="Crossfade",
            order=2,
            help_text=(
                "baked renders the fade into the sample itself, so it sounds identical in "
                "every sampler. sfz leaves the audio untouched and emits a loop_crossfade "
                "opcode instead, letting the sampler fade at playback time (V1's behavior) — "
                "only works in samplers that implement the opcode."
            ),
        ),
    )
    loop_crossfade_shape: LoopCrossfadeShape = Field(
        default="linear",
        json_schema_extra=ui_hint(
            group="Crossfade",
            order=3,
            help_text=(
                "Loop crossfade curve; baked mode only, since in sfz mode the curve is the "
                "sampler's choice. linear holds a steady level on the phase-matched, highly "
                "correlated material the loop finder selects; equal_power preserves power on "
                "decorrelated (noisy) sustains but lifts correlated material ~3 dB."
            ),
        ),
    )


class OutputConfig(BaseModel):
    """Output audio container/encoding and instrument-level SFZ playback settings."""

    model_config = ConfigDict(extra="forbid", strict=True)

    container: ContainerFormat = Field(
        default="flac",
        json_schema_extra=ui_hint(
            group="Output", order=0, help_text="Output audio container format."
        ),
    )
    sample_format: SampleFormat = Field(
        default="pcm24",
        json_schema_extra=ui_hint(
            group="Output", order=1, help_text="Output bit depth/encoding."
        ),
    )
    sample_rate: int | None = Field(
        default=None,
        ge=8_000,
        le=192_000,
        json_schema_extra=ui_hint(
            group="Output",
            order=2,
            unit="Hz",
            help_text="Resample target; omit to keep the source rate.",
        ),
    )
    collapse_to_mono: bool = Field(
        default=False,
        json_schema_extra=ui_hint(
            group="Output", order=3, help_text="Mix stereo input down to mono on load."
        ),
    )
    ampeg_release: float = Field(
        default=0.5,
        ge=0.0,
        json_schema_extra=ui_hint(
            group="Output",
            order=4,
            unit="s",
            step=0.05,
            help_text="Sampler release time written into the SFZ.",
        ),
    )


class StageConfig(BaseModel):
    """One entry in the ordered DSP stage chain.

    ``params`` is intentionally untyped JSON here — concrete parameter models are defined
    alongside each stage under ``autosampler.dsp`` and registered in ``dsp/registry.py``
    (steps 3-5). The pipeline builder validates ``params`` against the model registered for
    ``id`` once those stages exist; until then this model only guarantees a well-formed,
    reorderable, toggleable chain shape.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    enabled: bool = True
    params: dict[str, JSONValue] = Field(default_factory=dict)


def default_stage_chain() -> list[StageConfig]:
    """Canonical stage order for a new project.

    Transient shaping runs *before* normalize (fixes V1 bug 10, where the shaper ran after
    the peak ceiling so ``peak_ceiling_db`` was not actually the final peak). The new optional
    stages (eq, stereo, limiter) default to disabled so a fresh project reproduces V1-equivalent
    processing out of the box; the chain is reorderable and toggleable from the UI with no code
    change once the DSP registry (step 3+) exists.
    """
    return [
        StageConfig(id="dc"),
        StageConfig(id="trim"),
        StageConfig(id="transient", enabled=False),
        StageConfig(id="normalize"),
        StageConfig(id="eq", enabled=False),
        StageConfig(id="stereo", enabled=False),
        StageConfig(id="limiter", enabled=False),
        StageConfig(id="loop"),
    ]


class SampleOverride(BaseModel):
    """A persisted manual decision about one ``(note, velocity)`` sample.

    Overrides always beat auto-detection: the pipeline hands them to `dsp.loop` *instead of*
    running detection for that sample, rather than letting detection run and then patching the
    result. That ordering is what makes them correct in ``baked`` crossfade mode — the fade has
    to be rendered at the loop points that will actually be used, and a fade baked at
    auto-detected points could not be moved afterwards.

    Only loop points are overridable today, which is what the waveform editor (step 11) edits.
    New override fields belong here, and `pipeline.graph.LoopStep` is where a config override is
    translated into the primitives `dsp.loop` takes.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    note: int = Field(
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Overrides", order=0, help_text="MIDI note this override applies to."
        ),
    )
    velocity: int = Field(
        ge=0,
        le=127,
        json_schema_extra=ui_hint(
            group="Overrides",
            order=1,
            help_text="Recorded MIDI velocity this override applies to.",
        ),
    )
    loop_start: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra=ui_hint(
            group="Overrides",
            order=2,
            unit="frames",
            help_text="Manual loop start in frames; replaces detection for this sample.",
        ),
    )
    loop_end: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra=ui_hint(
            group="Overrides",
            order=3,
            unit="frames",
            help_text="Manual loop end in frames; replaces detection for this sample.",
        ),
    )
    loop_disabled: bool = Field(
        default=False,
        json_schema_extra=ui_hint(
            group="Overrides",
            order=4,
            help_text="Force this sample to have no loop at all, whatever detection would find.",
        ),
    )

    @model_validator(mode="after")
    def _check_loop_points(self) -> Self:
        """Reject a half-specified or self-contradictory loop override.

        Returns:
            The validated model.

        Raises:
            ValueError: if only one of ``loop_start``/``loop_end`` is set, if they are not
                strictly ordered, or if explicit points are combined with ``loop_disabled``.
        """
        if (self.loop_start is None) != (self.loop_end is None):
            raise ValueError(
                "loop_start and loop_end must be set together — a loop needs both ends; "
                f"got loop_start={self.loop_start}, loop_end={self.loop_end}"
            )
        if self.loop_start is not None and self.loop_end is not None:
            if self.loop_end <= self.loop_start:
                raise ValueError(
                    f"loop_end ({self.loop_end}) must be greater than "
                    f"loop_start ({self.loop_start})"
                )
            if self.loop_disabled:
                raise ValueError(
                    "loop_disabled cannot be combined with explicit loop points — "
                    "drop the points to disable the loop, or drop loop_disabled to use them"
                )
        return self

    @property
    def key(self) -> tuple[int, int]:
        """The ``(note, velocity)`` pair this override addresses."""
        return (self.note, self.velocity)

    @property
    def loop_points(self) -> tuple[int, int] | None:
        """``(loop_start, loop_end)`` if both are set, else ``None``."""
        if self.loop_start is None or self.loop_end is None:
            return None
        return (self.loop_start, self.loop_end)


class ProjectConfig(BaseModel):
    """Everything needed to process one instrument.

    Lives at ``<instrument_dir>/project.toml``. Recording and output-selection sections are
    required — there is no sensible default for values that must match an actual MIDI
    recording — everything else has a default matching V1's out-of-the-box behavior.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    instrument_name: str | None = Field(
        default=None,
        json_schema_extra=ui_hint(
            order=0,
            help_text=(
                "Overrides the instrument name used in output filenames and the SFZ; "
                "defaults to the project folder name if unset."
            ),
        ),
    )
    recording: RecordingConfig
    selection: SelectionConfig
    crossfade: CrossfadeConfig = Field(default_factory=CrossfadeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    stages: list[StageConfig] = Field(default_factory=default_stage_chain)
    overrides: list[SampleOverride] = Field(
        default_factory=list,
        json_schema_extra=ui_hint(
            group="Overrides",
            order=5,
            help_text="Per-sample manual decisions that always beat auto-detection.",
        ),
    )

    @model_validator(mode="after")
    def _check_velocity_layers_out(self) -> Self:
        """Ensure velocity_layers_out does not exceed the number of recorded layers."""
        if self.selection.velocity_layers_out > self.recording.velocity_layers:
            raise ValueError(
                f"selection.velocity_layers_out ({self.selection.velocity_layers_out}) cannot "
                f"exceed recording.velocity_layers ({self.recording.velocity_layers})"
            )
        return self

    @model_validator(mode="after")
    def _check_unique_overrides(self) -> Self:
        """Reject two overrides addressing the same sample.

        Returns:
            The validated model.

        Raises:
            ValueError: if any ``(note, velocity)`` pair appears twice. Last-one-wins would be
                exactly the kind of silent config failure `extra="forbid"` exists to prevent.
        """
        seen: set[tuple[int, int]] = set()
        for override in self.overrides:
            if override.key in seen:
                raise ValueError(
                    f"duplicate override for note={override.note} velocity={override.velocity}"
                )
            seen.add(override.key)
        return self

    def override_table(self) -> dict[tuple[int, int], SampleOverride]:
        """Return the overrides indexed by ``(note, velocity)`` for O(1) per-sample lookup."""
        return {override.key: override for override in self.overrides}
