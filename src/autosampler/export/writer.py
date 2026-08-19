"""Turns a finished run into an instrument on disk: which files, where, and what they say.

Split deliberately in two. `plan_export` decides everything — which samples survive selection,
what each one is called, which key and velocity zone it covers, and the full text of both SFZ
documents — without touching the filesystem, so step 9's ``/api/sfz/preview`` can show exactly
what a render would write. `write_export` then does nothing but encode audio and write text.

Two things this module must *not* do, both decided upstream and easy to undo by accident:

- **No resampling and no rescaling of loop points.** `pipeline.executor.load_instrument_audio`
  resamples to ``output.sample_rate`` before any DSP runs, so loop points are already expressed
  in frames of the audio being written here. `io.writer.write_audio`'s own resample is a no-op
  safety net at this point.
- **No ``loop_crossfade`` opcode in ``baked`` mode.** The fade is in the audio already.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from autosampler.analysis.metrics import estimate_rt_decay
from autosampler.config.schema import OutputConfig, ProjectConfig
from autosampler.domain.models import Sample, SampleSet
from autosampler.domain.notes import midi_to_note_name, select_notes, select_velocities
from autosampler.domain.units import frames_to_seconds
from autosampler.export.reports import VELOCITY_RANGE_FILENAME, velocity_range_report
from autosampler.export.sfz import (
    ReleaseRegion,
    SfzDocument,
    SustainRegion,
    build_release_document,
    build_sustain_document,
)
from autosampler.export.zones import (
    KeyZone,
    VelocityZone,
    compute_key_zones,
    compute_velocity_zones,
    select_velocities_segmented,
)
from autosampler.io.writer import write_audio
from autosampler.pipeline.executor import RunResult

SAMPLES_DIRNAME = "samples"
RELEASE_DIRNAME = "release"
SUSTAIN_SFZ_SUFFIX = "_sustain.sfz"
RELEASE_SFZ_SUFFIX = "_release.sfz"
RELEASE_SAMPLE_SUFFIX = "_rel"


class ExportError(RuntimeError):
    """Raised when a run cannot be turned into an instrument."""


@dataclass(frozen=True, slots=True)
class SampleFile:
    """One audio file the export will write, and the sample it comes from."""

    sample: Sample
    relative_path: Path
    """Where the file goes, relative to the instrument's own output directory."""


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Everything a render would write, decided but not yet written."""

    instrument_name: str
    sustain: SfzDocument
    release: SfzDocument | None
    files: tuple[SampleFile, ...]
    measured_dynamic_range_db: float | None
    """Set only when velocity-mode normalize ran; drives the velocity-range report."""

    @property
    def sustain_sfz_name(self) -> str:
        """Filename of the sustain SFZ."""
        return f"{self.instrument_name}{SUSTAIN_SFZ_SUFFIX}"

    @property
    def release_sfz_name(self) -> str:
        """Filename of the release SFZ."""
        return f"{self.instrument_name}{RELEASE_SFZ_SUFFIX}"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What an export actually put on disk."""

    instrument_name: str
    output_dir: Path
    sustain_sfz: Path
    release_sfz: Path | None
    report: Path | None
    audio_files: int
    sustain_regions: int
    release_regions: int


def selected_notes(sustain: SampleSet, config: ProjectConfig) -> list[int]:
    """Return the MIDI notes that reach the output, ascending.

    Args:
        sustain: The processed sustain samples.
        config: The project configuration.

    Returns:
        The notes present in `sustain` that fall inside the configured range, thinned to
        ``selection.note_percentage``.
    """
    selection = config.selection
    in_range = [
        note for note in sustain.notes() if selection.min_note <= note <= selection.max_note
    ]
    return select_notes(in_range, selection.note_percentage)


def selected_velocities(sustain: SampleSet, config: ProjectConfig) -> list[int]:
    """Return the recorded velocity layers that reach the output, ascending.

    ``selection.velocity_map`` wins when present, since a per-range layer count is a more
    specific request than a single count.

    Args:
        sustain: The processed sustain samples.
        config: The project configuration.

    Returns:
        The selected MIDI velocities.
    """
    recorded = sustain.velocities()
    segments = config.selection.parsed_velocity_map()
    if segments is not None:
        return select_velocities_segmented(recorded, segments)
    return select_velocities(recorded, config.selection.velocity_layers_out)


def _sample_filename(
    instrument_name: str, note: int, velocity: int, extension: str, *, release: bool
) -> str:
    """Build one sample's filename, e.g. ``keybass_C4_v127.flac``."""
    suffix = RELEASE_SAMPLE_SUFFIX if release else ""
    return (
        f"{instrument_name}_{midi_to_note_name(note)}_v{velocity}{suffix}.{extension}"
    )


def _loop_crossfade_seconds(sample: Sample, config: ProjectConfig) -> float | None:
    """Return the ``loop_crossfade`` opcode value for `sample`, or ``None`` to omit it.

    Emitted only in ``sfz`` crossfade mode, and taken from the length `dsp.loop` actually
    settled on for this sample after clamping — not from ``crossfade.loop_crossfade_ms``,
    which is only the request.
    """
    if config.crossfade.loop_crossfade_mode != "sfz":
        return None
    if sample.loop_crossfade_frames <= 0:
        return None
    return frames_to_seconds(sample.loop_crossfade_frames, sample.sample_rate)


def _iter_selected(
    samples: SampleSet,
    key_zones: Sequence[KeyZone],
    velocity_zones: Sequence[VelocityZone],
) -> list[tuple[Sample, KeyZone, VelocityZone]]:
    """Pair each selected sample with its zones, note-outer and velocity-inner.

    Samples the selection asks for but the set does not hold are skipped rather than raising:
    a truncated or partial recording should still export what it does have.
    """
    paired: list[tuple[Sample, KeyZone, VelocityZone]] = []
    for key_zone in key_zones:
        for velocity_zone in velocity_zones:
            sample = samples.get(key_zone.note, velocity_zone.velocity)
            if sample is not None:
                paired.append((sample, key_zone, velocity_zone))
    return paired


def plan_export(result: RunResult, config: ProjectConfig) -> ExportPlan:
    """Decide the whole instrument — filenames, zones, and both SFZ documents — on paper.

    Args:
        result: A completed run from `pipeline.executor.run_instrument`.
        config: The project configuration the run used.

    Returns:
        The `ExportPlan` describing everything a render would write.

    Raises:
        ExportError: if selection leaves no sustain samples at all, which would otherwise
            write an empty, unloadable instrument.
    """
    notes = selected_notes(result.audio.sustain, config)
    velocities = selected_velocities(result.audio.sustain, config)
    key_zones = compute_key_zones(notes, config.selection.min_note, config.selection.max_note)
    velocity_zones = compute_velocity_zones(velocities, config.crossfade.crossfade_percent)

    extension = config.output.container
    dynamic_range_db = (
        result.dynamic_range_db
        if result.dynamic_range_db is not None
        else config.output.velocity_dynamic_range_db
    )
    samples_dir = Path(SAMPLES_DIRNAME)
    release_dir = samples_dir / RELEASE_DIRNAME

    files: list[SampleFile] = []
    sustain_regions: list[SustainRegion] = []
    for sample, key_zone, velocity_zone in _iter_selected(
        result.audio.sustain, key_zones, velocity_zones
    ):
        filename = _sample_filename(
            result.instrument_name, sample.note, sample.velocity, extension, release=False
        )
        files.append(SampleFile(sample=sample, relative_path=samples_dir / filename))
        loop = (
            (sample.loop_start, sample.loop_end)
            if sample.loop_start is not None and sample.loop_end is not None
            else None
        )
        sustain_regions.append(
            SustainRegion(
                filename=filename,
                key=key_zone,
                velocity=velocity_zone,
                loop=loop,
                loop_crossfade_s=(
                    None if loop is None else _loop_crossfade_seconds(sample, config)
                ),
            )
        )

    if not sustain_regions:
        raise ExportError(
            f"selection left no sustain samples for '{result.instrument_name}': "
            f"notes {config.selection.min_note}-{config.selection.max_note} at "
            f"{config.selection.note_percentage}% matched none of the recorded set"
        )

    release_regions: list[ReleaseRegion] = []
    for sample, key_zone, velocity_zone in _iter_selected(
        result.audio.release, key_zones, velocity_zones
    ):
        filename = _sample_filename(
            result.instrument_name, sample.note, sample.velocity, extension, release=True
        )
        files.append(SampleFile(sample=sample, relative_path=release_dir / filename))
        release_regions.append(
            ReleaseRegion(
                filename=filename,
                key=key_zone,
                velocity=velocity_zone,
                rt_decay=estimate_rt_decay(sample.audio, sample.sample_rate),
            )
        )

    ampeg_release = config.output.ampeg_release
    return ExportPlan(
        instrument_name=result.instrument_name,
        sustain=build_sustain_document(
            sustain_regions, ampeg_release=ampeg_release, dynamic_range_db=dynamic_range_db
        ),
        release=(
            build_release_document(
                release_regions,
                ampeg_release=ampeg_release,
                dynamic_range_db=dynamic_range_db,
            )
            if release_regions
            else None
        ),
        files=tuple(files),
        measured_dynamic_range_db=result.dynamic_range_db,
    )


def write_export(plan: ExportPlan, output_root: Path, output: OutputConfig) -> ExportResult:
    """Write everything `plan` describes under ``output_root/<instrument_name>/``.

    Args:
        plan: The plan from `plan_export`.
        output_root: Directory the instrument folder is created inside.
        output: Container/bit-depth settings for the audio files.

    Returns:
        The `ExportResult` naming what was written.
    """
    out_dir = output_root / plan.instrument_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in plan.files:
        write_audio(
            out_dir / entry.relative_path,
            entry.sample.audio,
            entry.sample.sample_rate,
            output,
        )

    sustain_path = out_dir / plan.sustain_sfz_name
    sustain_path.write_text(plan.sustain.render(), encoding="utf-8")

    release_path: Path | None = None
    if plan.release is not None:
        release_path = out_dir / plan.release_sfz_name
        release_path.write_text(plan.release.render(), encoding="utf-8")

    report_path: Path | None = None
    if plan.measured_dynamic_range_db is not None:
        report_path = out_dir / VELOCITY_RANGE_FILENAME
        report_path.write_text(
            velocity_range_report(plan.instrument_name, plan.measured_dynamic_range_db),
            encoding="utf-8",
        )

    return ExportResult(
        instrument_name=plan.instrument_name,
        output_dir=out_dir,
        sustain_sfz=sustain_path,
        release_sfz=release_path,
        report=report_path,
        audio_files=len(plan.files),
        sustain_regions=len(plan.sustain.regions),
        release_regions=0 if plan.release is None else len(plan.release.regions),
    )


def export_instrument(
    result: RunResult, config: ProjectConfig, output_root: Path
) -> ExportResult:
    """Plan and write one finished run in a single call.

    Args:
        result: A completed run from `pipeline.executor.run_instrument`.
        config: The project configuration the run used.
        output_root: Directory the instrument folder is created inside.

    Returns:
        The `ExportResult` naming what was written.

    Raises:
        ExportError: if selection leaves no sustain samples.
    """
    return write_export(plan_export(result, config), output_root, config.output)
