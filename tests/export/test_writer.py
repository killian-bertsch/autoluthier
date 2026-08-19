"""Export layout: which files land where, and what the opcodes say about them."""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from autosampler.config.schema import ProjectConfig
from autosampler.domain.models import InstrumentAudio, SampleSet
from autosampler.export.reports import VELOCITY_RANGE_FILENAME
from autosampler.export.writer import (
    ExportError,
    ExportPlan,
    ExportResult,
    export_instrument,
    plan_export,
    selected_velocities,
)
from autosampler.pipeline.executor import RunResult, run_instrument
from tests.pipeline.conftest import Instrument, write_instrument


def _run(folder: Path, **config_overrides: object) -> tuple[RunResult, ProjectConfig]:
    instrument: Instrument = write_instrument(folder, **config_overrides)  # type: ignore[arg-type]
    return run_instrument(folder, instrument.config, workers=1), instrument.config


def _export(tmp_path: Path, **config_overrides: object) -> tuple[ExportResult, ProjectConfig]:
    result, config = _run(tmp_path / "keybass", **config_overrides)
    return export_instrument(result, config, tmp_path / "out"), config


class TestLayout:
    def test_files_land_where_v1_put_them(self, tmp_path: Path) -> None:
        export, _ = _export(tmp_path)
        out = tmp_path / "out" / "keybass"
        assert export.output_dir == out
        assert (out / "keybass_sustain.sfz").is_file()
        assert (out / "keybass_release.sfz").is_file()
        assert (out / "samples" / "keybass_C4_v127.flac").is_file()
        assert (out / "samples" / "release" / "keybass_C4_v127_rel.flac").is_file()

    def test_every_planned_file_is_written_and_counted(self, tmp_path: Path) -> None:
        export, _ = _export(tmp_path)
        written = sorted(p.name for p in (export.output_dir / "samples").rglob("*.flac"))
        assert len(written) == export.audio_files
        assert export.audio_files == export.sustain_regions + export.release_regions

    def test_the_extension_follows_the_configured_container(self, tmp_path: Path) -> None:
        export, _ = _export(tmp_path, output={"container": "wav", "sample_format": "pcm24"})
        assert (export.output_dir / "samples" / "keybass_C4_v127.wav").is_file()
        assert "sample=keybass_C4_v127.wav" in export.sustain_sfz.read_text()

    def test_instrument_name_is_honored_rather_than_discarded(self, tmp_path: Path) -> None:
        """V1 bug 11: the field was parsed and then overwritten with the folder name."""
        export, _ = _export(tmp_path, instrument_name="Fender Rhodes")
        assert export.output_dir == tmp_path / "out" / "Fender Rhodes"
        assert export.sustain_sfz.name == "Fender Rhodes_sustain.sfz"
        assert (export.output_dir / "samples" / "Fender Rhodes_C4_v127.flac").is_file()

    def test_an_instrument_with_no_release_pass_gets_no_release_file(
        self, tmp_path: Path
    ) -> None:
        result, config = _run(tmp_path / "keybass", with_release=False)
        export = export_instrument(result, config, tmp_path / "out")
        assert export.release_sfz is None
        assert export.release_regions == 0
        assert not (export.output_dir / "samples" / "release").exists()

    def test_planning_writes_nothing(self, tmp_path: Path) -> None:
        result, config = _run(tmp_path / "keybass")
        out = tmp_path / "out"
        plan_export(result, config)
        assert not out.exists()

    def test_exporting_twice_overwrites_in_place(self, tmp_path: Path) -> None:
        result, config = _run(tmp_path / "keybass")
        first = export_instrument(result, config, tmp_path / "out")
        second = export_instrument(result, config, tmp_path / "out")
        assert first.output_dir == second.output_dir
        assert first.audio_files == second.audio_files


class TestLoopOpcodes:
    def test_baked_mode_omits_loop_crossfade_because_the_fade_is_in_the_audio(
        self, tmp_path: Path
    ) -> None:
        export, _ = _export(tmp_path, crossfade={"loop_crossfade_mode": "baked"})
        text = export.sustain_sfz.read_text()
        assert "loop_start=" in text
        assert "loop_crossfade" not in text

    def test_sfz_mode_emits_each_samples_own_clamped_length(self, tmp_path: Path) -> None:
        result, config = _run(
            tmp_path / "keybass",
            crossfade={"loop_crossfade_mode": "sfz", "loop_crossfade_ms": 20.0},
        )
        plan = plan_export(result, config)
        sample = result.audio.sustain.samples[0]
        expected = sample.loop_crossfade_frames / sample.sample_rate
        assert sample.loop_crossfade_frames > 0
        assert f"loop_crossfade={expected:.4f}" in plan.sustain.render()

    def test_a_disabled_loop_drops_every_loop_opcode_for_that_sample(
        self, tmp_path: Path
    ) -> None:
        result, config = _run(
            tmp_path / "keybass",
            overrides=[{"note": 60, "velocity": 127, "loop_disabled": True}],
        )
        plan = plan_export(result, config)
        blocks = plan.sustain.render().split("<region>")
        disabled = next(b for b in blocks if "sample=keybass_C4_v127.flac" in b)
        assert "loop_start" not in disabled
        assert sum("loop_start" in b for b in blocks) == len(plan.sustain.regions) - 1


class TestNoRescaling:
    def test_loop_points_stay_in_frames_of_the_written_audio(self, tmp_path: Path) -> None:
        """Resampling happens at load, so export must not touch loop points or rates again.

        The source is 8 kHz and the output 16 kHz, so a loop point that had been rescaled — or
        never scaled — would land in the first half of the file instead of in the 85%-97%
        search window near its end.
        """
        result, config = _run(tmp_path / "keybass", output={"sample_rate": 16000})
        export = export_instrument(result, config, tmp_path / "out")
        text = export.sustain_sfz.read_text()
        written = export.output_dir / "samples" / "keybass_C4_v127.flac"
        info = sf.info(str(written))
        assert info.samplerate == 16000
        assert info.frames == 16000
        loop_ends = [
            int(line.removeprefix("loop_end="))
            for line in text.splitlines()
            if line.startswith("loop_end=")
        ]
        assert loop_ends
        assert all(8000 < end <= info.frames for end in loop_ends)


class TestSelection:
    def test_note_percentage_thins_the_exported_notes(self, tmp_path: Path) -> None:
        export, _ = _export(tmp_path, selection={"note_percentage": 34.0})
        # 3 recorded notes, 34% -> 1 note; V1's tie-break keeps the middle one.
        assert export.sustain_regions == 2
        assert (export.output_dir / "samples" / "keybass_C#4_v127.flac").is_file()

    def test_velocity_map_wins_over_velocity_layers_out(self, tmp_path: Path) -> None:
        result, config = _run(
            tmp_path / "keybass",
            selection={"velocity_layers_out": 2, "velocity_map": "0-127:1"},
        )
        assert selected_velocities(result.audio.sustain, config) == [127]

    def test_a_selection_matching_nothing_fails_instead_of_writing_an_empty_instrument(
        self, tmp_path: Path
    ) -> None:
        result, config = _run(tmp_path / "keybass")
        empty = RunResult(
            instrument_name=result.instrument_name,
            audio=InstrumentAudio(sustain=SampleSet([]), release=SampleSet([])),
            dynamic_range_db=None,
            duration_s=0.0,
        )
        with pytest.raises(ExportError, match="no sustain samples"):
            plan_export(empty, config)

    def test_a_selected_sample_the_set_does_not_hold_is_skipped(self, tmp_path: Path) -> None:
        result, config = _run(tmp_path / "keybass")
        kept = [s for s in result.audio.sustain if (s.note, s.velocity) != (61, 127)]
        partial = RunResult(
            instrument_name=result.instrument_name,
            audio=InstrumentAudio(sustain=SampleSet(kept), release=result.audio.release),
            dynamic_range_db=None,
            duration_s=0.0,
        )
        plan = plan_export(partial, config)
        assert "keybass_C#4_v127.flac" not in plan.sustain.render()
        assert len(plan.sustain.regions) == len(result.audio.sustain) - 1


class TestDynamicRange:
    def test_velocity_mode_writes_the_measured_range_into_opcode_and_report(
        self, tmp_path: Path
    ) -> None:
        result, config = _run(
            tmp_path / "keybass",
            stages=[{"id": "dc"}, {"id": "normalize", "params": {"mode": "velocity"}}],
        )
        assert result.dynamic_range_db is not None
        export = export_instrument(result, config, tmp_path / "out")
        assert export.report is not None
        assert export.report.name == VELOCITY_RANGE_FILENAME
        assert f"{result.dynamic_range_db:.1f} dB" in export.report.read_text()
        expected = 10.0 ** (-result.dynamic_range_db / 20.0)
        assert f"amp_velcurve_1={expected:.6f}" in export.sustain_sfz.read_text()

    def test_other_modes_fall_back_to_the_configured_range_and_write_no_report(
        self, tmp_path: Path
    ) -> None:
        export, _ = _export(tmp_path, output={"velocity_dynamic_range_db": 20.0})
        assert export.report is None
        assert "amp_velcurve_1=0.100000" in export.sustain_sfz.read_text()

    def test_the_same_range_reaches_both_files(self, tmp_path: Path) -> None:
        export, _ = _export(tmp_path, output={"velocity_dynamic_range_db": 30.0})
        assert export.release_sfz is not None
        curve = "amp_velcurve_1=0.031623"
        assert curve in export.sustain_sfz.read_text()
        assert curve in export.release_sfz.read_text()


class TestPlanMatchesWhatIsWritten:
    def test_the_written_text_is_exactly_the_planned_document(self, tmp_path: Path) -> None:
        """Step 9's SFZ preview shows the plan, so it has to be what lands on disk."""
        result, config = _run(tmp_path / "keybass")
        plan: ExportPlan = plan_export(result, config)
        export = export_instrument(result, config, tmp_path / "out")
        assert export.sustain_sfz.read_text() == plan.sustain.render()
        assert plan.release is not None
        assert export.release_sfz is not None
        assert export.release_sfz.read_text() == plan.release.render()
