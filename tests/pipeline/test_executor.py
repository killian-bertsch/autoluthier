"""Loading, running, parallelism, and the progress stream."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autosampler.config.schema import StageConfig
from autosampler.domain.models import InstrumentAudio
from autosampler.pipeline import executor as executor_module
from autosampler.pipeline.events import (
    Event,
    ProgressReporter,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
    StepStarted,
)
from autosampler.pipeline.executor import (
    PipelineError,
    load_instrument_audio,
    resolve_workers,
    run_chain,
    run_instrument,
)
from autosampler.pipeline.graph import build_chain, partition_chain
from tests.pipeline.conftest import (
    HOLD_TIME,
    RELEASE_TIME,
    SAMPLE_RATE,
    Instrument,
    event_frames,
    write_instrument,
)


def fingerprint(audio: InstrumentAudio) -> list[tuple[object, ...]]:
    """A byte-exact signature of every sample: audio bytes plus loop metadata."""
    return [
        (
            sample.note,
            sample.velocity,
            sample.sample_rate,
            sample.audio.shape,
            sample.audio.tobytes(),
            sample.loop_start,
            sample.loop_end,
            sample.loop_crossfade_frames,
        )
        for group in (audio.sustain, audio.release)
        for sample in group
    ]


class TestLoad:
    def test_slices_every_recorded_pair_at_the_recorded_lengths(
        self, instrument: Instrument
    ) -> None:
        audio = load_instrument_audio(instrument.folder, instrument.config)
        hold_frames, release_frames = event_frames()
        assert len(audio.sustain) == len(instrument.notes) * len(instrument.velocities)
        assert len(audio.release) == len(audio.sustain)
        assert {s.n_frames for s in audio.sustain} == {hold_frames}
        assert {s.n_frames for s in audio.release} == {release_frames}

    def test_missing_sustain_render_is_a_clear_error(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", with_sustain=False)
        with pytest.raises(PipelineError, match="no sustain render"):
            load_instrument_audio(instrument.folder, instrument.config)

    def test_release_is_optional(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", with_release=False)
        audio = load_instrument_audio(instrument.folder, instrument.config)
        assert len(audio.sustain) == 6
        assert len(audio.release) == 0

    def test_mismatched_release_sample_rate_is_rejected(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", release_sample_rate=16000)
        with pytest.raises(PipelineError, match="sample rate mismatch"):
            load_instrument_audio(instrument.folder, instrument.config)

    def test_release_timing_overrides_drive_the_release_pass(self, tmp_path: Path) -> None:
        """A release render taken at its own hold/release timing is sliced by that timing."""
        instrument = write_instrument(
            tmp_path / "kb",
            recording={"release_hold_time": 0.25, "release_release_time": 0.75},
        )
        audio = load_instrument_audio(instrument.folder, instrument.config)
        hold_frames, _ = event_frames()
        assert {s.n_frames for s in audio.sustain} == {hold_frames}
        assert {s.n_frames for s in audio.release} == {round(0.75 * SAMPLE_RATE)}

    def test_notes_outside_the_selection_range_are_never_loaded(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", selection={"min_note": 61})
        audio = load_instrument_audio(instrument.folder, instrument.config)
        assert audio.sustain.notes() == [61, 62]

    def test_stereo_stays_stereo(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", stereo=True)
        audio = load_instrument_audio(instrument.folder, instrument.config)
        assert {s.n_channels for s in audio.sustain} == {2}

    def test_collapse_to_mono(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb", stereo=True, output={"collapse_to_mono": True}
        )
        audio = load_instrument_audio(instrument.folder, instrument.config)
        assert {s.n_channels for s in audio.sustain} == {1}

    def test_resampling_happens_at_load_so_the_chain_runs_at_the_output_rate(
        self, tmp_path: Path
    ) -> None:
        """Loop points must be expressed in frames of the audio that gets written."""
        instrument = write_instrument(tmp_path / "kb", output={"sample_rate": 16000})
        audio = load_instrument_audio(instrument.folder, instrument.config)
        hold_frames, _ = event_frames()
        assert {s.sample_rate for s in audio.sustain} == {16000}
        assert {s.n_frames for s in audio.sustain} == {hold_frames * 2}
        run_chain(audio, build_chain(instrument.config), workers=1)
        looped = [s for s in audio.sustain if s.loop_end is not None]
        assert looped
        assert all(s.loop_end is not None and s.loop_end <= s.n_frames for s in looped)


class TestRunInstrument:
    def test_default_chain_loops_sustain_only(self, instrument: Instrument) -> None:
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        assert all(s.loop_start is not None for s in result.audio.sustain)
        assert all(s.loop_start is None for s in result.audio.release)

    def test_dc_offset_is_removed_from_both_sets(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", stages=[StageConfig(id="dc")])
        pristine = load_instrument_audio(instrument.folder, instrument.config)
        assert all(abs(float(s.audio.mean())) > 0.01 for s in pristine.sustain)
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        for group in (result.audio.sustain, result.audio.release):
            assert all(abs(float(s.audio.mean())) < 1e-7 for s in group)

    def test_velocity_mode_reports_a_dynamic_range_for_the_sfz(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb", stages=[StageConfig(id="normalize", params={"mode": "velocity"})]
        )
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        assert result.dynamic_range_db is not None
        assert result.dynamic_range_db > 0.0

    def test_lufs_mode_has_no_dynamic_range_to_report(self, instrument: Instrument) -> None:
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        assert result.dynamic_range_db is None

    def test_instrument_name_from_config_is_honored(self, tmp_path: Path) -> None:
        """V1 parsed instrument_name and then overwrote it with the folder name (bug 11)."""
        instrument = write_instrument(tmp_path / "kb", instrument_name="Grand Keybass")
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        assert result.instrument_name == "Grand Keybass"

    def test_instrument_name_falls_back_to_the_folder_name(self, instrument: Instrument) -> None:
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        assert result.instrument_name == "keybass"

    def test_a_sustain_only_stage_leaves_release_samples_untouched(
        self, tmp_path: Path
    ) -> None:
        instrument = write_instrument(
            tmp_path / "kb",
            stages=[StageConfig(id="transient", params={"attack_db": 6.0})],
        )
        pristine = load_instrument_audio(instrument.folder, instrument.config)
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        for before, after in zip(pristine.release, result.audio.release, strict=True):
            assert np.array_equal(before.audio, after.audio)
        assert any(
            not np.array_equal(before.audio, after.audio)
            for before, after in zip(pristine.sustain, result.audio.sustain, strict=True)
        )

    def test_trim_shortens_every_sample_in_both_sets(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb", stages=[StageConfig(id="trim", params={"pre_trim_ms": 50.0})]
        )
        result = run_instrument(instrument.folder, instrument.config, workers=1)
        hold_frames, release_frames = event_frames()
        cut = round(0.05 * SAMPLE_RATE)
        assert {s.n_frames for s in result.audio.sustain} == {hold_frames - cut}
        assert {s.n_frames for s in result.audio.release} == {release_frames - cut}


class TestParallelism:
    def test_resolve_workers_never_exceeds_the_sample_count(self) -> None:
        assert resolve_workers(8, 3) == 3
        assert resolve_workers(1, 100) == 1
        assert resolve_workers(None, 1) == 1
        assert resolve_workers(-4, 10) == 1

    def test_parallel_result_is_bit_identical_to_serial(self, instrument: Instrument) -> None:
        config = instrument.config.model_copy(
            update={
                "stages": [
                    StageConfig(id="dc"),
                    StageConfig(id="trim", params={"pre_trim_ms": 5.0}),
                    StageConfig(id="transient", params={"attack_db": 4.0}),
                    StageConfig(id="normalize", params={"mode": "velocity"}),
                    StageConfig(id="eq", params={"freq_hz": 500.0, "gain_db": 3.0}),
                    StageConfig(id="limiter"),
                    StageConfig(id="loop"),
                ]
            }
        )
        serial = run_instrument(instrument.folder, config, workers=1)
        parallel = run_instrument(instrument.folder, config, workers=3)
        assert fingerprint(serial.audio) == fingerprint(parallel.audio)
        assert serial.dynamic_range_db == parallel.dynamic_range_db

    def test_the_pool_is_actually_used_and_chunks_the_work(
        self, instrument: Instrument, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards the parallel path itself: the identity test above is vacuous without it."""
        submits = 0
        real_pool = executor_module.ProcessPoolExecutor

        class CountingPool(real_pool):  # type: ignore[valid-type, misc]
            def submit(
                self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
            ) -> Future[Any]:
                nonlocal submits
                submits += 1
                return super().submit(fn, *args, **kwargs)

        monkeypatch.setattr(executor_module, "ProcessPoolExecutor", CountingPool)
        result = run_instrument(instrument.folder, instrument.config, workers=3)
        # 6 sustain + 6 release through "dc -> trim", then 6 sustain through "loop".
        assert submits == 18
        assert all(s.loop_start is not None for s in result.audio.sustain)

    def test_a_single_worker_starts_no_pool_at_all(
        self, instrument: Instrument, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("workers=1 must not start a process pool")

        monkeypatch.setattr(executor_module, "ProcessPoolExecutor", fail)
        run_instrument(instrument.folder, instrument.config, workers=1)

    def test_parallel_write_back_keeps_the_sample_set_index_valid(
        self, instrument: Instrument
    ) -> None:
        """A worker returns new arrays; `SampleSet`'s index must still find the live objects."""
        result = run_instrument(instrument.folder, instrument.config, workers=3)
        for sample in result.audio.sustain.samples:
            found = result.audio.sustain.get(sample.note, sample.velocity)
            assert found is sample
            assert found.loop_start is not None


class TestProgressEvents:
    def test_stream_is_bracketed_monotonic_and_covers_every_step(
        self, instrument: Instrument
    ) -> None:
        events: list[Event] = []
        run_instrument(instrument.folder, instrument.config, workers=1, sink=events.append)

        assert isinstance(events[0], RunStarted)
        assert isinstance(events[-1], RunCompleted)

        expected_steps = 1 + len(partition_chain(build_chain(instrument.config)))
        started = [e for e in events if isinstance(e, StepStarted)]
        completed = [e for e in events if isinstance(e, StepCompleted)]
        assert len(started) == expected_steps == events[0].total_steps
        assert [e.index for e in started] == list(range(expected_steps))
        assert [e.index for e in completed] == [e.index for e in started]
        assert [e.label for e in started] == ["load", "dc -> trim", "normalize", "loop"]

        fractions = [
            e.fraction for e in events if isinstance(e, StepProgress | StepCompleted)
        ]
        assert fractions == sorted(fractions)
        assert fractions[-1] == 1.0

    def test_load_step_counts_both_source_files(self, instrument: Instrument) -> None:
        events: list[Event] = []
        load_instrument_audio(
            instrument.folder,
            instrument.config,
            reporter=ProgressReporter(events.append, total_steps=1),
        )
        load_started = next(e for e in events if isinstance(e, StepStarted))
        assert load_started.total_units == 2

    def test_a_per_sample_step_reports_every_sample_it_touches(
        self, instrument: Instrument
    ) -> None:
        events: list[Event] = []
        run_instrument(instrument.folder, instrument.config, workers=1, sink=events.append)
        loop_started = next(e for e in events if isinstance(e, StepStarted) and e.label == "loop")
        loop_progress = [
            e for e in events if isinstance(e, StepProgress) and e.label == "loop"
        ]
        assert loop_started.total_units == 6  # sustain only; release is never looped
        assert loop_progress[-1].completed_units == 6

    def test_failure_is_announced_and_then_re_raised(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", with_sustain=False)
        events: list[Event] = []
        with pytest.raises(PipelineError):
            run_instrument(instrument.folder, instrument.config, sink=events.append)
        failed = events[-1]
        assert isinstance(failed, RunFailed)
        assert failed.error_type == "PipelineError"
        assert not any(isinstance(e, RunCompleted) for e in events)

    def test_run_completed_reports_the_full_sample_count(self, instrument: Instrument) -> None:
        events: list[Event] = []
        run_instrument(instrument.folder, instrument.config, workers=1, sink=events.append)
        completed = events[-1]
        assert isinstance(completed, RunCompleted)
        assert completed.sample_count == 12
        assert completed.duration_s > 0.0


class TestEmptyChain:
    def test_a_chain_with_every_stage_disabled_is_a_no_op_load(
        self, instrument: Instrument
    ) -> None:
        config = instrument.config.model_copy(
            update={"stages": [StageConfig(id="dc", enabled=False)]}
        )
        pristine = load_instrument_audio(instrument.folder, config)
        result = run_instrument(instrument.folder, config, workers=1)
        assert fingerprint(pristine) == fingerprint(result.audio)
        assert result.dynamic_range_db is None


def test_hold_and_release_times_match_the_fixture_contract() -> None:
    """Guards the fixture itself: the render is built with the timing the config declares."""
    assert (HOLD_TIME, RELEASE_TIME) == (1.0, 0.5)
