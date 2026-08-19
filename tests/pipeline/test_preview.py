"""Preview: exact mode reproduces a full run bit-for-bit; subset mode trades that for speed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autosampler.config.schema import StageConfig
from autosampler.domain.models import InstrumentAudio
from autosampler.pipeline.events import Event, StepProgress
from autosampler.pipeline.executor import load_instrument_audio, run_instrument
from autosampler.pipeline.graph import build_chain
from autosampler.pipeline.preview import (
    PreviewSelection,
    run_preview,
    select_preview,
)
from tests.pipeline.conftest import Instrument, write_instrument

VELOCITY_NORMALIZE = [
    StageConfig(id="dc"),
    StageConfig(id="normalize", params={"mode": "velocity"}),
    StageConfig(id="loop"),
]


def signature(audio: InstrumentAudio, selection: PreviewSelection) -> dict[tuple[int, int], object]:
    """Byte-exact signature of the selected samples only."""
    return {
        (s.note, s.velocity): (
            s.audio.tobytes(),
            s.loop_start,
            s.loop_end,
            s.loop_crossfade_frames,
        )
        for group in (audio.sustain, audio.release)
        for s in group
        if selection.contains(s.note, s.velocity)
    }


class TestSelection:
    def test_one_note_is_the_middle_and_one_layer_is_the_loudest(
        self, instrument: Instrument
    ) -> None:
        audio = load_instrument_audio(instrument.folder, instrument.config)
        selection = select_preview(audio, note_count=1, velocity_count=1)
        assert selection.notes == (61,)
        assert selection.velocities == (127,)

    def test_asking_for_more_than_exists_returns_everything(
        self, instrument: Instrument
    ) -> None:
        audio = load_instrument_audio(instrument.folder, instrument.config)
        selection = select_preview(audio, note_count=99, velocity_count=99)
        assert selection.notes == (60, 61, 62)
        assert selection.velocities == (1, 127)

    def test_contains_is_the_grid(self) -> None:
        selection = PreviewSelection(notes=(60, 62), velocities=(127,))
        assert selection.contains(60, 127)
        assert not selection.contains(61, 127)
        assert not selection.contains(60, 1)


class TestExactMode:
    def test_previewed_samples_are_bit_identical_to_a_full_run(
        self, tmp_path: Path
    ) -> None:
        instrument = write_instrument(tmp_path / "kb", stages=VELOCITY_NORMALIZE)
        chain = build_chain(instrument.config)
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(60, 62), velocities=(1, 127))

        preview = run_preview(loaded, chain, selection, mode="exact")
        full = run_instrument(instrument.folder, instrument.config, workers=1)

        assert len(preview.audio.sustain) == 4
        assert signature(preview.audio, selection) == signature(full.audio, selection)

    def test_dynamic_range_matches_the_full_run(self, tmp_path: Path) -> None:
        instrument = write_instrument(tmp_path / "kb", stages=VELOCITY_NORMALIZE)
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(60,), velocities=(127,))
        preview = run_preview(loaded, build_chain(instrument.config), selection)
        full = run_instrument(instrument.folder, instrument.config, workers=1)
        assert preview.dynamic_range_db == full.dynamic_range_db

    def test_the_loaded_instrument_is_never_mutated(self, instrument: Instrument) -> None:
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        before = [s.audio.tobytes() for s in loaded.sustain]
        selection = select_preview(loaded)
        run_preview(loaded, build_chain(instrument.config), selection, mode="exact")
        assert [s.audio.tobytes() for s in loaded.sustain] == before
        assert all(s.loop_start is None for s in loaded.sustain)

    def test_release_samples_in_the_selection_are_previewed_too(
        self, instrument: Instrument
    ) -> None:
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(60,), velocities=(127,))
        preview = run_preview(loaded, build_chain(instrument.config), selection)
        assert len(preview.audio.release) == 1


class TestSubsetMode:
    def test_subset_barrier_statistics_differ_from_the_full_set(self, tmp_path: Path) -> None:
        """The honest reason exact mode exists: a group gain is not a per-sample property.

        The fixture's levels rise with note, so a subset of two notes has a different group mean
        than all three, and normalize scales the previewed samples differently.
        """
        instrument = write_instrument(tmp_path / "kb", stages=VELOCITY_NORMALIZE)
        chain = build_chain(instrument.config)
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(60, 62), velocities=(1, 127))

        subset = run_preview(loaded, chain, selection, mode="subset")
        full = run_instrument(instrument.folder, instrument.config, workers=1)

        assert subset.mode == "subset"
        assert signature(subset.audio, selection) != signature(full.audio, selection)

    def test_without_a_barrier_the_cheap_path_is_already_exact(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb", stages=[StageConfig(id="dc"), StageConfig(id="loop")]
        )
        chain = build_chain(instrument.config)
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(61,), velocities=(127,))

        for mode in ("exact", "subset"):
            preview = run_preview(loaded, chain, selection, mode=mode)  # type: ignore[arg-type]
            full = run_instrument(instrument.folder, instrument.config, workers=1)
            assert preview.mode == "exact"
            assert signature(preview.audio, selection) == signature(full.audio, selection)


class TestPreviewProgress:
    def test_fractions_stay_monotonic_across_the_two_chain_runs(
        self, instrument: Instrument
    ) -> None:
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        events: list[Event] = []
        run_preview(
            loaded,
            build_chain(instrument.config),
            select_preview(loaded),
            mode="exact",
            sink=events.append,
        )
        fractions = [e.fraction for e in events if isinstance(e, StepProgress)]
        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(1.0)


class TestOverridesInPreview:
    def test_an_override_shows_up_in_the_preview(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb",
            stages=[StageConfig(id="loop")],
            overrides=[{"note": 61, "velocity": 127, "loop_start": 1000, "loop_end": 3000}],
        )
        loaded = load_instrument_audio(instrument.folder, instrument.config)
        selection = PreviewSelection(notes=(61,), velocities=(127,))
        preview = run_preview(loaded, build_chain(instrument.config), selection)
        previewed = preview.audio.sustain.samples[0]
        assert (previewed.loop_start, previewed.loop_end) == (1000, 3000)
        assert not np.array_equal(previewed.audio, loaded.sustain.samples[0].audio)
