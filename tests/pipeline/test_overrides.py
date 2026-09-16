"""Per-sample overrides: they replace detection rather than patching its result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from autoluthier.config.schema import ProjectConfig, SampleOverride, StageConfig
from autoluthier.config.toml_io import load_project, save_project
from autoluthier.pipeline.executor import load_instrument_audio, run_chain, run_instrument
from autoluthier.pipeline.graph import build_chain
from tests.pipeline.conftest import Instrument, make_config, write_instrument

LOOP_ONLY = [StageConfig(id="loop")]


def _detected_points(instrument: Instrument) -> tuple[int, int]:
    """Run plain detection on the first sustain sample and return its loop points."""
    audio = load_instrument_audio(instrument.folder, instrument.config)
    run_chain(audio, build_chain(instrument.config), workers=1)
    sample = audio.sustain.samples[0]
    assert sample.loop_start is not None and sample.loop_end is not None
    return sample.loop_start, sample.loop_end


class TestOverrideModel:
    def test_half_specified_loop_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="set together"):
            SampleOverride(note=60, velocity=127, loop_start=100)

    def test_unordered_loop_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than"):
            SampleOverride(note=60, velocity=127, loop_start=500, loop_end=100)

    def test_disabled_plus_points_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be combined"):
            SampleOverride(
                note=60, velocity=127, loop_start=100, loop_end=500, loop_disabled=True
            )

    def test_duplicate_overrides_for_one_sample_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate override"):
            make_config(
                overrides=[
                    {"note": 60, "velocity": 127, "loop_disabled": True},
                    {"note": 60, "velocity": 127, "loop_start": 10, "loop_end": 20},
                ]
            )

    def test_two_overrides_for_different_samples_are_fine(self) -> None:
        config = make_config(
            overrides=[
                {"note": 60, "velocity": 127, "loop_disabled": True},
                {"note": 61, "velocity": 127, "loop_start": 10, "loop_end": 20},
            ]
        )
        assert set(config.override_table()) == {(60, 127), (61, 127)}

    def test_overrides_round_trip_through_toml(self, tmp_path: Path) -> None:
        config = make_config(
            overrides=[{"note": 61, "velocity": 1, "loop_start": 128, "loop_end": 4096}]
        )
        path = save_project(config, tmp_path / "project.toml")
        assert load_project(path).override_table()[(61, 1)].loop_points == (128, 4096)


class TestForcedLoopPoints:
    def _config_with_override(
        self, instrument: Instrument, start: int, end: int
    ) -> ProjectConfig:
        return instrument.config.model_copy(
            update={
                "stages": LOOP_ONLY,
                "overrides": [
                    SampleOverride(
                        note=instrument.notes[0],
                        velocity=instrument.velocities[0],
                        loop_start=start,
                        loop_end=end,
                    )
                ],
            }
        )

    def test_override_beats_detection(self, instrument: Instrument) -> None:
        detected = _detected_points(instrument)
        forced = (1000, 3000)
        assert forced != detected
        config = self._config_with_override(instrument, *forced)
        result = run_instrument(instrument.folder, config, workers=1)
        first = result.audio.sustain.samples[0]
        assert (first.loop_start, first.loop_end) == forced
        # Every other sample still gets auto-detection.
        others = result.audio.sustain.samples[1:]
        assert all((s.loop_start, s.loop_end) != forced for s in others)

    def test_baked_fade_lands_on_the_overridden_points(self, instrument: Instrument) -> None:
        """The seam identity ``out[end-1] == original[start-1]`` must hold at the *override*.

        This is why an override replaces detection instead of patching its result: a fade baked
        at auto-detected points could not be moved afterwards.
        """
        start, end = 1000, 3000
        config = self._config_with_override(instrument, start, end)
        pristine = load_instrument_audio(instrument.folder, config)
        result = run_instrument(instrument.folder, config, workers=1)
        processed = result.audio.sustain.samples[0]
        original = pristine.sustain.samples[0]
        assert processed.loop_crossfade_frames > 0
        assert processed.audio[end - 1] == original.audio[start - 1]

    def test_override_past_the_end_of_the_sample_names_the_sample(
        self, instrument: Instrument
    ) -> None:
        config = self._config_with_override(instrument, 100, 10**6)
        with pytest.raises(ValueError, match="note=60 velocity=1"):
            run_instrument(instrument.folder, config, workers=1)

    def test_override_on_a_release_sample_is_ignored(self, instrument: Instrument) -> None:
        """Release tails are never looped, so an override there has nothing to act on."""
        config = self._config_with_override(instrument, 100, 500)
        result = run_instrument(instrument.folder, config, workers=1)
        assert all(s.loop_start is None for s in result.audio.release)


class TestDisabledLoop:
    def test_disabling_leaves_the_sample_unlooped_and_unfaded(
        self, instrument: Instrument
    ) -> None:
        config = instrument.config.model_copy(
            update={
                "stages": LOOP_ONLY,
                "overrides": [
                    SampleOverride(note=60, velocity=1, loop_disabled=True),
                ],
            }
        )
        pristine = load_instrument_audio(instrument.folder, config)
        result = run_instrument(instrument.folder, config, workers=1)
        first = result.audio.sustain.samples[0]
        assert (first.loop_start, first.loop_end, first.loop_crossfade_frames) == (None, None, 0)
        assert np.array_equal(first.audio, pristine.sustain.samples[0].audio)
        assert all(s.loop_start is not None for s in result.audio.sustain.samples[1:])


class TestOverridesUnderParallelism:
    def test_parallel_and_serial_agree_with_overrides_applied(self, tmp_path: Path) -> None:
        instrument = write_instrument(
            tmp_path / "kb",
            stages=LOOP_ONLY,
            overrides=[
                {"note": 60, "velocity": 1, "loop_start": 1000, "loop_end": 3000},
                {"note": 61, "velocity": 127, "loop_disabled": True},
            ],
        )
        serial = run_instrument(instrument.folder, instrument.config, workers=1)
        parallel = run_instrument(instrument.folder, instrument.config, workers=3)
        for left, right in zip(serial.audio.sustain, parallel.audio.sustain, strict=True):
            assert (left.loop_start, left.loop_end) == (right.loop_start, right.loop_end)
            assert left.audio.tobytes() == right.audio.tobytes()
        forced = parallel.audio.sustain.get(60, 1)
        disabled = parallel.audio.sustain.get(61, 127)
        assert forced is not None and (forced.loop_start, forced.loop_end) == (1000, 3000)
        assert disabled is not None and disabled.loop_start is None
