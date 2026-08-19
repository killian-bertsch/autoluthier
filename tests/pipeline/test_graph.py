"""Chain building: the two special-cased ids, validation, targets, and segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from autosampler.config.schema import StageConfig
from autosampler.domain.models import Sample
from autosampler.dsp.loop import LoopParams
from autosampler.pipeline.graph import (
    BarrierSegment,
    BufferStep,
    ChainConfigError,
    LoopStep,
    NormalizeStep,
    PreparedSteps,
    SampleSegment,
    build_chain,
    partition_chain,
    split_at_last_barrier,
    targets_release,
    targets_sustain,
)
from tests.pipeline.conftest import make_config


class TestBuildChain:
    def test_default_chain_drops_disabled_stages(self) -> None:
        chain = build_chain(make_config())
        assert [step.stage_id for step in chain] == ["dc", "trim", "normalize", "loop"]

    def test_enabled_optional_stages_appear_in_configured_order(self) -> None:
        config = make_config(
            stages=[
                StageConfig(id="eq", params={"freq_hz": 800.0}),
                StageConfig(id="normalize"),
                StageConfig(id="limiter"),
            ]
        )
        assert [step.stage_id for step in build_chain(config)] == ["eq", "normalize", "limiter"]

    def test_unknown_stage_id_names_the_id_and_the_known_ones(self) -> None:
        config = make_config(stages=[StageConfig(id="reverb")])
        with pytest.raises(ChainConfigError, match="reverb"):
            build_chain(config)

    def test_invalid_params_fail_before_any_audio_is_read(self) -> None:
        config = make_config(stages=[StageConfig(id="trim", params={"pre_trim_ms": -5.0})])
        with pytest.raises(ChainConfigError, match="trim"):
            build_chain(config)

    def test_unknown_param_key_is_rejected(self) -> None:
        config = make_config(stages=[StageConfig(id="trim", params={"pre_trim_msec": 5.0})])
        with pytest.raises(ChainConfigError):
            build_chain(config)

    def test_repeated_buffer_stage_is_allowed(self) -> None:
        config = make_config(
            stages=[
                StageConfig(id="eq", params={"freq_hz": 200.0}),
                StageConfig(id="eq", params={"freq_hz": 4000.0}),
            ]
        )
        assert [step.stage_id for step in build_chain(config)] == ["eq", "eq"]

    @pytest.mark.parametrize("stage_id", ["loop", "normalize"])
    def test_repeated_non_idempotent_stage_is_rejected(self, stage_id: str) -> None:
        config = make_config(stages=[StageConfig(id=stage_id), StageConfig(id=stage_id)])
        with pytest.raises(ChainConfigError, match="more than once"):
            build_chain(config)

    def test_disabled_duplicate_does_not_trip_the_uniqueness_check(self) -> None:
        config = make_config(
            stages=[StageConfig(id="loop"), StageConfig(id="loop", enabled=False)]
        )
        assert [step.stage_id for step in build_chain(config)] == ["loop"]


class TestLoopStepWiring:
    """`loop`'s crossfade settings come from `CrossfadeConfig`, never from its own params."""

    def test_crossfade_settings_come_from_the_crossfade_section(self) -> None:
        config = make_config(
            crossfade={
                "loop_crossfade_ms": 7.5,
                "loop_crossfade_mode": "sfz",
                "loop_crossfade_shape": "equal_power",
            },
            stages=[StageConfig(id="loop", params={"min_loop_ms": 25.0})],
        )
        step = build_chain(config)[0]
        assert isinstance(step, LoopStep)
        assert step.crossfade_ms == 7.5
        assert step.crossfade_mode == "sfz"
        assert step.crossfade_shape == "equal_power"
        assert step.params == LoopParams(min_loop_ms=25.0)

    def test_crossfade_length_is_not_accepted_as_a_loop_param(self) -> None:
        config = make_config(
            stages=[StageConfig(id="loop", params={"loop_crossfade_ms": 20.0})]
        )
        with pytest.raises(ChainConfigError):
            build_chain(config)

    def test_overrides_reach_the_loop_step_indexed_by_note_and_velocity(self) -> None:
        config = make_config(
            overrides=[{"note": 61, "velocity": 127, "loop_start": 100, "loop_end": 500}]
        )
        step = build_chain(config)[-1]
        assert isinstance(step, LoopStep)
        assert step.overrides[(61, 127)].loop_points == (100, 500)


class TestTargets:
    def test_transient_is_sustain_only_matching_v1(self) -> None:
        config = make_config(stages=[StageConfig(id="transient")])
        step = build_chain(config)[0]
        assert isinstance(step, BufferStep)
        assert targets_sustain(step)
        assert not targets_release(step)

    @pytest.mark.parametrize("stage_id", ["dc", "trim", "eq", "stereo", "limiter"])
    def test_buffer_stages_that_apply_to_both_sets(self, stage_id: str) -> None:
        config = make_config(stages=[StageConfig(id=stage_id)])
        step = build_chain(config)[0]
        assert targets_sustain(step)
        assert targets_release(step)

    def test_loop_is_sustain_only(self) -> None:
        step = build_chain(make_config(stages=[StageConfig(id="loop")]))[0]
        assert targets_sustain(step)
        assert not targets_release(step)


class TestPartitionChain:
    def test_consecutive_per_sample_steps_share_one_segment(self) -> None:
        segments = partition_chain(build_chain(make_config()))
        assert [type(segment) for segment in segments] == [
            SampleSegment,
            BarrierSegment,
            SampleSegment,
        ]
        assert [segment.stage_ids for segment in segments] == [
            ("dc", "trim"),
            ("normalize",),
            ("loop",),
        ]

    def test_label_reads_as_the_stage_run(self) -> None:
        segments = partition_chain(build_chain(make_config()))
        assert segments[0].label == "dc -> trim"

    def test_a_chain_without_a_barrier_is_one_segment(self) -> None:
        config = make_config(stages=[StageConfig(id="dc"), StageConfig(id="loop")])
        assert len(partition_chain(build_chain(config))) == 1

    def test_a_leading_barrier_yields_no_empty_segment(self) -> None:
        config = make_config(stages=[StageConfig(id="normalize"), StageConfig(id="dc")])
        segments = partition_chain(build_chain(config))
        assert [type(segment) for segment in segments] == [BarrierSegment, SampleSegment]

    def test_empty_chain_partitions_to_nothing(self) -> None:
        assert partition_chain([]) == []


class TestSplitAtLastBarrier:
    def test_head_ends_with_the_barrier(self) -> None:
        chain = build_chain(make_config())
        head, tail = split_at_last_barrier(chain)
        assert [step.stage_id for step in head] == ["dc", "trim", "normalize"]
        assert [step.stage_id for step in tail] == ["loop"]
        assert isinstance(head[-1], NormalizeStep)

    def test_no_barrier_leaves_an_empty_head(self) -> None:
        chain = build_chain(make_config(stages=[StageConfig(id="dc"), StageConfig(id="loop")]))
        head, tail = split_at_last_barrier(chain)
        assert head == []
        assert [step.stage_id for step in tail] == ["dc", "loop"]


class TestPreparedSteps:
    def test_steps_apply_in_chain_order(self) -> None:
        config = make_config(
            stages=[
                StageConfig(id="dc"),
                StageConfig(id="trim", params={"pre_trim_ms": 10.0}),
            ]
        )
        prepared = PreparedSteps(
            [step for step in build_chain(config) if not isinstance(step, NormalizeStep)]
        )
        sample = Sample(
            note=60,
            velocity=127,
            audio=np.full(800, 0.5, dtype=np.float32),
            sample_rate=8000,
        )
        prepared.apply(sample)
        assert sample.n_frames == 720  # 10 ms at 8 kHz cut from the front
        assert abs(float(sample.audio.mean())) < 1e-6  # DC ran too
