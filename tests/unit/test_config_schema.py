"""Tests for autosampler.config.schema."""

import pytest
from pydantic import ValidationError

from autosampler.config.schema import (
    CrossfadeConfig,
    ProjectConfig,
    RecordingConfig,
    SelectionConfig,
    default_stage_chain,
    parse_velocity_map,
)


def _minimal_kwargs() -> dict[str, object]:
    return {
        "recording": RecordingConfig(
            velocity_layers=4, semitone_interval=1, hold_time=10.0, release_time=0.5
        ),
        "selection": SelectionConfig(velocity_layers_out=4),
    }


def test_minimal_project_applies_defaults() -> None:
    config = ProjectConfig(**_minimal_kwargs())
    assert config.instrument_name is None
    assert config.recording.start_note == 21
    assert config.recording.end_note == 108
    assert config.selection.min_note == 21
    assert config.crossfade.crossfade_percent == 0.0
    assert config.output.container == "flac"
    assert config.output.sample_format == "pcm24"
    assert len(config.stages) == len(default_stage_chain())


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        RecordingConfig(
            velocity_layers=4,
            semitone_interval=1,
            hold_time=10.0,
            release_time=0.5,
            bogus_field=1,
        )


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(**_minimal_kwargs(), typo_field="oops")


def test_bool_rejected_for_int_field() -> None:
    """V1 bug 4: bool is an int subclass, so TOML `true` silently became `1`."""
    with pytest.raises(ValidationError):
        RecordingConfig(
            velocity_layers=True,
            semitone_interval=1,
            hold_time=10.0,
            release_time=0.5,
        )


def test_int_accepted_for_float_field() -> None:
    recording = RecordingConfig(
        velocity_layers=4, semitone_interval=1, hold_time=10, release_time=0
    )
    assert recording.hold_time == 10.0
    assert isinstance(recording.hold_time, float)


def test_start_note_after_end_note_raises() -> None:
    with pytest.raises(ValidationError):
        RecordingConfig(
            velocity_layers=4,
            semitone_interval=1,
            hold_time=10.0,
            release_time=0.5,
            start_note=80,
            end_note=40,
        )


def test_min_note_after_max_note_raises() -> None:
    with pytest.raises(ValidationError):
        SelectionConfig(velocity_layers_out=1, min_note=80, max_note=40)


def test_velocity_layers_out_exceeds_recorded_raises() -> None:
    kwargs = _minimal_kwargs()
    kwargs["selection"] = SelectionConfig(velocity_layers_out=99)
    with pytest.raises(ValidationError):
        ProjectConfig(**kwargs)


def test_velocity_map_parses() -> None:
    assert parse_velocity_map("0-63:1, 64-127:5") == [(0, 63, 1), (64, 127, 5)]


def test_velocity_map_sorted_by_lo() -> None:
    assert parse_velocity_map("64-127:5, 0-63:1") == [(0, 63, 1), (64, 127, 5)]


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-segment",
        "0-63",
        "200-300:1",
        "63-0:1",
        "0-63:0",
        "",
    ],
)
def test_velocity_map_invalid_raises(raw: str) -> None:
    with pytest.raises(ValueError, match=r".*"):
        parse_velocity_map(raw)


def test_velocity_map_invalid_rejected_on_selection() -> None:
    with pytest.raises(ValidationError):
        SelectionConfig(velocity_layers_out=1, velocity_map="garbage")


def test_selection_parsed_velocity_map_none_by_default() -> None:
    selection = SelectionConfig(velocity_layers_out=1)
    assert selection.parsed_velocity_map() is None


def test_selection_parsed_velocity_map_returns_segments() -> None:
    selection = SelectionConfig(velocity_layers_out=1, velocity_map="0-63:1, 64-127:5")
    assert selection.parsed_velocity_map() == [(0, 63, 1), (64, 127, 5)]


def test_crossfade_percent_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        CrossfadeConfig(crossfade_percent=150.0)


def test_default_stage_chain_order_and_flags() -> None:
    chain = default_stage_chain()
    ids = [stage.id for stage in chain]
    assert ids == ["dc", "trim", "transient", "normalize", "eq", "stereo", "limiter", "loop"]
    disabled_by_default = {stage.id for stage in chain if not stage.enabled}
    assert disabled_by_default == {"transient", "eq", "stereo", "limiter"}


def test_json_schema_carries_ui_hints() -> None:
    schema = ProjectConfig.model_json_schema()
    recording_schema = schema["$defs"]["RecordingConfig"]["properties"]
    assert recording_schema["hold_time"]["unit"] == "s"
    assert recording_schema["hold_time"]["group"] == "Recording"
    assert "help" in recording_schema["hold_time"]
