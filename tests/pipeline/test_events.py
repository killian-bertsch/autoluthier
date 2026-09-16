"""Event payloads and the reporter's step numbering / monotonic fraction."""

from __future__ import annotations

import json

import pytest

from autoluthier.pipeline.events import (
    Event,
    ProgressReporter,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
    StepStarted,
)


def collect() -> tuple[list[Event], ProgressReporter]:
    events: list[Event] = []
    return events, ProgressReporter(events.append, total_steps=3)


class TestPayloads:
    def test_every_payload_is_json_serializable_and_carries_its_kind(self) -> None:
        events: list[Event] = [
            RunStarted(instrument="kb", total_steps=3),
            StepStarted(
                index=0, label="dc -> trim", step_kind="stages", stage_ids=("dc",), total_units=4
            ),
            StepProgress(
                index=0, label="dc", completed_units=2, total_units=4, fraction=0.25
            ),
            StepCompleted(index=0, label="dc", duration_s=0.1, fraction=0.33),
            RunCompleted(
                instrument="kb", sample_count=6, duration_s=1.0, dynamic_range_db=12.0
            ),
            RunFailed(instrument="kb", message="boom", error_type="PipelineError"),
        ]
        for event in events:
            payload = event.payload()
            assert payload["event"] == event.kind
            assert json.loads(json.dumps(payload)) == payload

    def test_stage_ids_serialize_as_a_list(self) -> None:
        payload = StepStarted(
            index=1, label="loop", step_kind="stages", stage_ids=("loop",), total_units=6
        ).payload()
        assert payload["stage_ids"] == ["loop"]

    def test_base_event_has_no_payload_of_its_own(self) -> None:
        with pytest.raises(NotImplementedError):
            Event().payload()


class TestReporter:
    def test_no_sink_is_silent_rather_than_an_error(self) -> None:
        reporter = ProgressReporter(total_steps=2)
        reporter.run_started("kb")
        handle = reporter.begin_step("load", step_kind="load", total_units=1)
        handle.advance()
        handle.complete()

    def test_step_indices_increment_across_steps(self) -> None:
        events, reporter = collect()
        for label in ("load", "dc", "loop"):
            reporter.begin_step(label, step_kind="stages", total_units=1).complete()
        assert [e.index for e in events if isinstance(e, StepStarted)] == [0, 1, 2]

    def test_fraction_is_monotonic_and_reaches_one(self) -> None:
        events, reporter = collect()
        reporter.run_started("kb")
        for label in ("load", "dc -> trim", "loop"):
            handle = reporter.begin_step(label, step_kind="stages", total_units=4)
            for _ in range(4):
                handle.advance()
            handle.complete()
        fractions = [
            e.fraction for e in events if isinstance(e, StepProgress | StepCompleted)
        ]
        assert fractions == sorted(fractions)
        assert fractions[-1] == 1.0

    def test_fraction_never_exceeds_one_even_if_over_advanced(self) -> None:
        events, reporter = collect()
        handle = reporter.begin_step("load", step_kind="load", total_units=2)
        handle.advance(99)
        assert all(e.fraction <= 1.0 for e in events if isinstance(e, StepProgress))
        assert handle.completed_units == 2

    def test_fraction_does_not_go_backwards_when_a_step_reports_late(self) -> None:
        _, reporter = collect()
        handle = reporter.begin_step("dc", step_kind="stages", total_units=2)
        handle.advance()
        peak = reporter.fraction_for(1.0)
        assert reporter.fraction_for(0.0) == peak

    def test_a_zero_unit_step_counts_as_complete_rather_than_dividing_by_zero(self) -> None:
        events, reporter = collect()
        handle = reporter.begin_step("loop", step_kind="stages", total_units=0)
        handle.advance()
        progress = [e for e in events if isinstance(e, StepProgress)]
        assert progress[0].fraction == 1.0 / 3.0

    def test_run_failed_reports_the_exception_type(self) -> None:
        events, reporter = collect()
        reporter.run_failed("kb", ValueError("bad"))
        failed = events[-1]
        assert isinstance(failed, RunFailed)
        assert (failed.message, failed.error_type) == ("bad", "ValueError")

    def test_total_steps_is_at_least_one(self) -> None:
        assert ProgressReporter(total_steps=0).total_steps == 1
