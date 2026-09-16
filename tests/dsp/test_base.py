"""Tests for dsp.base: the Stage protocol and StageContext."""

from __future__ import annotations

from autoluthier.dsp.base import Stage, StageContext
from autoluthier.dsp.dc import DcRemoveParams, DcRemoveStage
from autoluthier.dsp.trim import TrimParams, TrimStage


class TestStageContext:
    def test_holds_sample_rate(self) -> None:
        ctx = StageContext(sample_rate=44_100)
        assert ctx.sample_rate == 44_100


class TestStageProtocol:
    def test_dc_remove_stage_satisfies_protocol(self) -> None:
        assert isinstance(DcRemoveStage(DcRemoveParams()), Stage)

    def test_trim_stage_satisfies_protocol(self) -> None:
        assert isinstance(TrimStage(TrimParams()), Stage)
