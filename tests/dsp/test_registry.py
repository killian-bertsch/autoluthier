"""Tests for dsp.registry: stage lookup and param-validated construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autosampler.dsp.dc import DcRemoveStage
from autosampler.dsp.registry import build_stage, get_registration
from autosampler.dsp.trim import TrimStage


class TestGetRegistration:
    def test_dc_registered(self) -> None:
        registration = get_registration("dc")
        assert registration.factory is DcRemoveStage

    def test_trim_registered(self) -> None:
        registration = get_registration("trim")
        assert registration.factory is TrimStage

    def test_unregistered_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="eq"):
            get_registration("eq")


class TestBuildStage:
    def test_builds_dc_stage(self) -> None:
        stage = build_stage("dc", {})
        assert isinstance(stage, DcRemoveStage)

    def test_builds_trim_stage_with_params(self) -> None:
        stage = build_stage("trim", {"pre_trim_ms": 5.0})
        assert isinstance(stage, TrimStage)

    def test_rejects_unknown_param(self) -> None:
        with pytest.raises(ValidationError):
            build_stage("trim", {"pre_trim_ms": 5.0, "bogus": 1})

    def test_rejects_unregistered_stage_id(self) -> None:
        with pytest.raises(KeyError):
            build_stage("nonexistent", {})
