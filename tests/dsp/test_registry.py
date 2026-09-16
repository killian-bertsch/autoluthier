"""Tests for dsp.registry: stage lookup and param-validated construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoluthier.dsp.dc import DcRemoveStage
from autoluthier.dsp.eq import EqStage
from autoluthier.dsp.limiter import LimiterStage
from autoluthier.dsp.registry import all_registrations, build_stage, get_registration
from autoluthier.dsp.stereo import StereoStage
from autoluthier.dsp.transient import TransientStage
from autoluthier.dsp.trim import TrimStage


class TestGetRegistration:
    def test_dc_registered(self) -> None:
        registration = get_registration("dc")
        assert registration.factory is DcRemoveStage

    def test_trim_registered(self) -> None:
        registration = get_registration("trim")
        assert registration.factory is TrimStage

    def test_eq_registered(self) -> None:
        assert get_registration("eq").factory is EqStage

    def test_stereo_registered(self) -> None:
        assert get_registration("stereo").factory is StereoStage

    def test_transient_registered(self) -> None:
        assert get_registration("transient").factory is TransientStage

    def test_limiter_registered(self) -> None:
        assert get_registration("limiter").factory is LimiterStage

    def test_unregistered_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="loop"):
            get_registration("loop")


class TestAllRegistrations:
    def test_covers_every_buffer_stage(self) -> None:
        ids = set(all_registrations())
        assert ids == {"dc", "trim", "eq", "stereo", "transient", "limiter"}

    def test_returns_same_entries_as_get_registration(self) -> None:
        assert all_registrations()["trim"] is get_registration("trim")


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
