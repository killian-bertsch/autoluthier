"""Tests for the `autosampler ui` stub (the real server arrives in step 9)."""

from __future__ import annotations

from autosampler.cli.app import app
from tests.cli.conftest import runner


def test_ui_stub_explains_and_exits_nonzero() -> None:
    result = runner.invoke(app, ["ui"])
    assert result.exit_code == 1
    assert "not available yet" in result.output


def test_ui_stub_accepts_its_future_flags() -> None:
    result = runner.invoke(app, ["ui", "--host", "0.0.0.0", "--port", "9001", "--dev"])
    assert result.exit_code == 1
