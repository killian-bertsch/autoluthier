"""Tests for the `autoluthier ui` command.

`uvicorn.run` blocks forever serving requests, so every test here monkeypatches it (and
`webbrowser.open`, so a test run never actually pops a browser window) and just checks that
`ui_command` builds and launches the server with the right arguments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoluthier.cli import ui as ui_module
from autoluthier.cli.app import app
from tests.cli.conftest import runner


@pytest.fixture(autouse=True)
def _no_real_server(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Stub out `uvicorn.run` and `webbrowser.open`, recording what they were called with."""
    calls: list[tuple[str, int]] = []

    def fake_run(served_app: object, *, host: str, port: int) -> None:
        del served_app
        calls.append((host, port))

    monkeypatch.setattr(ui_module.uvicorn, "run", fake_run)
    monkeypatch.setattr(ui_module.webbrowser, "open", lambda url: calls.append(("open", url)))  # type: ignore[arg-type]
    return calls


def test_ui_starts_the_server_on_default_host_and_port(
    _no_real_server: list[tuple[str, int]],
) -> None:
    result = runner.invoke(app, ["ui"])
    assert result.exit_code == 0, result.output
    assert ("127.0.0.1", 8000) in _no_real_server


def test_ui_honors_host_and_port_flags(_no_real_server: list[tuple[str, int]]) -> None:
    result = runner.invoke(app, ["ui", "--host", "0.0.0.0", "--port", "9001", "--dev"])
    assert result.exit_code == 0, result.output
    assert ("0.0.0.0", 9001) in _no_real_server


def test_ui_no_open_skips_the_browser(_no_real_server: list[tuple[str, int]]) -> None:
    result = runner.invoke(app, ["ui", "--no-open"])
    assert result.exit_code == 0, result.output
    assert not any(call[0] == "open" for call in _no_real_server)


def test_ui_opens_the_real_frontend(_no_real_server: list[tuple[str, int]]) -> None:
    """Step 10 landed a real `frontend/index.html`, so the normal run opens `/`, not `/docs`."""
    result = runner.invoke(app, ["ui"])
    assert result.exit_code == 0, result.output
    assert "No frontend build found" not in result.output
    assert ("open", "http://127.0.0.1:8000/") in _no_real_server


def test_ui_reports_missing_frontend(
    _no_real_server: list[tuple[str, int]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `frontend/` is ever absent (e.g. a stripped-down checkout), fall back to `/docs`."""
    monkeypatch.setattr(ui_module, "FRONTEND_DIR", tmp_path)
    result = runner.invoke(app, ["ui"])
    assert "No frontend build found" in result.output
    assert ("open", "http://127.0.0.1:8000/docs") in _no_real_server
