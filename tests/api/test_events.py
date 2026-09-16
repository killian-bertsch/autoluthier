"""Tests for the ``/events`` route's SSE generator.

Deliberately doesn't drive this through `TestClient`'s HTTP transport: the endpoint's whole
point is to stay open indefinitely, and this environment's test transport buffers a streaming
response until its generator finishes rather than yielding bytes incrementally — fine for a
real ASGI server (uvicorn genuinely streams), useless for testing an endpoint that never
finishes on purpose. `_event_stream` is exercised directly instead, against a minimal fake
`Request` exposing only what it actually reads (`request.app.state.broadcaster` and
`await request.is_disconnected()`), which is deterministic and fast.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from autoluthier.pipeline.events import RunStarted
from autoluthier.server.routes.events import _event_stream
from autoluthier.server.sse import EventBroadcaster


@dataclass
class _FakeApp:
    broadcaster: EventBroadcaster

    def __post_init__(self) -> None:
        self.state = SimpleNamespace(broadcaster=self.broadcaster)


@dataclass
class _FakeRequest:
    """Stands in for `starlette.Request`: only `app` and `is_disconnected` are read."""

    app: _FakeApp
    disconnect_after: int | None = None
    _calls: int = field(default=0, init=False)

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self.disconnect_after is not None and self._calls > self.disconnect_after


def test_stream_yields_a_published_event() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        request = _FakeRequest(app=_FakeApp(broadcaster))

        stream = _event_stream(request)
        first = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)  # let the generator run up to subscribe()
        broadcaster.publish("job-1", RunStarted(instrument="keybass", total_steps=1))

        line = await asyncio.wait_for(first, timeout=1.0)
        assert line.startswith(b"data: ")
        payload = json.loads(line.removeprefix(b"data: ").decode())
        assert payload == {
            "job_id": "job-1",
            "event": "run_started",
            "instrument": "keybass",
            "total_steps": 1,
        }
        await stream.aclose()

    asyncio.run(scenario())


def test_stream_stops_once_the_client_disconnects() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        request = _FakeRequest(app=_FakeApp(broadcaster), disconnect_after=0)

        with pytest.raises(StopAsyncIteration):
            await _event_stream(request).__anext__()

    asyncio.run(scenario())


def test_stream_unsubscribes_on_close() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        request = _FakeRequest(app=_FakeApp(broadcaster))

        stream = _event_stream(request)
        first = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)
        assert len(broadcaster._subscribers) == 1

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(broadcaster._subscribers) == 0

    asyncio.run(scenario())
