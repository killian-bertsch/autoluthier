"""``GET /events`` — Server-Sent Events stream of every pipeline event from every running job."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from autosampler.server.sse import EventBroadcaster

router = APIRouter()

_KEEPALIVE_INTERVAL_S = 15.0


def _broadcaster(request: Request) -> EventBroadcaster:
    broadcaster: EventBroadcaster = request.app.state.broadcaster
    return broadcaster


async def _event_stream(request: Request) -> AsyncIterator[bytes]:
    """Yield SSE-formatted event lines until the client disconnects.

    A keep-alive comment line is sent whenever no real event arrives within
    `_KEEPALIVE_INTERVAL_S`, so an idle connection isn't mistaken for a dead one by a proxy or
    the browser's own reconnect logic.
    """
    broadcaster = _broadcaster(request)
    queue = broadcaster.subscribe()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_INTERVAL_S)
            except TimeoutError:
                yield b": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(payload)}\n\n".encode()
    finally:
        broadcaster.unsubscribe(queue)


@router.get("/events")
def stream_events(request: Request) -> StreamingResponse:
    """Stream every published pipeline event as Server-Sent Events.

    Returns:
        A ``text/event-stream`` response that stays open until the client disconnects.
    """
    return StreamingResponse(_event_stream(request), media_type="text/event-stream")
