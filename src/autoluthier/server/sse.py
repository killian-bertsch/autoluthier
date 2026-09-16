"""Bridges pipeline events from a background job thread to SSE clients over asyncio.

A job runs synchronously in a worker thread (`server/jobs.py`); this is the one place server
code has to cross from a thread back into the asyncio loop, since `asyncio.Queue` isn't
thread-safe on its own — `loop.call_soon_threadsafe` is what makes `publish` callable from any
thread.
"""

from __future__ import annotations

import asyncio

from autoluthier.config.schema import JSONValue
from autoluthier.pipeline.events import Event

SsePayload = dict[str, JSONValue]
"""A pipeline event's payload with its owning ``job_id`` merged in."""


class EventBroadcaster:
    """Fans out job events to every currently connected SSE client."""

    def __init__(self) -> None:
        """Start with no bound loop and no subscribers."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[SsePayload]] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop `publish` will schedule onto.

        Args:
            loop: The running event loop, captured once at server startup.
        """
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[SsePayload]:
        """Register a new SSE client.

        Returns:
            A queue that will receive every event published from now on.
        """
        queue: asyncio.Queue[SsePayload] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SsePayload]) -> None:
        """Drop a client's queue, e.g. once its connection closes.

        Args:
            queue: The queue returned by `subscribe`.
        """
        self._subscribers.discard(queue)

    def publish(self, job_id: str, event: Event) -> None:
        """Send `event` to every subscriber. Safe to call from any thread.

        Does nothing if no loop has been bound yet (e.g. a job somehow starts before the
        server's startup hook runs) rather than raising into the calling worker thread.

        Args:
            job_id: The job this event belongs to, merged into the payload so a client
                watching multiple jobs (or none) can tell them apart.
            event: The event to broadcast.
        """
        if self._loop is None:
            return
        payload: SsePayload = {"job_id": job_id, **event.payload()}
        for queue in list(self._subscribers):
            self._loop.call_soon_threadsafe(queue.put_nowait, payload)
