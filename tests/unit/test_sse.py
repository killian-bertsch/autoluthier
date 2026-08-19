"""Tests for autosampler.server.sse.EventBroadcaster.

Exercised directly against asyncio rather than through an HTTP round-trip: a job thread calls
`publish` from outside the event loop exactly like these tests do, so this is what actually
needs to be correct, and it's fast and deterministic without a real server or client.
"""

from __future__ import annotations

import asyncio
import threading

from autosampler.pipeline.events import RunCompleted, RunStarted
from autosampler.server.sse import EventBroadcaster


def test_publish_before_bind_loop_is_a_no_op() -> None:
    """A job somehow starting before the server's startup hook must not raise into it."""
    broadcaster = EventBroadcaster()
    broadcaster.publish("job-1", RunStarted(instrument="keybass", total_steps=1))


def test_subscriber_receives_a_published_event() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        queue = broadcaster.subscribe()

        broadcaster.publish("job-1", RunStarted(instrument="keybass", total_steps=3))

        payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert payload == {
            "job_id": "job-1",
            "event": "run_started",
            "instrument": "keybass",
            "total_steps": 3,
        }

    asyncio.run(scenario())


def test_unsubscribed_queue_receives_nothing_further() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        queue = broadcaster.subscribe()
        broadcaster.unsubscribe(queue)

        broadcaster.publish("job-1", RunStarted(instrument="keybass", total_steps=1))
        await asyncio.sleep(0)
        assert queue.empty()

    asyncio.run(scenario())


def test_multiple_subscribers_each_get_their_own_copy() -> None:
    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        first = broadcaster.subscribe()
        second = broadcaster.subscribe()

        event = RunCompleted(instrument="keybass", sample_count=4, duration_s=1.0)
        broadcaster.publish("job-1", event)

        for queue in (first, second):
            payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            assert payload["event"] == "run_completed"
            assert payload["sample_count"] == 4

    asyncio.run(scenario())


def test_publish_is_safe_from_a_background_thread() -> None:
    """The actual production shape: a job thread calls `publish`, never the event loop itself."""

    async def scenario() -> None:
        broadcaster = EventBroadcaster()
        broadcaster.bind_loop(asyncio.get_running_loop())
        queue = broadcaster.subscribe()

        def worker() -> None:
            broadcaster.publish("job-1", RunStarted(instrument="keybass", total_steps=1))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        payload = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert payload["job_id"] == "job-1"

    asyncio.run(scenario())
