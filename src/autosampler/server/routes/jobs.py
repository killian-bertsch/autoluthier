"""``POST /api/jobs``, ``GET /api/jobs/{id}`` — run a full instrument job in the background."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from autosampler.server.jobs import new_job_id, run_job
from autosampler.server.sse import EventBroadcaster
from autosampler.server.state import AppState, JobRecord

router = APIRouter()


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.autosampler
    return state


def _broadcaster(request: Request) -> EventBroadcaster:
    broadcaster: EventBroadcaster = request.app.state.broadcaster
    return broadcaster


class StartJobRequest(BaseModel):
    """Body of ``POST /api/jobs``."""

    output_dir: str = "output"
    workers: int | None = None


class JobStarted(BaseModel):
    """What ``POST /api/jobs`` returns immediately, before any work has happened."""

    job_id: str


class JobStatusResponse(BaseModel):
    """What ``GET /api/jobs/{id}`` returns."""

    id: str
    status: str
    error: str | None
    output_dir: str | None
    events: list[dict[str, Any]]


@router.post("/api/jobs")
def start_job(body: StartJobRequest, request: Request) -> JobStarted:
    """Start a full run+export of the currently open instrument in the background.

    Progress streams over ``GET /events``, each event tagged with this job's id;
    ``GET /api/jobs/{id}`` is the polling fallback for a client that missed the stream.

    Raises:
        autosampler.server.state.SessionError: mapped to 404 if no instrument is open.

    Returns:
        The new job's id.
    """
    state = _state(request)
    loaded = state.require_current()
    job = JobRecord(id=new_job_id())
    state.jobs[job.id] = job
    thread = threading.Thread(
        target=run_job,
        args=(_broadcaster(request), job, loaded.folder, loaded.config, Path(body.output_dir)),
        kwargs={"workers": body.workers},
        daemon=True,
    )
    thread.start()
    return JobStarted(job_id=job.id)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JobStatusResponse:
    """Return one job's current status, for polling or reconnect.

    Args:
        job_id: A job id from `POST /api/jobs`.
        request: The current request, used to reach server state.

    Raises:
        HTTPException: 404 if no job with this id exists.

    Returns:
        The job's current status, error (if any), and the events recorded so far.
    """
    job = _state(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        error=job.error,
        output_dir=job.output_dir,
        events=job.events,
    )
