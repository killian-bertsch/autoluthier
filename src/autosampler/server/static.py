"""Mounts the frontend's static assets, once they exist (step 10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def mount_frontend(app: FastAPI, frontend_dir: Path) -> bool:
    """Mount `frontend_dir` at ``/``, if it holds a built frontend.

    The frontend (step 10) doesn't exist yet, so this is a no-op until then rather than an
    error — the server is fully usable via its JSON API and ``/docs`` in the meantime.

    Args:
        app: The FastAPI app to mount onto.
        frontend_dir: Directory holding ``index.html`` and its assets.

    Returns:
        ``True`` if a frontend was found and mounted.
    """
    if not (frontend_dir / "index.html").is_file():
        return False
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return True
