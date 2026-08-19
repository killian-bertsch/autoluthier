"""Maps domain exceptions to HTTP responses, so routes can let them propagate.

Every route below this module calls straight into `pipeline`/`export`/`config`/`helpers`
functions and lets their exceptions bubble up rather than wrapping each call in its own
try/except — these handlers are what turns that into the right status code exactly once,
instead of once per route.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from autosampler.config.toml_io import ProjectConfigError
from autosampler.config.workspace import WorkspaceIndexError
from autosampler.export.writer import ExportError
from autosampler.helpers.concat_layers import ConcatError
from autosampler.helpers.prenormalize import PrenormalizeError
from autosampler.pipeline.executor import PipelineError
from autosampler.pipeline.graph import ChainConfigError
from autosampler.server.state import SampleNotFoundError, SessionError

_NOT_FOUND: tuple[type[Exception], ...] = (SessionError, SampleNotFoundError)
_BAD_REQUEST: tuple[type[Exception], ...] = (
    ProjectConfigError,
    ChainConfigError,
    PipelineError,
    ExportError,
    WorkspaceIndexError,
    PrenormalizeError,
    ConcatError,
)


ExceptionHandler = Callable[[Request, Exception], JSONResponse]


def _as_json(status_code: int) -> ExceptionHandler:
    """Build a Starlette exception handler that reports `exc` as ``{"detail": str(exc)}``."""

    def handler(request: Request, exc: Exception) -> JSONResponse:
        del request
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


def install_error_handlers(app: FastAPI) -> None:
    """Register exception handlers on `app` mapping domain errors to HTTP status codes.

    Args:
        app: The FastAPI app to register handlers on.
    """
    not_found = _as_json(404)
    bad_request = _as_json(400)
    for exc_type in _NOT_FOUND:
        app.add_exception_handler(exc_type, not_found)
    for exc_type in _BAD_REQUEST:
        app.add_exception_handler(exc_type, bad_request)
