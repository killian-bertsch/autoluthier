"""``GET /api/schema`` — the project config's JSON Schema, driving the frontend's forms."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from autosampler.config.schema import ProjectConfig

router = APIRouter()


@router.get("/api/schema")
def get_schema() -> dict[str, Any]:
    """Return `ProjectConfig`'s JSON Schema, hints and all.

    A field defined once in `config.schema` appears here automatically — this is what lets the
    frontend (step 10) render parameter forms without hand-writing one per field.

    Returns:
        The schema, as `ProjectConfig.model_json_schema` produces it.
    """
    return ProjectConfig.model_json_schema()
