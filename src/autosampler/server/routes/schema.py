"""``GET /api/schema`` — JSON Schemas that drive the frontend's auto-rendered forms.

`ProjectConfig.model_json_schema()` covers every field *except* the DSP stage chain:
`StageConfig.params` (config/schema.py) is deliberately a generic ``dict[str, JSONValue]``, not
a union of every stage's params model, so a field belongs to exactly one place — the stage
module that owns it (`dsp/registry.py`'s convention). That means the stage chain editor needs
its own schemas, keyed by stage id, which is what `/api/schema/stages` provides. `normalize`
and `loop` are included even though they aren't in `dsp/registry.py` (they operate on a whole
`SampleSet`, not one buffer, so they're not a `Stage`) — the frontend only cares that every id
`default_stage_chain()` can produce has *some* params schema to render.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from autosampler.config.schema import ProjectConfig
from autosampler.dsp.loop import LoopParams
from autosampler.dsp.normalize import NormalizeParams
from autosampler.dsp.registry import all_registrations

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


@router.get("/api/schema/stages")
def get_stage_schemas() -> dict[str, dict[str, Any]]:
    """Return each DSP stage id's params JSON Schema, for the stage chain editor.

    Returns:
        Stage id -> that stage's params model's JSON Schema.
    """
    schemas = {
        stage_id: registration.params_model.model_json_schema()
        for stage_id, registration in all_registrations().items()
    }
    schemas["normalize"] = NormalizeParams.model_json_schema()
    schemas["loop"] = LoopParams.model_json_schema()
    return schemas
