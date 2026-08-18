"""Maps a `StageConfig.id` to its Pydantic params model and `Stage` class.

Registering both together is what lets ``config/schema.py::StageConfig.params`` stay a
generic JSON dict (step 1) while still being validated against a concrete model once a stage
exists — and lets a stage be added to ``default_stage_chain()`` purely by id, with no other
code change. Only buffer-level `Stage`s (see ``dsp/base.py``) live here; ``normalize`` is
registered nowhere yet, since it operates on a whole `SampleSet` rather than one buffer —
``pipeline/graph.py`` (step 6) will special-case set-level stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from autosampler.dsp.base import Stage
from autosampler.dsp.dc import DcRemoveParams, DcRemoveStage
from autosampler.dsp.trim import TrimParams, TrimStage


@dataclass(frozen=True)
class StageRegistration:
    """One registry entry: a stage id's params model and how to build the stage from it."""

    params_model: type[BaseModel]
    factory: Callable[..., Stage]


_REGISTRY: dict[str, StageRegistration] = {
    "dc": StageRegistration(DcRemoveParams, DcRemoveStage),
    "trim": StageRegistration(TrimParams, TrimStage),
}


def get_registration(stage_id: str) -> StageRegistration:
    """Look up the params model + factory registered for `stage_id`.

    Args:
        stage_id: A `StageConfig.id` value, e.g. ``"dc"`` or ``"trim"``.

    Returns:
        The registered `StageRegistration`.

    Raises:
        KeyError: if `stage_id` isn't registered (eq/stereo/transient/limiter land in step 4,
            loop in step 5).
    """
    try:
        return _REGISTRY[stage_id]
    except KeyError:
        raise KeyError(
            f"No DSP stage registered for id={stage_id!r}. Registered: {sorted(_REGISTRY)}"
        ) from None


def build_stage(stage_id: str, params: dict[str, object]) -> Stage:
    """Validate `params` against the registered model and construct the stage.

    Args:
        stage_id: A `StageConfig.id` value.
        params: Raw params dict, e.g. from `StageConfig.params`.

    Returns:
        A constructed `Stage`, ready to `apply()`.

    Raises:
        KeyError: if `stage_id` isn't registered.
        pydantic.ValidationError: if `params` doesn't match the registered model.
    """
    registration = get_registration(stage_id)
    validated = registration.params_model.model_validate(params)
    return registration.factory(validated)
