"""Maps a `StageConfig.id` to its Pydantic params model and `Stage` class.

Registering both together is what lets ``config/schema.py::StageConfig.params`` stay a
generic JSON dict (step 1) while still being validated against a concrete model once a stage
exists — and lets a stage be added to ``default_stage_chain()`` purely by id, with no other
code change. Only buffer-level `Stage`s (see ``dsp/base.py``) live here; ``normalize`` is
registered nowhere yet, since it operates on a whole `SampleSet` rather than one buffer —
``pipeline/graph.py`` (step 6) will special-case set-level stages.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from autosampler.dsp.base import SampleTarget, Stage
from autosampler.dsp.dc import DcRemoveParams, DcRemoveStage
from autosampler.dsp.eq import EqParams, EqStage
from autosampler.dsp.limiter import LimiterParams, LimiterStage
from autosampler.dsp.stereo import StereoParams, StereoStage
from autosampler.dsp.transient import TransientParams, TransientStage
from autosampler.dsp.trim import TrimParams, TrimStage


@dataclass(frozen=True)
class StageRegistration:
    """One registry entry: a stage id's params model, how to build it, and what it applies to."""

    params_model: type[BaseModel]
    factory: Callable[..., Stage]
    targets: SampleTarget = "both"


_REGISTRY: dict[str, StageRegistration] = {
    "dc": StageRegistration(DcRemoveParams, DcRemoveStage),
    "trim": StageRegistration(TrimParams, TrimStage),
    "eq": StageRegistration(EqParams, EqStage),
    "stereo": StageRegistration(StereoParams, StereoStage),
    # Sustain only, matching V1: its `process()` iterated `data.sustain`, whatever its
    # docstring said (see the V1 notes in ../../../CLAUDE.md).
    "transient": StageRegistration(TransientParams, TransientStage, targets="sustain"),
    "limiter": StageRegistration(LimiterParams, LimiterStage),
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


def build_stage(stage_id: str, params: Mapping[str, object]) -> Stage:
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
