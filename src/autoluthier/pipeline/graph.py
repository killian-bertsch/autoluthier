"""Turns ``ProjectConfig.stages`` into an ordered, executable chain of steps.

Three kinds of step come out of this module, because not every DSP operation has the same
shape (see `dsp/base.py`):

- `BufferStep` — a registered `dsp.base.Stage`: one buffer in, one buffer out, given its own
  params. Per-sample and independent, so it parallelizes.
- `LoopStep` — per-sample too, but it produces loop points as well as audio and takes its
  crossfade settings from `CrossfadeConfig` rather than from its own ``StageConfig.params``,
  so it isn't a `Stage` and isn't in the registry.
- `NormalizeStep` — a **barrier**: lufs/rms mode matches notes within each velocity layer and
  velocity mode matches layers within each note, so one shared gain per group cannot be
  computed without every member of that group already processed. It has to see the whole set.

`build_chain` is therefore not a uniform ``build_stage(id, params)`` loop: ``"normalize"`` and
``"loop"`` are special-cased, exactly as `dsp/normalize.py` and `dsp/loop.py` said they would
have to be. `partition_chain` then groups the result into segments the executor can run —
maximal runs of per-sample steps between barriers.

Grouping consecutive per-sample steps into one segment is what keeps the parallel path cheap: a
worker applies *every* step in the segment to a sample while it holds it, so a sample's audio
crosses the process boundary twice per segment instead of twice per stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from autoluthier.config.schema import (
    JSONValue,
    LoopCrossfadeMode,
    LoopCrossfadeShape,
    ProjectConfig,
    SampleOverride,
)
from autoluthier.domain.models import Sample
from autoluthier.dsp.base import SampleTarget, Stage, StageContext
from autoluthier.dsp.loop import LoopParams, LoopPoints, apply_loop_to_sample
from autoluthier.dsp.normalize import NormalizeParams
from autoluthier.dsp.registry import build_stage, get_registration

NORMALIZE_STAGE_ID = "normalize"
LOOP_STAGE_ID = "loop"
_UNIQUE_STAGE_IDS = frozenset({NORMALIZE_STAGE_ID, LOOP_STAGE_ID})


class ChainConfigError(ValueError):
    """Raised when ``ProjectConfig.stages`` cannot be turned into an executable chain."""


@dataclass(frozen=True, slots=True)
class BufferStep:
    """A registered per-buffer `Stage`, kept as ``(id, params)`` rather than as an instance.

    The raw params are carried instead of a built `Stage` so a step is trivially picklable for
    a worker process, and so the stage object — which may cache filter coefficients — is built
    once per worker task rather than shared across processes.
    """

    stage_id: str
    params: Mapping[str, JSONValue]
    targets: SampleTarget

    def build(self) -> Stage:
        """Construct the `Stage` this step describes.

        Returns:
            A ready-to-apply `Stage`.
        """
        return build_stage(self.stage_id, self.params)


@dataclass(frozen=True, slots=True)
class LoopStep:
    """Loop detection plus crossfade for one sustain sample.

    All three crossfade settings come from `CrossfadeConfig`, not from this stage's
    ``StageConfig.params`` — the crossfade is a property of the loop *crossfade*, and
    `LoopParams` owns detection only.
    """

    stage_id: ClassVar[str] = LOOP_STAGE_ID
    targets: ClassVar[SampleTarget] = "sustain"

    params: LoopParams
    crossfade_ms: float
    crossfade_mode: LoopCrossfadeMode
    crossfade_shape: LoopCrossfadeShape
    overrides: Mapping[tuple[int, int], SampleOverride] = field(default_factory=dict)

    def apply(self, sample: Sample) -> None:
        """Detect or override this sample's loop, then crossfade it.

        Args:
            sample: The sustain sample to process, mutated in place.
        """
        override = self.overrides.get((sample.note, sample.velocity))
        points = None
        if override is not None and override.loop_points is not None:
            points = LoopPoints(*override.loop_points)
        apply_loop_to_sample(
            sample,
            self.params,
            crossfade_ms=self.crossfade_ms,
            crossfade_mode=self.crossfade_mode,
            crossfade_shape=self.crossfade_shape,
            forced_points=points,
            disabled=override is not None and override.loop_disabled,
        )


@dataclass(frozen=True, slots=True)
class NormalizeStep:
    """The one barrier step: it needs every sample in a group before it can scale any of them."""

    stage_id: ClassVar[str] = NORMALIZE_STAGE_ID
    targets: ClassVar[SampleTarget] = "both"

    params: NormalizeParams


PerSampleStep = BufferStep | LoopStep
"""A step that transforms one sample without looking at any other."""

ChainStep = BufferStep | LoopStep | NormalizeStep
"""One entry in the executable chain."""


@dataclass(frozen=True, slots=True)
class SampleSegment:
    """A maximal run of consecutive per-sample steps, executed as one parallel pass."""

    steps: tuple[PerSampleStep, ...]

    @property
    def stage_ids(self) -> tuple[str, ...]:
        """The ids of the steps in this segment, in execution order."""
        return tuple(step.stage_id for step in self.steps)

    @property
    def label(self) -> str:
        """Human-readable name for progress reporting, e.g. ``"dc -> trim"``."""
        return " -> ".join(self.stage_ids)


@dataclass(frozen=True, slots=True)
class BarrierSegment:
    """A single set-level step that has to see every sample at once."""

    step: NormalizeStep

    @property
    def stage_ids(self) -> tuple[str, ...]:
        """The id of the barrier step."""
        return (self.step.stage_id,)

    @property
    def label(self) -> str:
        """Human-readable name for progress reporting."""
        return self.step.stage_id


Segment = SampleSegment | BarrierSegment
"""One unit of execution: a parallel pass over samples, or a serial set-level pass."""


def targets_sustain(step: ChainStep) -> bool:
    """Whether `step` applies to sustain samples.

    Args:
        step: The chain step to test.

    Returns:
        ``True`` if sustain samples go through this step.
    """
    return step.targets in ("sustain", "both")


def targets_release(step: ChainStep) -> bool:
    """Whether `step` applies to release samples.

    Args:
        step: The chain step to test.

    Returns:
        ``True`` if release samples go through this step.
    """
    return step.targets in ("release", "both")


def _build_step(
    stage_id: str, params: Mapping[str, JSONValue], config: ProjectConfig
) -> ChainStep:
    """Build one chain step, special-casing the two ids that aren't registry `Stage`s."""
    if stage_id == NORMALIZE_STAGE_ID:
        return NormalizeStep(params=NormalizeParams.model_validate(dict(params)))
    if stage_id == LOOP_STAGE_ID:
        return LoopStep(
            params=LoopParams.model_validate(dict(params)),
            crossfade_ms=config.crossfade.loop_crossfade_ms,
            crossfade_mode=config.crossfade.loop_crossfade_mode,
            crossfade_shape=config.crossfade.loop_crossfade_shape,
            overrides=config.override_table(),
        )
    try:
        registration = get_registration(stage_id)
    except KeyError as exc:
        known = sorted({*_UNIQUE_STAGE_IDS, "dc", "trim", "eq", "stereo", "transient", "limiter"})
        raise ChainConfigError(
            f"unknown stage id {stage_id!r} in the stage chain; known ids: {known}"
        ) from exc
    # Validate now so a bad params dict fails before any audio is read, not mid-render in a
    # worker process where the traceback is far less useful.
    registration.params_model.model_validate(dict(params))
    return BufferStep(stage_id=stage_id, params=dict(params), targets=registration.targets)


def build_chain(config: ProjectConfig) -> list[ChainStep]:
    """Build the ordered, validated chain of steps described by `config.stages`.

    Disabled entries are dropped. Every stage's params are validated here, so a malformed
    project fails before any audio is read.

    Args:
        config: The project configuration.

    Returns:
        The chain, in execution order.

    Raises:
        ChainConfigError: on an unknown stage id, invalid params, or a repeated ``normalize``
            or ``loop`` entry. Repeating those two is rejected because neither is idempotent:
            a second ``loop`` would re-detect on already-crossfaded audio and bake a second
            fade over the first, and a second ``normalize`` would re-derive group gains from
            its own output. Repeating a buffer stage (two ``eq`` bands, say) is fine and
            allowed.
    """
    chain: list[ChainStep] = []
    seen_unique: set[str] = set()
    for entry in config.stages:
        if not entry.enabled:
            continue
        if entry.id in _UNIQUE_STAGE_IDS:
            if entry.id in seen_unique:
                raise ChainConfigError(
                    f"stage {entry.id!r} appears more than once in the chain; it is not "
                    "idempotent and must appear at most once"
                )
            seen_unique.add(entry.id)
        try:
            chain.append(_build_step(entry.id, entry.params, config))
        except ChainConfigError:
            raise
        except ValueError as exc:
            raise ChainConfigError(f"invalid params for stage {entry.id!r}: {exc}") from exc
    return chain


def partition_chain(chain: Sequence[ChainStep]) -> list[Segment]:
    """Group `chain` into segments: maximal runs of per-sample steps, split by barriers.

    Args:
        chain: The chain from `build_chain`.

    Returns:
        The segments, in execution order.
    """
    segments: list[Segment] = []
    pending: list[PerSampleStep] = []
    for step in chain:
        if isinstance(step, NormalizeStep):
            if pending:
                segments.append(SampleSegment(tuple(pending)))
                pending = []
            segments.append(BarrierSegment(step))
        else:
            pending.append(step)
    if pending:
        segments.append(SampleSegment(tuple(pending)))
    return segments


def split_at_last_barrier(
    chain: Sequence[ChainStep],
) -> tuple[list[ChainStep], list[ChainStep]]:
    """Split `chain` after its last barrier step.

    `pipeline.preview` uses this to stay exact: everything up to and including the last barrier
    has to run over the whole sample set (that's what a barrier means), while everything after
    it is per-sample and can run on the previewed subset alone.

    Args:
        chain: The chain from `build_chain`.

    Returns:
        ``(head, tail)``, where ``head`` ends with the last barrier step and is empty if the
        chain has no barrier at all.
    """
    last_barrier = -1
    for index, step in enumerate(chain):
        if isinstance(step, NormalizeStep):
            last_barrier = index
    if last_barrier < 0:
        return [], list(chain)
    return list(chain[: last_barrier + 1]), list(chain[last_barrier + 1 :])


class PreparedSteps:
    """A segment's steps with their `Stage`s built once, ready to apply to many samples.

    Built inside the worker process (or in-process on the serial path), so filter coefficient
    caches are reused across every sample in the task instead of being rebuilt per sample.
    """

    def __init__(self, steps: Sequence[PerSampleStep]) -> None:
        """Build the stages for `steps`.

        Args:
            steps: The per-sample steps of one segment, in execution order.
        """
        self._ops: list[Stage | LoopStep] = [
            step if isinstance(step, LoopStep) else step.build() for step in steps
        ]

    def apply(self, sample: Sample) -> None:
        """Run every step over `sample`, mutating it in place.

        Args:
            sample: The sample to process.
        """
        ctx = StageContext(sample_rate=sample.sample_rate)
        for op in self._ops:
            if isinstance(op, LoopStep):
                op.apply(sample)
            else:
                sample.audio = op.apply(sample.audio, ctx)
