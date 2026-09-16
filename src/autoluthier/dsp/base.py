"""Common interface for per-buffer DSP stages: a params model, a class, and `apply`.

Every stage from this framework onward that transforms one sample in isolation (`dc`, `trim`
now; `eq`/`stereo`/`transient`/`limiter` in step 4) implements `Stage`: constructed from its
own validated params, then applied to one sample's audio buffer at a time. Stages that need
cross-sample context — `normalize`'s per-velocity-layer or per-note grouping, `loop`'s
whole-sample search — don't fit this shape and are deliberately *not* `Stage`s; they expose
their own module-level functions operating on a `SampleSet` (see `dsp/normalize.py`).
`pipeline/graph.py` (step 6) is where the two kinds of stage get threaded into one ordered
chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

SampleTarget = Literal["sustain", "release", "both"]
"""Which samples a stage applies to.

V1 fixed this per processor and V2 keeps its choices: DC removal and trim ran over every
sample, the transient shaper only over sustain (its docstring claimed otherwise; the code is
what shipped), and the loop finder only over sustain since release tails are never looped. The
step-4 stages (eq, stereo, limiter) shape tone and level, which a release tail needs as much as
a sustain, so they target both. The mapping lives in `dsp/registry.py` next to each stage's
params model, because it is stage metadata rather than something the pipeline should decide.
"""


@dataclass(frozen=True, slots=True)
class StageContext:
    """Per-sample context available to a `Stage.apply` call."""

    sample_rate: int


@runtime_checkable
class Stage(Protocol):
    """A DSP stage that transforms one sample's audio buffer, given its own params."""

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return the processed buffer; shape ``(n_frames,)`` or ``(n_frames, n_channels)``."""
        ...
