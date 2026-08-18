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
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


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
