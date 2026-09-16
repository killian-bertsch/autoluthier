"""Cut a fixed duration from the front of a sample, clamped to the sample's own length.

Ports V1's ``TrimProcessor`` (``src/processors/trim.py``), fixing bug 5: ``pre_trim_ms``
longer than the sample used to produce an empty array, which a downstream mean()/RMS call
then turned into NaN written straight into the FLAC. The clamp here guarantees at least one
frame survives a trim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autoluthier.config.hints import ui_hint
from autoluthier.domain.units import ms_to_frames
from autoluthier.dsp.base import StageContext


class TrimParams(BaseModel):
    """Parameters for the ``trim`` DSP stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pre_trim_ms: float = Field(
        default=0.0,
        ge=0.0,
        json_schema_extra=ui_hint(
            group="Trim",
            order=0,
            unit="ms",
            help_text="Milliseconds to cut from the front of every sample.",
        ),
    )


class TrimStage:
    """Cut ``params.pre_trim_ms`` from the front of the buffer; never empties it."""

    def __init__(self, params: TrimParams) -> None:
        self._params = params

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` with the front trimmed, clamped to leave at least one frame.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
            ctx: Stage context; ``sample_rate`` converts ``pre_trim_ms`` to a frame count.

        Returns:
            The trimmed buffer, unchanged if ``pre_trim_ms`` is 0 or the input is empty.
        """
        if audio.shape[0] == 0:
            return audio
        cut = ms_to_frames(self._params.pre_trim_ms, ctx.sample_rate)
        cut = max(0, min(cut, audio.shape[0] - 1))
        return audio[cut:]
