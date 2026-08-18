"""DC offset removal — always the first stage in the default chain, no configurable params."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from autosampler.dsp.base import StageContext


class DcRemoveParams(BaseModel):
    """No parameters — DC removal is unconditional."""

    model_config = ConfigDict(extra="forbid", strict=True)


class DcRemoveStage:
    """Subtract each channel's mean, accumulated in float64 for precision."""

    def __init__(self, params: DcRemoveParams) -> None:
        self._params = params

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` with its per-channel DC offset removed.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
            ctx: Stage context (unused — DC removal doesn't depend on sample rate).

        Returns:
            The DC-free buffer, float32.
        """
        del ctx
        audio64 = audio.astype(np.float64)
        mean = audio64.mean(axis=0)
        return (audio64 - mean).astype(np.float32)
