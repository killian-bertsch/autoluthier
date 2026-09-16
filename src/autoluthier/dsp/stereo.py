"""Stereo width control via mid-side scaling. New in V2 — V1 had no stereo stage."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autoluthier.config.hints import ui_hint
from autoluthier.dsp.base import StageContext

_STEREO_CHANNELS = 2


class StereoParams(BaseModel):
    """Parameters for the ``stereo`` DSP stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    width: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        json_schema_extra=ui_hint(
            group="Stereo",
            order=0,
            help_text="0 = mono, 1 = unchanged, 2 = maximally wide.",
        ),
    )


class StereoStage:
    """Scale the mid-side decomposition's side signal by `width`.

    Mono input passes through unchanged — there is no side signal to widen.
    """

    def __init__(self, params: StereoParams) -> None:
        self._params = params

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` with its stereo width scaled.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, 2)``.
            ctx: Stage context (unused — width scaling doesn't depend on sample rate).

        Returns:
            The width-scaled buffer, float32. Unchanged if `audio` is mono.

        Raises:
            ValueError: if `audio` has more than 2 channels.
        """
        del ctx
        if audio.ndim == 1:
            return audio
        if audio.shape[1] != _STEREO_CHANNELS:
            raise ValueError(
                f"stereo stage supports mono or 2-channel audio, got {audio.shape[1]} channels"
            )

        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)
        mid = (left + right) / 2.0
        side = (left - right) / 2.0 * self._params.width

        out = np.empty_like(audio, dtype=np.float32)
        out[:, 0] = (mid + side).astype(np.float32)
        out[:, 1] = (mid - side).astype(np.float32)
        return out
