"""Lookahead true-peak limiter. New in V2 — V1 had no limiter (only a post-normalize ceiling).

"True peak" means peaks that occur *between* samples, invisible to a plain `max(abs(x))` but
audible after D/A reconstruction or lossy transcoding — detected here the standard way (as in
ITU-R BS.1770): 4x polyphase oversampling (`scipy.signal.resample_poly`), which reveals
inter-sample peaks the discrete samples alone would miss.

Because this is an offline, whole-buffer renderer rather than a real-time processor,
"lookahead" needs no output latency: the required gain at every oversampled position is known
before any output is produced, so the gain envelope can look at genuinely future samples via
plain array slicing.

Pipeline, all at the oversampled rate:
  1. `needed_gain[i] = min(1, ceiling_linear / true_peak[i])` — the gain that would just touch
     the ceiling at position `i`, never boosting past 1.0.
  2. A forward-looking minimum over `lookahead_ms` (`scipy.ndimage.minimum_filter1d`) ensures
     the gain is already reduced before a peak inside that window is reached.
  3. A peak-hold over `release_ms`, applied to the *attenuation* `1 - gain` (reusing
     `dsp/_envelope.py`), holds a dip and releases it back toward 1.0 gradually rather than
     snapping — the same peak-hold math the transient shaper uses, on the inverted signal.

Each step only ever pushes gain at or below what step 1 already required at that instant, so
the ceiling is respected throughout; downsampling back to the original rate takes the minimum
gain within each oversampled block (never a single arbitrarily-chosen phase) for the same
reason.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly

from autoluthier.config.hints import ui_hint
from autoluthier.dsp._envelope import envelope_coeff, peak_hold_envelope
from autoluthier.dsp.base import StageContext

_OVERSAMPLE = 4


class LimiterParams(BaseModel):
    """Parameters for the ``limiter`` DSP stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ceiling_db: float = Field(
        default=-0.3,
        le=0.0,
        json_schema_extra=ui_hint(
            group="Limiter", order=0, unit="dB", help_text="True-peak ceiling; never exceeded."
        ),
    )
    lookahead_ms: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        json_schema_extra=ui_hint(
            group="Limiter",
            order=1,
            unit="ms",
            help_text="How far ahead to anticipate an oncoming peak.",
        ),
    )
    release_ms: float = Field(
        default=50.0,
        ge=0.0,
        le=1_000.0,
        json_schema_extra=ui_hint(
            group="Limiter",
            order=2,
            unit="ms",
            help_text="How gradually gain recovers back toward unity after a peak.",
        ),
    )


def _forward_min_window(gain: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """`out[i] = min(gain[i : i + window])`, via a forward-shifted `minimum_filter1d`."""
    if window <= 1:
        return gain
    window = window if window % 2 == 1 else window + 1  # odd size keeps the origin shift exact
    return minimum_filter1d(gain, size=window, origin=-(window // 2), mode="nearest")


class LimiterStage:
    """Lookahead true-peak limiter."""

    def __init__(self, params: LimiterParams) -> None:
        self._params = params

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` limited so its true peak never exceeds `params.ceiling_db`.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
            ctx: Stage context; ``sample_rate`` sets the oversampled lookahead/release windows.

        Returns:
            The limited buffer, float32.
        """
        n = audio.shape[0]
        if n == 0:
            return audio

        x64 = audio.astype(np.float64)
        oversampled_rate = ctx.sample_rate * _OVERSAMPLE
        oversampled = resample_poly(x64, _OVERSAMPLE, 1, axis=0)
        true_peak = (
            np.abs(oversampled) if oversampled.ndim == 1 else np.max(np.abs(oversampled), axis=1)
        )

        ceiling_linear = 10.0 ** (self._params.ceiling_db / 20.0)
        needed_gain = np.where(
            true_peak > 0.0, np.minimum(1.0, ceiling_linear / np.maximum(true_peak, 1e-300)), 1.0
        )

        lookahead_samples = round(self._params.lookahead_ms / 1000.0 * oversampled_rate)
        gain_lookahead = _forward_min_window(needed_gain, lookahead_samples)

        coeff_release = envelope_coeff(self._params.release_ms, oversampled_rate)
        held_attenuation = peak_hold_envelope(1.0 - gain_lookahead, coeff_release)
        gain_oversampled = 1.0 - held_attenuation

        gain = gain_oversampled.reshape(n, _OVERSAMPLE).min(axis=1)
        out = x64 * gain if x64.ndim == 1 else x64 * gain[:, np.newaxis]
        return out.astype(np.float32)
