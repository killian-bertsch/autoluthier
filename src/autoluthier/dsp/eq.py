"""Biquad EQ: highpass, lowpass, low-shelf, high-shelf, and peaking, via `scipy.signal.sosfilt`.

New in V2 — V1 had no EQ stage. Coefficients follow the standard Audio EQ Cookbook (Robert
Bristow-Johnson) / Web Audio API `BiquadFilterNode` formulas, parameterized by `q` for every
filter type (including the shelves, rather than the cookbook's alternate slope parameter, so
one field means the same thing across all five types).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import sosfilt

from autoluthier.config.hints import ui_hint
from autoluthier.dsp.base import StageContext

FilterType = Literal["lowpass", "highpass", "low_shelf", "high_shelf", "peaking"]


class EqParams(BaseModel):
    """Parameters for the ``eq`` DSP stage: one biquad section."""

    model_config = ConfigDict(extra="forbid", strict=True)

    filter_type: FilterType = Field(
        default="peaking",
        json_schema_extra=ui_hint(group="EQ", order=0, help_text="Biquad filter shape."),
    )
    freq_hz: float = Field(
        default=1_000.0,
        gt=0.0,
        le=40_000.0,
        json_schema_extra=ui_hint(
            group="EQ", order=1, unit="Hz", log=True, help_text="Corner/center frequency."
        ),
    )
    q: float = Field(
        default=0.7071,
        gt=0.0,
        le=20.0,
        json_schema_extra=ui_hint(group="EQ", order=2, help_text="Filter Q (resonance/width)."),
    )
    gain_db: float = Field(
        default=0.0,
        ge=-24.0,
        le=24.0,
        json_schema_extra=ui_hint(
            group="EQ",
            order=3,
            unit="dB",
            help_text="Boost/cut; ignored for lowpass and highpass.",
        ),
    )


def _biquad_sos(
    filter_type: FilterType, freq_hz: float, q: float, gain_db: float, sample_rate: int
) -> NDArray[np.float64]:
    """Compute one second-order section for `filter_type`, normalized so a0 == 1.

    Args:
        filter_type: Biquad shape.
        freq_hz: Corner/center frequency; must be below the Nyquist frequency.
        q: Filter Q.
        gain_db: Boost/cut for shelf and peaking types; ignored for lowpass/highpass.
        sample_rate: Sample rate in Hz.

    Returns:
        A single SOS row, shape ``(1, 6)``, ready for `scipy.signal.sosfilt`.

    Raises:
        ValueError: if `freq_hz` is at or above the Nyquist frequency for `sample_rate`.
    """
    nyquist = sample_rate / 2.0
    if freq_hz >= nyquist:
        raise ValueError(f"freq_hz={freq_hz} must be below the Nyquist frequency {nyquist}")

    w0 = 2.0 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    a_amp = 10.0 ** (gain_db / 40.0)

    if filter_type == "lowpass":
        b = ((1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0)
        a = (1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
    elif filter_type == "highpass":
        b = ((1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0)
        a = (1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
    elif filter_type == "peaking":
        b = (1.0 + alpha * a_amp, -2.0 * cos_w0, 1.0 - alpha * a_amp)
        a = (1.0 + alpha / a_amp, -2.0 * cos_w0, 1.0 - alpha / a_amp)
    else:
        sqrt_a = np.sqrt(a_amp)
        shelf_alpha = sin_w0 / 2.0 * np.sqrt((a_amp + 1.0 / a_amp) * (1.0 / q - 1.0) + 2.0)
        if filter_type == "low_shelf":
            b = (
                a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha),
                2.0 * a_amp * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
                a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha),
            )
            a = (
                (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha,
                -2.0 * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
                (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha,
            )
        else:  # high_shelf
            b = (
                a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha),
                -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
                a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha),
            )
            a = (
                (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * shelf_alpha,
                2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
                (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * shelf_alpha,
            )

    a0 = a[0]
    sos = np.array([[b[0] / a0, b[1] / a0, b[2] / a0, 1.0, a[1] / a0, a[2] / a0]])
    return sos


class EqStage:
    """One biquad EQ section, applied with `scipy.signal.sosfilt`."""

    def __init__(self, params: EqParams) -> None:
        self._params = params
        self._sos_cache: dict[int, NDArray[np.float64]] = {}

    def _sos_for(self, sample_rate: int) -> NDArray[np.float64]:
        sos = self._sos_cache.get(sample_rate)
        if sos is None:
            sos = _biquad_sos(
                self._params.filter_type,
                self._params.freq_hz,
                self._params.q,
                self._params.gain_db,
                sample_rate,
            )
            self._sos_cache[sample_rate] = sos
        return sos

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` filtered by this section's biquad.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
            ctx: Stage context; ``sample_rate`` determines the biquad coefficients.

        Returns:
            The filtered buffer, float32.
        """
        if audio.shape[0] == 0:
            return audio
        sos = self._sos_for(ctx.sample_rate)
        filtered = sosfilt(sos, audio.astype(np.float64), axis=0)
        return filtered.astype(np.float32)
