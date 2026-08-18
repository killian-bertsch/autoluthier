"""SPL-style transient shaping: attack/sustain split from a vectorized dual envelope.

Ports V1's ``TransientShaperProcessor`` (``src/processors/transient_shaper.py``) with one
deliberate fix: the fast envelope is a peak-hold (instant attack, exponential release) as in
V1, but the slow envelope is now a proper one-pole follower with a *slow attack too* — not
another peak-hold. V1 used a peak-hold for both, differing only in release rate, which makes
``max(0, fast - slow)`` provably always zero (both jump to a new peak equally instantly, and
`coeff_fast <= coeff_slow` means slow can never fall below fast by induction), so V1's
`attack_db` had no audible effect in any input. See ``dsp/_envelope.py`` for both follower
shapes.

Also implements the decisions-table call that the shaper's ``tanh`` soft-clip becomes an
*optional* control (``saturate``) rather than V1's always-on behavior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from autosampler.config.hints import ui_hint
from autosampler.dsp._envelope import envelope_coeff, one_pole_follower, peak_hold_envelope
from autosampler.dsp.base import StageContext

_SLOW_TO_FAST_RATIO = 20.0
_TRANSIENT_SIGNAL_FLOOR = 1e-12


class TransientParams(BaseModel):
    """Parameters for the ``transient`` DSP stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    attack_db: float = Field(
        default=6.0,
        ge=-12.0,
        le=12.0,
        json_schema_extra=ui_hint(
            group="Transient", order=0, unit="dB", help_text="Gain on the transient portion."
        ),
    )
    sustain_db: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        json_schema_extra=ui_hint(
            group="Transient", order=1, unit="dB", help_text="Gain on the body/sustain portion."
        ),
    )
    speed_ms: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        json_schema_extra=ui_hint(
            group="Transient",
            order=2,
            unit="ms",
            help_text="Fast envelope time constant; the slow envelope is 20x this.",
        ),
    )
    saturate: bool = Field(
        default=False,
        json_schema_extra=ui_hint(
            group="Transient",
            order=3,
            help_text="Soft-clip the output with tanh (V1 always did this; now optional).",
        ),
    )


class TransientStage:
    """Dual-envelope transient shaper."""

    def __init__(self, params: TransientParams) -> None:
        self._params = params

    def apply(
        self, audio: NDArray[np.float32], ctx: StageContext
    ) -> NDArray[np.float32]:
        """Return `audio` with attack/sustain gain applied per the dual-envelope split.

        Envelope detection uses the mid (channel-averaged) signal; the resulting gain curve
        is applied identically to every channel.

        Args:
            audio: Buffer to process, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
            ctx: Stage context; ``sample_rate`` sets the envelope time constants.

        Returns:
            The shaped buffer, float32.
        """
        if audio.shape[0] == 0:
            return audio

        x = audio.astype(np.float64)
        mid = x if x.ndim == 1 else x.mean(axis=1)
        amp = np.abs(mid)

        coeff_fast = envelope_coeff(self._params.speed_ms, ctx.sample_rate)
        coeff_slow = envelope_coeff(self._params.speed_ms * _SLOW_TO_FAST_RATIO, ctx.sample_rate)

        fast_env = peak_hold_envelope(amp, coeff_fast)
        slow_env = one_pole_follower(amp, coeff_slow)

        diff = np.maximum(0.0, fast_env - slow_env)
        max_diff = float(diff.max()) if diff.size else 0.0
        transient_sig = (
            diff / max_diff if max_diff > _TRANSIENT_SIGNAL_FLOOR else np.zeros_like(diff)
        )
        sustain_sig = 1.0 - transient_sig

        attack_lin = 10.0 ** (self._params.attack_db / 20.0)
        sustain_lin = 10.0 ** (self._params.sustain_db / 20.0)
        gain = attack_lin * transient_sig + sustain_lin * sustain_sig

        out = x * gain if x.ndim == 1 else x * gain[:, np.newaxis]
        if self._params.saturate:
            out = np.tanh(out)
        return np.asarray(out, dtype=np.float32)
