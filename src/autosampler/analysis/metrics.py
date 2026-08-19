"""Measurements taken *from* processed audio, as opposed to transformations applied to it.

`estimate_rt_decay` is the first one and arrives with export (step 7), which needs it for the
release SFZ's ``rt_decay`` opcode. It lives here rather than in ``export`` because it measures
audio — ``export`` decides file layout and opcode text, and should not also be doing DSP — and
because the analysis view (step 11) wants the same numbers on screen.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from autosampler.domain.units import ms_to_frames

RT_DECAY_WINDOW_MS = 20.0
"""Non-overlapping RMS window used for the decay fit, from V1."""

RT_DECAY_FIT_FRACTION = 0.8
"""Fraction of the windows the slope is fitted over: the tail is mostly noise floor, and
including it flattens the fit toward 0 dB/s."""

RT_DECAY_MIN = 1.0
RT_DECAY_MAX = 24.0
RT_DECAY_DEFAULT = 6.0
"""Fallback for a sample too short to hold two windows, where no slope can be fitted."""

_MIN_FIT_WINDOWS = 2
_SILENCE_FLOOR = 1e-9


def estimate_rt_decay(audio: NDArray[np.float32], sample_rate: int) -> float:
    """Estimate how fast a release sample decays, in dB per second.

    V1's method, preserved: take the RMS of consecutive 20 ms windows, convert to dB, fit a
    straight line through the first 80% of them, and report the magnitude of its (negative)
    slope clamped to ``[1, 24]`` dB/s. The result feeds the SFZ ``rt_decay`` opcode, which
    tells the sampler how much to attenuate a release sample whose note was held a while.

    The measurement is invariant to any constant gain — a slope in dB does not move when every
    window shifts by the same amount — so it does not depend on where in the chain normalize
    ran.

    Args:
        audio: The release sample, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        sample_rate: Sample rate of `audio` in Hz.

    Returns:
        Decay rate in dB/s, clamped to ``[1, 24]``; `RT_DECAY_DEFAULT` when the sample is too
        short to hold two windows.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    window = max(1, ms_to_frames(RT_DECAY_WINDOW_MS, sample_rate))
    n_windows = mono.shape[0] // window
    if n_windows < _MIN_FIT_WINDOWS:
        return RT_DECAY_DEFAULT

    windowed = mono[: n_windows * window].astype(np.float64).reshape(n_windows, window)
    rms = np.sqrt(np.mean(windowed**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, _SILENCE_FLOOR))

    times = np.arange(n_windows, dtype=np.float64) * (window / sample_rate)
    n_fit = max(_MIN_FIT_WINDOWS, int(n_windows * RT_DECAY_FIT_FRACTION))
    slope = float(np.polyfit(times[:n_fit], db[:n_fit], 1)[0])
    return float(np.clip(-slope, RT_DECAY_MIN, RT_DECAY_MAX))
