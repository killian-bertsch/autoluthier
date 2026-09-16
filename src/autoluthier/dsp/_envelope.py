"""Envelope followers shared by `transient.py` and `limiter.py`.

Two different follower shapes live here:

- `peak_hold_envelope` — instant attack, exponential release: ``env[i] = max(amp[i], coeff *
  env[i-1])``. `limiter.py` uses this for release-stage gain recovery (hold an attenuation dip,
  then relax it slowly), and `transient.py` uses it for its fast envelope.
- `one_pole_follower` — slow attack *and* slow release, a plain exponential moving average:
  ``env[i] = coeff * env[i-1] + (1 - coeff) * amp[i]``. `transient.py` uses this for its slow
  envelope — it must genuinely lag a peak-hold on attack, or `max(0, fast - slow)` can never
  go positive (both would jump to a new peak instantly, since a peak-hold's attack is instant
  by construction — this was V1's actual bug: its slow envelope was *also* a peak-hold,
  differing from the fast one only in release rate, which makes the transient signal
  provably always zero, by induction on ``coeff_fast <= coeff_slow``).

`peak_hold_envelope`'s recursive form doesn't vectorize directly, but unrolling it gives a
closed form:

    env[i] = coeff**i * max(amp[k] * coeff**-k for k in range(i + 1))

i.e. ``coeff**i`` times a running max of ``amp * coeff**-arange(n)``. Computed in one shot
this overflows float64 quickly whenever ``coeff`` is much less than 1 (fast envelopes), since
``coeff**-i`` grows exponentially — so it processes fixed-size blocks sized from ``coeff``
itself (``coeff**-block_size`` capped well under the float64 ceiling) and carries the previous
block's final envelope value into the next block's floor term:
``env_local[i] = coeff**i * max(running_max[i], carry * coeff)``.

`one_pole_follower` is a linear recursion, so `scipy.signal.lfilter` vectorizes it directly —
no overflow risk, no block-wise trick needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

# Keeps coeff**-block_size within roughly exp(_LOG_HEADROOM), far under float64's ~exp(709)
# overflow ceiling, leaving generous margin for the multiplication that follows.
_LOG_HEADROOM = 250.0


def envelope_coeff(time_ms: float, sample_rate: int) -> float:
    """First-order IIR smoothing coefficient for a given time constant; 0 if `time_ms` <= 0."""
    if time_ms <= 0.0:
        return 0.0
    tc_samples = (time_ms / 1000.0) * sample_rate
    return float(np.exp(-1.0 / max(tc_samples, 1.0)))


def peak_hold_envelope(amp: NDArray[np.float64], coeff: float) -> NDArray[np.float64]:
    """Vectorized equivalent of ``env[i] = max(amp[i], coeff * env[i-1])``.

    Args:
        amp: Non-negative signal to peak-hold, shape ``(n,)``.
        coeff: Decay coefficient in ``[0, 1)``; 0 means no hold (``env == amp``).

    Returns:
        The envelope, same shape as `amp`.
    """
    n = amp.shape[0]
    env = np.empty(n, dtype=np.float64)
    if n == 0:
        return env
    if coeff <= 0.0:
        return amp.copy()

    log_c = float(np.log(coeff))
    block_size = n if log_c == 0.0 else max(1, int(_LOG_HEADROOM / -log_c))

    carry = 0.0
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = amp[start:end]
        exponents = np.arange(block.shape[0], dtype=np.float64)
        c_pow = coeff**exponents
        c_pow_inv = coeff ** (-exponents)
        running_max = np.maximum.accumulate(block * c_pow_inv)
        floor = carry * coeff
        block_env = c_pow * np.maximum(running_max, floor)
        env[start:end] = block_env
        carry = float(block_env[-1])

    return env


def one_pole_follower(amp: NDArray[np.float64], coeff: float) -> NDArray[np.float64]:
    """Vectorized equivalent of ``env[i] = coeff * env[i-1] + (1 - coeff) * amp[i]``.

    Unlike `peak_hold_envelope`, this lags a rising signal as much as a falling one — an
    ordinary exponential moving average, computed directly via `scipy.signal.lfilter`.

    Args:
        amp: Signal to follow, shape ``(n,)``.
        coeff: Smoothing coefficient in ``[0, 1)``; 0 means no smoothing (``env == amp``).

    Returns:
        The envelope, same shape as `amp`.
    """
    if amp.shape[0] == 0 or coeff <= 0.0:
        return amp.copy()
    filtered = lfilter([1.0 - coeff], [1.0, -coeff], amp)
    return np.asarray(filtered, dtype=np.float64)
