"""Tests for dsp._envelope: the vectorized peak-hold, checked against a naive reference loop."""

from __future__ import annotations

import numpy as np
import pytest

from autosampler.dsp._envelope import envelope_coeff, one_pole_follower, peak_hold_envelope


def _naive_peak_hold(amp: np.ndarray, coeff: float) -> np.ndarray:
    """Direct transcription of V1's per-sample loop — the ground truth being vectorized."""
    env = np.zeros_like(amp)
    prev = 0.0
    for i in range(amp.shape[0]):
        prev = max(amp[i], coeff * prev)
        env[i] = prev
    return env


class TestEnvelopeCoeff:
    def test_zero_time_is_zero_coeff(self) -> None:
        assert envelope_coeff(0.0, 44_100) == 0.0

    def test_negative_time_is_zero_coeff(self) -> None:
        assert envelope_coeff(-5.0, 44_100) == 0.0

    def test_positive_time_gives_coeff_in_unit_interval(self) -> None:
        coeff = envelope_coeff(10.0, 44_100)
        assert 0.0 < coeff < 1.0

    def test_longer_time_constant_gives_higher_coeff(self) -> None:
        fast = envelope_coeff(5.0, 44_100)
        slow = envelope_coeff(100.0, 44_100)
        assert slow > fast


class TestPeakHoldEnvelope:
    def test_empty_input(self) -> None:
        result = peak_hold_envelope(np.zeros(0), 0.9)
        assert result.shape == (0,)

    def test_zero_coeff_returns_amp_unchanged(self) -> None:
        amp = np.array([0.1, 0.9, 0.2, 0.05])
        result = peak_hold_envelope(amp, 0.0)
        np.testing.assert_array_equal(result, amp)

    @pytest.mark.parametrize("coeff", [0.3, 0.9, 0.99, 0.9999])
    def test_matches_naive_loop_single_block(self, coeff: float) -> None:
        rng = np.random.default_rng(0)
        amp = rng.uniform(0.0, 1.0, 500)
        vectorized = peak_hold_envelope(amp, coeff)
        naive = _naive_peak_hold(amp, coeff)
        np.testing.assert_allclose(vectorized, naive, atol=1e-6)

    def test_matches_naive_loop_across_many_blocks(self) -> None:
        """A fast coefficient forces a small block size — verify the block-carry logic."""
        rng = np.random.default_rng(1)
        amp = rng.uniform(0.0, 1.0, 20_000)
        coeff = 0.5  # small block size relative to buffer length: many block boundaries
        vectorized = peak_hold_envelope(amp, coeff)
        naive = _naive_peak_hold(amp, coeff)
        np.testing.assert_allclose(vectorized, naive, atol=1e-6)

    def test_matches_naive_loop_with_impulse(self) -> None:
        """A single spike, then silence — checks the hold-and-decay shape directly."""
        amp = np.zeros(1000)
        amp[10] = 1.0
        coeff = 0.95
        vectorized = peak_hold_envelope(amp, coeff)
        naive = _naive_peak_hold(amp, coeff)
        np.testing.assert_allclose(vectorized, naive, atol=1e-6)
        assert vectorized[10] == pytest.approx(1.0)
        assert vectorized[999] < vectorized[10]  # decayed by the end

    def test_never_overflows_for_very_fast_decay(self) -> None:
        """Coeff near the low end of the realistic range, over many samples: no inf/nan."""
        amp = np.ones(100_000) * 0.5
        result = peak_hold_envelope(amp, 0.37)
        assert np.all(np.isfinite(result))


class TestOnePoleFollower:
    def test_empty_input(self) -> None:
        result = one_pole_follower(np.zeros(0), 0.9)
        assert result.shape == (0,)

    def test_zero_coeff_returns_amp_unchanged(self) -> None:
        amp = np.array([0.1, 0.9, 0.2, 0.05])
        np.testing.assert_array_equal(one_pole_follower(amp, 0.0), amp)

    def test_matches_naive_recursion(self) -> None:
        rng = np.random.default_rng(2)
        amp = rng.uniform(0.0, 1.0, 500)
        coeff = 0.95
        result = one_pole_follower(amp, coeff)

        naive = np.zeros_like(amp)
        prev = 0.0
        for i in range(amp.shape[0]):
            prev = coeff * prev + (1.0 - coeff) * amp[i]
            naive[i] = prev
        np.testing.assert_allclose(result, naive, atol=1e-9)

    def test_lags_a_step_rise_unlike_peak_hold(self) -> None:
        """The whole point of this follower: it must NOT jump instantly on a rising step."""
        amp = np.concatenate([np.zeros(100), np.ones(100)])
        coeff = 0.9
        slow = one_pole_follower(amp, coeff)
        fast = peak_hold_envelope(amp, coeff)
        assert slow[100] < fast[100]  # peak-hold jumps instantly; the follower lags behind
        assert slow[100] == pytest.approx(1.0 - coeff, abs=1e-9)
