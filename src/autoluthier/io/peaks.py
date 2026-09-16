"""Precomputed min/max waveform peaks for the browser waveform editor (step 11).

wavesurfer.js decodes audio client-side by default, which is far too slow for a multi-minute
render (see CLAUDE.md's frontend-specifics note). Precomputing (min, max) pairs server-side at a
few zoom levels lets the browser hand wavesurfer's own ``peaks`` option pre-reduced data instead
of decoding raw audio itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_BIN_COUNTS: tuple[int, ...] = (200, 1000, 5000)
"""Target bin counts for the three precomputed zoom levels: overview, mid, detail."""


@dataclass(frozen=True, slots=True)
class PeakLevel:
    """One zoom level: per-channel min/max pairs over ``samples_per_bin``-sized bins."""

    samples_per_bin: int
    mins: NDArray[np.float32]
    """Shape ``(n_channels, n_bins)``."""
    maxs: NDArray[np.float32]
    """Shape ``(n_channels, n_bins)``."""


def _channels_first(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return `audio` as ``(n_channels, n_frames)``, whatever its input shape."""
    return audio.reshape(1, -1) if audio.ndim == 1 else audio.T


def compute_peak_level(audio: NDArray[np.float32], bin_count: int) -> PeakLevel:
    """Reduce `audio` to at most `bin_count` (min, max) pairs per channel.

    A short trailing partial bin is padded up to a full bin by repeating the last frame, so
    every bin holds the same number of samples — the padding can only narrow that bin's own
    min/max toward its last real value, never invent energy that wasn't there.

    Args:
        audio: Audio, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        bin_count: Target number of bins; the actual count is clamped to ``[1, n_frames]``.

    Returns:
        The computed `PeakLevel`.
    """
    by_channel = _channels_first(audio)
    n_channels, n_frames = by_channel.shape
    if n_frames == 0:
        empty = np.zeros((n_channels, 0), dtype=np.float32)
        return PeakLevel(samples_per_bin=1, mins=empty, maxs=empty)

    bins = max(1, min(bin_count, n_frames))
    samples_per_bin = -(-n_frames // bins)  # ceil division
    padded_len = samples_per_bin * bins
    if padded_len > n_frames:
        padded = np.pad(by_channel, ((0, 0), (0, padded_len - n_frames)), mode="edge")
    else:
        padded = by_channel

    reshaped = padded.reshape(n_channels, bins, samples_per_bin)
    return PeakLevel(
        samples_per_bin=samples_per_bin,
        mins=reshaped.min(axis=2).astype(np.float32),
        maxs=reshaped.max(axis=2).astype(np.float32),
    )


def compute_peaks(
    audio: NDArray[np.float32], bin_counts: Sequence[int] = DEFAULT_BIN_COUNTS
) -> list[PeakLevel]:
    """Compute peak levels at each of `bin_counts`, coarsest to finest.

    Args:
        audio: Audio to reduce.
        bin_counts: Target bin count per zoom level.

    Returns:
        One `PeakLevel` per requested bin count, in the given order.
    """
    return [compute_peak_level(audio, count) for count in bin_counts]
