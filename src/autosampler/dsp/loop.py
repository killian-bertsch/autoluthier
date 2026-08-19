"""Loop-point detection and the loop crossfade for sustain samples.

Ports V1's ``LoopFinderProcessor`` (``src/processors/loop_finder.py``): the search window,
the candidate cap, and the ``0.65 * amp_similarity + 0.35 * max(correlation, 0)`` score are
reproduced exactly, with the per-sample Python zero-crossing scan replaced by a vectorized
``np.flatnonzero`` pass and the local-RMS pass replaced by a cumulative-sum-of-squares lookup.

Two things are genuinely new here: the crossfade is V2's job rather than the sampler's (by
default), and bug 7 is gone.

**Bug 7 is fixed by removing the coupling that caused it.** V1 shrank the search window by the
crossfade length — ``tail_frac = max(crossfade_frames / n, 0.03)``, ``region_end_frac =
min(1 - tail_frac, 0.97)`` — because the crossfade was left to the *sampler* (V1 emitted an
SFZ ``loop_crossfade`` opcode), so audio had to be reserved *after* ``loop_end`` for the
sampler to fade into. Once ``loop_crossfade_ms`` exceeded 15% of the sample length,
``region_end_frac`` dropped below ``region_start_frac``, the window inverted, and looping was
silently disabled for **every** sample. V2 decouples the two instead. The search window is now a
pair of plain config fractions that do not depend on the crossfade length in any way, so no
crossfade setting can collapse it in *either* mode; a crossfade too long for the audio available
is clamped to fit (`crossfade_frames`) rather than disabling looping. Baked mode additionally
needs nothing after ``loop_end`` at all, since it fades using the material *preceding*
``loop_start`` (see `bake_crossfade`).

**The crossfade can be baked or delegated** — ``CrossfadeConfig.loop_crossfade_mode``,
defaulting to ``"baked"``. ``loop_crossfade_ms`` comes from `CrossfadeConfig` too, not
duplicated into `LoopParams`.

In ``"baked"`` mode, that many frames of the loop's tail are crossfaded into the audio
immediately before ``loop_start``, so the last frame of the loop equals the frame just before
``loop_start`` and the wrap is sample-continuous. It sounds identical in every sampler, needs
no opcode support, and export must **not** also emit ``loop_crossfade`` — the fade is already
in the audio, and asking the sampler to fade again would double it.

In ``"sfz"`` mode the audio is left completely untouched and V1's arrangement returns: export
emits ``loop_crossfade`` and the sampler fades at playback time. That keeps the loop region
bit-identical to the recording, at the cost of only working in samplers that implement the
opcode (ARIA/Cakewalk do; not every sampler does, and one that ignores it simply loops with no
fade). Either way ``sample.loop_crossfade_frames`` records the length actually settled on, so
export has one truthful per-sample number to write and the UI has one to display.

**Bug 7 does not come back in ``sfz`` mode.** V1's mistake was not delegating the fade, it was
*shrinking the search window* to reserve a tail for the sampler. V2 never shrinks the window;
it clamps the requested length per sample against the tail detection happened to leave
(`crossfade_frames`), so the opcode can never claim a longer fade than the audio behind it, and
an over-long request costs fade length instead of costing every loop in the instrument. With
the default ``region_end_frac`` of 0.97 the tail is always at least 3% of the sample, so any
crossfade up to that (300 ms on a 10 s sample) is honored exactly; wanting more tail than that
is a reason to lower ``region_end_frac`` — an explicit knob rather than a hidden coupling.

The fade **curve** is configurable (``CrossfadeConfig.loop_crossfade_shape``) and defaults to
``"linear"``, not to the equal-power curve. That default is deliberate: the two segments being
summed are the ones `find_loop_points` picked precisely *because* they correlate — matched
phase at upward zero crossings, with waveform correlation carrying 0.35 of the score — and
measured correlation on a test tone is 1.000000. An equal-power curve (``sin^2 + cos^2 == 1``)
preserves power only for *uncorrelated* summands; on correlated in-phase material it sums
toward ``sqrt(2)``, a measured **+2.84 dB** lift at the fade midpoint, i.e. a level surge once
per loop cycle. A linear curve (``fade_in + fade_out == 1``) measures +0.00 dB on the same
signal. ``"equal_power"`` remains available for noise-dominated sustains, where the summands
genuinely are decorrelated and it is the better choice.

Like `dsp/normalize.py`, this module operates on a whole `SampleSet` and returns loop points
alongside transformed audio, so it deliberately does not implement the `dsp.base.Stage`
protocol and is *not* registered in `dsp/registry.py`; ``pipeline/graph.py`` (step 6)
special-cases the ``"loop"`` id. See `dsp/base.py`'s docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autosampler.config.hints import ui_hint
from autosampler.config.schema import LoopCrossfadeMode, LoopCrossfadeShape
from autosampler.domain.models import SampleSet
from autosampler.domain.units import ms_to_frames

_RMS_WINDOW_MS = 10.0
_CORRELATION_WINDOW_MS = 20.0
_MAX_CANDIDATES = 10
_AMP_SIMILARITY_WEIGHT = 0.65
_CORRELATION_WEIGHT = 0.35
_AMP_EPSILON = 1e-10
_CORRELATION_NORM_FLOOR = 1e-12
_MIN_CORRELATION_FRAMES = 4
_MIN_CROSSINGS = 2  # a loop needs a start and an end
_MIN_FRAMES_FOR_CROSSING = 2  # a crossing is defined by a pair of adjacent frames
_NO_SCORE = -1.0


class LoopParams(BaseModel):
    """Parameters for the ``loop`` DSP stage.

    These three were hardcoded constants in V1's `find_loop_point` signature. The crossfade
    length is deliberately *not* here — it already exists once as
    ``CrossfadeConfig.loop_crossfade_ms`` and is passed to `apply_loop` separately, so the
    "every field defined exactly once" rule holds.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    region_start_frac: float = Field(
        default=0.85,
        ge=0.0,
        lt=1.0,
        json_schema_extra=ui_hint(
            group="Loop",
            order=0,
            step=0.01,
            fine_step=0.001,
            help_text="Start of the loop-point search window, as a fraction of sample length.",
        ),
    )
    region_end_frac: float = Field(
        default=0.97,
        gt=0.0,
        le=1.0,
        json_schema_extra=ui_hint(
            group="Loop",
            order=1,
            step=0.01,
            fine_step=0.001,
            help_text="End of the loop-point search window, as a fraction of sample length.",
        ),
    )
    min_loop_ms: float = Field(
        default=50.0,
        gt=0.0,
        json_schema_extra=ui_hint(
            group="Loop",
            order=2,
            unit="ms",
            step=1.0,
            help_text="Minimum distance between the detected loop start and loop end.",
        ),
    )

    @model_validator(mode="after")
    def _check_window_ordered(self) -> LoopParams:
        """Reject an inverted search window.

        Returns:
            The validated model.

        Raises:
            ValueError: if ``region_start_frac`` is not strictly below ``region_end_frac``.
        """
        if self.region_start_frac >= self.region_end_frac:
            raise ValueError(
                f"region_start_frac={self.region_start_frac} must be below "
                f"region_end_frac={self.region_end_frac}"
            )
        return self


@dataclass(frozen=True, slots=True)
class LoopPoints:
    """A detected loop as frame indices: the loop plays ``[start, end)``."""

    start: int
    end: int


def _to_mono(audio: NDArray[np.float32]) -> NDArray[np.float64]:
    """Downmix to a 1-D float64 array for analysis (V1 also analyzed the channel mean)."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    return mono.astype(np.float64)


def upward_zero_crossings(
    mono: NDArray[np.float64], start: int, stop: int
) -> NDArray[np.intp]:
    """Find frames in ``[start, stop)`` where the signal crosses zero going upward.

    Vectorized replacement for V1's per-frame Python scan, with the same definition of a
    crossing: the first frame ``i`` where ``mono[i - 1] < 0`` and ``mono[i] >= 0``.

    Args:
        mono: 1-D analysis signal.
        start: First frame of the search window.
        stop: One past the last frame of the search window.

    Returns:
        Ascending absolute frame indices of the upward crossings, possibly empty.
    """
    region = mono[start:stop]
    if region.size < _MIN_FRAMES_FOR_CROSSING:
        return np.empty(0, dtype=np.intp)
    local = np.flatnonzero((region[:-1] < 0.0) & (region[1:] >= 0.0)) + 1
    return local + start


def _local_rms(
    mono: NDArray[np.float64], indices: NDArray[np.intp], window: int
) -> NDArray[np.float64]:
    """RMS of a `window`-long span centered on each index, clipped to the signal's bounds.

    Args:
        mono: 1-D analysis signal.
        indices: Frame indices to measure around.
        window: Window length in frames; the span is ``[i - window // 2, i + window // 2)``,
            exactly as V1 sliced it.

    Returns:
        One RMS value per index; ``0.0`` wherever the clipped span is empty.
    """
    n = mono.size
    half = window // 2
    starts = np.maximum(0, indices - half)
    ends = np.minimum(n, indices + half)
    lengths = ends - starts
    cumsq = np.concatenate(([0.0], np.cumsum(mono**2)))
    # Clamped at zero because differencing a cumulative sum can land marginally negative.
    sums = np.maximum(cumsq[ends] - cumsq[starts], 0.0)
    return np.where(lengths > 0, np.sqrt(sums / np.maximum(lengths, 1)), 0.0)


def _correlation(mono: NDArray[np.float64], i: int, j: int, half: int) -> float:
    """Normalized dot product of the windows centered on `i` and `j`, in ``[-1, 1]``.

    Args:
        mono: 1-D analysis signal.
        i: Center frame of the first window.
        j: Center frame of the second window.
        half: Half the correlation window length, in frames.

    Returns:
        The correlation, or ``0.0`` if either window is too short or effectively silent.
    """
    n = mono.size
    wi = mono[max(0, i - half) : min(n, i + half)]
    wj = mono[max(0, j - half) : min(n, j + half)]
    length = min(wi.size, wj.size)
    if length < _MIN_CORRELATION_FRAMES:
        return 0.0
    wi = wi[:length]
    wj = wj[:length]
    norm = float(np.linalg.norm(wi) * np.linalg.norm(wj))
    if norm < _CORRELATION_NORM_FLOOR:
        return 0.0
    return float(np.dot(wi, wj) / norm)


def find_loop_points(
    audio: NDArray[np.float32], sample_rate: int, params: LoopParams
) -> LoopPoints | None:
    """Search the tail of `audio` for the best-matching pair of upward zero crossings.

    Both returned frames are upward zero crossings, so the loop wrap joins two points of
    matching waveform phase. Candidates are capped at the first and last `_MAX_CANDIDATES`
    crossings in the window, keeping scoring O(100) regardless of how many crossings exist.

    Args:
        audio: Sample buffer, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        sample_rate: Sample rate in Hz.
        params: Validated `LoopParams`.

    Returns:
        The best-scoring `LoopPoints`, or ``None`` if the sample is too short, the window
        holds fewer than two crossings, or no pair clears ``min_loop_ms``.
    """
    mono = _to_mono(audio)
    n = mono.size
    start_idx = int(n * params.region_start_frac)
    end_idx = int(n * params.region_end_frac)
    min_gap = ms_to_frames(params.min_loop_ms, sample_rate)

    if end_idx - start_idx < min_gap:
        return None

    crossings = upward_zero_crossings(mono, start_idx, end_idx)
    if crossings.size < _MIN_CROSSINGS:
        return None

    amplitudes = _local_rms(mono, crossings, max(1, ms_to_frames(_RMS_WINDOW_MS, sample_rate)))
    corr_half = max(1, ms_to_frames(_CORRELATION_WINDOW_MS, sample_rate)) // 2

    tail = max(0, crossings.size - _MAX_CANDIDATES)
    start_cands, start_amps = crossings[:_MAX_CANDIDATES], amplitudes[:_MAX_CANDIDATES]
    end_cands, end_amps = crossings[tail:], amplitudes[tail:]

    gaps = end_cands[np.newaxis, :] - start_cands[:, np.newaxis]
    amp_diff = np.abs(start_amps[:, np.newaxis] - end_amps[np.newaxis, :])
    amp_sum = start_amps[:, np.newaxis] + end_amps[np.newaxis, :] + _AMP_EPSILON
    amp_similarity = 1.0 - amp_diff / amp_sum

    scores = np.full(gaps.shape, _NO_SCORE, dtype=np.float64)
    for i in range(start_cands.size):
        for j in range(end_cands.size):
            if gaps[i, j] < min_gap:
                continue
            correlation = _correlation(mono, int(start_cands[i]), int(end_cands[j]), corr_half)
            scores[i, j] = _AMP_SIMILARITY_WEIGHT * amp_similarity[i, j] + (
                _CORRELATION_WEIGHT * max(correlation, 0.0)
            )

    # Scores are non-negative, so `_NO_SCORE` surviving means no pair cleared `min_gap`.
    # `argmax` breaks ties toward the first in C order — the same pair V1's strictly-greater
    # comparison kept while looping start-outer, end-inner.
    flat_best = int(np.argmax(scores))
    if float(scores.flat[flat_best]) < 0.0:
        best_start, best_end = int(crossings[0]), int(crossings[-1])
    else:
        i_best, j_best = np.unravel_index(flat_best, scores.shape)
        best_start, best_end = int(start_cands[i_best]), int(end_cands[j_best])

    if best_end - best_start < min_gap:
        return None
    return LoopPoints(start=best_start, end=best_end)


def crossfade_frames(
    points: LoopPoints, crossfade_ms: float, sample_rate: int, *, headroom_frames: int
) -> int:
    """Clamp a requested crossfade length to what this loop can actually accommodate.

    Two limits always apply. The fade must fit inside the loop, so it cannot exceed
    ``end - start``. And it needs real audio *outside* the loop to fade with, on whichever side
    the fade reads from — that quantity is `headroom_frames`, and it is the one thing that
    differs between the two crossfade modes:

    - ``baked`` reads the material immediately *before* ``loop_start`` (see `bake_crossfade`),
      so its headroom is ``points.start``.
    - ``sfz`` hands the job to the sampler, which fades using the tail *after* ``loop_end``,
      so its headroom is ``n_frames - points.end``.

    Exceeding either limit shortens the fade. It never disables the loop, which is what V1 did
    (bug 7) — and it is why `apply_loop` clamps rather than shrinking the search window, in
    ``sfz`` mode just as much as in ``baked`` mode.

    Args:
        points: The detected loop.
        crossfade_ms: Requested crossfade length in milliseconds.
        sample_rate: Sample rate in Hz.
        headroom_frames: Usable frames outside the loop on the side the fade reads from.

    Returns:
        The crossfade length in frames, in ``[0, min(headroom_frames, end - start)]``.
    """
    requested = ms_to_frames(crossfade_ms, sample_rate)
    return max(0, min(requested, headroom_frames, points.end - points.start))


def fade_curves(
    frames: int, shape: LoopCrossfadeShape
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the ``(fade_in, fade_out)`` pair for a `frames`-long crossfade.

    The ramp runs over ``(0, 1]`` rather than ``[0, 1]`` so the final frame is at full
    crossover — that endpoint is what makes the loop wrap sample-continuous.

    Args:
        frames: Crossfade length in frames; must be positive.
        shape: ``"linear"`` (``fade_in + fade_out == 1``, level-preserving on correlated
            material) or ``"equal_power"`` (``fade_in**2 + fade_out**2 == 1``,
            power-preserving on decorrelated material). See the module docstring for why
            linear is the default here.

    Returns:
        Two 1-D float64 arrays of length `frames`.
    """
    ramp = np.arange(1, frames + 1, dtype=np.float64) / frames
    if shape == "equal_power":
        return np.sin(0.5 * np.pi * ramp), np.cos(0.5 * np.pi * ramp)
    return ramp, 1.0 - ramp


def bake_crossfade(
    audio: NDArray[np.float32],
    points: LoopPoints,
    *,
    crossfade_ms: float,
    sample_rate: int,
    shape: LoopCrossfadeShape = "linear",
) -> tuple[NDArray[np.float32], int]:
    """Crossfade the loop's tail into the material before ``loop_start``, in the audio itself.

    Over the last `xf` frames of the loop, ``[end - xf, end)``, the original tail is faded out
    while ``[start - xf, start)`` is faded in, using the curve `shape` selects. The fade
    reaches full crossover exactly on the loop's final frame, so
    ``out[end - 1] == audio[start - 1]`` and the wrap to ``start`` continues the waveform with
    no step. Nothing after ``end`` is touched or required, which is why the search window no
    longer has to reserve a tail (see the module docstring, bug 7).

    Args:
        audio: Sample buffer, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        points: The detected loop.
        crossfade_ms: Requested crossfade length; clamped by `crossfade_frames`.
        sample_rate: Sample rate in Hz.
        shape: Fade curve, from ``CrossfadeConfig.loop_crossfade_shape``.

    Returns:
        ``(processed_audio, frames_applied)``. The input buffer is returned unchanged, with
        ``0``, when the clamped length is zero.
    """
    frames = crossfade_frames(
        points, crossfade_ms, sample_rate, headroom_frames=points.start
    )
    if frames == 0:
        return audio, 0

    fade_in, fade_out = fade_curves(frames, shape)
    if audio.ndim != 1:
        fade_in = fade_in[:, np.newaxis]
        fade_out = fade_out[:, np.newaxis]

    tail = audio[points.end - frames : points.end].astype(np.float64)
    pre_start = audio[points.start - frames : points.start].astype(np.float64)

    out = audio.copy()
    out[points.end - frames : points.end] = (tail * fade_out + pre_start * fade_in).astype(
        np.float32
    )
    return out, frames


def apply_loop(
    sustain: SampleSet,
    params: LoopParams,
    *,
    crossfade_ms: float,
    crossfade_mode: LoopCrossfadeMode = "baked",
    crossfade_shape: LoopCrossfadeShape = "linear",
) -> None:
    """Detect loop points for every sustain sample, and crossfade them per `crossfade_mode`.

    In ``baked`` mode the fade is rendered into ``sample.audio`` here, so it sounds the same in
    every sampler and export emits no crossfade opcode. In ``sfz`` mode the audio is left
    completely untouched and the fade becomes export's problem: the clamped length lands in
    ``sample.loop_crossfade_frames`` for export to write as a ``loop_crossfade`` opcode, which
    is V1's arrangement — with the difference that an over-long request is clamped per sample
    instead of collapsing the search window and disabling every loop (bug 7).

    Samples with no detectable loop keep ``loop_start``/``loop_end`` at ``None`` and are left
    untouched; export (step 7) then omits their loop opcodes. Release samples are never
    looped, so they are not passed here at all — matching V1.

    Args:
        sustain: Sustain samples; ``loop_start``, ``loop_end``, and ``loop_crossfade_frames``
            are set, and ``audio`` is mutated in ``baked`` mode only.
        params: Validated `LoopParams`.
        crossfade_ms: Crossfade length, from ``CrossfadeConfig.loop_crossfade_ms``.
        crossfade_mode: ``"baked"`` or ``"sfz"``, from
            ``CrossfadeConfig.loop_crossfade_mode``.
        crossfade_shape: Fade curve, from ``CrossfadeConfig.loop_crossfade_shape``; ignored in
            ``sfz`` mode, where the curve is the sampler's choice.
    """
    for sample in sustain:
        points = find_loop_points(sample.audio, sample.sample_rate, params)
        if points is None:
            continue

        if crossfade_mode == "baked":
            sample.audio, frames = bake_crossfade(
                sample.audio,
                points,
                crossfade_ms=crossfade_ms,
                sample_rate=sample.sample_rate,
                shape=crossfade_shape,
            )
        else:
            # The sampler fades using the tail after loop_end, so that tail is the headroom.
            frames = crossfade_frames(
                points,
                crossfade_ms,
                sample.sample_rate,
                headroom_frames=sample.n_frames - points.end,
            )

        sample.loop_start = points.start
        sample.loop_end = points.end
        sample.loop_crossfade_frames = frames
