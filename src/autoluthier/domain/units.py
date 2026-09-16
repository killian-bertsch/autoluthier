"""Sample-rate-relative unit conversions shared by ``io``, ``dsp``, and ``export``.

Centralizing seconds/milliseconds <-> frame-count conversion here means every stage that
turns a config duration (``hold_time``, ``pre_trim_ms``, ``loop_crossfade_ms``, ...) into a
frame count agrees on the same rounding rule — Python's built-in ``round``, matching V1's
``int(round(...))``.
"""

from __future__ import annotations


def seconds_to_frames(seconds: float, sample_rate: int) -> int:
    """Convert a duration in seconds to a frame count at ``sample_rate``.

    Args:
        seconds: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        The nearest integer frame count.
    """
    return round(seconds * sample_rate)


def ms_to_frames(milliseconds: float, sample_rate: int) -> int:
    """Convert a duration in milliseconds to a frame count at ``sample_rate``.

    Args:
        milliseconds: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        The nearest integer frame count.
    """
    return seconds_to_frames(milliseconds / 1000.0, sample_rate)


def frames_to_seconds(frames: int, sample_rate: int) -> float:
    """Convert a frame count at ``sample_rate`` to a duration in seconds.

    Args:
        frames: Frame count.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    return frames / sample_rate
