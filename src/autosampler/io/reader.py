"""Audio file reading: float32 in-memory arrays and source-file discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

_AUDIO_EXTENSIONS = (".wav", ".flac")


def find_audio_file(folder: Path, stem: str) -> Path | None:
    """Return the first of ``{stem}.wav``/``{stem}.flac`` that exists under ``folder``.

    Args:
        folder: Instrument folder to search.
        stem: Base filename without extension, e.g. ``"sustain"`` or ``"release"``.

    Returns:
        The matching path, or ``None`` if neither extension exists.
    """
    for ext in _AUDIO_EXTENSIONS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_audio(path: Path) -> tuple[NDArray[np.float32], int]:
    """Read an audio file into a float32 array and its sample rate.

    Returns a 1-D ``(n_frames,)`` array for mono files and ``(n_frames, n_channels)`` for
    multichannel — never a forced 2-D shape for mono, so downstream ``ndim == 1`` checks
    stay meaningful. The whole file is loaded into memory; V2 does not stream.

    Args:
        path: Path to a ``.wav`` or ``.flac`` file.

    Returns:
        The audio samples and the file's sample rate in Hz.
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)
