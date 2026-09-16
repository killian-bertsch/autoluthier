"""Audio file writing: container/bit-depth selection and optional resample.

Delegates encoding to ``soundfile``/libsndfile. FLAC only supports up to 24-bit integer PCM
(no ``PCM_32``, no ``FLOAT``) — ``resolve_subtype`` raises early rather than letting
libsndfile fail with an opaque error deep inside a batch render.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample

from autoluthier.config.schema import ContainerFormat, OutputConfig, SampleFormat

_SUBTYPE_BY_FORMAT: dict[SampleFormat, str] = {
    "pcm16": "PCM_16",
    "pcm24": "PCM_24",
    "pcm32": "PCM_32",
    "float32": "FLOAT",
}

_FLAC_SUPPORTED_FORMATS: frozenset[SampleFormat] = frozenset({"pcm16", "pcm24"})


def resolve_subtype(container: ContainerFormat, sample_format: SampleFormat) -> str:
    """Map a ``(container, sample_format)`` pair to a libsndfile subtype string.

    Args:
        container: Output container format.
        sample_format: Output bit depth/encoding.

    Returns:
        The libsndfile subtype name, e.g. ``"PCM_24"``.

    Raises:
        ValueError: if ``container`` cannot encode ``sample_format`` (FLAC supports only
            16- and 24-bit PCM).
    """
    if container == "flac" and sample_format not in _FLAC_SUPPORTED_FORMATS:
        raise ValueError(
            f"FLAC cannot encode sample_format={sample_format!r}; use pcm16 or pcm24, "
            "or switch output.container to 'wav'"
        )
    return _SUBTYPE_BY_FORMAT[sample_format]


def resample(
    audio: NDArray[np.float32], source_rate: int, target_rate: int
) -> NDArray[np.float32]:
    """Resample ``audio`` from ``source_rate`` to ``target_rate`` (FFT-based).

    Args:
        audio: Audio to resample, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        source_rate: Sample rate of ``audio`` in Hz.
        target_rate: Desired sample rate in Hz.

    Returns:
        The resampled float32 audio; the input unchanged if the rates already match.
    """
    if source_rate == target_rate:
        return audio
    target_frames = round(audio.shape[0] * target_rate / source_rate)
    resampled = scipy_resample(audio, target_frames, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def encode_audio(
    audio: NDArray[np.float32], sample_rate: int, output: OutputConfig
) -> bytes:
    """Encode ``audio`` in memory per ``output``'s container/bit-depth/resample settings.

    Clips to ``[-1, 1]`` before encoding, matching V1's contract for the integer PCM formats
    this pipeline writes. Used by `write_audio` and by the server (step 9), which serves a
    sample's audio over HTTP without ever putting it on disk.

    Args:
        audio: Audio to encode, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        sample_rate: Sample rate of ``audio`` in Hz.
        output: Output container/bit-depth/resample configuration.

    Returns:
        The encoded file's bytes.
    """
    subtype = resolve_subtype(output.container, output.sample_format)
    if output.sample_rate is not None and output.sample_rate != sample_rate:
        audio = resample(audio, sample_rate, output.sample_rate)
        sample_rate = output.sample_rate

    data = np.clip(audio, -1.0, 1.0)
    buffer = BytesIO()
    sf.write(buffer, data, sample_rate, format=output.container.upper(), subtype=subtype)
    return buffer.getvalue()


def write_audio(
    path: Path, audio: NDArray[np.float32], sample_rate: int, output: OutputConfig
) -> Path:
    """Encode ``audio`` to ``path`` per ``output``'s container/bit-depth/resample settings.

    Args:
        path: Destination file path; parent directories are created as needed.
        audio: Audio to write, shape ``(n_frames,)`` or ``(n_frames, n_channels)``.
        sample_rate: Sample rate of ``audio`` in Hz.
        output: Output container/bit-depth/resample configuration.

    Returns:
        The path written to.
    """
    data = encode_audio(audio, sample_rate, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
