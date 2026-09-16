"""Streaming peak-normalization of raw source renders, before they ever reach the pipeline.

Ports V1's ``normalize_source.py``. This runs on whole, un-sliced ``sustain``/``release``
renders, which can be many minutes long — the two-pass streaming approach (measure the peak,
then apply one linear gain block by block) is kept rather than routing through
`io.reader.read_audio`/`io.writer.write_audio`, which load a file whole into memory by design
for the (much shorter) per-sample slices the rest of the pipeline works with.

Unlike V1, which always wrote 24-bit FLAC regardless of the source container, this preserves
each source file's own container and subtype: a WAV source stays WAV, a FLAC source stays FLAC,
at whatever bit depth it already had. Pre-normalization only changes level.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from autoluthier.io.reader import find_audio_file

BLOCK_SIZE = 65536
"""Frames per streaming block: large enough to amortize per-call overhead, small enough that
peak measurement and gain application never hold a whole render in memory."""

TARGET_DB_DEFAULT = -6.0
_STEMS = ("sustain", "release")


@dataclass(frozen=True, slots=True)
class PrenormalizeReport:
    """The outcome of measuring (and optionally normalizing) one source file."""

    path: Path
    peak_db: float | None
    """`None` for a silent file, which is skipped rather than divided by zero."""
    gain_db: float | None
    applied: bool
    """Whether the gain was actually written; `False` in dry-run mode or for a silent file."""


class PrenormalizeError(RuntimeError):
    """Raised when a source file cannot be measured or rewritten."""


def _find_peak_db(path: Path) -> float | None:
    """Return the streaming absolute peak of `path` in dBFS, or `None` if it is silent."""
    peak_linear = 0.0
    with sf.SoundFile(str(path)) as reader:
        for block in reader.blocks(blocksize=BLOCK_SIZE, dtype="float64"):
            block_peak = float(np.max(np.abs(block))) if block.size else 0.0
            peak_linear = max(peak_linear, block_peak)
    if peak_linear == 0.0:
        return None
    return 20.0 * math.log10(peak_linear)


def _apply_gain(src: Path, dst: Path, gain_linear: float) -> None:
    """Copy `src` to `dst`, scaling every sample by `gain_linear`, preserving format/subtype."""
    with sf.SoundFile(str(src)) as reader, sf.SoundFile(
        str(dst),
        mode="w",
        samplerate=reader.samplerate,
        channels=reader.channels,
        format=reader.format,
        subtype=reader.subtype,
    ) as writer:
        for block in reader.blocks(blocksize=BLOCK_SIZE, dtype="float64"):
            writer.write(block * gain_linear)


def normalize_file(path: Path, *, target_db: float, dry_run: bool) -> PrenormalizeReport:
    """Measure `path`'s peak and, unless `dry_run`, rewrite it to `target_db` peak in place.

    The rewrite goes to a temp file in the same directory and is moved over the original with
    `Path.replace`, so a failure partway through leaves the original file untouched.

    Args:
        path: Audio file to normalize.
        target_db: Desired peak level in dBFS.
        dry_run: If `True`, only measure — never write.

    Returns:
        The `PrenormalizeReport` for this file.

    Raises:
        PrenormalizeError: if the file can be read but not rewritten.
    """
    peak_db = _find_peak_db(path)
    if peak_db is None:
        return PrenormalizeReport(path=path, peak_db=None, gain_db=None, applied=False)

    gain_db = target_db - peak_db
    if dry_run:
        return PrenormalizeReport(path=path, peak_db=peak_db, gain_db=gain_db, applied=False)

    gain_linear = 10.0 ** (gain_db / 20.0)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix + ".tmp")
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        _apply_gain(path, tmp_path, gain_linear)
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise PrenormalizeError(f"failed to normalize {path}: {exc}") from exc
    return PrenormalizeReport(path=path, peak_db=peak_db, gain_db=gain_db, applied=True)


def find_source_files(source_dir: Path) -> list[Path]:
    """Find every ``sustain``/``release`` render in the immediate subdirectories of `source_dir`.

    Args:
        source_dir: Parent directory holding one subfolder per instrument.

    Returns:
        Matching files, sorted by their containing folder's name then stem.
    """
    files: list[Path] = []
    if not source_dir.is_dir():
        return files
    for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        for stem in _STEMS:
            found = find_audio_file(entry, stem)
            if found is not None:
                files.append(found)
    return files


def prenormalize_sources(
    source_dir: Path, *, target_db: float = TARGET_DB_DEFAULT, dry_run: bool = False
) -> list[PrenormalizeReport]:
    """Normalize every ``sustain``/``release`` render under `source_dir` to `target_db` peak.

    Args:
        source_dir: Parent directory holding one subfolder per instrument.
        target_db: Desired peak level in dBFS.
        dry_run: If `True`, only report what would change — never write.

    Returns:
        One `PrenormalizeReport` per file found, in discovery order. Empty if `source_dir`
        holds no ``sustain``/``release`` renders.
    """
    return [
        normalize_file(path, target_db=target_db, dry_run=dry_run)
        for path in find_source_files(source_dir)
    ]
