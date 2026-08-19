"""Concatenates numbered raw layer recordings into per-instrument ``sustain``/``release`` renders.

Ports V1's ``convert_samples.sh``, in Python rather than bash+ffmpeg — the plan's "Python, not
bash" decision — which also fixes bug **12** structurally rather than by patching the shell
syntax: V1's ``STEREO_INSTRUMENTS=("ag", "wg", "pearl")`` kept the commas as part of each
element, so ``ag``/``wg`` never matched and were silently downmixed to mono. Here
`stereo_instruments` is a plain Python ``frozenset[str]``, which cannot suffer that class of
bug — there is no quoting step where a stray comma could survive.

Input convention (unchanged from V1): files named ``{n} {InstrumentName}.{ext}`` for sustain
layers (``n`` = 1, 2, 3, ...) and ``R {InstrumentName}.{ext}`` for the release tail, `ext` being
``wav`` or ``flac``. Layers are concatenated in numeric order, downmixed to mono unless the
instrument name is in `stereo_instruments`, then peak-normalized to `target_db`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

TARGET_DB_DEFAULT = -6.0
DEFAULT_LAYER_NUMBERS = (1, 2, 3)
_OUTPUT_SUBTYPE = "PCM_24"
_LAYER_NAME_RE = re.compile(r"^(\d+) (.+)\.(wav|flac)$")


class ConcatError(RuntimeError):
    """Raised when an instrument's layers cannot be concatenated."""


@dataclass(frozen=True, slots=True)
class ConcatReport:
    """What was written (or, if `error` is set, why nothing was written) for one instrument."""

    instrument: str
    sustain_path: Path | None
    release_path: Path | None
    layers_found: int
    layers_missing: list[int]
    stereo: bool
    error: str | None = None


def _discover_instruments(input_dir: Path, layer_numbers: tuple[int, ...]) -> list[str]:
    """Return instrument names, in discovery order, from files naming the first layer number."""
    first = layer_numbers[0]
    names: list[str] = []
    for entry in sorted(input_dir.iterdir(), key=lambda p: p.name):
        match = _LAYER_NAME_RE.match(entry.name)
        if match and int(match.group(1)) == first:
            names.append(match.group(2))
    return names


def _find_layer(input_dir: Path, instrument: str, number: int) -> Path | None:
    for ext in ("wav", "flac"):
        candidate = input_dir / f"{number} {instrument}.{ext}"
        if candidate.is_file():
            return candidate
    return None


def _find_release(input_dir: Path, instrument: str) -> Path | None:
    for ext in ("wav", "flac"):
        for candidate in (
            input_dir / f"R {instrument}.{ext}",
            input_dir / f"R  {instrument}.{ext}",
        ):
            if candidate.is_file():
                return candidate
    return None


def _read(path: Path) -> tuple[NDArray[np.float64], int]:
    audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def _to_mono(audio: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average all channels down to one, matching a standard mono downmix."""
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def _concat(
    paths: list[Path], *, stereo: bool
) -> tuple[NDArray[np.float64], int]:
    """Read and concatenate `paths` along the frame axis, downmixing to mono unless `stereo`.

    Raises:
        ConcatError: if the layers don't share a sample rate.
    """
    chunks: list[NDArray[np.float64]] = []
    sample_rate: int | None = None
    for path in paths:
        audio, rate = _read(path)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ConcatError(
                f"{path} is {rate} Hz but earlier layers are {sample_rate} Hz — "
                "all layers of one instrument must share a sample rate"
            )
        chunks.append(audio if stereo else _to_mono(audio))
    if sample_rate is None:
        raise ConcatError("no layers to concatenate")
    return np.concatenate(chunks, axis=0), sample_rate


def _peak_normalize(audio: NDArray[np.float64], target_db: float) -> NDArray[np.float64]:
    """Scale `audio` so its absolute peak sits at `target_db` dBFS; silence passes through."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio
    gain = 10.0 ** ((target_db - 20.0 * np.log10(peak)) / 20.0)
    return np.asarray(audio * gain, dtype=np.float64)


def _write(path: Path, audio: NDArray[np.float64], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(audio, -1.0, 1.0).astype(np.float32),
        sample_rate,
        format="FLAC",
        subtype=_OUTPUT_SUBTYPE,
    )


def concat_instrument(
    input_dir: Path,
    output_dir: Path,
    instrument: str,
    *,
    stereo_instruments: frozenset[str] = frozenset(),
    target_db: float = TARGET_DB_DEFAULT,
    layer_numbers: tuple[int, ...] = DEFAULT_LAYER_NUMBERS,
) -> ConcatReport:
    """Concatenate and normalize one instrument's layers into ``output_dir/<instrument>/``.

    Args:
        input_dir: Directory holding the numbered layer files.
        output_dir: Directory instrument subfolders are created inside.
        instrument: Instrument name, as it appears in the layer filenames.
        stereo_instruments: Instrument names to keep stereo; everything else is downmixed to
            mono. A plain `frozenset`, so bug 12 (a shell array that silently matched nothing)
            cannot recur.
        target_db: Peak level in dBFS to normalize sustain and release to, independently.
        layer_numbers: Sustain layer numbers, in concatenation order.

    Returns:
        The `ConcatReport` for this instrument.

    Raises:
        ConcatError: if no sustain layers are found, or found layers disagree on sample rate.
    """
    stereo = instrument in stereo_instruments
    found = [(n, _find_layer(input_dir, instrument, n)) for n in layer_numbers]
    present = [(n, p) for n, p in found if p is not None]
    missing = [n for n, p in found if p is None]
    if not present:
        raise ConcatError(f"no sustain layers found for {instrument!r} in {input_dir}")

    sustain, sustain_rate = _concat([p for _, p in present], stereo=stereo)
    sustain = _peak_normalize(sustain, target_db)
    sustain_path = output_dir / instrument / "sustain.flac"
    _write(sustain_path, sustain, sustain_rate)

    release_path: Path | None = None
    release_src = _find_release(input_dir, instrument)
    if release_src is not None:
        release, release_rate = _concat([release_src], stereo=stereo)
        release = _peak_normalize(release, target_db)
        release_path = output_dir / instrument / "release.flac"
        _write(release_path, release, release_rate)

    return ConcatReport(
        instrument=instrument,
        sustain_path=sustain_path,
        release_path=release_path,
        layers_found=len(present),
        layers_missing=missing,
        stereo=stereo,
    )


def concat_layers(
    input_dir: Path,
    output_dir: Path,
    *,
    stereo_instruments: frozenset[str] = frozenset(),
    target_db: float = TARGET_DB_DEFAULT,
    layer_numbers: tuple[int, ...] = DEFAULT_LAYER_NUMBERS,
) -> list[ConcatReport]:
    """Discover and concatenate every instrument's layers under `input_dir`.

    An instrument whose layers disagree on sample rate (or has none at all) does not abort the
    batch — it comes back as a `ConcatReport` with `error` set, matching V1's per-instrument
    error handling in ``main.py``: skip and continue, never silently drop the failure.

    Args:
        input_dir: Directory holding the numbered layer files for every instrument.
        output_dir: Directory instrument subfolders are created inside.
        stereo_instruments: Instrument names to keep stereo.
        target_db: Peak level in dBFS to normalize to.
        layer_numbers: Sustain layer numbers, in concatenation order.

    Returns:
        One `ConcatReport` per discovered instrument, successful or not.
    """
    reports: list[ConcatReport] = []
    for instrument in _discover_instruments(input_dir, layer_numbers):
        try:
            reports.append(
                concat_instrument(
                    input_dir,
                    output_dir,
                    instrument,
                    stereo_instruments=stereo_instruments,
                    target_db=target_db,
                    layer_numbers=layer_numbers,
                )
            )
        except ConcatError as exc:
            reports.append(
                ConcatReport(
                    instrument=instrument,
                    sustain_path=None,
                    release_path=None,
                    layers_found=0,
                    layers_missing=list(layer_numbers),
                    stereo=instrument in stereo_instruments,
                    error=str(exc),
                )
            )
    return reports
