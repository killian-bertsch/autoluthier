"""Which samples reach the output SFZ, and which key/velocity span each one covers.

Two separate jobs, in this order:

1. **Thinning** — a full recording is often denser than the instrument needs to be. The
   even-spacing selectors live in `domain.notes` (they are shared with `pipeline.preview`) and
   are imported here rather than reimplemented; V1's bug 8 — a set-based index dedup that could
   silently return fewer layers than asked for — is already fixed there.
   `select_velocities_segmented` is the one selector that belongs only to export, so it lives
   here: it drives ``selection.velocity_map``, which asks for a different layer count per
   velocity range.

2. **Zoning** — the surviving notes and velocities are handed contiguous, gapless SFZ ranges,
   with boundaries at the midpoints between neighbors. This is V1's math verbatim: it is what
   determines which sample a sampler reaches for at any given ``(key, velocity)``, so it is the
   part the golden-file test pins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from autosampler.domain.notes import select_velocities

MIN_VELOCITY = 1
MAX_VELOCITY = 127


@dataclass(frozen=True, slots=True)
class KeyZone:
    """The key span one recorded note is stretched across in the output SFZ."""

    note: int
    lokey: int
    hikey: int


@dataclass(frozen=True, slots=True)
class VelocityZone:
    """The velocity span one recorded layer covers, plus its optional crossfade ramps.

    ``lovel``/``hivel`` are the *sounding* bounds, already widened by the crossfade. ``xfin``
    and ``xfout`` are the ``(lo, hi)`` ranges over which the layer fades in and out; each is
    ``None`` when there is no neighbor on that side or the crossfade rounds to zero width, in
    which case the corresponding opcodes are simply omitted.
    """

    velocity: int
    lovel: int
    hivel: int
    xfin: tuple[int, int] | None
    xfout: tuple[int, int] | None


def select_velocities_segmented(
    velocities: Sequence[int], segments: Sequence[tuple[int, int, int]]
) -> list[int]:
    """Pick velocity layers per range, as ``selection.velocity_map`` describes.

    Each ``(lo, hi, n)`` segment keeps `n` evenly spaced layers from the recorded velocities
    falling in ``[lo, hi]``, so a map like ``"0-63:1, 64-127:5"`` can be sparse where the
    instrument barely changes and dense where it does. A segment holding fewer than `n`
    recorded layers keeps all of them — a soft ceiling, not an error.

    Segments are allowed to overlap, so a velocity chosen by two of them appears once in the
    result. That deduplication is deliberate and is *not* V1's bug 8: the bug was dropping
    layers a single evenly-spaced selection had actually asked for, which is now impossible
    (see `domain.notes.select_evenly`).

    Args:
        velocities: Recorded MIDI velocities, ascending.
        segments: Parsed ``(lo, hi, n)`` segments from
            `config.schema.SelectionConfig.parsed_velocity_map`.

    Returns:
        The selected MIDI velocities, ascending and deduplicated.
    """
    selected: set[int] = set()
    for lo, hi, count in segments:
        in_range = [velocity for velocity in velocities if lo <= velocity <= hi]
        if not in_range:
            continue
        selected.update(select_velocities(in_range, count))
    return sorted(selected)


def compute_key_zones(notes: Sequence[int], min_note: int, max_note: int) -> list[KeyZone]:
    """Stretch each selected note across the keys nearest to it.

    Boundaries sit at the midpoint between adjacent selected notes; the outermost edges clamp
    to `min_note`/`max_note` so the requested range is covered with no gaps and no overlaps,
    however sparse the selection is.

    Args:
        notes: Selected MIDI notes, ascending.
        min_note: Lowest key the instrument should respond to.
        max_note: Highest key the instrument should respond to.

    Returns:
        One `KeyZone` per note, in the same order.
    """
    last = len(notes) - 1
    return [
        KeyZone(
            note=note,
            lokey=min_note if i == 0 else (notes[i - 1] + note) // 2 + 1,
            hikey=max_note if i == last else (note + notes[i + 1]) // 2,
        )
        for i, note in enumerate(notes)
    ]


def compute_velocity_zones(
    velocities: Sequence[int], crossfade_percent: float
) -> list[VelocityZone]:
    """Split ``1-127`` between the selected layers, optionally overlapping them.

    Natural boundaries sit at the midpoint between adjacent selected velocities, with the
    outermost edges clamped to ``1`` and ``127``. Each zone then grows outward by
    ``crossfade_percent`` of its own natural width — inward toward its neighbors only, never
    past the ends of the velocity range — and the overlap becomes the ``xfin``/``xfout`` ramps
    so adjacent layers blend instead of switching abruptly.

    Args:
        velocities: Selected MIDI velocities, ascending.
        crossfade_percent: Percent of each zone's natural width that overlaps its neighbors;
            ``0`` gives hard boundaries.

    Returns:
        One `VelocityZone` per velocity, in the same order.
    """
    last = len(velocities) - 1
    zones: list[VelocityZone] = []
    for i, velocity in enumerate(velocities):
        lo_natural = (
            MIN_VELOCITY if i == 0 else (velocities[i - 1] + velocity) // 2 + 1
        )
        hi_natural = (
            MAX_VELOCITY if i == last else (velocity + velocities[i + 1]) // 2
        )
        overlap = int(crossfade_percent / 100.0 * (hi_natural - lo_natural + 1))
        has_lower, has_upper = i > 0, i < last
        lovel = max(MIN_VELOCITY, lo_natural - overlap) if has_lower else MIN_VELOCITY
        hivel = min(MAX_VELOCITY, hi_natural + overlap) if has_upper else MAX_VELOCITY
        zones.append(
            VelocityZone(
                velocity=velocity,
                lovel=lovel,
                hivel=hivel,
                xfin=(lovel, lo_natural) if has_lower and overlap > 0 else None,
                xfout=(hi_natural, hivel) if has_upper and overlap > 0 else None,
            )
        )
    return zones
