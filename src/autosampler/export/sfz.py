"""The SFZ opcode model and its rendering to text.

An SFZ file is a flat list of headers and ``name=value`` opcodes, so it is tempting to build
one with f-strings. This module builds a small model first — `OpcodeLine`, `SfzRegion`,
`SfzDocument` — for two reasons. The golden-file test compares rendered *text* against V1's
output byte for byte, so line grouping (``lokey`` and ``hikey`` share a line; ``sample`` gets
its own) is part of the contract and belongs somewhere explicit rather than scattered through
string concatenation. And step 9's ``/api/sfz/preview`` wants the same document the writer is
about to put on disk, without writing it.

Region contents come from `export.zones` (which key and velocity span each sample covers) plus
the per-sample facts the pipeline measured. Two of those deserve care:

- ``amp_velcurve_1`` is the gain at velocity 1 relative to 127, ``10 ** (-range_db / 20)``. In
  velocity normalize mode `range_db` is what the pipeline *measured* from the audio; otherwise
  it is `OutputConfig.velocity_dynamic_range_db`, the value the user asked for.
- ``loop_crossfade`` is emitted **only** in ``sfz`` crossfade mode, and its value is that
  region's own clamped fade length, never the project-wide ``crossfade.loop_crossfade_ms``
  request. In ``baked`` mode the fade is already in the audio and the opcode must be absent —
  asking the sampler to fade again would apply it twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from autosampler.export.zones import KeyZone, VelocityZone

Opcode = tuple[str, str]
"""One ``name=value`` pair, with the value already formatted for output."""

_OPCODE_SEPARATOR = "  "
"""What separates two opcodes sharing a line, matching V1's output."""

SUSTAIN_DEFAULT_PATH = "samples/"
RELEASE_DEFAULT_PATH = "samples/release/"


@dataclass(frozen=True, slots=True)
class OpcodeLine:
    """One line of the SFZ: one or more opcodes rendered side by side."""

    opcodes: tuple[Opcode, ...]

    def render(self) -> str:
        """Return this line's text."""
        return _OPCODE_SEPARATOR.join(f"{name}={value}" for name, value in self.opcodes)


@dataclass(frozen=True, slots=True)
class SfzRegion:
    """One ``<region>`` block: the opcode lines that describe a single sample."""

    lines: tuple[OpcodeLine, ...]


@dataclass(frozen=True, slots=True)
class SfzDocument:
    """A whole SFZ file: one ``<control>``, one ``<group>``, and its ``<region>`` blocks."""

    default_path: str
    group: tuple[OpcodeLine, ...]
    regions: tuple[SfzRegion, ...]

    def render(self) -> str:
        """Return the complete SFZ text, ending in a newline."""
        lines = ["<control>", f"default_path={self.default_path}", "", "<group>"]
        lines.extend(line.render() for line in self.group)
        lines.append("")
        for region in self.regions:
            lines.append("<region>")
            lines.extend(line.render() for line in region.lines)
            lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SustainRegion:
    """Everything the sustain SFZ needs to know about one exported sample."""

    filename: str
    key: KeyZone
    velocity: VelocityZone
    loop: tuple[int, int] | None = None
    loop_crossfade_s: float | None = None


@dataclass(frozen=True, slots=True)
class ReleaseRegion:
    """Everything the release SFZ needs to know about one exported release sample."""

    filename: str
    key: KeyZone
    velocity: VelocityZone
    rt_decay: float


def amp_velcurve_1(dynamic_range_db: float) -> float:
    """Return the ``amp_velcurve_1`` value for a given dynamic range.

    Args:
        dynamic_range_db: How much quieter velocity 1 should play than velocity 127.

    Returns:
        The linear gain at velocity 1, ``10 ** (-dynamic_range_db / 20)``.
    """
    return float(10.0 ** (-dynamic_range_db / 20.0))


def _placement_lines(key: KeyZone, velocity: VelocityZone) -> list[OpcodeLine]:
    """Return the key/velocity lines shared by sustain and release regions."""
    lines = [
        OpcodeLine(
            (
                ("lokey", str(key.lokey)),
                ("hikey", str(key.hikey)),
                ("pitch_keycenter", str(key.note)),
            )
        ),
        OpcodeLine((("lovel", str(velocity.lovel)), ("hivel", str(velocity.hivel)))),
    ]
    if velocity.xfin is not None:
        lo, hi = velocity.xfin
        lines.append(OpcodeLine((("xfin_lovel", str(lo)), ("xfin_hivel", str(hi)))))
    if velocity.xfout is not None:
        lo, hi = velocity.xfout
        lines.append(OpcodeLine((("xfout_lovel", str(lo)), ("xfout_hivel", str(hi)))))
    return lines


def build_sustain_region(region: SustainRegion) -> SfzRegion:
    """Turn one `SustainRegion` into its ``<region>`` block."""
    lines = [
        OpcodeLine((("sample", region.filename),)),
        *_placement_lines(region.key, region.velocity),
    ]
    if region.loop is not None:
        loop_start, loop_end = region.loop
        lines.append(OpcodeLine((("loop_mode", "loop_continuous"),)))
        lines.append(OpcodeLine((("loop_start", str(loop_start)),)))
        lines.append(OpcodeLine((("loop_end", str(loop_end)),)))
        if region.loop_crossfade_s is not None:
            lines.append(
                OpcodeLine((("loop_crossfade", f"{region.loop_crossfade_s:.4f}"),))
            )
    return SfzRegion(tuple(lines))


def build_release_region(region: ReleaseRegion) -> SfzRegion:
    """Turn one `ReleaseRegion` into its ``<region>`` block."""
    lines = [
        OpcodeLine((("sample", region.filename),)),
        *_placement_lines(region.key, region.velocity),
        OpcodeLine((("rt_decay", f"{region.rt_decay:.2f}"),)),
    ]
    return SfzRegion(tuple(lines))


def build_sustain_document(
    regions: Sequence[SustainRegion], *, ampeg_release: float, dynamic_range_db: float
) -> SfzDocument:
    """Assemble the sustain SFZ document.

    Args:
        regions: The exported sustain samples, in output order.
        ampeg_release: Sampler release time, from `config.schema.OutputConfig`.
        dynamic_range_db: Range behind ``amp_velcurve_1`` — measured by velocity-mode
            normalize when it ran, otherwise the configured fallback.

    Returns:
        The renderable document.
    """
    group = (
        OpcodeLine((("ampeg_release", f"{ampeg_release:.3f}"),)),
        OpcodeLine((("amp_velcurve_1", f"{amp_velcurve_1(dynamic_range_db):.6f}"),)),
    )
    return SfzDocument(
        default_path=SUSTAIN_DEFAULT_PATH,
        group=group,
        regions=tuple(build_sustain_region(region) for region in regions),
    )


def build_release_document(
    regions: Sequence[ReleaseRegion], *, ampeg_release: float, dynamic_range_db: float
) -> SfzDocument:
    """Assemble the release SFZ document.

    ``trigger=release`` makes these regions sound on note-off. ``ampeg_attack`` is set to the
    sustain's ``ampeg_release`` so the release tail fades in exactly as the held note fades
    out, instead of the two overlapping or leaving a gap.

    Args:
        regions: The exported release samples, in output order.
        ampeg_release: The sustain's release time, reused as this file's attack.
        dynamic_range_db: Range behind ``amp_velcurve_1``, as for the sustain document.

    Returns:
        The renderable document.
    """
    group = (
        OpcodeLine((("trigger", "release"),)),
        OpcodeLine((("ampeg_attack", f"{ampeg_release:.3f}"),)),
        OpcodeLine((("ampeg_sustain", "0"),)),
        OpcodeLine((("ampeg_decay", "1.000"),)),
        OpcodeLine((("ampeg_release", "0.300"),)),
        OpcodeLine((("amp_velcurve_1", f"{amp_velcurve_1(dynamic_range_db):.6f}"),)),
    )
    return SfzDocument(
        default_path=RELEASE_DEFAULT_PATH,
        group=group,
        regions=tuple(build_release_region(region) for region in regions),
    )
