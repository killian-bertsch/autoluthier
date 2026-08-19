"""The cross-cutting guard that V2's rewrite did not change V1's instrument math.

``keybass_sustain.sfz`` and ``keybass_release.sfz`` in this directory were produced by **V1**
(``Auto-sampler_CLI``, its own venv) from the render `keybass_render` builds, using the V1
``metadata.toml`` equivalent of ``keybass_project.toml``. Nothing here imports V1 — the whole
point of the clean break — so the comparison is against checked-in text rather than a live run.

The sustain file must match V2 **byte for byte**. That covers a lot at once: key zones,
velocity zones and their crossfade ramps, ``velocity_map`` segment selection, filenames and
note naming, ``amp_velcurve_1``, and the detected loop points and crossfade length, which come
out of `dsp.loop`'s port of V1's search. ``loop_crossfade_mode = "sfz"`` is set in the config
precisely to isolate that: V2's default ``baked`` mode deliberately omits the opcode because
the fade is in the audio, so comparing under the default would confuse an intended change with
a regression.

The release file matches with **one intended difference**: V2 also emits ``xfin``/``xfout``
there. V1 widened release velocity zones by ``crossfade_percent`` but never wrote the ramps,
so two release tails sounded together at full level across the overlap band. That was neither
a catalogued bug nor part of the audio math held verbatim, so it was raised as an open question
and fixed with the user's sign-off. The tests below pin both halves of that: the release file
is identical once the ramp lines are removed, *and* the ramps that were added match the sustain
file's for the same velocity.

To regenerate the golden files after an intended change:

    # in Auto-sampler_CLI_2
    uv run python -c "import sys; sys.path.insert(0, 'tests/golden'); \
        from pathlib import Path; import keybass_render as k; \
        k.write_golden_instrument(Path('/tmp/golden/keybass'))"
    # write the V1 metadata.toml equivalent into /tmp/golden/keybass, then, in Auto-sampler_CLI:
    .venv/bin/python main.py /tmp/golden --output-dir /tmp/golden/out
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

from autosampler.config.toml_io import load_project
from autosampler.export.writer import ExportPlan, plan_export
from autosampler.pipeline.executor import run_instrument
from tests.golden.keybass_render import INSTRUMENT_NAME, write_golden_instrument

GOLDEN_DIR = Path(__file__).parent
_XFADE_LINE = re.compile(r"^xf(in|out)_(lo|hi)vel=")


def _without_xfade_lines(sfz: str) -> str:
    return "\n".join(line for line in sfz.splitlines() if not _XFADE_LINE.match(line)) + "\n"


def _region_blocks(sfz: str) -> list[list[str]]:
    """Split rendered SFZ text into one list of opcode lines per ``<region>``."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in sfz.splitlines():
        if line == "<region>":
            current = []
            blocks.append(current)
        elif current is not None and line:
            current.append(line)
    return blocks


def _opcodes(block: list[str]) -> dict[str, str]:
    return {
        name: value
        for line in block
        for name, _, value in (pair.partition("=") for pair in line.split("  "))
    }


@pytest.fixture(scope="module")
def golden_plan(tmp_path_factory: pytest.TempPathFactory) -> ExportPlan:
    """Run the golden instrument through V2 and return the export it would write."""
    folder = write_golden_instrument(tmp_path_factory.mktemp("golden") / INSTRUMENT_NAME)
    config = load_project(GOLDEN_DIR / "keybass_project.toml")
    result = run_instrument(folder, config, workers=1)
    return plan_export(result, config)


def test_sustain_sfz_matches_v1_byte_for_byte(golden_plan: ExportPlan) -> None:
    expected = (GOLDEN_DIR / "keybass_sustain.sfz").read_text(encoding="utf-8")
    assert golden_plan.sustain.render() == expected


def test_release_sfz_matches_v1_apart_from_the_added_crossfade_ramps(
    golden_plan: ExportPlan,
) -> None:
    assert golden_plan.release is not None
    expected = (GOLDEN_DIR / "keybass_release.sfz").read_text(encoding="utf-8")
    assert _without_xfade_lines(golden_plan.release.render()) == expected


def test_release_crossfade_ramps_match_the_sustain_zones(golden_plan: ExportPlan) -> None:
    assert golden_plan.release is not None
    sustain_by_vel = {
        opcodes["lovel"]: opcodes
        for opcodes in map(_opcodes, _region_blocks(golden_plan.sustain.render()))
    }
    release_blocks = _region_blocks(golden_plan.release.render())
    assert release_blocks
    for opcodes in map(_opcodes, release_blocks):
        sustain = sustain_by_vel[opcodes["lovel"]]
        for opcode in ("xfin_lovel", "xfin_hivel", "xfout_lovel", "xfout_hivel"):
            assert opcodes.get(opcode) == sustain.get(opcode)


def test_release_sfz_actually_gained_the_ramps(golden_plan: ExportPlan) -> None:
    """Guard the guard: the diff-modulo-ramps test is only meaningful if ramps exist."""
    assert golden_plan.release is not None
    assert "xfin_lovel=" in golden_plan.release.render()


def test_v1_golden_still_carries_no_ramps() -> None:
    """The checked-in V1 text must stay ramp-free, or the comparison above proves nothing."""
    v1_release = (GOLDEN_DIR / "keybass_release.sfz").read_text(encoding="utf-8")
    assert "xfin_lovel=" not in v1_release
    assert "xfin_lovel=" in (GOLDEN_DIR / "keybass_sustain.sfz").read_text(encoding="utf-8")


def test_key_zones_are_contiguous_and_gapless(golden_plan: ExportPlan) -> None:
    config = load_project(GOLDEN_DIR / "keybass_project.toml")
    blocks = [_opcodes(block) for block in _region_blocks(golden_plan.sustain.render())]
    spans = sorted({(int(o["lokey"]), int(o["hikey"])) for o in blocks})
    assert spans[0][0] == config.selection.min_note
    assert spans[-1][1] == config.selection.max_note
    for (_, hikey), (lokey, _) in pairwise(spans):
        assert lokey == hikey + 1
