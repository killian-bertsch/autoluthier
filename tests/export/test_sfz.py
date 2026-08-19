"""The opcode model and its rendered text."""

from __future__ import annotations

import pytest

from autosampler.export.sfz import (
    OpcodeLine,
    ReleaseRegion,
    SfzDocument,
    SfzRegion,
    SustainRegion,
    amp_velcurve_1,
    build_release_document,
    build_release_region,
    build_sustain_document,
    build_sustain_region,
)
from autosampler.export.zones import KeyZone, VelocityZone

KEY = KeyZone(note=60, lokey=58, hikey=62)
VEL = VelocityZone(velocity=100, lovel=80, hivel=120, xfin=None, xfout=None)
VEL_XF = VelocityZone(velocity=100, lovel=80, hivel=120, xfin=(80, 88), xfout=(112, 120))


def _lines(region: SfzRegion) -> list[str]:
    return [line.render() for line in region.lines]


class TestOpcodeLine:
    def test_a_single_opcode_renders_as_name_equals_value(self) -> None:
        assert OpcodeLine((("lokey", "48"),)).render() == "lokey=48"

    def test_opcodes_sharing_a_line_are_separated_by_two_spaces(self) -> None:
        assert OpcodeLine((("lovel", "1"), ("hivel", "31"))).render() == "lovel=1  hivel=31"


class TestDocumentRendering:
    def test_headers_and_blank_lines_follow_v1s_layout(self) -> None:
        document = SfzDocument(
            default_path="samples/",
            group=(OpcodeLine((("ampeg_release", "0.500"),)),),
            regions=(SfzRegion((OpcodeLine((("sample", "a.flac"),)),)),),
        )
        assert document.render() == (
            "<control>\n"
            "default_path=samples/\n"
            "\n"
            "<group>\n"
            "ampeg_release=0.500\n"
            "\n"
            "<region>\n"
            "sample=a.flac\n"
        )

    def test_a_document_with_no_regions_still_renders_its_header(self) -> None:
        rendered = SfzDocument(default_path="samples/", group=(), regions=()).render()
        assert rendered == "<control>\ndefault_path=samples/\n\n<group>\n"


class TestAmpVelcurve:
    @pytest.mark.parametrize(
        ("range_db", "expected"),
        [(0.0, 1.0), (20.0, 0.1), (40.0, 0.01), (60.0, 0.001)],
    )
    def test_matches_the_closed_form(self, range_db: float, expected: float) -> None:
        """10 ** (-range_db / 20), computed here rather than by calling the implementation."""
        assert amp_velcurve_1(range_db) == pytest.approx(expected)

    def test_the_group_opcode_is_written_with_six_decimals(self) -> None:
        document = build_sustain_document([], ampeg_release=0.5, dynamic_range_db=40.0)
        assert "amp_velcurve_1=0.010000" in document.render()


class TestSustainRegions:
    def test_placement_opcodes_are_grouped_onto_v1s_lines(self) -> None:
        region = build_sustain_region(SustainRegion(filename="a.flac", key=KEY, velocity=VEL))
        assert _lines(region) == [
            "sample=a.flac",
            "lokey=58  hikey=62  pitch_keycenter=60",
            "lovel=80  hivel=120",
        ]

    def test_crossfade_ramps_appear_only_when_the_zone_has_them(self) -> None:
        region = build_sustain_region(
            SustainRegion(filename="a.flac", key=KEY, velocity=VEL_XF)
        )
        assert _lines(region)[3:] == [
            "xfin_lovel=80  xfin_hivel=88",
            "xfout_lovel=112  xfout_hivel=120",
        ]

    def test_a_sample_with_no_loop_gets_no_loop_opcodes(self) -> None:
        rendered = "\n".join(
            _lines(build_sustain_region(SustainRegion(filename="a.flac", key=KEY, velocity=VEL)))
        )
        assert "loop_" not in rendered

    def test_a_looped_sample_gets_mode_start_and_end(self) -> None:
        region = build_sustain_region(
            SustainRegion(filename="a.flac", key=KEY, velocity=VEL, loop=(1000, 2000))
        )
        assert _lines(region)[3:] == [
            "loop_mode=loop_continuous",
            "loop_start=1000",
            "loop_end=2000",
        ]

    def test_loop_crossfade_is_written_in_seconds_to_four_decimals(self) -> None:
        region = build_sustain_region(
            SustainRegion(
                filename="a.flac",
                key=KEY,
                velocity=VEL,
                loop=(1000, 2000),
                loop_crossfade_s=0.02,
            )
        )
        assert _lines(region)[-1] == "loop_crossfade=0.0200"

    def test_a_crossfade_without_a_loop_is_not_written(self) -> None:
        """The opcode only means anything attached to a loop; a stray one would confuse."""
        region = build_sustain_region(
            SustainRegion(filename="a.flac", key=KEY, velocity=VEL, loop_crossfade_s=0.02)
        )
        assert "loop_crossfade" not in "\n".join(_lines(region))


class TestReleaseRegions:
    def test_rt_decay_is_written_with_two_decimals(self) -> None:
        region = build_release_region(
            ReleaseRegion(filename="a_rel.flac", key=KEY, velocity=VEL, rt_decay=8.567)
        )
        assert _lines(region)[-1] == "rt_decay=8.57"

    def test_release_regions_carry_the_same_placement_opcodes_as_sustain(self) -> None:
        release = build_release_region(
            ReleaseRegion(filename="a_rel.flac", key=KEY, velocity=VEL_XF, rt_decay=6.0)
        )
        sustain = build_sustain_region(
            SustainRegion(filename="a.flac", key=KEY, velocity=VEL_XF)
        )
        assert _lines(release)[1:4] == _lines(sustain)[1:4]

    def test_release_regions_never_loop(self) -> None:
        region = build_release_region(
            ReleaseRegion(filename="a_rel.flac", key=KEY, velocity=VEL, rt_decay=6.0)
        )
        assert "loop_" not in "\n".join(_lines(region))

    def test_the_group_makes_regions_trigger_on_note_off(self) -> None:
        rendered = build_release_document(
            [], ampeg_release=0.045, dynamic_range_db=40.0
        ).render()
        assert "trigger=release" in rendered
        assert "ampeg_attack=0.045" in rendered

    def test_the_release_file_points_at_the_release_sample_directory(self) -> None:
        rendered = build_release_document(
            [], ampeg_release=0.5, dynamic_range_db=40.0
        ).render()
        assert "default_path=samples/release/" in rendered
