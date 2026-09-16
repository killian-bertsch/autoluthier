"""Zone math and selection: the part that decides which sample a sampler reaches for."""

from __future__ import annotations

from itertools import pairwise

import pytest

from autoluthier.config.schema import parse_velocity_map
from autoluthier.export.zones import (
    MAX_VELOCITY,
    MIN_VELOCITY,
    compute_key_zones,
    compute_velocity_zones,
    select_velocities_segmented,
)

RECORDED_6 = [1, 26, 51, 77, 102, 127]


class TestKeyZones:
    def test_boundaries_sit_at_midpoints_between_neighbors(self) -> None:
        zones = compute_key_zones([48, 52, 56], min_note=48, max_note=60)
        assert [(z.lokey, z.hikey) for z in zones] == [(48, 50), (51, 54), (55, 60)]

    def test_outer_edges_clamp_to_the_configured_range(self) -> None:
        zones = compute_key_zones([50, 55], min_note=40, max_note=70)
        assert zones[0].lokey == 40
        assert zones[-1].hikey == 70

    def test_a_single_note_covers_the_whole_range(self) -> None:
        zones = compute_key_zones([60], min_note=21, max_note=108)
        assert [(z.note, z.lokey, z.hikey) for z in zones] == [(60, 21, 108)]

    def test_no_notes_yields_no_zones(self) -> None:
        assert compute_key_zones([], min_note=21, max_note=108) == []

    @pytest.mark.parametrize("interval", [1, 2, 3, 5, 7])
    def test_zones_are_contiguous_and_gapless_over_the_whole_range(self, interval: int) -> None:
        notes = list(range(36, 85, interval))
        zones = compute_key_zones(notes, min_note=36, max_note=84)
        assert zones[0].lokey == 36
        assert zones[-1].hikey == 84
        for lower, upper in pairwise(zones):
            assert upper.lokey == lower.hikey + 1

    @pytest.mark.parametrize("interval", [1, 2, 3, 5, 7])
    def test_every_zone_contains_its_own_note(self, interval: int) -> None:
        notes = list(range(36, 85, interval))
        for zone in compute_key_zones(notes, min_note=36, max_note=84):
            assert zone.lokey <= zone.note <= zone.hikey


class TestVelocityZones:
    def test_natural_boundaries_with_no_crossfade(self) -> None:
        zones = compute_velocity_zones([1, 64, 127], crossfade_percent=0.0)
        assert [(z.lovel, z.hivel) for z in zones] == [(1, 32), (33, 95), (96, 127)]
        assert all(z.xfin is None and z.xfout is None for z in zones)

    def test_zones_are_contiguous_and_gapless_with_no_crossfade(self) -> None:
        zones = compute_velocity_zones(RECORDED_6, crossfade_percent=0.0)
        assert zones[0].lovel == MIN_VELOCITY
        assert zones[-1].hivel == MAX_VELOCITY
        for lower, upper in pairwise(zones):
            assert upper.lovel == lower.hivel + 1

    def test_crossfade_widens_each_zone_by_a_percent_of_its_own_width(self) -> None:
        # Middle zone's natural span is 33..95, i.e. 63 wide; 20% of that is int(12.6) == 12.
        zones = compute_velocity_zones([1, 64, 127], crossfade_percent=20.0)
        middle = zones[1]
        assert (middle.lovel, middle.hivel) == (33 - 12, 95 + 12)
        assert middle.xfin == (21, 33)
        assert middle.xfout == (95, 107)

    def test_outermost_edges_never_widen_past_the_velocity_range(self) -> None:
        zones = compute_velocity_zones(RECORDED_6, crossfade_percent=100.0)
        assert (zones[0].lovel, zones[0].xfin) == (MIN_VELOCITY, None)
        assert (zones[-1].hivel, zones[-1].xfout) == (MAX_VELOCITY, None)

    def test_a_crossfade_rounding_to_zero_width_emits_no_ramp(self) -> None:
        # A 2-velocity split gives the upper zone a 64-wide span; 1% of it truncates to 0.
        zones = compute_velocity_zones([1, 127], crossfade_percent=1.0)
        assert all(z.xfin is None and z.xfout is None for z in zones)

    def test_a_single_layer_covers_the_whole_velocity_range(self) -> None:
        zones = compute_velocity_zones([127], crossfade_percent=50.0)
        assert [(z.lovel, z.hivel, z.xfin, z.xfout) for z in zones] == [(1, 127, None, None)]

    def test_every_zone_contains_its_own_velocity(self) -> None:
        for percent in (0.0, 20.0, 50.0):
            for zone in compute_velocity_zones(RECORDED_6, crossfade_percent=percent):
                assert zone.lovel <= zone.velocity <= zone.hivel


class TestSegmentedSelection:
    def test_each_segment_keeps_its_own_layer_count(self) -> None:
        segments = parse_velocity_map("0-63:2, 64-127:3")
        assert select_velocities_segmented(RECORDED_6, segments) == [1, 51, 77, 102, 127]

    def test_a_segment_asking_for_more_than_it_holds_keeps_all_of_them(self) -> None:
        segments = parse_velocity_map("0-63:10")
        assert select_velocities_segmented(RECORDED_6, segments) == [1, 26, 51]

    def test_a_segment_matching_no_recorded_layer_contributes_nothing(self) -> None:
        segments = parse_velocity_map("2-25:2, 64-127:1")
        assert select_velocities_segmented(RECORDED_6, segments) == [127]

    def test_overlapping_segments_deduplicate(self) -> None:
        segments = parse_velocity_map("0-127:6, 64-127:3")
        assert select_velocities_segmented(RECORDED_6, segments) == RECORDED_6

    def test_a_single_layer_segment_takes_its_loudest(self) -> None:
        """V1's tie-break: one layer covering a range should be the loud one, not the middle."""
        segments = parse_velocity_map("0-63:1")
        assert select_velocities_segmented(RECORDED_6, segments) == [51]

    def test_no_recorded_velocities_yields_nothing(self) -> None:
        assert select_velocities_segmented([], parse_velocity_map("0-127:4")) == []
