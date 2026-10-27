"""Bridges across junction gaps, along flat ground.

Slope rasters are built by hand: a flat road corridor through steep
ground, so where the router may and may not go is known in advance.
"""

import numpy as np
from shapely.geometry import LineString

from sar_geo_mcp.terrain.bridges import BRIDGE_ASSUMPTIONS, junction_bridges

PIXEL_M = 2.0


def ground(size=120, corridor_row=60, corridor_deg=4.0, hillside_deg=30.0):
    """Steep hillside with one flat east-west road corridor 4 px wide."""
    slope = np.full((size, size), hillside_deg)
    slope[corridor_row - 2:corridor_row + 2, :] = corridor_deg
    return slope


def test_a_gap_on_the_road_is_bridged_along_it():
    left = LineString([(5, 60), (40, 60)])
    right = LineString([(55, 60), (100, 60)])           # 15 px = 30 m gap
    found = junction_bridges([left, right], [left, right], ground(), PIXEL_M)
    assert len(found) == 1, "the two ends find each other once, not twice"
    bridge = found[0]
    assert 28 <= bridge["length_m"] <= 34
    assert bridge["steepest_deg"] <= 4.0
    assert all(abs(row - 60) <= 2 for _, row in bridge["path"])


def test_a_gap_across_steep_ground_is_not_bridged():
    """Two benches one above the other on the hillside: 30 m apart, but
    nothing flat joins them."""
    upper = LineString([(10, 30), (100, 30)])
    lower = LineString([(10, 45), (100, 45)])
    slope = np.full((120, 120), 30.0)
    assert junction_bridges([upper, lower], [upper, lower], slope, PIXEL_M) == []


def test_steep_end_pixels_do_not_block_a_flat_junction():
    """A traced line ends on the cut bank: its last few pixels read steep
    while the junction between the lines is flat. Measured on real ground,
    judging the whole path bridged none of the circled junctions."""
    slope = ground()
    slope[56:64, 38:42] = 24.0          # the bank where the left line stops
    slope[56:64, 54:58] = 24.0          # and where the right one starts
    left = LineString([(5, 60), (40, 60)])
    right = LineString([(56, 60), (100, 60)])
    assert len(junction_bridges([left, right], [left, right], slope, PIXEL_M)) == 1


def test_lines_already_touching_or_too_far_apart_are_left_alone():
    slope = ground()
    left = LineString([(5, 60), (40, 60)])
    touching = LineString([(40.5, 60), (100, 60)])
    assert junction_bridges([left, touching], [left, touching], slope, PIXEL_M) == []
    far = LineString([(70, 60), (110, 60)])             # 30 px = 60 m
    assert junction_bridges([left, far], [left, far], slope, PIXEL_M) == []


def test_a_line_end_can_bridge_to_a_mapped_way():
    slope = ground()
    line = LineString([(5, 60), (40, 60)])
    mapped = LineString([(60, 0), (60, 119)])           # a mapped road crossing
    slope[:, 58:62] = 4.0
    found = junction_bridges([line], [line, mapped], slope, PIXEL_M)
    assert len(found) == 1 and found[0]["to"] == 1


def test_assumptions_say_a_bridge_is_not_a_detection():
    assert BRIDGE_ASSUMPTIONS["is_not"].startswith("a detection")
    assert "not seen" in BRIDGE_ASSUMPTIONS["is_not"]
