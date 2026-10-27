"""Candidate benches from the elevation, and the ones OSM already knows.

A cut road is a bench: a run of ground locally FLATTER than the hillside
around it, with a cut bank above and fill below. That is a signature in real
elevation, which is why this runs on dem.tif and not on a picture of it.

Nothing here calls anything a road. A fault scarp, a pipeline cut, a cattle
terrace and a logging grade are indistinguishable by this method.
"""

import numpy as np
from shapely.geometry import LineString, mapping

from sar_geo_mcp.terrain.lineaments import (
    LINEAMENT_ASSUMPTIONS,
    bench_response,
    find_lineaments,
    subtract_known_ways,
)

PIXEL_M = 1.0


def hillside_with_bench(rows=160, cols=160, *, at=80, width=4, grade=0.0):
    """A plane falling east, with a level bench cut across it north-south.

    `grade` tilts the bench along its own length, the way a real road climbs.
    """
    ground = np.tile(np.linspace(80.0, 0.0, cols), (rows, 1)).astype("float64")
    for row in range(rows):
        target = ground[row, at] + grade * row
        ground[row, at:at + width] = target
    return ground.astype("float32")


def plain_hillside(rows=160, cols=160):
    return np.tile(np.linspace(80.0, 0.0, cols), (rows, 1)).astype("float32")


def hillside_with_nodata_strip(rows=160, cols=160, *, at=80, width=4):
    """The same falling plane as `plain_hillside`, with a strip of pixels set
    to NaN in exactly the shape `hillside_with_bench` cuts a real bench into -
    same location, same width, same length - so the two are a matched pair.
    """
    ground = plain_hillside(rows=rows, cols=cols).astype("float64")
    ground[:, at:at + width] = np.nan
    return ground.astype("float32")


def test_bench_response_is_positive_on_the_bench_and_flat_elsewhere():
    response = bench_response(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    on_bench = response[80, 80:84].mean()
    off_bench = response[80, 20:40].mean()
    assert on_bench > 5.0
    assert abs(off_bench) < 1.0


def test_a_plain_hillside_yields_no_lineaments():
    found = find_lineaments(plain_hillside(), PIXEL_M, track_width_m=4.0)
    assert found == []


def test_a_cut_bench_is_found_with_its_bearing_and_length():
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    assert len(found) == 1
    bench = found[0]
    assert bench["length_m"] > 100.0
    # Running north-south: bearing near 0 or 180 depending on traced direction.
    assert min(bench["end_to_end_bearing_deg"] % 180.0,
               180.0 - bench["end_to_end_bearing_deg"] % 180.0) < 15.0
    assert bench["sinuosity"] < 1.1
    assert 2.0 < bench["mean_width_m"] < 12.0


def test_a_climbing_bench_is_still_found():
    found = find_lineaments(hillside_with_bench(grade=0.05), PIXEL_M, track_width_m=4.0)
    assert len(found) == 1


def test_length_m_traces_to_the_components_true_extent():
    """`_trace` used to stop about one TRACE_BIN_M short at each end, because
    a bin centroid sits inside the component's true extreme (there is no
    pixel beyond the boundary to balance the mean). A 160 px bench used to
    trace to 149.0 m; fixed, it traces to within 1 m of its true extent.
    """
    rows = 260
    extent = 160
    start = 50
    ground = plain_hillside(rows=rows, cols=160)
    ground[start:start + extent, 80:84] = ground[start:start + extent, 80][:, None]
    found = find_lineaments(ground, PIXEL_M, track_width_m=4.0, min_length_m=1.0)
    assert len(found) == 1
    # extent-1 m is the true centre-to-centre span of an `extent`-pixel run.
    assert abs(found[0]["length_m"] - (extent - 1)) < 1.0


def test_a_short_scar_is_below_the_length_floor():
    ground = plain_hillside()
    ground[80:90, 80:84] = ground[80:90, 80][:, None]
    found = find_lineaments(ground, PIXEL_M, track_width_m=4.0)
    assert found == []


def test_a_known_way_along_the_bench_removes_it():
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    # The bench sits at column 80 of a raster whose origin is (0, 0) in 3857,
    # pixel 1 m, rows counting south: x = 80, y from 0 down to -160.
    along = {"geometry": mapping(LineString([(82.0, -10.0), (82.0, -150.0)])),
             "properties": {"name": "Old Haul Road"}}
    residue, _ = subtract_known_ways(found, [along], tolerance_m=15.0,
                                     transform=_transform(), latitude_deg=37.4)
    assert residue == []


def test_a_way_forty_metres_off_does_not_remove_it():
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    elsewhere = {"geometry": mapping(LineString([(122.0, -10.0), (122.0, -150.0)])),
                 "properties": {"name": "Ridge Trail"}}
    residue, _ = subtract_known_ways(found, [elsewhere], tolerance_m=15.0,
                                     transform=_transform(), latitude_deg=37.4)
    assert len(residue) == 1
    assert residue[0]["nearest_known_way"]["name"] == "Ridge Trail"
    assert 30.0 < residue[0]["nearest_known_way"]["distance_m"] < 50.0


def test_a_route_without_geometry_does_not_shift_the_names():
    """osm.json carries records with no geometry; the pairing must survive it.

    Filtering geometries into their own list and zipping that back against the
    unfiltered routes slips by one per skipped record, which reports a real
    distance under the wrong way's name.
    """
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    routes = [
        {"geometry": None, "properties": {"name": "A node with no way geometry"}},
        {"geometry": mapping(LineString([(122.0, -10.0), (122.0, -150.0)])),
         "properties": {"name": "Ridge Trail"}},
    ]
    residue, _ = subtract_known_ways(found, routes, tolerance_m=15.0,
                                     transform=_transform(), latitude_deg=37.4)
    assert len(residue) == 1
    assert residue[0]["nearest_known_way"]["name"] == "Ridge Trail"
    assert 30.0 < residue[0]["nearest_known_way"]["distance_m"] < 50.0


def test_a_nodata_strip_does_not_manufacture_a_lineament():
    """A DEM nodata boundary must never read as a bench.

    `bench_response` smooths local slope over a neighbourhood; if nodata is
    filled with 0 before smoothing, a NaN strip looks exactly like a real cut
    bench to the method - locally flat (fudged to 0), with a higher
    neighbourhood pulled in from the surrounding hillside - and produces a
    lineament where nothing was cut. This fixture is shaped identically to
    `hillside_with_bench`'s real bench to prove the two are indistinguishable
    under the old `nan_to_num` fill.
    """
    found = find_lineaments(hillside_with_nodata_strip(), PIXEL_M, track_width_m=4.0)
    assert found == []


def test_a_bench_survives_a_nodata_strip_elsewhere():
    ground = hillside_with_bench().copy()
    ground[:, 20:24] = np.nan  # nodata, well clear of the bench at column 80
    found = find_lineaments(ground, PIXEL_M, track_width_m=4.0)
    assert len(found) == 1
    assert found[0]["length_m"] > 100.0


def test_bench_response_is_nan_not_zero_at_a_nodata_pixel():
    ground = hillside_with_bench().copy()
    ground[50, 50] = np.nan  # an isolated nodata pixel, off the bench
    response = bench_response(ground, PIXEL_M, track_width_m=4.0)
    assert np.isnan(response[50, 50])


def test_assumptions_refuse_to_call_anything_a_road():
    text = " ".join(str(value) for value in LINEAMENT_ASSUMPTIONS.values()).lower()
    assert "unverified" in text
    assert "scarp" in text or "indistinguishable" in text


def _transform():
    from rasterio.transform import from_origin
    return from_origin(0.0, 0.0, PIXEL_M, PIXEL_M)


# ---------------------------------------------------------------------------
# Components too winding for one axis: split into centreline branches.
#
# Drawn straight onto the mask, because a level bench on a plane can only
# run along the contour - a tree of spurs or a road that bends back cannot
# be cut into one. The shapes are 5 px strokes, about a 4 m tread.
# ---------------------------------------------------------------------------

def drawn(*polylines, size=300):
    """A bench mask of 5 px strokes along the given (row, col) polylines."""
    from scipy import ndimage
    from skimage.draw import line

    centre = np.zeros((size, size), bool)
    for points in polylines:
        for (r0, c0), (r1, c1) in zip(points, points[1:]):
            rows, cols = line(r0, c0, r1, c1)
            centre[rows, cols] = True
    return ndimage.binary_dilation(centre, iterations=2)


TRUNK_WITH_SPURS = ([(20, 150), (250, 150)], [(80, 150), (40, 60)],
                    [(160, 150), (200, 240)])
#: Up, across, and back down: about 590 m of tread drawn.
CROOK = ([(250, 30), (40, 30), (40, 250), (200, 250)],)
#: Five legs joined by hairpins - every leg bins onto the same axis.
SWITCHBACK = ([(20, 40), (60, 240), (100, 40), (140, 240), (180, 40)],)
#: Winds round on itself: a coil, not a road.
SPIRAL = ([(150, 150), (150, 250), (50, 250), (50, 50), (250, 50), (250, 270)],)


def test_a_trunk_with_two_spurs_comes_back_as_one_trunk_and_two_spurs():
    """Cut at both junctions the trunk is three arms; they carry straight
    on through the junctions, so they are joined back into one line, and
    each spur - leaving at an angle - stays its own."""
    from sar_geo_mcp.terrain.lineaments import centreline_branches

    branches = sorted(centreline_branches(drawn(*TRUNK_WITH_SPURS), PIXEL_M),
                      key=lambda branch: -branch["length_m"])
    assert len(branches) == 3
    trunk, *spurs = branches
    assert len(trunk["arms"]) == 3, "the trunk's three arms, joined"
    assert 220 <= trunk["length_m"] <= 235, "230 m drawn"
    assert all(len(spur["arms"]) == 1 for spur in spurs)
    assert all(90 <= spur["length_m"] <= 105 for spur in spurs)
    assert all(3.0 <= branch["mean_width_m"] <= 6.0 for branch in branches)


def test_two_roads_crossing_come_back_as_two_whole_roads():
    from sar_geo_mcp.terrain.lineaments import centreline_branches

    cross = drawn([(20, 150), (280, 150)], [(150, 20), (150, 280)])
    branches = centreline_branches(cross, PIXEL_M)
    assert len(branches) == 2
    assert all(250 <= branch["length_m"] <= 262 for branch in branches)


def test_a_road_that_bends_back_is_split_not_dropped():
    from sar_geo_mcp.terrain.lineaments import lineaments_from_mask

    found = lineaments_from_mask(drawn(*CROOK), PIXEL_M)
    assert len(found) == 1
    assert found[0]["traced_by"] == "centreline"
    assert abs(found[0]["length_m"] - 590.0) < 15.0


def test_a_switchback_squashed_onto_one_axis_comes_back_as_its_legs():
    """Measured before: traced 204 m straight at sinuosity 1.0, then dropped
    as too wide - it never failed the winding test at all."""
    from sar_geo_mcp.terrain.lineaments import lineaments_from_mask

    found = lineaments_from_mask(drawn(*SWITCHBACK), PIXEL_M)
    assert len(found) == 4, "four legs; the fifth short piece is a hairpin turn"
    assert all(item["traced_by"] == "centreline" for item in found)
    assert all(180.0 <= item["length_m"] <= 215.0 for item in found)


def test_a_coil_is_still_dropped():
    from sar_geo_mcp.terrain.lineaments import lineaments_from_mask

    assert lineaments_from_mask(drawn(*SPIRAL), PIXEL_M) == []


def test_a_blob_is_not_split_into_roads():
    from sar_geo_mcp.terrain.lineaments import lineaments_from_mask

    mask = np.zeros((300, 300), bool)
    mask[100:180, 100:180] = True
    assert lineaments_from_mask(mask, PIXEL_M) == []


def test_a_straight_bench_is_still_traced_along_its_axis():
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    assert found and all(item["traced_by"] == "axis" for item in found)


def test_a_diagonal_staircase_has_no_junctions():
    """Counting neighbours would call a staircase step a junction - three
    skeleton pixels round it - and chop a diagonal road into stubs."""
    from sar_geo_mcp.terrain.lineaments import _junctions

    stairs = np.zeros((20, 20), bool)
    for step in range(9):
        stairs[step * 2, step * 2] = stairs[step * 2, step * 2 + 1] = True
        stairs[step * 2 + 1, step * 2 + 1] = True
    assert not _junctions(stairs).any()

    tee = np.zeros((20, 20), bool)
    tee[10, 2:18] = True
    tee[2:10, 10] = True
    assert _junctions(tee).sum() >= 1


def test_a_mostly_mapped_bench_keeps_the_stretch_that_runs_on_past_the_map():
    """Measured before: a bench two thirds along a mapped road was dropped
    whole, the unmapped third with it."""
    found = find_lineaments(hillside_with_bench(rows=300), PIXEL_M, track_width_m=4.0)
    assert len(found) == 1 and found[0]["length_m"] > 290
    mapped_top = {"geometry": mapping(LineString([(82.0, 0.0), (82.0, -180.0)])),
                  "properties": {"name": "Old Haul Road"}}
    residue, _ = subtract_known_ways(found, [mapped_top], tolerance_m=15.0,
                                     transform=_transform(), latitude_deg=0.0)
    assert len(residue) == 1
    piece = residue[0]
    assert piece["clipped_from_m"] == found[0]["length_m"]
    assert 0.55 < piece["mapped_share"] < 0.7
    assert 95 < piece["length_m"] < 115, "rows 195-299 lie past the 15 m buffer"
    ys = [y for _, y in piece["geometry"]["coordinates"]]
    assert max(ys) <= -194.0, "only the unmapped stretch, south of the road's end"


def test_a_short_unmapped_tail_is_not_offered():
    found = find_lineaments(hillside_with_bench(), PIXEL_M, track_width_m=4.0)
    nearly_all = {"geometry": mapping(LineString([(82.0, 0.0), (82.0, -130.0)])),
                  "properties": {"name": "Old Haul Road"}}
    residue, _ = subtract_known_ways(found, [nearly_all], tolerance_m=15.0,
                                     transform=_transform(), latitude_deg=0.0)
    assert residue == [], "about 15 m runs on past the buffer - under MIN_LENGTH_M"


def test_a_hairpin_chained_through_its_foot_comes_back_as_its_two_legs():
    """Down one leg, across the foot, back up the other: chained, it winds
    11x its chord and fails. Regrouped in order, the first leg keeps the
    foot and the second stands alone - two legs, not three scraps."""
    from sar_geo_mcp.terrain.lineaments import _regrouped

    def part(points):
        length = sum(np.hypot(x1 - x0, y1 - y0)
                     for (x0, y0), (x1, y1) in zip(points, points[1:]))
        return {"pixels": points, "length_m": float(length), "mean_width_m": 4.0}

    down = part([(0.0, float(y)) for y in range(0, 101)])
    foot = part([(float(x), 100.0) for x in range(1, 21)])
    up = part([(20.0, float(y)) for y in range(99, -1, -1)])
    legs = _regrouped([down, foot, up], (0, 0), PIXEL_M, 4.0, 50.0)
    assert [round(leg["length_m"]) for leg in legs] == [120, 99]
    assert [leg["arms_joined"] for leg in legs] == [2, 1]
