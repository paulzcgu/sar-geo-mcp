import math

import pytest

from sar_geo_mcp.geo import (
    bbox_center_latitude,
    buffer_bbox,
    polygon_bounds_wgs84,
    to_web_mercator,
)

SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.34, 37.30],
            [-122.30, 37.30],
            [-122.30, 37.34],
            [-122.34, 37.34],
            [-122.34, 37.30],
        ]
    ],
}


def test_polygon_bounds():
    assert polygon_bounds_wgs84(SQUARE) == pytest.approx((-122.34, 37.30, -122.30, 37.34))


def test_polygon_bounds_rejects_non_polygon():
    with pytest.raises(ValueError, match="expected Polygon"):
        polygon_bounds_wgs84({"type": "Point", "coordinates": [0, 0]})


def test_web_mercator_origin_is_zero():
    assert to_web_mercator((0.0, 0.0, 0.0, 0.0)) == pytest.approx((0, 0, 0, 0), abs=1e-6)


def test_web_mercator_matches_known_values():
    # Verified against CalTopo's own WMS during design exploration.
    result = to_web_mercator((-122.34, 37.30, -122.30, 37.34))

    assert result[0] == pytest.approx(-13618826.50, abs=1.0)
    assert result[1] == pytest.approx(4481005.82, abs=1.0)
    assert result[2] == pytest.approx(-13614373.72, abs=1.0)
    assert result[3] == pytest.approx(4486604.96, abs=1.0)


def test_bbox_center_latitude():
    assert bbox_center_latitude((-122.34, 37.30, -122.30, 37.34)) == pytest.approx(37.32)


def test_buffer_corrects_for_mercator_scale():
    bbox = (0.0, 0.0, 1000.0, 1000.0)
    latitude = 37.32

    buffered = buffer_bbox(bbox, ground_meters=200.0, latitude_deg=latitude)

    expected = 200.0 / math.cos(math.radians(latitude))
    assert buffered[0] == pytest.approx(-expected)
    assert buffered[2] == pytest.approx(1000.0 + expected)
    assert expected > 200.0  # the correction inflates, never shrinks


def test_buffer_at_equator_is_uncorrected():
    buffered = buffer_bbox((0.0, 0.0, 100.0, 100.0), ground_meters=50.0, latitude_deg=0.0)

    assert buffered[0] == pytest.approx(-50.0)


def test_ground_meters_per_mercator_unit_shrinks_off_equator():
    from sar_geo_mcp.geo import ground_meters_per_mercator_unit

    assert ground_meters_per_mercator_unit(0.0) == pytest.approx(1.0)
    assert ground_meters_per_mercator_unit(37.32) == pytest.approx(0.79526, abs=1e-4)


def test_ground_conversion_inverts_the_buffer_inflation():
    """buffer_bbox inflates by 1/cos; this must undo it exactly."""
    from sar_geo_mcp.geo import ground_meters_per_mercator_unit

    latitude = 37.32
    buffered = buffer_bbox((0.0, 0.0, 0.0, 0.0), ground_meters=200.0, latitude_deg=latitude)
    mercator_delta = buffered[2]

    assert mercator_delta * ground_meters_per_mercator_unit(latitude) == pytest.approx(200.0)


def test_geometry_bounds_handles_linestring():
    from sar_geo_mcp.geo import geometry_bounds_wgs84

    line = {"type": "LineString", "coordinates": [[-122.34, 37.30], [-122.30, 37.34]]}

    assert geometry_bounds_wgs84(line) == pytest.approx((-122.34, 37.30, -122.30, 37.34))


def test_geometry_bounds_handles_multipolygon():
    from sar_geo_mcp.geo import geometry_bounds_wgs84

    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-122.34, 37.30], [-122.33, 37.30], [-122.33, 37.31], [-122.34, 37.30]]],
            [[[-122.31, 37.33], [-122.30, 37.33], [-122.30, 37.34], [-122.31, 37.33]]],
        ],
    }

    assert geometry_bounds_wgs84(multi) == pytest.approx((-122.34, 37.30, -122.30, 37.34))


def test_geometry_bounds_rejects_unsupported_type():
    from sar_geo_mcp.geo import geometry_bounds_wgs84

    with pytest.raises(ValueError, match="unsupported geometry type"):
        geometry_bounds_wgs84({"type": "GeometryCollection", "geometries": []})


def test_geometry_bounds_rejects_empty_coordinates():
    from sar_geo_mcp.geo import geometry_bounds_wgs84

    with pytest.raises(ValueError, match="no coordinates"):
        geometry_bounds_wgs84({"type": "Polygon", "coordinates": []})


# ---------------------------------------------------------------------------
# buffer_geometry_wgs84: the polygon CalTopo's Buffer tool draws around a line
# ---------------------------------------------------------------------------

#: 0.02 degrees of latitude is 2,226 m of ground at any longitude.
NORTH_SOUTH_LINE = {"type": "LineString",
                    "coordinates": [[-122.32, 37.30], [-122.32, 37.32]]}
METRES_PER_DEGREE_LATITUDE = 111_320.0


def _metres_per_degree_longitude(latitude: float) -> float:
    return METRES_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude))


def test_a_buffered_line_is_a_ribbon_twice_the_distance_wide():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    result = buffer_geometry_wgs84(NORTH_SOUTH_LINE, 15.0)
    polygon = result["geometry"]
    assert polygon["type"] == "Polygon"
    ring = polygon["coordinates"][0]
    # The east and west edges of the ribbon, in ground metres.
    longitudes = [x for x, _y in ring]
    width_m = (max(longitudes) - min(longitudes)) * _metres_per_degree_longitude(37.31)
    assert width_m == pytest.approx(30.0, rel=0.01)
    assert result["distance_m"] == 15.0
    assert result["width_m"] == 30.0
    # 2,226 m by 30 m is 6.68 ha, plus a 15 m disc for the two round ends.
    assert result["area_ha"] == pytest.approx(6.75, abs=0.1)


def test_a_buffered_line_has_round_ends_like_caltopo():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    ring = buffer_geometry_wgs84(NORTH_SOUTH_LINE, 15.0)["geometry"]["coordinates"][0]
    latitudes = [y for _x, y in ring]
    # A round cap reaches distance_m PAST the end of the line; a flat cap
    # would stop at it.
    overshoot_m = (max(latitudes) - 37.32) * METRES_PER_DEGREE_LATITUDE
    assert overshoot_m == pytest.approx(15.0, rel=0.05)
    assert len(ring) > 20  # a rectangle would be 5


def test_two_touching_branches_buffer_into_one_polygon():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    fork = {"type": "MultiLineString", "coordinates": [
        [[-122.32, 37.30], [-122.32, 37.31]],
        [[-122.32, 37.31], [-122.31, 37.315]],
    ]}
    assert buffer_geometry_wgs84(fork, 15.0)["geometry"]["type"] == "Polygon"


def test_branches_further_apart_than_the_width_are_a_multipolygon():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    apart = {"type": "MultiLineString", "coordinates": [
        [[-122.32, 37.30], [-122.32, 37.31]],
        [[-122.30, 37.30], [-122.30, 37.31]],
    ]}
    assert buffer_geometry_wgs84(apart, 15.0)["geometry"]["type"] == "MultiPolygon"


def test_a_polygon_is_widened_too():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    grown = buffer_geometry_wgs84(SQUARE, 100.0)
    assert grown["geometry"]["type"] == "Polygon"
    # The square is 3,543 m by 4,453 m of ground (1,578 ha); a 100 m ring
    # round it adds about 160 ha.
    assert 1700 < grown["area_ha"] < 1800


@pytest.mark.parametrize("distance", [0, -1, 1001, float("nan"), "wide"])
def test_a_distance_outside_the_band_is_refused(distance):
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    with pytest.raises(ValueError, match="1000"):
        buffer_geometry_wgs84(NORTH_SOUTH_LINE, distance)


def test_a_point_and_an_empty_geometry_are_refused():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    with pytest.raises(ValueError, match="Point"):
        buffer_geometry_wgs84({"type": "Point", "coordinates": [-122.32, 37.30]}, 15.0)
    with pytest.raises(ValueError, match="empty"):
        buffer_geometry_wgs84({"type": "LineString", "coordinates": []}, 15.0)


def test_a_lat_lon_swapped_pair_is_refused_naming_the_order():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    swapped = {"type": "LineString", "coordinates": [[37.30, -122.32], [37.32, -122.32]]}
    with pytest.raises(ValueError, match="longitude, latitude"):
        buffer_geometry_wgs84(swapped, 15.0)


def test_a_nan_coordinate_is_refused_rather_than_crashing():
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    broken = {"type": "LineString", "coordinates": [[-122.32, float("nan")], [-122.32, 37.32]]}
    with pytest.raises(ValueError, match="not a number"):
        buffer_geometry_wgs84(broken, 15.0)


# ---------------------------------------------------------------------------
# single_ring: the ribbon round a loop, cut open so CalTopo keeps the hole
# ---------------------------------------------------------------------------

#: A closed way, about 500 m square at 37.30 N - a loop trail in miniature.
LOOP = {"type": "LineString", "coordinates": [
    [-122.3200, 37.3000], [-122.3143, 37.3000], [-122.3143, 37.3045],
    [-122.3200, 37.3045], [-122.3200, 37.3000],
]}


def _filled(geometry: dict):
    """The ground a renderer paints for `geometry`, holes honoured, whether
    the holes are separate rings or cut open along a bridge."""
    from shapely.geometry import shape
    from shapely.validation import make_valid

    made = make_valid(shape(geometry))
    parts = getattr(made, "geoms", [made])
    from shapely.ops import unary_union
    return unary_union([p for p in parts if p.geom_type in ("Polygon", "MultiPolygon")])


def test_a_buffered_loop_encloses_ground_as_a_hole_by_default():
    """The honest answer: a ribbon round a closed way is a donut."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    polygon = buffer_geometry_wgs84(LOOP, 15.0)["geometry"]

    assert polygon["type"] == "Polygon"
    assert len(polygon["coordinates"]) == 2  # outer ring and the loop's hole


def test_single_ring_cuts_the_hole_open_instead_of_dropping_it():
    """CalTopo carries one ring per polygon, so a hole reaches it only as a
    bridge cut into the outer ring - and the hole has to stay empty."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    cut = buffer_geometry_wgs84(LOOP, 15.0, single_ring=True)["geometry"]

    assert cut["type"] == "Polygon"
    assert len(cut["coordinates"]) == 1
    donut = _filled(buffer_geometry_wgs84(LOOP, 15.0)["geometry"])
    assert _filled(cut).symmetric_difference(donut).area == pytest.approx(0.0, abs=1e-12)


def test_a_cut_ribbon_still_measures_the_ribbon_not_the_ground_it_rings():
    """The area a team is being asked to walk does not change because the
    drawing did."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    assert (buffer_geometry_wgs84(LOOP, 15.0, single_ring=True)["area_ha"]
            == pytest.approx(buffer_geometry_wgs84(LOOP, 15.0)["area_ha"]))


def test_single_ring_leaves_a_ribbon_that_encloses_nothing_alone():
    """An open way has no hole to cut, so nothing is done to it."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    assert (buffer_geometry_wgs84(NORTH_SOUTH_LINE, 15.0, single_ring=True)["geometry"]
            == buffer_geometry_wgs84(NORTH_SOUTH_LINE, 15.0)["geometry"])


def test_every_part_of_a_multipolygon_is_cut_open():
    """Branches further apart than the width buffer into separate polygons;
    a loop among them still has to be cut."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    loop_and_line = {"type": "MultiLineString", "coordinates": [
        LOOP["coordinates"], [[-122.30, 37.30], [-122.30, 37.31]],
    ]}

    cut = buffer_geometry_wgs84(loop_and_line, 15.0, single_ring=True)["geometry"]

    assert cut["type"] == "MultiPolygon"
    assert [len(rings) for rings in cut["coordinates"]] == [1, 1]
    donut = _filled(buffer_geometry_wgs84(loop_and_line, 15.0)["geometry"])
    assert _filled(cut).symmetric_difference(donut).area == pytest.approx(0.0, abs=1e-12)


def test_a_way_that_rings_ground_twice_has_both_holes_cut():
    """The Anniversary Trail shape: one way, two enclosed pieces of ground.
    Bridging the second hole must not undo the bridge to the first."""
    from sar_geo_mcp.geo import buffer_geometry_wgs84

    two_loops = {"type": "MultiLineString", "coordinates": [
        LOOP["coordinates"],
        [[-122.3143, 37.3000], [-122.3086, 37.3000], [-122.3086, 37.3045],
         [-122.3143, 37.3045], [-122.3143, 37.3000]],
    ]}
    assert len(buffer_geometry_wgs84(two_loops, 15.0)["geometry"]["coordinates"]) == 3

    cut = buffer_geometry_wgs84(two_loops, 15.0, single_ring=True)["geometry"]

    assert len(cut["coordinates"]) == 1
    donut = _filled(buffer_geometry_wgs84(two_loops, 15.0)["geometry"])
    assert _filled(cut).symmetric_difference(donut).area == pytest.approx(0.0, abs=1e-12)
