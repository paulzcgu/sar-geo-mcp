"""Parking candidates off a staged directory: the filter, the point, the
order, and what is said when the staged cache predates the query."""

from __future__ import annotations

import json
import math

import pytest

from sar_geo_mcp.planning.parking import NEAREST_WAY_CEILING_M, parking_candidates
from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

# Windy Hill, the area the day-one work was measured on.
CENTRE = (37.35687, -122.24093)

# Ground metres per Web Mercator unit at CENTRE's latitude - the same
# conversion `parking_candidates` uses, needed here to build a route a known
# ground distance from a lot's edge.
_SCALE = math.cos(math.radians(CENTRE[0]))


def _metric(lon: float, lat: float) -> tuple[float, float]:
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)


def _square_3857(lon: float, lat: float, size_m: float) -> dict:
    x, y = _metric(lon, lat)
    half = size_m / 2.0
    ring = [(x - half, y - half), (x + half, y - half),
            (x + half, y + half), (x - half, y + half), (x - half, y - half)]
    return {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]}


def _l_shape_3857(lon: float, lat: float, size_m: float) -> dict:
    """An L: its centroid falls in the notch, outside the polygon."""
    x, y = _metric(lon, lat)
    s = size_m
    ring = [(x, y), (x + s, y), (x + s, y + s / 3), (x + s / 3, y + s / 3),
            (x + s / 3, y + s), (x, y + s), (x, y)]
    return {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]}


def _bowtie_multipolygon_3857(lon: float, lat: float, size_m: float) -> dict:
    """Two triangles touching at a point - the shape `make_valid` produces
    for a bowtie way, per `terrain/linear._repaired_polygon`."""
    x, y = _metric(lon, lat)
    s = size_m
    left = [(x - s, y - s), (x, y), (x - s, y + s), (x - s, y - s)]
    right = [(x + s, y - s), (x, y), (x + s, y + s), (x + s, y - s)]
    return {"type": "MultiPolygon",
            "coordinates": [[[list(pt) for pt in left]], [[list(pt) for pt in right]]]}


def _staged(tmp_path, sites, *, routes=(), query_version: int = OSM_QUERY_VERSION):
    directory = tmp_path / "staged"
    directory.mkdir()
    latitude, longitude = CENTRE
    (directory / "subject.json").write_text(json.dumps({
        "label": "aoi", "geometry": {
            "type": "Polygon", "coordinates": [[
                [longitude - 0.02, latitude - 0.02], [longitude + 0.02, latitude - 0.02],
                [longitude + 0.02, latitude + 0.02], [longitude - 0.02, latitude + 0.02],
                [longitude - 0.02, latitude - 0.02]]]}}), encoding="utf-8")
    (directory / "osm.json").write_text(json.dumps({
        "routes": list(routes), "parking": sites, "query_version": query_version}),
        encoding="utf-8")
    return directory


def _route(route_id, name, route_type, *, edge_x, centre_y, half_span=300.0):
    """A north-south LineString `edge_x` Web Mercator UNITS east of
    `centre_y`, running well past any lot built by `_square_3857` on that
    same centre - so the polygon-to-line distance is exactly the horizontal
    gap, not shortened by the line ending early."""
    return {"id": route_id, "kind": "route",
            "geometry": {"type": "LineString",
                        "coordinates": [[edge_x, centre_y - half_span],
                                        [edge_x, centre_y + half_span]]},
            "properties": {"route_type": route_type, "name": name, "osm_id": 1,
                          "surface": None}}


def _lot_with_gap_route(lon, lat, lot_size_m, gap_units, *, name="Alpine Road",
                        route_type="ROAD", route_id="route_1"):
    """A square lot at (lon, lat) and a named route running parallel to its
    east edge, `gap_units` Web Mercator units beyond it - so the true
    polygon-to-line distance is exactly `gap_units`, convertible to ground
    metres via `_SCALE`."""
    x, y = _metric(lon, lat)
    edge_x = x + lot_size_m / 2.0 + gap_units
    route = _route(route_id, name, route_type, edge_x=edge_x, centre_y=y)
    return _square_3857(lon, lat, lot_size_m), route


def _site(site_id, kind, geometry, *, name=None, flags=(), surface=None, capacity=None):
    return {"id": site_id, "kind": kind, "geometry": geometry,
            "properties": {"name": name, "osm_id": 1, "osm_type": "way",
                           "surface": surface, "capacity": capacity, "flags": list(flags)}}


def test_lots_come_back_nearest_first_with_ground_facts(tmp_path):
    latitude, longitude = CENTRE
    near = _site("parking-w1", "parking", _square_3857(longitude + 0.002, latitude, 60),
                 name="Near lot", surface="asphalt", capacity="40")
    far = _site("parking-w2", "parking", _square_3857(longitude + 0.02, latitude, 60),
                name="Far lot")

    payload = parking_candidates(_staged(tmp_path, [far, near]))

    assert [c["id"] for c in payload["candidates"]] == ["parking-w1", "parking-w2"]
    first = payload["candidates"][0]
    assert first["name"] == "Near lot"
    assert first["surface"] == "asphalt"
    assert 150 < first["distance_from_center_m"] < 250
    # East of the centre, so a bearing near 90.
    assert 80 < first["bearing_from_center_deg"] < 100
    # A 60-unit-wide square in EPSG:3857 at this latitude is a smaller
    # ground square (Web Mercator inflates east-west distance by 1/cos(lat)
    # at 37 N): confirmed against both the scale-factor conversion this
    # module uses and an independent WGS84 geodesic polygon-area check
    # (pyproj's Geod.polygon_area_perimeter), which agree at ~2270-2275 m2.
    assert 2000 < first["area_m2"] < 2600
    assert first["geometry"]["type"] == "Polygon"
    # WGS84 out, whatever came in: this goes on a CalTopo map.
    assert -123 < first["geometry"]["coordinates"][0][0][0] < -121


def test_the_point_is_inside_the_lot_even_when_the_centroid_is_not(tmp_path):
    from shapely.geometry import Point, shape

    latitude, longitude = CENTRE
    lot = _site("parking-w1", "parking", _l_shape_3857(longitude, latitude, 200))

    payload = parking_candidates(_staged(tmp_path, [lot]))

    point = payload["candidates"][0]["point"]
    polygon = shape(payload["candidates"][0]["geometry"])
    assert polygon.contains(Point(point["longitude"], point["latitude"]))


def test_the_point_is_inside_a_multipolygon_lot(tmp_path):
    """A repaired bowtie (see `terrain/linear._repaired_polygon`) can come
    back as a MultiPolygon; the marker still has to land on asphalt."""
    from shapely.geometry import Point, shape

    latitude, longitude = CENTRE
    lot = _site("parking-w1", "parking", _bowtie_multipolygon_3857(longitude, latitude, 100))

    payload = parking_candidates(_staged(tmp_path, [lot]))

    candidate = payload["candidates"][0]
    point = candidate["point"]
    polygon = shape(candidate["geometry"])
    assert polygon.geom_type == "MultiPolygon"
    assert polygon.contains(Point(point["longitude"], point["latitude"]))
    # area is the sum of both lobes, not just the one the point landed in.
    assert candidate["area_m2"] > 0


def test_a_hardstand_holding_a_mapped_lot_is_not_offered_twice(tmp_path):
    latitude, longitude = CENTRE
    school = _site("hardstand-w9", "hardstand", _square_3857(longitude, latitude, 300),
                   name="Sequoias School", flags=["parking_unverified"])
    lot = _site("parking-w10", "parking", _square_3857(longitude, latitude, 60),
                name="Sequoias lot")

    payload = parking_candidates(_staged(tmp_path, [school, lot]))

    assert [c["id"] for c in payload["candidates"]] == ["parking-w10"]


def test_a_hardstand_with_no_lot_in_it_is_offered_and_flagged(tmp_path):
    latitude, longitude = CENTRE
    school = _site("hardstand-w9", "hardstand", _square_3857(longitude, latitude, 300),
                   name="Sequoias School", flags=["parking_unverified"])

    payload = parking_candidates(_staged(tmp_path, [school]))

    assert [c["id"] for c in payload["candidates"]] == ["hardstand-w9"]
    assert "parking_unverified" in payload["candidates"][0]["flags"]


def test_the_limit_reports_what_it_dropped(tmp_path):
    latitude, longitude = CENTRE
    sites = [_site(f"parking-w{n}", "parking",
                   _square_3857(longitude + 0.002 * n, latitude, 60), name=f"Lot {n}")
             for n in range(1, 6)]

    payload = parking_candidates(_staged(tmp_path, sites), limit=2)

    assert [c["id"] for c in payload["candidates"]] == ["parking-w1", "parking-w2"]
    assert payload["summary"] == {"count": 2, "total_before_limit": 5, "truncated_to": 2}


def test_max_distance_drops_the_far_ones(tmp_path):
    latitude, longitude = CENTRE
    near = _site("parking-w1", "parking", _square_3857(longitude + 0.002, latitude, 60))
    far = _site("parking-w2", "parking", _square_3857(longitude + 0.05, latitude, 60))

    payload = parking_candidates(_staged(tmp_path, [near, far]), max_distance_m=1000)

    assert [c["id"] for c in payload["candidates"]] == ["parking-w1"]


def test_ground_staged_before_the_parking_query_says_so(tmp_path):
    """"No lots" and "lots were never fetched" are different answers, and a
    planner acts differently on each."""
    payload = parking_candidates(_staged(tmp_path, [], query_version=2))

    assert payload["candidates"] == []
    assert any("version 2" in warning and "re-stage" in warning
               for warning in payload["warnings"]), payload["warnings"]


def test_missing_osm_json_is_unknown_not_empty(tmp_path):
    directory = _staged(tmp_path, [])
    (directory / "osm.json").unlink()

    payload = parking_candidates(directory)

    assert payload["candidates"] == []
    assert any("osm.json" in warning for warning in payload["warnings"])


# --- nearest_way ------------------------------------------------------------


def test_a_lot_beside_a_named_road_gets_it(tmp_path):
    latitude, longitude = CENTRE
    lot_geometry, road = _lot_with_gap_route(
        longitude, latitude, 60, 15.0, name="Alpine Road", route_type="ROAD")
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[road]))

    nearest_way = payload["candidates"][0]["nearest_way"]
    assert nearest_way["name"] == "Alpine Road"
    assert nearest_way["route_type"] == "ROAD"


def test_nearest_way_distance_is_ground_metres_and_roughly_right(tmp_path):
    latitude, longitude = CENTRE
    gap_units = 15.0
    expected_ground_m = gap_units * _SCALE
    lot_geometry, road = _lot_with_gap_route(longitude, latitude, 60, gap_units)
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[road]))

    distance_m = payload["candidates"][0]["nearest_way"]["distance_m"]
    assert expected_ground_m - 1.0 < distance_m < expected_ground_m + 1.0


def test_an_unnamed_way_is_never_chosen_even_when_nearer(tmp_path):
    latitude, longitude = CENTRE
    lot_geometry, unnamed_road = _lot_with_gap_route(
        longitude, latitude, 60, 5.0, name=None, route_type="ROAD", route_id="route_near")
    _, named_road = _lot_with_gap_route(
        longitude, latitude, 60, 50.0, name="Alpine Road", route_type="ROAD",
        route_id="route_far")
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(
        _staged(tmp_path, [lot], routes=[unnamed_road, named_road]))

    nearest_way = payload["candidates"][0]["nearest_way"]
    assert nearest_way["name"] == "Alpine Road"


def test_a_blank_name_is_treated_the_same_as_no_name(tmp_path):
    latitude, longitude = CENTRE
    lot_geometry, blank_named_road = _lot_with_gap_route(
        longitude, latitude, 60, 5.0, name="   ", route_type="ROAD")
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[blank_named_road]))

    assert payload["candidates"][0]["nearest_way"] is None


def test_a_named_way_past_the_ceiling_gives_none(tmp_path):
    latitude, longitude = CENTRE
    gap_units = (NEAREST_WAY_CEILING_M + 50.0) / _SCALE
    lot_geometry, road = _lot_with_gap_route(longitude, latitude, 60, gap_units)
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[road]))

    assert payload["candidates"][0]["nearest_way"] is None


def test_a_trail_beats_a_farther_road(tmp_path):
    latitude, longitude = CENTRE
    lot_geometry, trail = _lot_with_gap_route(
        longitude, latitude, 60, 10.0, name="Anniversary Trail", route_type="TRAIL",
        route_id="route_trail")
    _, road = _lot_with_gap_route(
        longitude, latitude, 60, 40.0, name="Alpine Road", route_type="ROAD",
        route_id="route_road")
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[trail, road]))

    nearest_way = payload["candidates"][0]["nearest_way"]
    assert nearest_way["name"] == "Anniversary Trail"
    assert nearest_way["route_type"] == "TRAIL"


@pytest.mark.parametrize("excluded_type", ["WATERWAY", "BARRIER", "WATER_EDGE"])
def test_a_waterway_or_barrier_is_never_chosen_even_when_nearest(tmp_path, excluded_type):
    latitude, longitude = CENTRE
    lot_geometry, excluded = _lot_with_gap_route(
        longitude, latitude, 60, 5.0, name="Coyote Creek", route_type=excluded_type,
        route_id="route_near")
    _, named_road = _lot_with_gap_route(
        longitude, latitude, 60, 50.0, name="Alpine Road", route_type="ROAD",
        route_id="route_far")
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(
        _staged(tmp_path, [lot], routes=[excluded, named_road]))

    nearest_way = payload["candidates"][0]["nearest_way"]
    assert nearest_way["name"] == "Alpine Road"


def test_ties_break_by_name_then_route_type(tmp_path):
    latitude, longitude = CENTRE
    x, y = _metric(longitude, latitude)
    lot_geometry = _square_3857(longitude, latitude, 60)
    gap_units = 15.0
    edge_x = x + 30.0 + gap_units
    # Two ways at the exact same position relative to the lot - genuinely
    # tied on distance - differing only in name and class, so nothing but
    # the name/route_type tie-break can decide which is reported.
    beta = _route("route_beta", "Beta Road", "ROAD", edge_x=edge_x, centre_y=y)
    alpha = _route("route_alpha", "Alpha Trail", "TRAIL", edge_x=edge_x, centre_y=y)
    lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot], routes=[beta, alpha]))

    nearest_way = payload["candidates"][0]["nearest_way"]
    assert nearest_way["name"] == "Alpha Trail"
    assert abs(nearest_way["distance_m"] - gap_units * _SCALE) < 1.0


def test_a_lot_with_no_routes_at_all_gives_none(tmp_path):
    latitude, longitude = CENTRE
    lot = _site("parking-w1", "parking", _square_3857(longitude, latitude, 60),
               name="Near lot")

    payload = parking_candidates(_staged(tmp_path, [lot]))

    assert payload["candidates"][0]["nearest_way"] is None


def test_nearest_way_key_is_present_on_every_candidate_row(tmp_path):
    latitude, longitude = CENTRE
    lot_geometry, road = _lot_with_gap_route(longitude, latitude, 60, 15.0)
    near_lot = _site("parking-w1", "parking", lot_geometry, name="Near lot")
    far_lot = _site("parking-w2", "parking",
                    _square_3857(longitude + 0.02, latitude, 60), name="Far lot")

    payload = parking_candidates(_staged(tmp_path, [near_lot, far_lot], routes=[road]))

    assert len(payload["candidates"]) == 2
    for candidate in payload["candidates"]:
        assert "nearest_way" in candidate
    ids_with_way = {c["id"]: c["nearest_way"] for c in payload["candidates"]}
    assert ids_with_way["parking-w1"] is not None
    assert ids_with_way["parking-w2"] is None



def test_ground_staged_at_version_3_reads_its_parking_without_a_warning(tmp_path):
    """Version 3 is the query that fetched parking; version 4 only added
    distribution power lines. A version-3 file's lots are read, and nothing
    says they cannot be."""
    latitude, longitude = CENTRE
    lot = _site("parking-w1", "parking", _square_3857(longitude + 0.002, latitude, 60))
    payload = parking_candidates(_staged(tmp_path, [lot], query_version=3))

    assert [c["id"] for c in payload["candidates"]] == ["parking-w1"]
    assert not any("re-stage" in warning for warning in payload["warnings"]), \
        payload["warnings"]


def test_ground_with_no_query_version_is_called_version_1(tmp_path):
    """The first query wrote no version; every other tool calls such a file
    version 1, and so must this one."""
    directory = _staged(tmp_path, [])
    payload = json.loads((directory / "osm.json").read_text(encoding="utf-8"))
    del payload["query_version"]
    (directory / "osm.json").write_text(json.dumps(payload), encoding="utf-8")

    warnings = parking_candidates(directory)["warnings"]

    assert any("version 1" in w for w in warnings), warnings
    assert not any("version 0" in w for w in warnings), warnings
