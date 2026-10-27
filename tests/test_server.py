"""The MCP tool surface.

Exercised through fastmcp's in-memory client rather than by calling the
functions directly: the point of these tests is the ADAPTER - that each tool
is registered, that its schema is what a model will see, and that it returns
JSON-shaped data rather than Python objects a client cannot read. The
computation underneath has its own tests in the modules it lives in.

Every CalTopo call is stubbed. A test suite that reaches caltopo.com would
be slow, credential-dependent, and would fail on an aeroplane; and the thing
worth testing here is the adapter, not their uptime.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402

from sar_geo_mcp.config import CaltopoConfig  # noqa: E402
from sar_geo_mcp.server import build_server  # noqa: E402

# El Corte, the ground this project was built against.
NEAR = (37.4022, -122.3164)

SQUARE = {
    "type": "Polygon",
    "coordinates": [[
        [-122.32, 37.40], [-122.30, 37.40],
        [-122.30, 37.42], [-122.32, 37.42], [-122.32, 37.40],
    ]],
}

FEATURES = [
    {"id": "a1", "properties": {"class": "Assignment", "letter": "QA",
                                "number": "103", "title": "QA103"},
     "geometry": SQUARE},
    {"id": "m1", "properties": {"class": "Marker", "title": "CP",
                                "marker-symbol": "cp"},
     # CalTopo writes four ordinates - lon, lat, and two it fills with zero.
     "geometry": {"type": "Point", "coordinates": [-122.24695, 37.3618, 0, 0]}},
]


def stub_caltopo(monkeypatch, features=FEATURES):
    """Every public CalTopo read answers with `features`, never networked.

    The stub answers in CalTopo's full-reply shape - features plus a change
    token - because the state layer reads the token to decide whether a
    later refresh can be a cheap probe. Map state is read when a tool is
    called, not when the server is built, so a test may call this again to
    put a different map behind a server already built.
    """
    monkeypatch.setattr(
        "sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
        lambda map_id, host, since=0: {"result": {
            "state": {"features": features}, "timestamp": 1_787_989_336_147}},
    )


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A server whose CalTopo reads are stubbed, never networked."""
    stub_caltopo(monkeypatch)
    return build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))


@pytest.fixture
def uncached_server(monkeypatch):
    """The same server run with caching disabled (`cache_dir=None`)."""
    stub_caltopo(monkeypatch)
    return build_server(CaltopoConfig(cache_dir=None))


async def call(server, name, **arguments):
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
    return result.data


def server_reader(tmp_path):
    """A `MapReader` sharing the cache dir the `server` fixture built.

    `build_server` does not hand back the reader it constructs, so a test
    that needs to stage against the same cache the server reads from
    builds its own reader pointed at the same directory.
    """
    from sar_geo_mcp.caltopo.reader import MapReader

    return MapReader(CaltopoConfig(cache_dir=tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Registration.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_every_tool_is_registered_with_a_description(server):
    """A tool a model cannot understand is a tool it will misuse. fastmcp
    takes the description straight from the docstring, so an undocumented
    tool is a real defect, not a style issue."""
    async with Client(server) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert names == {
        "list_assignments", "get_assignment", "list_map_features",
        "parse_coordinate", "convert_coordinate", "distance_bearing",
        "point_in_geometry", "geometry_bounds", "buffer_geometry",
        "prefetch_map", "cache_status", "evict_cache",
        "prepare_terrain", "terrain_metrics", "coverage_for",
        "slope_stats", "elevation_profile",
        "identify_features", "list_feature_kinds",
        "prepare_area", "assignment_norms", "describe_boundary",
        "hasty_candidates", "get_candidate", "parking_candidates",
        "lz_candidates", "egress_routes", "travel_time_rings",
        "nearest_medical_facilities", "populated_areas",
        "urban_assignments", "caltopo_export",
        "site_facts", "boundary_clearance",
        "shaded_relief", "slope_shading", "unmarked_paths",
        "plan_medevac", "plan_initial_response", "plan_urban_search",
        "debrief_assignment",
        "key_status", "check_for_update", "save_export",
    }
    for tool in tools:
        assert tool.description and len(tool.description) > 40, (
            f"{tool.name} has no usable description"
        )


@pytest.mark.anyio
async def test_no_write_tool_is_exposed(server):
    """v1 is read-only by design: the write tier needs its governance kit
    (staging folder, ownership ledger, dry-run, map allowlist, a separate
    write credential) before an agent's mistake can reach a map that live
    teams navigate by. This fails loudly if a write tool appears before
    that machinery does."""
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}

    forbidden = {"add", "create", "edit", "delete", "move", "update", "write"}
    offenders = [n for n in names if any(n.startswith(verb) for verb in forbidden)]
    assert not offenders, f"write-shaped tools exposed without governance: {offenders}"


# ---------------------------------------------------------------------------
# CalTopo reads.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_assignments_returns_labels_kinds_and_centres(server):
    result = await call(server, "list_assignments", map_id="M1")

    assert len(result["assignments"]) == 1, "only Assignments belong in this list"
    entry = result["assignments"][0]
    assert entry["label"] == "QA103"
    assert entry["geometry_kind"] == "Polygon"
    assert entry["center"]["latitude"] == pytest.approx(37.41, abs=0.01)
    assert entry["center"]["longitude"] == pytest.approx(-122.31, abs=0.01)


@pytest.mark.anyio
async def test_get_assignment_carries_the_geometry_and_a_centre(server):
    result = await call(server, "get_assignment", map_id="M1", label="qa103")
    assert result["geometry"]["type"] == "Polygon", "matched case-insensitively"
    assert result["center"]["latitude"] == pytest.approx(37.41, abs=0.01)


@pytest.mark.anyio
async def test_list_map_features_summarises_geometry_rather_than_returning_it(server):
    """One GPS track can carry thousands of coordinates. Returning them
    would swamp a model's context to no purpose - `get_assignment` is where
    real geometry comes from."""
    result = await call(server, "list_map_features", map_id="M1")

    assert len(result["features"]) == 2
    for feature in result["features"]:
        assert "coordinates" not in json.dumps(feature)
        assert feature["geometry_kind"] in ("Polygon", "Point")


@pytest.mark.anyio
async def test_list_map_features_filters_by_class(server):
    result = await call(server, "list_map_features", map_id="M1",
                        feature_class="Marker")
    assert [f["title"] for f in result["features"]] == ["CP"]


@pytest.mark.anyio
async def test_list_map_features_carries_a_markers_symbol_and_point(server):
    payload = await call(server, "list_map_features", map_id="DEMOMAP",
                         feature_class="Marker")

    marker = payload["features"][0]
    assert marker["marker_symbol"] == "cp"
    # Two numbers, not the thousands the summarising exists to avoid.
    assert marker["point"] == {"latitude": 37.3618, "longitude": -122.24695}


@pytest.mark.anyio
async def test_list_map_features_point_is_none_for_a_non_point_marker_symbol_is_none_without_one(
        server):
    """A Marker whose geometry is not a Point (drawn, or malformed) must not
    raise; nor must an Assignment, whose class carries no `marker-symbol`
    at all."""
    result = await call(server, "list_map_features", map_id="M1",
                        feature_class="Assignment")
    assignment = result["features"][0]
    assert assignment["marker_symbol"] is None
    assert assignment["point"] is None


#: Assignments whose geometry no centre can be computed from: one drawn with
#: no coordinates, one with no geometry at all.
NO_COORDINATES = {"id": "b1", "properties": {"class": "Assignment", "letter": "ZZ",
                                             "number": "9", "title": "broken"},
                  "geometry": {"type": "Polygon", "coordinates": []}}
NO_GEOMETRY = {"id": "b2", "properties": {"class": "Assignment", "letter": "ZZ",
                                          "number": "8", "title": "empty"},
               "geometry": {}}


@pytest.mark.anyio
async def test_an_assignment_with_unusable_geometry_is_still_listed_without_a_centre(
        server, monkeypatch):
    """Its label is what a planner looks it up by, so a broken outline must
    not drop it from the list - nor take the good assignments down with it.
    The centre is null, never a made-up point."""
    stub_caltopo(monkeypatch, [*FEATURES, NO_COORDINATES, NO_GEOMETRY])

    result = await call(server, "list_assignments", map_id="M1")

    by_label = {entry["label"]: entry for entry in result["assignments"]}
    assert list(by_label) == ["QA103", "ZZ9", "ZZ8"]
    assert by_label["QA103"]["center"]["latitude"] == pytest.approx(37.41, abs=0.01)
    assert by_label["ZZ9"] == {"label": "ZZ9", "geometry_kind": "Polygon",
                               "title": "broken", "center": None}
    assert by_label["ZZ8"] == {"label": "ZZ8", "geometry_kind": "unknown",
                               "title": "empty", "center": None}


@pytest.mark.anyio
async def test_get_assignment_with_unusable_geometry_answers_with_a_null_centre(
        server, monkeypatch):
    """The geometry as CalTopo sent it, a null centre, and the provenance -
    not a crash in the centre computation."""
    stub_caltopo(monkeypatch, [*FEATURES, NO_COORDINATES])

    result = await call(server, "get_assignment", map_id="M1", label="zz9")

    assert result["id"] == "b1"
    assert result["geometry"] == {"type": "Polygon", "coordinates": []}
    assert result["center"] is None
    assert result["provenance"]["source"] in ("network", "cache", "cache-revalidated")


@pytest.mark.anyio
async def test_get_assignment_with_no_geometry_at_all_is_refused_naming_it(
        server, monkeypatch):
    """No geometry at all is refused by the Assignment model before any
    centre is looked for - by the feature's id, so the planner can find it
    on the map. (Listing it still works; see above.)"""
    from fastmcp.exceptions import ToolError

    stub_caltopo(monkeypatch, [*FEATURES, NO_GEOMETRY])

    with pytest.raises(ToolError, match="'b2' has no geometry"):
        await call(server, "get_assignment", map_id="M1", label="ZZ8")


# ---------------------------------------------------------------------------
# Geo accessories.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("text", [
    "37.4159, -122.3080",
    "10S 561234 4141234",
    "10S EG 61234 41234",
    "37 24.95 N, 122 18.48 W",
])
async def test_parse_coordinate_reads_every_supported_format(server, text):
    result = await call(server, "parse_coordinate", text=text,
                        near_latitude=NEAR[0], near_longitude=NEAR[1])
    assert result["parsed"] is True
    assert result["latitude"] == pytest.approx(37.4159, abs=2e-3)
    assert result["longitude"] == pytest.approx(-122.3080, abs=2e-3)


@pytest.mark.anyio
async def test_parse_coordinate_reports_a_description_as_unparsed_not_an_error(server):
    """"Not a coordinate" is a complete answer. The tool must say so in
    data, so a model records the words rather than retrying or inventing a
    position."""
    result = await call(server, "parse_coordinate",
                        text="the second creek crossing above the bridge")
    assert result["parsed"] is False
    assert result["stated"] == "the second creek crossing above the bridge"
    assert "reason" in result


@pytest.mark.anyio
async def test_parse_coordinate_keeps_the_searchers_words_either_way(server):
    """`stated` is what makes an unparsed answer still useful downstream."""
    for text in ("37.4159, -122.3080", "somewhere near the saddle"):
        result = await call(server, "parse_coordinate", text=text)
        assert result["stated"] == text


@pytest.mark.anyio
async def test_convert_coordinate_answers_in_every_format_a_team_uses(server):
    result = await call(server, "convert_coordinate",
                        latitude=37.4159, longitude=-122.3080)

    assert result["utm"].startswith("10S ")
    assert result["mgrs"].startswith("10S EG ")
    assert "N," in result["degrees_decimal_minutes"]
    assert result["degrees_minutes_seconds"] == "37°24'57.2\"N 122°18'28.8\"W"
    assert result["decimal_degrees"].startswith("37.41")


@pytest.mark.anyio
async def test_a_converted_coordinate_parses_back_to_itself(server):
    """The round trip is the real assertion: formatting and parsing share
    the same grid tables, so agreement means both are consistent."""
    converted = await call(server, "convert_coordinate",
                           latitude=37.4159, longitude=-122.3080)
    for form in ("utm", "mgrs", "degrees_decimal_minutes", "degrees_minutes_seconds"):
        back = await call(server, "parse_coordinate", text=converted[form],
                          near_latitude=NEAR[0], near_longitude=NEAR[1])
        assert back["parsed"] is True, f"{form} did not survive the round trip"
        assert back["latitude"] == pytest.approx(37.4159, abs=1e-3)
        assert back["longitude"] == pytest.approx(-122.3080, abs=1e-3)


@pytest.mark.anyio
async def test_distance_bearing_is_true_ground_distance(server):
    """Not a planar measurement on a projection: Web Mercator over-reads by
    1/cos(latitude), which at this latitude is about 26% - an error this
    project has had to fix more than once."""
    result = await call(server, "distance_bearing",
                        from_latitude=37.40, from_longitude=-122.31,
                        to_latitude=37.41, to_longitude=-122.31)
    # One hundredth of a degree of latitude is ~1.11 km anywhere on Earth.
    assert result["metres"] == pytest.approx(1111, abs=5)
    assert result["bearing_degrees"] == pytest.approx(0.0, abs=0.5)


@pytest.mark.anyio
async def test_point_in_geometry_answers_inside_and_how_far_from_the_edge(server):
    inside = await call(server, "point_in_geometry",
                        latitude=37.41, longitude=-122.31, geometry=SQUARE)
    assert inside["inside"] is True

    outside = await call(server, "point_in_geometry",
                         latitude=37.45, longitude=-122.31, geometry=SQUARE)
    assert outside["inside"] is False
    # "Just outside" and "far outside" are different operational facts, so
    # the distance has to be in metres a person can act on.
    assert outside["distance_to_boundary_m"] > 1000


def ground_m(from_latitude, from_longitude, to_latitude, to_longitude):
    """The WGS84 geodesic distance in metres: the truth `point_in_geometry`'s
    distance is held to. A point due south of an east-west edge (or due
    west of a north-south one) is nearest the edge straight across, so this
    is its distance to that edge."""
    from pyproj import Geod

    return Geod(ellps="WGS84").inv(from_longitude, from_latitude,
                                   to_longitude, to_latitude)[2]


def square(west, south, east, north):
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


#: SQUARE's eastern neighbour, 0.02 degrees of longitude (~1.8 km) away.
EAST_PART = square(-122.28, 37.40, -122.26, 37.42)
TWO_PARTS = {"type": "MultiPolygon",
             "coordinates": [SQUARE["coordinates"], [EAST_PART]]}
#: SQUARE with a hole 0.004 degrees on a side (~350 x 445 m) at its centre.
HOLED = {"type": "Polygon", "coordinates": [
    SQUARE["coordinates"][0], square(-122.312, 37.408, -122.308, 37.412)]}


@pytest.mark.anyio
@pytest.mark.parametrize("latitude,longitude,edge", [
    # 0.002 degrees of longitude inside SQUARE's east edge, then outside it.
    (37.41, -122.302, (37.41, -122.30)),
    (37.41, -122.298, (37.41, -122.30)),
])
async def test_point_in_geometry_east_west_distance_is_ground_metres(
        server, latitude, longitude, edge):
    """0.002 degrees of longitude is ~177 m here. The east-west direction is
    the one the cos(latitude) scale gets right; a fix to the north-south
    one (B4, below) must keep it right."""
    result = await call(server, "point_in_geometry", latitude=latitude,
                        longitude=longitude, geometry=SQUARE)
    assert result["distance_to_boundary_m"] == pytest.approx(
        ground_m(latitude, longitude, *edge), rel=0.01)


@pytest.mark.anyio
async def test_point_in_geometry_reads_a_multipolygon_part_by_part(server):
    """A multi-part assignment: a point in the second part is inside it, and
    a point in the gap between the parts is outside, its distance the
    ground metres to the nearer part."""
    in_second = await call(server, "point_in_geometry", latitude=37.41,
                           longitude=-122.27, geometry=TWO_PARTS)
    assert in_second["inside"] is True

    between = await call(server, "point_in_geometry", latitude=37.41,
                         longitude=-122.29, geometry=TWO_PARTS)
    assert between["inside"] is False
    # 0.01 degrees of longitude to either part: ~885 m.
    assert between["distance_to_boundary_m"] == pytest.approx(
        ground_m(37.41, -122.29, 37.41, -122.30), rel=0.01)


@pytest.mark.anyio
async def test_point_in_geometry_a_point_in_a_hole_is_not_inside(server):
    """A preserve's inholding is not preserve ground."""
    in_hole = await call(server, "point_in_geometry", latitude=37.41,
                         longitude=-122.31, geometry=HOLED)
    assert in_hole["inside"] is False

    in_ring = await call(server, "point_in_geometry", latitude=37.41,
                         longitude=-122.315, geometry=HOLED)
    assert in_ring["inside"] is True


#: A square at 60N, where cos(latitude) is one half.
NORTHERN = {"type": "Polygon", "coordinates": [square(10.0, 59.99, 10.02, 60.01)]}


@pytest.mark.anyio
@pytest.mark.parametrize("geometry,latitude,longitude,edge", [
    # 0.002 degrees of latitude inside SQUARE's north edge: ~222 m, read 177 m.
    (SQUARE, 37.418, -122.31, (37.42, -122.31)),
    # The existing "outside" case: ~3,330 m, read 2,651 m.
    (SQUARE, 37.45, -122.31, (37.42, -122.31)),
    # 0.002 degrees inside the north edge at 60N: ~223 m, read 111 m.
    (NORTHERN, 60.008, 10.01, (60.01, 10.01)),
], ids=["inside-north-edge", "outside-north", "60N"])
async def test_point_in_geometry_north_south_distance_is_ground_metres(
        server, geometry, latitude, longitude, edge):
    result = await call(server, "point_in_geometry", latitude=latitude,
                        longitude=longitude, geometry=geometry)
    assert result["distance_to_boundary_m"] == pytest.approx(
        ground_m(latitude, longitude, *edge), rel=0.02)


@pytest.mark.anyio
async def test_point_in_geometry_inside_a_multipolygon_reads_its_edge_distance(server):
    """The centre of SQUARE is ~885 m from its nearest edge whether SQUARE
    comes as a Polygon or as one part of a MultiPolygon; 0.0 would put a
    clue deep inside the assignment on its boundary line."""
    truth = ground_m(37.41, -122.31, 37.41, -122.30)
    for geometry in (SQUARE, TWO_PARTS,
                     {"type": "MultiPolygon", "coordinates": [SQUARE["coordinates"]]}):
        result = await call(server, "point_in_geometry", latitude=37.41,
                            longitude=-122.31, geometry=geometry)
        assert result["inside"] is True
        assert result["distance_to_boundary_m"] == pytest.approx(truth, rel=0.01), (
            geometry["type"])


@pytest.mark.anyio
async def test_point_in_geometry_measures_to_the_edge_of_a_hole(server):
    """At the hole's centre the nearest boundary is the hole's own edge,
    0.002 degrees of longitude (~177 m) away - not the outer ring ~885 m off."""
    result = await call(server, "point_in_geometry", latitude=37.41,
                        longitude=-122.31, geometry=HOLED)
    assert result["inside"] is False
    assert result["distance_to_boundary_m"] == pytest.approx(
        ground_m(37.41, -122.31, 37.41, -122.308), rel=0.02)


@pytest.mark.anyio
async def test_point_in_geometry_measures_a_line_of_one_place_to_that_place(server):
    """A line whose positions are all one place is valid GeoJSON - a track
    part of a single fix - and is measured to as that place, alone or as the
    nearer part of a MultiLineString, rather than failing in GEOS ("point
    array must contain 0 or >1 elements") with nothing the caller passed."""
    truth = ground_m(37.41, -122.31, 37.41, -122.30)
    one_place = [[-122.30, 37.41], [-122.30, 37.41]]
    for geometry in ({"type": "LineString", "coordinates": one_place},
                     {"type": "MultiLineString", "coordinates": [
                         [[-122.28, 37.40], [-122.28, 37.42]], one_place]}):
        result = await call(server, "point_in_geometry", latitude=37.41,
                            longitude=-122.31, geometry=geometry)
        assert result["inside"] is False
        assert result["distance_to_boundary_m"] == pytest.approx(truth, rel=0.01), (
            geometry["type"])


@pytest.mark.parametrize("latitude,longitude,asks", [
    # Swapped, (37.41, -122.31) is on the globe: worth asking.
    (-122.31, 37.41, True),
    # Swapped, -122.31 is no latitude either: asking would send the caller
    # to a pair refused with the same question.
    (95.0, -122.31, False),
])
def test_require_point_asks_about_a_swap_only_when_swapping_would_help(
        latitude, longitude, asks):
    from sar_geo_mcp.geo import require_point

    with pytest.raises(ValueError, match="latitude") as refused:
        require_point(latitude, longitude)
    assert ("passed as the latitude" in str(refused.value)) is asks


@pytest.mark.anyio
@pytest.mark.parametrize("latitude,longitude,named", [
    (95.0, -122.31, "latitude"),
    (-95.0, -122.31, "latitude"),
    (37.41, -200.0, "longitude"),
])
async def test_point_in_geometry_refuses_an_out_of_range_coordinate(
        server, latitude, longitude, named):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match=named):
        await call(server, "point_in_geometry", latitude=latitude,
                   longitude=longitude, geometry=SQUARE)


def test_point_in_geometry_refuses_a_non_finite_latitude(tmp_path):
    """Over MCP the client turns NaN into null and validation refuses it; a
    scenario calling the plain function has no such guard."""
    from sar_geo_mcp.components.geometry import point_in_geometry
    from sar_geo_mcp.context import ServerContext

    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))
    with pytest.raises(ValueError, match="latitude"):
        point_in_geometry(ctx, latitude=float("nan"), longitude=-122.31, geometry=SQUARE)


@pytest.mark.anyio
async def test_geometry_bounds_returns_a_box_and_its_centre(server):
    result = await call(server, "geometry_bounds", geometry=SQUARE)
    assert result["west"] == pytest.approx(-122.32)
    assert result["north"] == pytest.approx(37.42)
    assert result["center"]["latitude"] == pytest.approx(37.41, abs=0.01)


@pytest.mark.anyio
async def test_buffer_geometry_answers_with_the_ribbon_and_its_width(server):
    line = {"type": "LineString", "coordinates": [[-122.32, 37.30], [-122.32, 37.32]]}
    result = await call(server, "buffer_geometry", geometry=line, distance_m=15)
    assert result["geometry"]["type"] == "Polygon"
    assert result["width_m"] == 30.0
    assert result["area_ha"] == pytest.approx(6.75, abs=0.1)


@pytest.mark.anyio
async def test_buffer_geometry_refuses_a_point_with_the_reason(server):
    with pytest.raises(Exception, match="Point"):
        await call(server, "buffer_geometry",
                   geometry={"type": "Point", "coordinates": [-122.32, 37.30]}, distance_m=15)


@pytest.mark.anyio
@pytest.mark.parametrize("half", [{"access_latitude": 37.4105},
                                  {"access_longitude": -122.345}])
async def test_boundary_clearance_refuses_half_an_access_point(server, half):
    """A lone latitude dropped in silence would leave out the approach - the
    private-land check the tool exists for - with nothing to say so."""
    from fastmcp.exceptions import ToolError
    from test_clearance import BOUNDARY, CENTRAL

    with pytest.raises(ToolError,
                       match="needs both access_latitude and access_longitude"):
        await call(server, "boundary_clearance", geometry=CENTRAL, boundary=BOUNDARY,
                   **half)


@pytest.mark.anyio
async def test_boundary_clearance_measures_the_approach_from_the_access_point(server):
    """Both halves given: the approach block, measured from (latitude,
    longitude) in that order - the lot west of the preserve is outside it,
    and the straight line in is only partly on preserve ground."""
    from pydantic_core import to_jsonable_python
    from test_clearance import BOUNDARY, CENTRAL

    from sar_geo_mcp.geo import boundary_clearance

    result = await call(server, "boundary_clearance", geometry=CENTRAL, boundary=BOUNDARY,
                        access_latitude=37.4105, access_longitude=-122.345)

    approach = result["approach"]
    assert approach["access_inside_boundary"] is False
    assert 0.0 < approach["fraction_inside"] < 1.0
    assert approach == to_jsonable_python(
        boundary_clearance(CENTRAL, BOUNDARY, access=(37.4105, -122.345))["approach"])


# ---------------------------------------------------------------------------
# Bad input is refused by name, before any work.
# ---------------------------------------------------------------------------

async def answer(server, name, **arguments):
    """`(is_error, text)` of a call, an error included rather than raised."""
    async with Client(server) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
    return result.is_error, result.content[0].text


@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review B47: Component tools pass raw Python error text and NaN through "
    "to MCP clients on bad geometry or coordinates"))
@pytest.mark.anyio
@pytest.mark.parametrize("name,arguments,names", [
    # Accepted today, answering NaN metres and bearing - invalid JSON.
    ("distance_bearing", {"from_latitude": 95, "from_longitude": -122.31,
                          "to_latitude": 37.41, "to_longitude": -122.31}, r"latitude.*90"),
    ("distance_bearing", {"from_latitude": "NaN", "from_longitude": -122.31,
                          "to_latitude": 37.41, "to_longitude": -122.31}, r"latitude"),
    # Accepted today as "95 0.000 N", with the null UTM/MGRS the description
    # says means "above 84N".
    ("convert_coordinate", {"latitude": 95, "longitude": -122.3}, r"latitude.*90"),
    ("convert_coordinate", {"latitude": "inf", "longitude": -122.3}, r"latitude"),
    ("convert_coordinate", {"latitude": "NaN", "longitude": -122.3}, r"latitude"),
    # "'NoneType' object has no attribute 'lower'" today.
    ("point_in_geometry", {"latitude": 37.41, "longitude": -122.31, "geometry": {}},
     r'GeoJSON|"type"|\btype\b'),
    ("boundary_clearance", {"geometry": {}, "boundary": SQUARE}, r'GeoJSON|"type"|\btype\b'),
    # "maximum recursion depth exceeded" today.
    ("geometry_bounds", {"geometry": {"type": "Polygon", "coordinates": "abc"}},
     r"coordinates"),
    # "cannot convert float NaN to integer" / "buffer distance must be finite" /
    # "Polygon geometry has no coordinates" today.
    ("prepare_area", {"center_latitude": 37.41, "center_longitude": -122.31,
                      "radius_m": 500, "resolution_m": "NaN"}, r"resolution_m"),
    ("prepare_area", {"center_latitude": "NaN", "center_longitude": -122.31,
                      "radius_m": 500}, r"center_latitude"),
    ("prepare_area", {"center_latitude": 95, "center_longitude": -122.31,
                      "radius_m": 500}, r"center_latitude.*90"),
    # "cannot convert float NaN to integer" today; a negative ceiling is
    # accepted in silence.
    ("evict_cache", {"ceiling_mb": "NaN"}, r"ceiling_mb"),
    ("evict_cache", {"ceiling_mb": -5}, r"ceiling_mb"),
], ids=["bearing-lat-95", "bearing-lat-nan", "convert-lat-95", "convert-lat-inf",
        "convert-lat-nan", "point-empty-geometry", "clearance-empty-geometry",
        "bounds-string-coordinates", "area-resolution-nan", "area-lat-nan", "area-lat-95",
        "evict-ceiling-nan", "evict-ceiling-negative"])
async def test_a_bad_number_or_geometry_is_refused_naming_the_argument(
        server, monkeypatch, name, arguments, names):
    """A model reads the error to correct its call: "cannot convert float NaN
    to integer" names nothing it passed, and a NaN answer is not JSON at all.
    Each is refused, naming the argument and what it must be."""
    def refuse(*args, **kwargs):
        raise AssertionError("refused input must never reach staging")

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", refuse)

    is_error, text = await answer(server, name, **arguments)

    assert is_error, f"accepted: {text}"
    assert re.search(names, text), text
    for leak in ("NoneType", "recursion", "cannot convert float", "must never reach"):
        assert leak not in text, text


# ---------------------------------------------------------------------------
# stdout is the protocol channel.
# ---------------------------------------------------------------------------

def test_logging_never_writes_to_the_protocol_channel():
    """Under stdio, stdout IS the JSON-RPC stream, and a single log line
    written there corrupts it.

    Importing caltopo_python installs a root StreamHandler pointing at
    stdout, so simply reading a CalTopo map would inject text into the
    protocol. Found by launching the server against a real stdio client and
    watching every call report "Failed to parse JSONRPC message from
    server" - no in-process test could have seen it, because in-process
    tests never touch the transport.
    """
    import logging
    import sys

    import caltopo_python  # noqa: F401  - the import IS the trigger

    from sar_geo_mcp.server import redirect_logging_to_stderr

    handler = logging.StreamHandler(sys.stdout)
    logging.root.addHandler(handler)
    try:
        moved = redirect_logging_to_stderr()
        assert moved, "nothing was moved - the guard is not finding handlers"

        offenders = [
            h for logger in [logging.root, *logging.Logger.manager.loggerDict.values()]
            for h in getattr(logger, "handlers", [])
            if getattr(h, "stream", None) is sys.stdout
        ]
        assert not offenders, f"still writing to the protocol channel: {offenders}"
    finally:
        logging.root.removeHandler(handler)


# ---------------------------------------------------------------------------
# Freshness (F2). Map state is the one thing here that changes during an
# operation, so every read has to say how old it is.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("tool,arguments", [
    ("list_assignments", {}),
    ("get_assignment", {"label": "QA103"}),
    ("list_map_features", {}),
])
async def test_every_caltopo_read_reports_how_old_its_data_is(server, tool, arguments):
    """A planner reasoning off forty-minute-old boundaries has to be able to
    see that they are forty minutes old. Same principle as recording a
    searcher's claim beside the computed value: say what you are handing
    over."""
    result = await call(server, tool, map_id="M1", **arguments)

    provenance = result["provenance"]
    assert provenance["age_seconds"] < 5
    assert provenance["source"] in ("network", "cache", "cache-revalidated")
    assert "membership_age_seconds" in provenance


@pytest.mark.anyio
async def test_a_second_read_inside_the_window_does_not_refetch(server, monkeypatch):
    """The saving this tier exists for: browsing a map - list, get, list
    features - was three full downloads of the same 2.75 MB payload."""
    await call(server, "list_assignments", map_id="M1")

    def refuse(map_id, host, since=0):
        raise AssertionError("second read should have been served from cache")

    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.fetch_public_map", refuse)
    result = await call(server, "list_map_features", map_id="M1")
    assert result["provenance"]["source"] == "cache"


@pytest.mark.anyio
async def test_max_age_zero_forces_a_freshness_check(server, monkeypatch):
    """What a computation will use: a listing may be slightly stale provided
    it says so, but a computed result may not be."""
    await call(server, "list_assignments", map_id="M1")

    probes = []

    def probe(map_id, host, since=0):
        probes.append(since)
        return {"result": {"state": {"features": []}, "timestamp": 2}}

    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.fetch_public_map", probe)
    await call(server, "list_assignments", map_id="M1", max_age_seconds=0)

    assert probes == [1_787_989_336_147], "must probe from the known token"


@pytest.mark.anyio
async def test_an_offline_read_is_served_but_marked_stale(server, monkeypatch):
    """A command post often has no connectivity. Refusing to answer is worse
    than answering with an age attached."""
    await call(server, "list_assignments", map_id="M1")

    def offline(map_id, host, since=0):
        raise OSError("network is unreachable")

    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.fetch_public_map", offline)
    result = await call(server, "list_assignments", map_id="M1", max_age_seconds=0)

    assert len(result["assignments"]) == 1, "the cached answer still gets served"
    assert result["provenance"]["stale"] is True
    assert "OSError" in result["provenance"]["reason"]


# ---------------------------------------------------------------------------
# Prefetch (F5).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_prefetch_defaults_to_an_estimate_that_downloads_nothing(server, tmp_path):
    """The first call must be free, or nobody will look at the estimate
    before committing a tethered phone to a few hundred megabytes."""
    result = await call(server, "prefetch_map", map_id="M1")

    assert result["executed"] is False
    assert result["tiles_missing"] > 0
    assert result["estimated_mb"] > 0


@pytest.mark.anyio
async def test_prefetch_says_so_when_panels_need_a_token_it_does_not_have(server):
    """Elevation needs no credentials. A caller without a WMS token should
    get what it can, and be told what it is missing and why."""
    result = await call(server, "prefetch_map", map_id="M1")

    assert result["panels_included"] is False
    assert "token" in result["note"]


@pytest.mark.anyio
async def test_cache_status_reports_what_is_held(server):
    result = await call(server, "cache_status")
    assert "by_tier" in result
    assert "mb" in result


@pytest.fixture
def recorded_prefetch(monkeypatch):
    """`run_prefetch` replaced by a recorder, so `execute=True` downloads
    nothing: what the tool asked it for is in the list it returns."""
    calls = []

    def record(map_id, features, cache_dir, budget_bytes, panel_sets=None,
               base_url=None, **kwargs):
        calls.append({"map_id": map_id, "features": len(features), "cache_dir": cache_dir,
                      "budget_bytes": budget_bytes, "panel_sets": panel_sets,
                      "base_url": base_url})
        return {"tiles_fetched": 3, "mb_written": 1.5, "budget_reached": False}

    monkeypatch.setattr("sar_geo_mcp.components.cache.run_prefetch", record)
    return calls


@pytest.mark.anyio
@pytest.mark.parametrize("budget_mb,budget_bytes", [
    (250, 250_000_000), (0.5, 500_000), (None, 500_000_000)])
async def test_prefetch_execute_hands_the_budget_down_in_bytes(
        server, tmp_path, recorded_prefetch, budget_mb, budget_bytes):
    """The operator approves megabytes, the fetch stops on bytes written: a
    unit slip here pulls far more down a tethered phone than was approved.
    Unset, the budget is the tool's 500 MB default."""
    arguments = {"map_id": "M1", "execute": True}
    if budget_mb is not None:
        arguments["budget_mb"] = budget_mb

    result = await call(server, "prefetch_map", **arguments)

    assert recorded_prefetch == [{
        "map_id": "M1", "features": len(FEATURES), "cache_dir": tmp_path / "cache",
        "budget_bytes": budget_bytes,
        # No WMS token on this server: elevation only, no panels.
        "panel_sets": None, "base_url": None}]
    assert result == {"tiles_fetched": 3, "mb_written": 1.5, "budget_reached": False,
                      "executed": True}


@pytest.mark.anyio
async def test_prefetch_execute_with_a_token_fetches_the_panels_too(
        monkeypatch, tmp_path, recorded_prefetch):
    from sar_geo_mcp.components.cache import PANEL_SETS

    stub_caltopo(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token="tok"))

    result = await call(server, "prefetch_map", map_id="M1", execute=True)

    assert result["executed"] is True
    (asked,) = recorded_prefetch
    assert asked["panel_sets"] == PANEL_SETS
    assert asked["base_url"] == "https://caltopo.com/api/tok/wms"

    # And include_panels=False leaves them out even with the token.
    await call(server, "prefetch_map", map_id="M1", execute=True, include_panels=False)
    assert recorded_prefetch[1]["panel_sets"] is None
    assert recorded_prefetch[1]["base_url"] is None


@pytest.mark.parametrize("component,arguments", [
    ("cache_status", {}),
    # execute=True: the disabled answer even when asked to delete, and ahead
    # of `tiles_in_use(cache_dir)`, which cannot take a None.
    ("evict_cache", {"ceiling_mb": 0, "execute": True}),
])
def test_cache_components_say_caching_is_disabled_without_a_cache(component, arguments):
    from sar_geo_mcp.components import cache
    from sar_geo_mcp.context import ServerContext

    ctx = ServerContext.build(CaltopoConfig(cache_dir=None))
    assert getattr(cache, component)(ctx, **arguments) == {
        "cache_dir": None, "note": "caching is disabled"}


@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review B48: prefetch_map raises TypeError when caching is disabled; "
    "cache_status and evict_cache return 'caching is disabled'"))
@pytest.mark.anyio
@pytest.mark.parametrize("execute", [False, True])
async def test_prefetch_without_a_cache_says_caching_is_disabled(
        uncached_server, monkeypatch, execute):
    """Like `cache_status` and `evict_cache`: a clear answer, never a Path(None)
    TypeError - and never a download with nowhere to keep it."""
    def refuse(*args, **kwargs):
        raise AssertionError("nothing may be downloaded with no cache to keep it in")

    monkeypatch.setattr("sar_geo_mcp.components.cache.run_prefetch", refuse)

    result = await call(uncached_server, "prefetch_map", map_id="M1", execute=execute)

    assert result["cache_dir"] is None
    assert result["executed"] is False
    assert "disabled" in result["note"]
    assert await call(uncached_server, "cache_status") == {
        "cache_dir": None, "note": "caching is disabled"}


# ---------------------------------------------------------------------------
# Feature identification (registry).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_feature_kinds_is_generated_from_the_registry(server):
    result = await call(server, "list_feature_kinds")

    kinds = {entry["kind"] for entry in result["kinds"]}
    assert kinds == {"steep_band", "clearing", "drainage", "cliff_band",
                     "route", "ingress", "ridge", "cliff_pitch"}
    for entry in result["kinds"]:
        assert entry["description"]
        assert isinstance(entry["assumptions"], dict)


@pytest.mark.anyio
async def test_identify_features_rejects_an_unknown_kind_by_name(server):
    """Kind validation runs BEFORE handle resolution, so a model that
    hallucinated a kind is corrected even when its handle is garbage."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="steep_band"):
        await call(server, "identify_features",
                   handle="not-a-handle", kinds=["bogus"])


@pytest.mark.anyio
async def test_identify_features_rejects_an_empty_kinds_list(server):
    """Selection is the tool's purpose; requesting nothing is a caller
    error, answered with what could have been requested."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="steep_band"):
        await call(server, "identify_features",
                   handle="not-a-handle", kinds=[])


# ---------------------------------------------------------------------------
# coverage_for edge cases.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_coverage_for_an_area_without_a_map_names_the_requirement(server, monkeypatch, tmp_path):
    """Spec: coverage_for needs a map id on the handle; without one the error
    must name that, not the area's immutability."""
    from pathlib import Path

    from sar_geo_mcp.staging import write_manifest

    def fake_stage(subject, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                   access="public", subject_kind="assignment", **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "subject.json").write_text(subject.model_dump_json(), encoding="utf-8")
        write_manifest(out_dir, map_id=map_id, assignment_label=subject.label, bbox=(0, 0, 1, 1),
                       dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4, panels={},
                       warnings=[], assignment_fingerprint=fingerprint, access=access,
                       subject_kind=subject_kind)
        return {"warnings": [], "dem_resolution_m": 10.0}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", fake_stage)

    from sar_geo_mcp.planning.area import prepare_area as prepare_area_impl

    reader = server_reader(tmp_path)
    prepared = prepare_area_impl(reader, center_latitude=37.41, center_longitude=-122.31, radius_m=300)

    with pytest.raises(Exception, match="staged without a map_id"):
        await call(server, "coverage_for", handle=prepared["handle"], track_names=["1211"])


# ---------------------------------------------------------------------------
# One reader per server.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_coverage_for_revalidates_the_map_exactly_once(server, monkeypatch, tmp_path):
    """It used to read the map twice per call: once to check the boundary
    and once more for the tracks, which are in the very state the boundary
    was checked against."""
    from pathlib import Path

    from sar_geo_mcp.caltopo.reader import MapReader
    from sar_geo_mcp.handles import Resolved

    reads = []
    original = MapReader.read

    def counting(self, map_id, *, public, max_age_seconds=300.0):
        reads.append((map_id, public, max_age_seconds))
        return original(self, map_id, public=public, max_age_seconds=max_age_seconds)

    monkeypatch.setattr(MapReader, "read", counting)

    def fake_resolve(handle, reader, *, revalidate=True):
        state = reader.read("M1", public=True, max_age_seconds=0)
        return Resolved(Path(tmp_path), {"current": True, "checked": True}, state)

    monkeypatch.setattr("sar_geo_mcp.context.resolve", fake_resolve)

    seen = []

    def fake_coverage(directory, names, features):
        # The real return type. This stub used to return a plain dict,
        # which `dict(result)` in the tool accepted - so the test passed
        # while every real call crashed on the dataclass.
        from types import SimpleNamespace
        seen.append(len(features))
        metrics = SimpleNamespace(model_dump=lambda mode=None: {"tracks": []})
        return SimpleNamespace(metrics=metrics, track_geojson={})

    monkeypatch.setattr("sar_geo_mcp.coverage.metrics.compute_coverage", fake_coverage)

    result = await call(server, "coverage_for", handle="M1_QA103", track_names=["101"])

    assert reads == [("M1", True, 0)]
    assert seen == [2]
    assert result["staleness"]["current"] is True


# ---------------------------------------------------------------------------
# The tools that read terrain_metrics.json, end to end through the server.
# ---------------------------------------------------------------------------

def _staged_valley(monkeypatch, tmp_path, features=()):
    """Real staged ground (the valley DEM) behind a stubbed handle, with the
    map state `coverage_for` reads its tracks from."""
    from types import SimpleNamespace

    from test_boundary import staged, valley_dem

    from sar_geo_mcp.handles import Resolved

    (tmp_path / "ground").mkdir()
    data_dir = staged(tmp_path / "ground", valley_dem())
    state = SimpleNamespace(features=list(features))
    monkeypatch.setattr("sar_geo_mcp.context.resolve",
                        lambda handle, reader, *, revalidate=True:
                        Resolved(data_dir, {"current": True, "checked": True}, state))
    return data_dir


@pytest.mark.anyio
async def test_terrain_metrics_saves_the_file_the_shape_tools_read(server, monkeypatch, tmp_path):
    """Nothing in this package wrote terrain_metrics.json - only the debrief
    app did - so over MCP alone `slope_stats` and `elevation_profile` could
    never run. The tool now saves what it computed."""
    data_dir = _staged_valley(monkeypatch, tmp_path)

    with pytest.raises(Exception, match="(?i)call the `terrain_metrics` tool"):
        await call(server, "slope_stats", handle="H", shape_id="drainage_0")

    metrics = await call(server, "terrain_metrics", handle="H")
    assert (data_dir / "terrain_metrics.json").is_file()
    drainage = next(s["id"] for s in metrics["shapes"] if s["id"].startswith("drainage_"))

    stats = await call(server, "slope_stats", handle="H", shape_id=drainage)
    assert stats["pixels"] > 0
    profile = await call(server, "elevation_profile", handle="H", drainage_id=drainage)
    assert len(profile["elevations_m"]) == len(profile["distances_m"]) > 0


@pytest.mark.anyio
async def test_coverage_for_returns_the_digest_of_a_real_computation(server, monkeypatch, tmp_path):
    """Regression: the tool called `dict()` on `compute_coverage`'s dataclass
    and crashed with "'CoverageResult' object is not iterable" on every real
    call. It answers with the metrics digest, and never the dense track
    geometry that is kept for a renderer."""
    from test_boundary import m

    track = {"properties": {"title": "1.1 AA", "class": "Shape"},
             "geometry": {"type": "LineString",
                          "coordinates": [m(50, 200) + [100.0, 0], m(350, 200) + [100.0, 600_000]]}}
    _staged_valley(monkeypatch, tmp_path, features=[track])

    with pytest.raises(Exception, match="(?i)call the `terrain_metrics` tool"):
        await call(server, "coverage_for", handle="H", track_names=["1"])

    await call(server, "terrain_metrics", handle="H")
    result = await call(server, "coverage_for", handle="H", track_names=["1"])

    assert result["assignment_kind"] == "area"
    assert 0 < result["covered_fraction"] < 1
    assert len(result["tracks"]) == 1
    assert "track_geojson" not in result
    assert result["staleness"]["current"] is True


def test_a_host_override_keeps_the_cache_and_the_token(monkeypatch):
    """`--caltopo-host` rebuilt the config from three fields and silently
    dropped the other two: staging then refused for want of a cache
    directory, and panels vanished."""
    import argparse

    from sar_geo_mcp.server import config_from_args

    monkeypatch.setenv("ZA_CACHE_DIR", "somewhere")
    monkeypatch.setenv("ZA_WMS_TOKEN", "tok")
    config = config_from_args(argparse.Namespace(caltopo_host="team.caltopo.com"))

    assert config.caltopo_host == "team.caltopo.com"
    assert config.cache_dir == Path("somewhere").absolute()
    assert config.wms_token == "tok"
