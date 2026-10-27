"""The anatomy of a boundary, measured the way the report measured 4,528
real ones: walk the edge every 10 m and ask which mapped feature is nearest
within tolerance.

Built on a 40x40 grid of 10 m pixels. Coordinates below are Web Mercator;
the helper turns them into the WGS84 GeoJSON the API takes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from test_grid import write_fetch_dir

from sar_geo_mcp.planning import boundary as boundary_module
from sar_geo_mcp.planning.boundary import (
    BoundaryAnatomy,
    MIN_RUN_M,
    MISSING_OSM_WARNING,
    SAMPLE_SPACING_M,
    describe_boundary,
    fold_ring_seam,
    ground_for,
    load_ground,
    merge_short_runs,
    ring_run_order,
    runs_from_samples,
    sample_boundary,
    straight_segments,
)
from sar_geo_mcp.planning.norms import load_norms
from sar_geo_mcp.terrain.features.base import DEM_VOID_WARNING
from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

ORIGIN_X, ORIGIN_Y = -13618826.5, 4486605.0
TRANSFORM = from_origin(ORIGIN_X, ORIGIN_Y, 10.0, 10.0)
_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
LATITUDE = 37.3392
GROUND_PER_UNIT = 0.7952  # ground metres per Mercator unit at this latitude


def m(x_units, y_units):
    """Mercator offsets from the raster origin -> WGS84 [lon, lat]."""
    lon, lat = _TO_WGS84.transform(ORIGIN_X + x_units, ORIGIN_Y - y_units)
    return [lon, lat]


def polygon(*offsets):
    ring = [m(*o) for o in offsets] + [m(*offsets[0])]
    return {"type": "Polygon", "coordinates": [ring]}


def route(rid, kind, coords_units, name=None):
    return {"id": rid, "kind": "route",
            "geometry": {"type": "LineString",
                         "coordinates": [[ORIGIN_X + x, ORIGIN_Y - y] for x, y in coords_units]},
            "properties": {"route_type": kind, "name": name, "osm_id": 1, "surface": None}}


def valley_dem(rows=40, cols=40):
    """A V-shaped valley with its floor along column 20, sloping north, so a
    drainage runs north-south through the middle of the raster."""
    col_distance = np.abs(np.arange(cols) - 20).astype("float64")
    across = col_distance[None, :].repeat(rows, axis=0) * 3.0
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return across + along


def staged(tmp_path, dem, routes=(), query_version=OSM_QUERY_VERSION):
    """A staged directory whose subject is the WHOLE raster; targets are
    supplied per call. `query_version` is what `stage_terrain` records, so
    the fixture is a CURRENT staging unless a test asks for an older one."""
    whole = polygon((0, 0), (400, 0), (400, 400), (0, 400))
    data_dir = write_fetch_dir(tmp_path, dem, TRANSFORM, whole, latitude=LATITUDE)
    payload = {"routes": list(routes)}
    if query_version is not None:
        payload["query_version"] = query_version
    (data_dir / "osm.json").write_text(json.dumps(payload), encoding="utf-8")
    return data_dir


def assignment(label, geometry):
    return {"id": label.lower(),
            "properties": {"class": "Assignment", "letter": label, "number": ""},
            "geometry": geometry}


TRAIL_SQUARE = ((100, 100), (300, 100), (300, 300), (100, 300))


@pytest.fixture(autouse=True)
def _fresh_ground_cache():
    """The Ground cache is keyed by directory, and every test stages its own
    under a fresh tmp_path; clearing keeps a test from holding a grid for a
    directory pytest has already removed."""
    ground_for.cache_clear()
    yield
    ground_for.cache_clear()


# --- the run merger, alone ---------------------------------------------------------

def test_short_runs_are_absorbed_into_their_longer_neighbour():
    runs = [{"follows": "trail", "start_m": 0, "end_m": 100},
            {"follows": "none", "start_m": 100, "end_m": 120},
            {"follows": "trail", "start_m": 120, "end_m": 300}]
    merged = merge_short_runs(runs, MIN_RUN_M)
    assert merged == [{"follows": "trail", "start_m": 0, "end_m": 300}]


def test_a_single_short_run_survives_when_it_is_all_there_is():
    runs = [{"follows": "none", "start_m": 0, "end_m": 12}]
    assert merge_short_runs(runs, MIN_RUN_M) == runs


def ring_composition(classes, spacing_m=10.0, total_m=200.0):
    """The class metres of a closed walk sampled at `spacing_m`, through the
    same pipeline `describe_boundary` uses for a ring."""
    samples = [(0.0, 0.0, i * spacing_m) for i in range(len(classes))]
    runs = runs_from_samples(samples, classes, spacing_m, total_m=total_m)
    runs = merge_short_runs(fold_ring_seam(runs, total_m), MIN_RUN_M, ring=True)
    runs = ring_run_order(fold_ring_seam(runs, total_m), total_m)
    composition: dict[str, float] = {}
    for run in runs:
        composition[run["follows"]] = (composition.get(run["follows"], 0.0)
                                       + run["end_m"] - run["start_m"])
    return composition


def test_where_a_short_run_sits_relative_to_the_seam_does_not_move_credit():
    """A ring has no ends. Treating index 0 and -1 as ends made a short run
    that landed beside the seam absorb into the only neighbour the list
    allowed, rather than into the longer of its two real ones - so rotating
    the start vertex by one sample moved 20 m between the classes."""
    classes = ["trail"] * 10 + ["road"] * 8 + ["none"] * 2
    turned = classes[-1:] + classes[:-1]      # the same ring, started 10 m earlier

    at_start, rotated = ring_composition(classes), ring_composition(turned)
    # The short 'none' run is absorbed by the trail across the seam, which is
    # longer than the road beside it, whichever end of the list it lands on.
    assert at_start == pytest.approx({"trail": 120.0, "road": 80.0})
    assert rotated == pytest.approx(at_start)
    assert sum(at_start.values()) == pytest.approx(200.0)
    assert sum(rotated.values()) == pytest.approx(200.0)


# --- a trail edge and a freehand edge -------------------------------------------------

def test_a_polygon_traced_along_a_trail_reports_trail_on_that_edge(tmp_path):
    # Trail runs west-east at y=100 units; the polygon's north edge sits on it,
    # its other three edges cross open ground.
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    target = polygon((100, 100), (300, 100), (300, 300), (100, 300))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())

    assert isinstance(anatomy, BoundaryAnatomy)
    comp = anatomy.boundary.composition_pct
    # A quarter of the perimeter is the north edge, and the 20 m trail
    # tolerance reaches 20 m around each of its two corners as well - the
    # second of those is past the ring's start vertex, and is counted here
    # because the seam is folded rather than cut.
    assert comp["trail"] == pytest.approx(30.8, abs=2.0)
    assert comp["none"] == pytest.approx(69.2, abs=2.0)
    assert anatomy.boundary.on_feature_pct == pytest.approx(30.8, abs=2.0)
    north = [r for r in anatomy.boundary.runs if r["follows"] == "trail"]
    assert north and north[0]["feature_id"] == "route_0" and north[0]["feature_name"] == "Ridge Trail"
    # Four 200-unit edges, every one a single straight segment >= 150 m.
    assert anatomy.boundary.straight_pct == pytest.approx(100.0, abs=1.0)
    assert anatomy.boundary.straight_freehand_pct == pytest.approx(75.0, abs=4.0)
    assert anatomy.shape == "block" and anatomy.axis_feature is None
    assert anatomy.size.area_ha == pytest.approx((200 * GROUND_PER_UNIT) ** 2 / 1e4, rel=0.02)
    assert anatomy.boundary.shared_pct is None
    assert "no map" in anatomy.boundary.shared_reason
    assert anatomy.assumptions["sample_spacing_m"] == SAMPLE_SPACING_M


def test_a_drainage_read_from_the_elevation_model_counts_without_osm(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    (data_dir / "osm.json").unlink()
    # West edge on the valley floor (x = 200 units), the rest open ground.
    target = polygon((200, 50), (350, 50), (350, 350), (200, 350))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())
    assert anatomy.boundary.composition_pct["drainage"] > 25.0
    assert any(w.startswith("osm.json missing") for w in anatomy.warnings)
    # The channel lies ON the west edge, not inside it: without the interior
    # buffer the intersection is a matter of floating-point luck.
    assert anatomy.interior["drainage_m"] > 200 * GROUND_PER_UNIT * 0.8


# --- corridors and interiors ------------------------------------------------------------

def test_a_thin_ribbon_around_a_trail_is_a_corridor_with_that_axis(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    target = polygon((50, 185), (350, 185), (350, 215), (50, 215))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())
    assert anatomy.shape == "corridor"
    assert anatomy.axis_feature == "trail" and anatomy.axis_ratio == pytest.approx(1.0, abs=0.1)
    assert anatomy.size.width_m == pytest.approx(30 * GROUND_PER_UNIT, rel=0.1)
    assert anatomy.interior["trail_m"] == pytest.approx(300 * GROUND_PER_UNIT, rel=0.05)
    assert [f["metric"] for f in anatomy.norm_flags] == ["area_ha"], anatomy.norm_flags


def test_a_line_target_is_a_line(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    line = {"type": "LineString", "coordinates": [m(50, 200), m(350, 200)]}
    anatomy = describe_boundary(data_dir, line, norms=load_norms())
    assert anatomy.shape == "line"
    assert anatomy.size.length_m == pytest.approx(300 * GROUND_PER_UNIT, rel=0.02)
    assert anatomy.size.area_ha is None
    assert anatomy.boundary.composition_pct["trail"] == pytest.approx(100.0, abs=3.0)
    assert [f["metric"] for f in anatomy.norm_flags] == ["length_m"]


# --- shared edges against the map -------------------------------------------------------

def test_shared_edges_are_measured_against_other_assignments_on_the_map(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    target = polygon((100, 100), (300, 100), (300, 300), (100, 300))
    neighbour = polygon((300, 100), (390, 100), (390, 300), (300, 300))   # shares the east edge
    features = [
        {"id": "a1", "properties": {"class": "Assignment", "letter": "AA", "number": ""},
         "geometry": target},
        {"id": "a2", "properties": {"class": "Assignment", "letter": "AB", "number": ""},
         "geometry": neighbour},
        {"id": "m1", "properties": {"class": "Marker"}, "geometry": {"type": "Point", "coordinates": m(5, 5)}},
    ]
    anatomy = describe_boundary(data_dir, target, norms=load_norms(),
                                map_features=features, exclude_label="AA")
    assert anatomy.boundary.shared_pct == pytest.approx(25.0, abs=4.0)
    assert anatomy.boundary.shared_with == ["AB"]


def test_a_target_outside_the_staged_extent_is_refused(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    far = polygon((900, 900), (1000, 900), (1000, 1000), (900, 1000))
    with pytest.raises(ValueError, match="staged extent"):
        describe_boundary(data_dir, far, norms=load_norms())


def test_the_extent_refusal_names_the_staged_bounds_in_wgs84(tmp_path):
    """Web Mercator metres are not something an agent can act on: the refusal
    also says where the staging is in the coordinates the caller passed."""
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    far = polygon((900, 900), (1000, 900), (1000, 1000), (900, 1000))
    with pytest.raises(ValueError) as error:
        describe_boundary(data_dir, far, norms=load_norms())
    message = str(error.value)
    assert "WGS84" in message
    west, south = m(0, 400)[0], m(0, 400)[1]
    east, north = m(400, 0)[0], m(400, 0)[1]
    for value in (west, south, east, north):
        assert f"{value:.5f}" in message, message


def test_a_line_neighbour_along_the_edge_shares_it(tmp_path):
    """A LineString assignment IS its own boundary; buffering it into a 100 m
    corridor first would put its edge 50 m off the line it was drawn on."""
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    target = polygon(*TRAIL_SQUARE)
    line = {"type": "LineString", "coordinates": [m(300, 100), m(300, 300)]}
    features = [assignment("AA", target), assignment("AB", line)]
    anatomy = describe_boundary(data_dir, target, norms=load_norms(),
                                map_features=features, exclude_label="AA")
    assert anatomy.boundary.shared_with == ["AB"]
    assert anatomy.boundary.shared_pct == pytest.approx(25.0, abs=4.0)


def test_a_line_neighbour_well_clear_of_the_edge_shares_nothing(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    target = polygon(*TRAIL_SQUARE)
    east = 300 + 57.0 / GROUND_PER_UNIT          # 57 ground metres east of the edge
    line = {"type": "LineString", "coordinates": [m(east, 100), m(east, 300)]}
    features = [assignment("AA", target), assignment("AB", line)]
    anatomy = describe_boundary(data_dir, target, norms=load_norms(),
                                map_features=features, exclude_label="AA")
    assert anatomy.boundary.shared_pct == 0.0
    assert anatomy.boundary.shared_with == []


def test_a_neighbour_that_only_meets_a_corner_is_not_a_shared_edge(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    target = polygon(*TRAIL_SQUARE)
    east = polygon((300, 100), (390, 100), (390, 300), (300, 300))      # shares the east edge
    corner = polygon((300, 300), (390, 300), (390, 390), (300, 390))    # meets one corner
    features = [assignment("AA", target), assignment("AB", east), assignment("AC", corner)]
    anatomy = describe_boundary(data_dir, target, norms=load_norms(),
                                map_features=features, exclude_label="AA")
    assert anatomy.boundary.shared_with == ["AB"]
    assert anatomy.boundary.shared_pct == pytest.approx(25.0, abs=4.0)


# --- rings, degenerate targets and multipolygons -------------------------------------


def test_where_the_ring_starts_does_not_change_what_it_follows(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    at_corner = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    # The same square, drawn from a vertex half way along its north edge.
    mid_edge = describe_boundary(
        data_dir, polygon((200, 100), (300, 100), (300, 300), (100, 300), (100, 100)),
        norms=load_norms())
    assert mid_edge.boundary.composition_pct["trail"] == pytest.approx(
        at_corner.boundary.composition_pct["trail"], abs=1.0)
    assert len([r for r in mid_edge.boundary.runs if r["follows"] == "trail"]) == 1
    assert len([r for r in at_corner.boundary.runs if r["follows"] == "trail"]) == 1
    assert sum(r["end_m"] - r["start_m"] for r in mid_edge.boundary.runs) == pytest.approx(
        mid_edge.size.perimeter_m, rel=1e-6)


def test_a_degenerate_target_is_refused(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    collapsed = polygon((100, 100), (100, 100), (100, 100))
    collinear = polygon((100, 100), (200, 100), (300, 100))
    for target in (collapsed, collinear):
        with pytest.raises(ValueError, match="degenerate"):
            describe_boundary(data_dir, target, norms=load_norms())


def test_a_multipolygon_target_is_described_as_its_largest_part(tmp_path):
    """A multipart assignment is usually one real block and a sliver left by
    an editing accident; the largest part described with a warning is a more
    useful answer than a refusal."""
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    small = polygon((50, 50), (150, 50), (150, 150), (50, 150))        # 100 x 100 units
    large = polygon((200, 200), (380, 200), (380, 380), (200, 380))    # 180 x 180 units
    multi = {"type": "MultiPolygon",
             "coordinates": [small["coordinates"], large["coordinates"]]}
    anatomy = describe_boundary(data_dir, multi, norms=load_norms())

    assert ("multipolygon target: described the largest of 2 parts "
            "(76% of the area); the others are not described") in anatomy.warnings
    assert anatomy.size.area_ha == pytest.approx((180 * GROUND_PER_UNIT) ** 2 / 1e4, rel=0.02)
    assert anatomy.size.perimeter_m == pytest.approx(4 * 180 * GROUND_PER_UNIT, rel=0.02)
    assert anatomy.shape == "block"


# --- one Ground per staged directory ---------------------------------------------------


def test_two_describes_on_one_directory_build_one_ground(tmp_path, monkeypatch):
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    calls = []
    real = boundary_module.load_ground

    def counted(directory):
        calls.append(directory)
        return real(directory)

    monkeypatch.setattr(boundary_module, "load_ground", counted)
    first = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    second = describe_boundary(data_dir, polygon((50, 200), (250, 200), (250, 380), (50, 380)),
                               norms=load_norms())
    assert len(calls) == 1
    assert first.boundary.composition_pct["trail"] > 0
    assert second.shape == "block"


def test_a_caller_supplied_ground_is_measured_against_without_a_load(tmp_path, monkeypatch):
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    ground = load_ground(data_dir)
    ground_for.cache_clear()

    def refuse(directory):
        raise AssertionError("describe_boundary loaded a Ground it was handed")

    monkeypatch.setattr(boundary_module, "load_ground", refuse)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms(),
                                ground=ground)
    assert anatomy.boundary.composition_pct["trail"] > 0


def test_slope_and_landform_come_from_the_grid(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    target = polygon((50, 50), (150, 50), (150, 350), (50, 350))   # the west sidewall
    anatomy = describe_boundary(data_dir, target, norms=load_norms())
    assert anatomy.interior["slope_mean_deg"] > 10.0
    assert anatomy.interior["slope_p90_deg"] >= anatomy.interior["slope_mean_deg"]
    assert abs(sum(anatomy.interior["landform_fractions"].values()) - 1.0) < 0.01


def test_a_ribbon_along_a_dem_drainage_is_a_corridor_on_that_drainage(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    (data_dir / "osm.json").unlink()
    target = polygon((185, 50), (215, 50), (215, 350), (185, 350))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())
    assert anatomy.shape == "corridor"
    assert anatomy.axis_feature == "drainage"
    assert anatomy.axis_ratio > 0.5


def test_a_target_thinner_than_a_pixel_falls_back_to_touched_pixels(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    target = polygon((100, 196), (300, 196), (300, 198), (100, 198))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())
    assert any("touched pixels" in w for w in anatomy.warnings)
    assert anatomy.interior["slope_mean_deg"] is not None
    assert anatomy.interior["landform_fractions"]


def test_the_warnings_say_when_a_channel_leaves_the_raster(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assert any(w.startswith("a drainage reaches the raster edge") for w in anatomy.warnings)
    assert any(w.startswith("a ridgeline reaches the raster edge") for w in anatomy.warnings)


def test_the_assumptions_say_how_the_edge_and_the_ids_were_read(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assumptions = anatomy.assumptions
    assert list(assumptions["tolerances_m"]) == [
        "road", "trail", "waterway", "barrier", "water_edge", "drainage", "ridge"]
    assert assumptions["interior_buffer_m"] == 0.5
    assert "counts as interior" in assumptions["interior_buffer_note"]
    assert "whole staged raster" in assumptions["feature_ids"]
    assert "axis_ratio is null" in assumptions["interior_note"]
    assert "interior is empty for a line target" in assumptions["interior_note"]


def test_the_assumptions_say_which_band_the_area_flag_used(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    target = polygon(*TRAIL_SQUARE)
    whole = describe_boundary(data_dir, target, norms=load_norms())
    assert "whole corpus" in whole.assumptions["area_band"]

    per_resource = describe_boundary(data_dir, target, norms=load_norms(),
                                     resource_type="dog_trail")
    assert "DOG_TRAIL" in per_resource.assumptions["area_band"]
    # 2.5 ha of block against a 0.7 ha median: above that resource's band,
    # where the whole corpus (2 to 31 ha) says nothing about it.
    flags = {f["metric"]: f for f in per_resource.norm_flags}
    assert "resource" in flags["area_ha"]["band"] and "above" in flags["area_ha"]["note"]
    assert "area_ha" not in {f["metric"] for f in whole.norm_flags}

    unknown = describe_boundary(data_dir, target, norms=load_norms(),
                                resource_type="SUBMARINE")
    assert "whole corpus" in unknown.assumptions["area_band"]


# --- the staged OSM query's version ------------------------------------------------


def test_a_directory_staged_with_an_older_osm_query_says_so(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)), query_version=None)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    stale = [w for w in anatomy.warnings if "older OpenStreetMap query" in w]
    assert len(stale) == 1
    assert "v1" in stale[0] and "force=True" in stale[0]
    assert "water edges" in stale[0]


def test_a_version_three_directory_says_only_distribution_lines_are_missing(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)), query_version=3)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    stale = [w for w in anatomy.warnings if "older OpenStreetMap query" in w]
    assert len(stale) == 1
    assert "v3" in stale[0] and "force=True" in stale[0]
    assert "power=minor_line" in stale[0] and "water edges" not in stale[0]


def test_a_version_four_directory_is_not_told_distribution_lines_are_missing(tmp_path):
    """Version 4 first fetched `power=minor_line`; a boundary reads nothing
    version 5 added (road class, access, gates), so a version-4 file is
    complete for it and gets no stale-query warning."""
    data_dir = staged(tmp_path, np.zeros((40, 40)), query_version=4)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assert not [w for w in anatomy.warnings if "OpenStreetMap query" in w]


def test_a_directory_staged_with_the_current_osm_query_is_silent(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)), query_version=OSM_QUERY_VERSION)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assert not [w for w in anatomy.warnings if "OpenStreetMap query" in w]


# --- the sampler, the runs and the straight segments, alone ----------------------------


def test_sample_boundary_steps_along_and_ends_on_the_far_end():
    from shapely.geometry import LineString

    samples = sample_boundary(LineString([(0, 0), (95, 0)]), 10.0)
    assert len(samples) == 11
    assert samples[0][2] == 0.0
    assert samples[-1][2] == pytest.approx(95.0)
    assert samples[-1][0] == pytest.approx(95.0)


def test_straight_segments_are_single_vertex_to_vertex_edges():
    from shapely.geometry import LineString

    square = LineString([(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)])
    assert straight_segments(square, 1.0, 150.0) == [
        (0.0, 200.0), (200.0, 400.0), (400.0, 600.0), (600.0, 800.0)]
    # A 200-unit edge is only 159 ground metres where a unit is 0.7952 m.
    assert straight_segments(square, GROUND_PER_UNIT, 150.0) == [
        pytest.approx((0.0, 200 * GROUND_PER_UNIT)),
        pytest.approx((200 * GROUND_PER_UNIT, 400 * GROUND_PER_UNIT)),
        pytest.approx((400 * GROUND_PER_UNIT, 600 * GROUND_PER_UNIT)),
        pytest.approx((600 * GROUND_PER_UNIT, 800 * GROUND_PER_UNIT))]


def test_runs_from_samples_gives_each_run_half_a_spacing_either_side():
    samples = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, 20.0), (0.0, 0.0, 30.0)]
    runs = runs_from_samples(samples, ["trail", "trail", "none", "none"], 10.0)
    assert runs == [{"follows": "trail", "start_m": 0.0, "end_m": 15.0},
                    {"follows": "none", "start_m": 15.0, "end_m": 30.0}]


def test_the_ground_cache_is_dropped_when_the_directory_is_re_staged(tmp_path):
    """`prepare_terrain` re-stages a moved assignment into the SAME directory,
    overwriting dem.tif and osm.json and rewriting manifest.json last. Keyed
    by the path alone the cache served the old raster's numbers under a
    staleness check that said 'current'."""
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    target = polygon(*TRAIL_SQUARE)
    before = describe_boundary(data_dir, target, norms=load_norms())
    assert before.boundary.composition_pct["trail"] > 0

    # A re-stage whose OSM fetch came back empty. Rewriting osm.json WITHOUT
    # touching the manifest is deliberately NOT an invalidation: manifest.json
    # is written last, and it is the barrier that says the rest of the
    # directory is already the new staging.
    (data_dir / "osm.json").unlink()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["warnings"] = ["re-staged"]
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    after = describe_boundary(data_dir, target, norms=load_norms())
    assert after.boundary.composition_pct["trail"] == 0.0
    assert after.boundary.on_feature_pct == pytest.approx(0.0)
    assert MISSING_OSM_WARNING in after.warnings


def test_a_dem_with_a_hole_says_the_flow_routines_filled_it(tmp_path):
    dem = valley_dem()
    dem[10:16, 10:16] = -999999.0        # the fetch directory's nodata value
    data_dir = staged(tmp_path, dem)
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assert DEM_VOID_WARNING in anatomy.warnings


def test_a_run_over_two_trails_is_named_by_the_one_it_mostly_follows(tmp_path):
    """The modal feature under the run's own samples, not the one nearest its
    midpoint: the north edge runs 100..300, of which route_1 holds 150."""
    west = route("route_0", "TRAIL", [(0, 100), (150, 100)], name="West Trail")
    east = route("route_1", "TRAIL", [(150, 100), (400, 100)], name="East Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[west, east])
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())

    trail_runs = [r for r in anatomy.boundary.runs if r["follows"] == "trail"]
    assert len(trail_runs) == 1
    assert trail_runs[0]["feature_id"] == "route_1"
    assert trail_runs[0]["feature_name"] == "East Trail"


def test_a_trail_along_each_edge_is_not_the_axis_of_the_corridor_between_them(tmp_path):
    """The half-metre interior skin holds a trail drawn on the edge, and a
    corridor with one along each side held it twice - an axis_ratio of 2.0 on
    a feature that runs down neither. The axis is chosen from the bare
    polygon, and cannot be longer than the corridor it is the axis of."""
    skin = 0.25 / GROUND_PER_UNIT     # a quarter metre outside each long edge
    west = route("route_0", "TRAIL", [(185 - skin, 50), (185 - skin, 350)])
    east = route("route_1", "TRAIL", [(215 + skin, 50), (215 + skin, 350)])
    data_dir = staged(tmp_path, valley_dem(), routes=[west, east])
    target = polygon((185, 50), (215, 50), (215, 350), (185, 350))
    anatomy = describe_boundary(data_dir, target, norms=load_norms())

    assert anatomy.shape == "corridor"
    # Both trails are inside the skin, and the interior report says so.
    assert anatomy.interior["trail_m"] == pytest.approx(600 * GROUND_PER_UNIT, rel=0.05)
    # The axis is the drainage on the valley floor the corridor runs along.
    assert anatomy.axis_feature == "drainage"
    assert 0.5 <= anatomy.axis_ratio <= 1.0


def test_a_closed_line_target_is_walked_as_a_ring(tmp_path):
    """A LineString drawn back to its own start is a ring; a line's runs are
    folded at the seam like a polygon's, and no rectangle is measured."""
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    closed = {"type": "LineString",
              "coordinates": [m(*p) for p in TRAIL_SQUARE] + [m(*TRAIL_SQUARE[0])]}
    anatomy = describe_boundary(data_dir, closed, norms=load_norms())

    assert anatomy.shape == "line"
    assert anatomy.size.area_ha is None and anatomy.size.width_m is None
    assert anatomy.size.length_m == pytest.approx(800 * GROUND_PER_UNIT, rel=0.02)
    # The trail crosses the start vertex and is reported once, not twice.
    assert len([r for r in anatomy.boundary.runs if r["follows"] == "trail"]) == 1
    assert sum(r["end_m"] - r["start_m"] for r in anatomy.boundary.runs) == pytest.approx(
        anatomy.size.length_m, rel=1e-6)


def test_the_assumptions_say_how_the_runs_are_ordered(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    anatomy = describe_boundary(data_dir, polygon(*TRAIL_SQUARE), norms=load_norms())
    assumptions = anatomy.assumptions
    assert assumptions["runs_note"] == (
        "on a closed boundary runs are ordered from the first vertex and one run may "
        "carry wraps: true with end_m past the perimeter; on a line the first run "
        "starts at 0")
    assert assumptions["corridor"]["axis_ratio"] == (
        "the axis class's interior metres as a fraction of the corridor's length, "
        "at most 1")


def test_a_ring_run_that_crosses_the_seam_is_reported_once():
    samples = [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, 20.0), (0.0, 0.0, 30.0)]
    runs = runs_from_samples(samples, ["trail", "none", "none", "trail"], 10.0, total_m=40.0)
    folded = ring_run_order(fold_ring_seam(runs, 40.0), 40.0)
    assert [r["follows"] for r in folded] == ["none", "trail"]
    assert folded[-1]["wraps"] is True
    assert folded[-1]["start_m"] == pytest.approx(25.0)
    assert folded[-1]["end_m"] == pytest.approx(45.0)
    assert sum(r["end_m"] - r["start_m"] for r in folded) == pytest.approx(40.0)


def test_straight_segments_walks_every_part_of_a_multi_part_line():
    """A line candidate's axis is a MultiLineString when its named way forks,
    and `list(geometry.coords)` raises NotImplementedError on one of those.
    The positions are along the concatenated walk, the same frame
    `sample_boundary` interpolates in, so the two still line up."""
    from shapely.geometry import MultiLineString

    walk = MultiLineString([[(ORIGIN_X, ORIGIN_Y), (ORIGIN_X + 300, ORIGIN_Y)],
                            [(ORIGIN_X, ORIGIN_Y - 100), (ORIGIN_X + 200, ORIGIN_Y - 100)]])

    segments = straight_segments(walk, GROUND_PER_UNIT, 10.0)

    assert segments == [pytest.approx((0.0, 300 * GROUND_PER_UNIT)),
                        pytest.approx((300 * GROUND_PER_UNIT, 500 * GROUND_PER_UNIT))]


def test_a_multi_part_line_target_is_described_rather_than_refused(tmp_path):
    """`describe_boundary` takes a candidate id, and a candidate's axis is a
    MultiLineString whenever its named way forks."""
    trail = route("route_0", "TRAIL", [(100, 100), (300, 100)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    forked = {"type": "MultiLineString",
              "coordinates": [[m(100, 100), m(300, 100)], [m(200, 100), m(200, 250)]]}

    anatomy = describe_boundary(data_dir, forked, norms=load_norms())

    assert anatomy.shape == "line"
    assert anatomy.size.length_m == pytest.approx(350 * GROUND_PER_UNIT, rel=0.02)
    assert anatomy.size.area_ha is None
    # The first part lies on the mapped trail, the second on nothing.
    assert anatomy.boundary.composition_pct["trail"] > 40.0
