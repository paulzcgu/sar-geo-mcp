import json
import math

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import shape as shapely_shape
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.planning.egress_graph import build_graph
from sar_geo_mcp.planning.travel_rings import (
    LARGEST_RADIUS_M, NO_EVT_WARNING, NO_OSM_WARNING, VOID_ORIGIN_ERROR,
    WATERWAY_KINDS_UNKNOWN_WARNING, _edge_advice, fastest_factor, radius_to_fit, rings_model,
    travel_time_rings)
from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, routes_for

# Staged ground: 10 m cells, as the rings are staged at (the egress tests' grid).
ORIGIN = (-13618826.5, 4486605.0)
LATITUDE = 37.34
CELL_M = 10.0
PIXEL = CELL_M / math.cos(math.radians(LATITUDE))    # Mercator units per 10 m cell
TO4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

#: LANDFIRE codes, one per lifeform the tests need (from lf2024_evt.csv).
HERB, SHRUB, WATER = 4407, 4401, 7292
#: Off-way speed on flat open ground: Tobler at grade 0 times 0.6, in m/min.
FLAT_OFF_WAY_M_PER_MIN = 6000 * math.exp(-0.175) * 0.6 / 60


# --- The two graph options -----------------------------------------------------


def _graph(elevation, **kw):
    no = np.zeros(elevation.shape, dtype=bool)
    options = {"on_way": no, "impassable": no, "on_way_factor": 1.0,
               "off_way_factor": 0.6, "max_grade": None, **kw}
    return build_graph(elevation, cell_m=CELL_M, **options)


def test_the_vegetation_scale_slows_only_ground_off_a_way():
    elevation = np.zeros((1, 3))
    on_way = np.array([[False, True, False]])
    scale = np.full((1, 3), 0.5)
    plain = _graph(elevation, on_way=on_way)
    scaled = _graph(elevation, on_way=on_way, off_way_scale=scale)
    assert scaled[1, 2] == pytest.approx(plain[1, 2] * 2)     # onto brush: twice as slow
    assert scaled[0, 1] == pytest.approx(plain[0, 1])         # onto the trail: unchanged


def test_without_either_option_the_graph_is_the_one_egress_builds():
    rng = np.random.default_rng(3)
    elevation = rng.uniform(0, 8, (6, 7))
    plain = _graph(elevation)
    same = _graph(elevation, off_way_scale=None, crossing=np.ones((6, 7), dtype=bool),
                  crossing_minutes=0.0)
    assert (plain != same).nnz == 0


def test_a_crossing_costs_its_minutes_to_get_into_and_nothing_along_or_out():
    elevation = np.zeros((1, 4))
    creek = np.array([[False, True, True, False]])
    plain = _graph(elevation)
    graph = _graph(elevation, crossing=creek, crossing_minutes=5.0)
    assert graph[0, 1] == pytest.approx(plain[0, 1] + 5.0)    # bank -> creek
    assert graph[1, 2] == pytest.approx(plain[1, 2])          # along the creek
    assert graph[2, 3] == pytest.approx(plain[2, 3])          # creek -> far bank


def test_a_diagonal_between_two_creek_corners_pays_the_crossing():
    # A 45-degree creek burned all_touched joins at its corners: (0,1) and (1,0).
    elevation = np.zeros((2, 2))
    creek = np.array([[False, True], [True, False]])
    plain = _graph(elevation)
    graph = _graph(elevation, crossing=creek, crossing_minutes=5.0)
    assert graph[0, 3] == pytest.approx(plain[0, 3] + 5.0)    # (0,0) -> (1,1)


# --- Staged ground -------------------------------------------------------------


def xy(row, col):
    """The EPSG:3857 centre of a cell (fractions allowed)."""
    return ORIGIN[0] + (col + 0.5) * PIXEL, ORIGIN[1] - (row + 0.5) * PIXEL


def line(route_type, *cells, **properties):
    """A route record through cell centres, every current key written."""
    props = {"route_type": route_type, "name": None, "osm_id": 1, "surface": None,
             "power": None, "highway": None, "access": None, "tracktype": None,
             "waterway": None, "tunnel": None, **properties}
    return {"id": "route_x", "kind": "route",
            "geometry": {"type": "LineString", "coordinates": [list(xy(*c)) for c in cells]},
            "properties": props}


def trail(*cells):
    return line("TRAIL", *cells, highway="path")


def waterway(kind, *cells, **properties):
    return line("WATERWAY", *cells, waterway=kind, **properties)


def stage(tmp_path, elevation, routes=(), *, osm=True, evt=None):
    """Staged ground; `evt` is a per-DEM-cell code array (written at 30 m, so
    each EVT pixel covers 3 x 3 cells and the array is read at its top-left)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows, cols = elevation.shape
    transform = from_origin(ORIGIN[0], ORIGIN[1], PIXEL, PIXEL)
    west, north = TO4326.transform(*ORIGIN)
    east, south = TO4326.transform(ORIGIN[0] + cols * PIXEL, ORIGIN[1] - rows * PIXEL)
    write_fetch_dir(tmp_path, elevation, transform,
                    make_wgs84_square(west, south, east, north), latitude=LATITUDE)
    if osm:
        (tmp_path / "osm.json").write_text(json.dumps(
            {"routes": list(routes), "parking": [], "gates": [],
             "query_version": OSM_QUERY_VERSION}), encoding="utf-8")
    if evt is not None:
        codes = np.asarray(evt, dtype="int16")[::3, ::3]
        with rasterio.open(tmp_path / "evt.tif", "w", driver="GTiff", height=codes.shape[0],
                           width=codes.shape[1], count=1, dtype="int16", crs="EPSG:3857",
                           transform=from_origin(ORIGIN[0], ORIGIN[1], 3 * PIXEL,
                                                 3 * PIXEL)) as dst:
            dst.write(codes, 1)
    return tmp_path


def herb(shape):
    return np.full(shape, HERB)


def run(directory, row, col, **kw):
    lon, lat = TO4326.transform(*xy(row, col))
    return travel_time_rings(directory, center_latitude=lat, center_longitude=lon, **kw)


def reach(result, direction, index=0):
    return result["rings"][index]["reach_by_direction_m"][direction]


# --- The rings ---------------------------------------------------------------


def test_on_flat_open_ground_the_ring_is_near_a_circle_of_tobler_off_path(tmp_path):
    flat = np.zeros((61, 61))
    result = run(stage(tmp_path, flat, evt=herb(flat.shape)), 30, 30, minutes=[4],
                 vegetation_factors={"Herb": 1.0})
    expected = 4 * FLAT_OFF_WAY_M_PER_MIN                        # about 200 m
    ring = result["rings"][0]
    for direction in ("N", "E", "S", "W"):
        assert reach(result, direction) == pytest.approx(expected, abs=CELL_M)
    # An 8-neighbour walk is at most ~8 % slow off its eight directions.
    assert ring["equivalent_radius_m"] == pytest.approx(expected, rel=0.1)
    assert ring["reaches_edge"] is False and ring["parts"] == 1
    assert result["warnings"] == []


def test_rings_come_back_ascending_nested_and_drawable(tmp_path):
    flat = np.zeros((61, 61))
    result = run(stage(tmp_path, flat, evt=herb(flat.shape)), 30, 30, minutes=[4, 1, 2, 2])
    assert [ring["minutes"] for ring in result["rings"]] == [1.0, 2.0, 4.0]
    shapes = [shapely_shape(ring["geometry"]) for ring in result["rings"]]
    assert all(shape.is_valid for shape in shapes)
    assert shapes[0].area < shapes[1].area < shapes[2].area
    assert shapes[2].buffer(1e-9).contains(shapes[1])
    outline = result["rings"][2]["outline"]
    assert outline["type"] == "LineString"
    assert outline["coordinates"][0] == outline["coordinates"][-1]           # a loop


def test_a_trail_stretches_the_ring_along_it(tmp_path):
    flat = np.zeros((41, 81))
    routes = [trail((20, 40), (20, 80))]                         # east from the location
    result = run(stage(tmp_path, flat, routes, evt=herb(flat.shape)), 20, 40, minutes=[4],
                 vegetation_factors={"Herb": 1.0})
    assert reach(result, "E") == pytest.approx(reach(result, "W") / 0.6, rel=0.08)
    assert result["rings"][0]["max_reach_bearing_deg"] == pytest.approx(90, abs=3)


def test_a_river_stops_the_ring_and_a_bridge_lets_it_across(tmp_path):
    flat = np.zeros((41, 61))
    river = waterway("river", (0, 36), (40, 36))                 # 60 m east, north-south
    walled = run(stage(tmp_path / "a", flat, [river], evt=herb(flat.shape)), 20, 30,
                 minutes=[5])
    assert reach(walled, "E") < 60.0
    bridge = trail((20, 30), (20, 60))
    bridged = run(stage(tmp_path / "b", flat, [river, bridge], evt=herb(flat.shape)),
                  20, 30, minutes=[5])
    assert reach(bridged, "E") > 150.0
    assert bridged["ground"]["water"]["rivers"] == 1


def test_a_creek_costs_its_crossing_minutes(tmp_path):
    flat = np.zeros((41, 81))
    creek = waterway("stream", (0, 46), (40, 46))
    plain = run(stage(tmp_path / "a", flat, evt=herb(flat.shape)), 20, 40, minutes=[6])
    free = run(stage(tmp_path / "b", flat, [creek], evt=herb(flat.shape)), 20, 40,
               minutes=[6], creek_crossing_minutes=0)
    slowed = run(stage(tmp_path / "c", flat, [creek], evt=herb(flat.shape)), 20, 40,
                 minutes=[6], creek_crossing_minutes=2)
    assert reach(free, "E") == pytest.approx(reach(plain, "E"))
    # Two minutes at the bank is about two minutes' walk less beyond it.
    lost = reach(plain, "E") - reach(slowed, "E")
    assert lost == pytest.approx(2 * FLAT_OFF_WAY_M_PER_MIN * 0.9, abs=2 * CELL_M)
    assert reach(slowed, "W") == pytest.approx(reach(plain, "W"))


def test_a_creek_in_a_culvert_is_not_crossed(tmp_path):
    flat = np.zeros((41, 81))
    culvert = waterway("stream", (0, 46), (40, 46), tunnel="culvert")
    plain = run(stage(tmp_path / "a", flat, evt=herb(flat.shape)), 20, 40, minutes=[6])
    piped = run(stage(tmp_path / "b", flat, [culvert], evt=herb(flat.shape)), 20, 40,
                minutes=[6])
    assert reach(piped, "E") == pytest.approx(reach(plain, "E"))
    assert piped["ground"]["water"]["culverts"] == 1


def test_a_river_staged_before_waterway_kinds_were_kept_is_read_as_a_creek(tmp_path):
    flat = np.zeros((41, 61))
    river = waterway("river", (0, 36), (40, 36))
    for key in ("waterway", "tunnel"):
        river["properties"].pop(key)
    result = run(stage(tmp_path, flat, [river], evt=herb(flat.shape)), 20, 30, minutes=[5],
                 creek_crossing_minutes=1)
    assert reach(result, "E") > 100.0                            # crossed, not stopped
    assert WATERWAY_KINDS_UNKNOWN_WARNING in result["warnings"]


def test_a_lakeshore_is_closed_and_a_dam_is_not_water(tmp_path):
    flat = np.zeros((41, 61))
    shore = line("WATER_EDGE", (0, 36), (40, 36))
    dam = waterway("dam", (0, 24), (40, 24))
    result = run(stage(tmp_path, flat, [shore, dam], evt=herb(flat.shape)), 20, 30,
                 minutes=[5])
    assert reach(result, "E") < 60.0
    assert reach(result, "W") > 150.0


def test_a_cliff_closes_ground_off_a_trail(tmp_path):
    ground = np.zeros((41, 61))
    ground[:, 36:] = 40.0                       # a 40 m step: far over 45 degrees at 10 m
    walled = run(stage(tmp_path / "a", ground, evt=herb(ground.shape)), 20, 30, minutes=[8])
    assert reach(walled, "E") < 70.0
    assert reach(walled, "W") > 250.0
    assert walled["ground"]["closed_share"] > 0
    # With the cliff slope set past the step, Tobler alone makes it slow, not shut.
    opened = run(tmp_path / "a", 20, 30, minutes=[8], cliff_slope_deg=89.9)
    assert opened["ground"]["closed_share"] == 0


def test_a_benched_trail_across_ground_steeper_than_a_cliff_stays_open(tmp_path):
    sidehill = np.mgrid[0:41, 0:61][0] * 12.0            # 50 degrees, rising south
    shut = run(stage(tmp_path / "a", sidehill, evt=herb(sidehill.shape)), 20, 30,
               minutes=[3])
    assert reach(shut, "E") < CELL_M * 1.5               # off-trail: nowhere to step
    bench = trail((20, 30), (20, 60))
    along = run(stage(tmp_path / "b", sidehill, [bench], evt=herb(sidehill.shape)), 20, 30,
                minutes=[3])
    assert reach(along, "E") > 100.0                     # the flat tread is walked
    assert reach(along, "W") < CELL_M * 1.5


def test_brush_shrinks_the_ring_and_open_water_closes_it(tmp_path):
    flat = np.zeros((42, 84))
    codes = herb(flat.shape)
    codes[:, :42] = SHRUB                                        # brush to the west
    result = run(stage(tmp_path / "a", flat, evt=codes), 21, 42, minutes=[4])
    assert reach(result, "E") / reach(result, "W") == pytest.approx(0.9 / 0.4, rel=0.15)
    assert result["ground"]["vegetation"]["Shrub"] == pytest.approx(0.5, abs=0.05)

    codes = herb(flat.shape)
    codes[:, 48:] = WATER                                        # a lake to the east
    wet = run(stage(tmp_path / "b", flat, evt=codes), 21, 42, minutes=[4])
    assert reach(wet, "E") < 70.0


def test_an_unreachable_lake_inside_a_ring_is_a_hole(tmp_path):
    flat = np.zeros((81, 63))
    codes = herb(flat.shape)
    codes[33:39, 27:36] = WATER                       # a 60 x 90 m pond 70 m north
    result = run(stage(tmp_path, flat, evt=codes), 45, 31, minutes=[6])
    geometry = shapely_shape(result["rings"][0]["geometry"])
    assert geometry.geom_type == "Polygon" and len(geometry.interiors) == 1


def test_speed_factor_shrinks_every_ring(tmp_path):
    flat = np.zeros((61, 61))
    directory = stage(tmp_path, flat, evt=herb(flat.shape))
    full = run(directory, 30, 30, minutes=[4])
    half = run(directory, 30, 30, minutes=[4], speed_factor=0.5)
    assert reach(half, "N") == pytest.approx(reach(full, "N") / 2, abs=CELL_M)
    assert half["speed_factor"] == 0.5


def test_a_ring_cut_by_the_staged_edge_says_so_and_names_a_radius(tmp_path):
    flat = np.zeros((21, 21))
    result = run(stage(tmp_path, flat, evt=herb(flat.shape)), 10, 10, minutes=[10])
    assert result["rings"][0]["reaches_edge"] is True
    assert any("reaches the edge of the staged ground" in warning
               and "radius_m=" in warning for warning in result["warnings"])


def test_the_edge_advice_names_a_fitting_radius_or_says_none_fits():
    assert "radius_m=250" in _edge_advice(10, 1.0)      # 1,000 m at top speed, the floor
    assert "radius_m=2,000" in _edge_advice(30, 1.0)    # 3,000 m less the 1,000 m buffer
    assert "cannot be drawn whole" in _edge_advice(120, 1.0)


@pytest.mark.parametrize("minutes", [52.6, 53.0, 54.99])
def test_the_edge_advice_never_names_a_radius_prepare_area_refuses(minutes):
    """5,260 to 5,499 m rounds up to a 4,500 m radius, which prepare_area
    refuses by a hair; the largest it stages, 4,499 m, still holds the
    walk, so that is the radius named."""
    assert radius_to_fit(minutes) == 4500.0
    advice = _edge_advice(minutes, 1.0)
    assert "radius_m=4,499 stages enough ground" in advice
    assert "4,500" not in advice


def test_past_the_largest_ground_the_edge_advice_names_the_largest_staged():
    advice = _edge_advice(55.0, 1.0)         # 5,500 m: one metre past 4,499 m + buffer
    assert "cannot be drawn whole" in advice
    assert "(4,499 m radius) reaches about 5,499 m" in advice
    assert LARGEST_RADIUS_M == 4499.0


def test_an_absurd_time_or_pace_says_none_fits_rather_than_overflowing():
    assert radius_to_fit(1e307, 1e308) == math.inf
    assert "cannot be drawn whole" in _edge_advice(1e307, 1e308)


def test_the_reach_bound_follows_the_quickest_ground_on_the_model():
    """A trail is the quickest ground on the default model; an off-way or
    vegetation override above it makes open ground quicker, and the bound
    the edge advice names must follow it."""
    assert fastest_factor(rings_model()) == 1.0
    assert fastest_factor(rings_model(speed_factor=0.5)) == 0.5
    assert fastest_factor(rings_model(off_way_factor=2.0)) == 2.0
    assert fastest_factor(rings_model(vegetation_factors={"Herb": 3.0})) == pytest.approx(1.8)
    assert fastest_factor(rings_model(off_way_factor=1.5, speed_factor=0.5)) == 0.75
    with pytest.raises(ValueError, match="off_way_factor"):
        rings_model(off_way_factor=0)


def test_an_edge_warning_with_quick_open_ground_names_a_radius_that_holds_it(tmp_path):
    """Off-way at twice Tobler's pace: 2,000 m in 10 minutes, so 1,000 m
    once the buffer is off - where the trail's bound said 250 m."""
    flat = np.zeros((21, 21))
    result = run(stage(tmp_path, flat, evt=herb(flat.shape)), 10, 10, minutes=[10],
                 off_way_factor=2.0)
    edge = [warning for warning in result["warnings"] if "reaches the edge" in warning]
    assert edge and "No one walks farther than 2,000 m in 10 minutes" in edge[0]
    assert "radius_m=1,000 stages enough ground" in edge[0]


def test_missing_osm_and_vegetation_are_warned_not_refused(tmp_path):
    flat = np.zeros((21, 21))
    result = run(stage(tmp_path, flat, osm=False), 10, 10, minutes=[1])
    assert NO_OSM_WARNING in result["warnings"] and NO_EVT_WARNING in result["warnings"]
    assert result["ground"]["vegetation"] is None


def test_a_location_on_a_void_is_refused(tmp_path):
    ground = np.zeros((21, 21))
    ground[10, 10] = np.nan
    with pytest.raises(ValueError, match="gap in the elevation model"):
        run(stage(tmp_path, ground), 10, 10, minutes=[1])
    assert "move it" in VOID_ORIGIN_ERROR


def test_ground_staged_too_fine_is_refused(tmp_path):
    flat = np.zeros((21, 21))
    with pytest.raises(ValueError, match="resolution_m=10.0"):
        run(stage(tmp_path, flat), 10, 10, minutes=[1], max_cells=100)


@pytest.mark.parametrize("kw, message", [
    ({"minutes": []}, "empty"),
    ({"minutes": [10, -5]}, "above zero"),
    ({"minutes": [float("nan")]}, "above zero"),
    ({"minutes": list(range(1, 10))}, "the most one call draws is 8"),
    ({"speed_factor": 0}, "speed_factor"),
    ({"off_way_factor": -1}, "off_way_factor"),
    ({"creek_crossing_minutes": -1}, "creek_crossing_minutes"),
    ({"cliff_slope_deg": 90}, "cliff_slope_deg"),
    ({"vegetation_factors": {"Water": 0.5}}, "open water is closed"),
    ({"vegetation_factors": {"Shrub": 0}}, "above zero"),
    ({"vegetation_factors": {"Shrubs": 0.5}}, "not a LANDFIRE lifeform"),
])
def test_bad_settings_are_refused_before_any_ground_is_read(tmp_path, kw, message):
    with pytest.raises(ValueError, match=message):
        travel_time_rings(tmp_path, center_latitude=LATITUDE, center_longitude=-122.3, **kw)


def test_the_assumptions_carry_every_factor_used(tmp_path):
    flat = np.zeros((21, 21))
    result = run(stage(tmp_path, flat), 10, 10, minutes=[1],
                 vegetation_factors={"Shrub": 0.3}, creek_crossing_minutes=8)
    assumptions = result["assumptions"]
    movement = assumptions["movement"]
    assert movement["vegetation_factors"]["Shrub"] == 0.3
    assert movement["vegetation_factors"]["Tree"] == 0.8
    assert movement["creek_crossing_minutes"] == 8.0
    assert movement["impassable_slope_deg"] == 45.0
    assert movement["source"] == "sar_geo_mcp.terrain.movement"
    assert "not_a_prediction" in assumptions and "closed_water" in assumptions


# --- The staged record keeps what the rings read ------------------------------


def test_a_route_record_always_carries_waterway_and_tunnel(tmp_path):
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"waterway": "River", "name": "Pescadero Creek"},
         "geometry": [{"lat": 37.3561, "lon": -122.2419}, {"lat": 37.3571, "lon": -122.2409}]},
        {"type": "way", "id": 2, "tags": {"waterway": "stream", "tunnel": "culvert"},
         "geometry": [{"lat": 37.3561, "lon": -122.2419}, {"lat": 37.3565, "lon": -122.2415}]},
        {"type": "way", "id": 3, "tags": {"highway": "path"},
         "geometry": [{"lat": 37.3561, "lon": -122.2419}, {"lat": 37.3565, "lon": -122.2415}]}]}
    to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = to3857.transform(-122.2440, 37.3540)
    x1, y1 = to3857.transform(-122.2390, 37.3590)
    routes = routes_for((x0, y0, x1, y1), tmp_path, get_json=lambda query: payload)
    by_osm = {route["properties"]["osm_id"]: route["properties"] for route in routes}
    assert by_osm[1]["waterway"] == "river" and by_osm[1]["tunnel"] is None
    assert by_osm[2]["waterway"] == "stream" and by_osm[2]["tunnel"] == "culvert"
    assert by_osm[3]["waterway"] is None and "tunnel" in by_osm[3]


# --- The quickest route in each direction -------------------------------------


def test_each_direction_says_how_long_on_trail_and_where_it_leaves(tmp_path):
    flat = np.zeros((61, 121))
    ridge = line("TRAIL", (30, 20), (30, 120), highway="path", name="Ridge Trail")
    result = run(stage(tmp_path, flat, [ridge], evt=herb(flat.shape)), 30, 20, minutes=[6])
    routes = result["rings"][0]["routes_by_direction"]

    east = routes["E"]                                           # straight down the trail
    assert east["off_trail_minutes"] == pytest.approx(0.0, abs=0.05)
    assert east["leaves_trail_at"] is None
    assert east["trail"] == {"name": "Ridge Trail", "route_type": "TRAIL", "highway": "path"}

    north_east = routes["NE"]                                    # trail, then off it
    assert north_east["trail_minutes"] > 0.5 and north_east["off_trail_minutes"] > 0.5
    assert north_east["trail_minutes"] + north_east["off_trail_minutes"] == pytest.approx(
        north_east["minutes"], abs=0.15)
    assert north_east["leaves_trail_at"]["type"] == "Point"
    assert north_east["route"]["type"] == "LineString"
    leave_lon = north_east["leaves_trail_at"]["coordinates"][0]
    start_lon = result["center"]["longitude"]
    end_lon = north_east["point"]["coordinates"][0]
    assert start_lon < leave_lon < end_lon                       # left partway along

    ring = result["rings"][0]
    assert {name: route["reach_m"] for name, route in routes.items()} == \
        ring["reach_by_direction_m"]


def test_a_route_that_never_touches_a_trail_says_so(tmp_path):
    flat = np.zeros((41, 41))
    result = run(stage(tmp_path, flat, evt=herb(flat.shape)), 20, 20, minutes=[3])
    north = result["rings"][0]["routes_by_direction"]["N"]
    assert north["trail"] is None and north["leaves_trail_at"] is None
    assert north["trail_minutes"] == 0.0
    assert north["off_trail_minutes"] == pytest.approx(north["minutes"], abs=0.05)


def test_a_creek_crossing_is_counted_as_off_trail_time(tmp_path):
    flat = np.zeros((41, 81))
    creek = waterway("stream", (0, 46), (40, 46))
    result = run(stage(tmp_path, flat, [creek], evt=herb(flat.shape)), 20, 40, minutes=[8],
                 creek_crossing_minutes=2)
    east = result["rings"][0]["routes_by_direction"]["E"]
    walked = east["reach_m"] / (FLAT_OFF_WAY_M_PER_MIN * 0.9)
    assert east["off_trail_minutes"] == pytest.approx(walked + 2, abs=0.5)


def test_a_direction_with_no_ground_in_reach_has_no_route(tmp_path):
    ground = np.zeros((41, 41))
    ground[:21, :] = 60.0                                       # a wall along the north
    ground[:, 21:] = 60.0                                       # and the east
    result = run(stage(tmp_path, ground, evt=herb(ground.shape)), 30, 10, minutes=[1])
    ring = result["rings"][0]
    for name, route in ring["routes_by_direction"].items():
        assert (route is None) == (ring["reach_by_direction_m"][name] == 0.0)
