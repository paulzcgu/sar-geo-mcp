import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.planning.egress import (EGRESS_MAX_CELLS, STEEP_ORIGIN_REASON,
                                         VOID_ORIGIN_REASON, _exit, _measure, _way_masks,
                                         check_params, egress_routes)
from sar_geo_mcp.planning.egress_graph import (MODES, build_graph, signed_tobler_kmh,
                                               solve_from, solve_staying_on_ways)
from sar_geo_mcp.planning.egress_roads import NO_OSM_ERROR
from sar_geo_mcp.planning.walking import NO_EVT_WARNING
from sar_geo_mcp.terrain.grid import load_assignment_grid
from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

# Staged ground for the entry-point tests: 10 m cells, as egress stages at.
ORIGIN = (-13618826.5, 4486605.0)
LATITUDE = 37.34
CELL_M = 10.0
PIXEL = CELL_M / math.cos(math.radians(LATITUDE))    # Mercator units per 10 m cell
TO4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
TO3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _solve(graph, elevation, origin):
    """Minutes and predecessors from `origin`, with the void read off the ground."""
    return solve_from(graph, elevation.shape, origin, void=np.isnan(elevation))


def _graph(elevation, mode, *, on_way=None, impassable=None):
    no = np.zeros(elevation.shape, dtype=bool)
    return build_graph(elevation, cell_m=10.0,
                       on_way=no if on_way is None else on_way,
                       impassable=no if impassable is None else impassable,
                       **MODES[mode])


def test_signed_tobler_is_fastest_on_a_gentle_downhill():
    rates = signed_tobler_kmh(np.array([-0.05, 0.0, 0.10, -0.10, -0.30]))
    assert rates[0] == pytest.approx(6.0)
    assert rates[0] > rates[1] > rates[2]
    assert rates[3] > rates[2]                        # down 10 % beats up 10 %
    assert rates[4] < rates[1]                        # steep down is slow too


def test_climbing_costs_more_than_descending_the_same_steps():
    elevation = np.tile(np.arange(20, dtype=float) * 3.0, (3, 1))   # 30 % ramp east
    graph = _graph(elevation, "walk_out")
    up = _solve(graph, elevation, (1, 0))[0][1, 19]
    down = _solve(graph, elevation, (1, 19))[0][1, 0]
    assert up > down


def test_a_way_is_quicker_than_the_same_ground_off_it():
    elevation = np.zeros((3, 30))
    on = np.ones(elevation.shape, dtype=bool)
    t_off = _solve(_graph(elevation, "walk_out"), elevation, (1, 0))[0][1, 29]
    t_on = _solve(_graph(elevation, "walk_out", on_way=on), elevation, (1, 0))[0][1, 29]
    assert t_on == pytest.approx(t_off * 0.6)


def test_the_ground_factor_is_the_destination_cells():
    elevation = np.zeros((1, 2))
    graph = _graph(elevation, "walk_out", on_way=np.array([[False, True]]))
    # 0 -> 1 steps onto the way (factor 1.0); 1 -> 0 steps off it (factor 0.6).
    assert graph[0, 1] == pytest.approx(graph[1, 0] * 0.6)


def test_minutes_on_flat_open_ground_are_tobler_times_the_off_way_factor():
    elevation = np.zeros((4, 4))
    minutes, _ = _solve(_graph(elevation, "walk_out"), elevation, (0, 0))
    assert minutes[3, 3] == pytest.approx(
        3 * 10 * math.sqrt(2) * 0.06 / (6 * math.exp(-0.175) * 0.6))


def test_a_litter_cannot_take_a_step_over_the_grade_limit():
    elevation = np.zeros((3, 10))
    elevation[:, 5:] = 50.0                       # a 5 m-per-metre step at 10 m cells
    carry = _graph(elevation, "carry_out")
    assert math.isinf(_solve(carry, elevation, (1, 0))[0][1, 9])     # not up
    assert math.isinf(_solve(carry, elevation, (1, 9))[0][1, 0])     # nor down
    walk = _graph(elevation, "walk_out")
    assert math.isfinite(_solve(walk, elevation, (1, 0))[0][1, 9])
    assert math.isfinite(_solve(walk, elevation, (1, 9))[0][1, 0])


@pytest.mark.parametrize("rise, allowed", [(3.9, True), (4.1, False)])
def test_the_step_limit_is_pinned_either_side_of_40_percent(rise, allowed):
    # One row, one step of 0.39 or 0.41 between cols 1 and 2. np.gradient
    # spreads that step over two cells, so the ground's slope is at most
    # rise / 20 = 0.205 everywhere: only the per-step rule can close it.
    elevation = np.array([[0.0, 0.0, rise, rise]])
    graph = _graph(elevation, "carry_out")
    assert bool(graph[1, 2] > 0) is allowed            # up the step
    assert bool(graph[2, 1] > 0) is allowed            # down the step
    assert graph[0, 1] > 0 and graph[2, 3] > 0     # the flat either side stays open


def test_a_litter_cannot_zigzag_up_a_face_steeper_than_the_limit():
    cols = np.arange(9, dtype=float)
    face = np.tile(cols * 5.0, (9, 1))            # 50 % east-facing; a diagonal step is 35 %
    carry = _graph(face, "carry_out")
    assert math.isinf(_solve(carry, face, (4, 0))[0][4, 8])
    assert math.isinf(_solve(carry, face, (4, 8))[0][4, 0])
    assert math.isfinite(_solve(_graph(face, "walk_out"), face, (4, 0))[0][4, 8])
    # Just under the limit the same face is open to a litter.
    gentle = np.tile(cols * 3.9, (9, 1))
    assert math.isfinite(_solve(_graph(gentle, "carry_out"), gentle, (4, 0))[0][4, 8])


def test_the_grounds_slope_is_its_steepest_not_its_steepest_axis():
    # 30 % south and 30 % east: 42 % down the fall line. Every straight step
    # is 30 %, under the step limit, so only the ground's slope closes it.
    rows, cols = np.mgrid[0:9, 0:9].astype(float)
    tilted = (rows + cols) * 3.0
    assert math.isinf(_solve(_graph(tilted, "carry_out"), tilted, (0, 0))[0][8, 8])


def test_a_litter_may_follow_a_trail_across_steep_ground_one_step_at_a_time():
    # A 60 % sidehill rising to the south. A benched trail runs east-west along
    # row 4 and reads 60 % on the DEM though its tread is flat; another trail
    # climbs straight up the fall line in column 6, each step 60 %.
    rows = np.mgrid[0:9, 0:9][0].astype(float)
    hill = rows * 6.0
    trail = np.zeros(hill.shape, dtype=bool)
    trail[4, :] = True
    trail[:, 6] = True
    carry = _graph(hill, "carry_out", on_way=trail)
    along, _ = _solve(carry, hill, (4, 0))
    assert math.isfinite(along[4, 5])                      # along the bench
    assert carry[4 * 9 + 1, 4 * 9 + 2] > 0                 # a trail cell is enterable
    # The same ground off the trail is closed: row 2 goes nowhere.
    off, _ = _solve(carry, hill, (2, 0))
    assert math.isinf(off[2, 5])
    # The trail up the fall line is 60 % a step, but a step along a trail is
    # belayed, never left (ruled 2026-10-24): the litter climbs and descends it.
    assert math.isfinite(along[0, 6]) and math.isfinite(along[8, 6])
    assert carry[4 * 9 + 6, 5 * 9 + 6] > 0 and carry[4 * 9 + 6, 3 * 9 + 6] > 0
    # Stepping onto that trail from the open face beside it is still 60 %.
    assert carry[4 * 9 + 5, 3 * 9 + 5] == 0


def test_a_litter_may_follow_a_45_degree_trail_burned_corner_to_corner():
    # A 60 % sidehill whose fall line runs diagonally, so the contour is the
    # 45-degree line row == col: 60 % / sqrt(2) per axis, 4.24 m per 10 m cell.
    # The trail on the contour is burned corner to corner (np.eye), so every
    # step along it is diagonal and both its corner cells are steep ground a
    # litter may not enter. Hard ground at a corner is enough for a trail.
    rows, cols = np.mgrid[0:9, 0:9].astype(float)
    hill = (rows - cols) * (6.0 / math.sqrt(2.0))
    trail = np.eye(9, dtype=bool)
    carry = _graph(hill, "carry_out", on_way=trail)
    along, _ = _solve(carry, hill, (0, 0))
    assert math.isfinite(along[8, 8])
    # Off the trail the face is still closed to a litter.
    assert math.isinf(along[0, 2]) and math.isinf(along[2, 0])


def test_the_diagonal_trail_rule_needs_both_ends_on_the_way():
    # Both corners of (0,0) -> (1,1) are steep 20 m knobs a litter may not
    # enter; the step itself is flat and lands on a trail cell.
    elevation = np.array([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    onto = np.zeros(elevation.shape, dtype=bool)
    onto[1, 1] = True
    assert _graph(elevation, "carry_out", on_way=onto)[0, 4] == 0      # from off the trail
    along = onto.copy()
    along[0, 0] = True
    assert _graph(elevation, "carry_out", on_way=along)[0, 4] > 0      # along it


@pytest.mark.parametrize("corners_are", ["void", "motorway"])
def test_a_diagonal_trail_still_cannot_squeeze_between_two_blocked_corners(corners_are):
    elevation = np.zeros((2, 2))
    closed = np.zeros(elevation.shape, dtype=bool)
    if corners_are == "void":
        elevation[1, 0] = elevation[0, 1] = np.nan
    else:
        closed[1, 0] = closed[0, 1] = True
    trail = np.eye(2, dtype=bool)
    for mode in ("walk_out", "carry_out"):
        graph = _graph(elevation, mode, on_way=trail, impassable=closed)
        assert graph[0, 3] == 0 and graph[3, 0] == 0


def test_ground_beside_a_void_is_not_closed_to_a_litter_on_its_unknown_slope():
    # np.gradient gives NaN beside a void. That slope is unknown, not steep:
    # the cell stays open and only the per-step limit applies to it.
    elevation = np.zeros((5, 5))
    elevation[2, 2] = np.nan
    minutes, _ = _solve(_graph(elevation, "carry_out"), elevation, (2, 0))
    assert math.isfinite(minutes[2, 1]) and math.isfinite(minutes[2, 3])


def test_a_dem_void_and_an_impassable_cell_are_never_crossed():
    elevation = np.zeros((5, 5))
    elevation[:, 2] = np.nan                      # a void column
    assert math.isinf(_solve(_graph(elevation, "walk_out"), elevation, (2, 0))[0][2, 4])
    # The same wall made of impassable cells (a motorway) on measured ground.
    ground = np.zeros((5, 5))
    wall = np.zeros(ground.shape, dtype=bool)
    wall[:, 2] = True
    graph = _graph(ground, "walk_out", impassable=wall)
    assert math.isinf(_solve(graph, ground, (2, 0))[0][2, 4])


@pytest.mark.parametrize("wall_is", ["motorway", "void"])
def test_no_diagonal_step_squeezes_between_two_blocked_corners(wall_is):
    # A one-cell wall on the exact diagonal: every cell of it touches the next
    # only at a corner, the way all_touched burns a 45-degree line.
    elevation = np.zeros((9, 9))
    closed = np.zeros(elevation.shape, dtype=bool)
    if wall_is == "void":
        elevation[np.eye(9, dtype=bool)] = np.nan
    else:
        closed = np.eye(9, dtype=bool)
    graph = _graph(elevation, "walk_out", impassable=closed)
    assert math.isinf(_solve(graph, elevation, (8, 0))[0][0, 8])


def test_a_diagonal_past_one_blocked_corner_is_still_open():
    elevation = np.zeros((3, 3))
    closed = np.zeros(elevation.shape, dtype=bool)
    closed[0, 1] = True
    graph = _graph(elevation, "walk_out", impassable=closed)
    assert graph[0, 4] > 0                         # (0, 0) -> (1, 1), past (1, 0)


def test_a_subject_on_a_motorway_can_walk_off_it():
    ground = np.zeros((5, 5))
    wall = np.zeros(ground.shape, dtype=bool)
    wall[:, 2] = True
    minutes, _ = _solve(_graph(ground, "walk_out", impassable=wall), ground, (2, 2))
    assert math.isfinite(minutes[2, 0]) and math.isfinite(minutes[2, 4])


def test_an_origin_on_a_void_reaches_nothing_not_even_itself():
    elevation = np.zeros((5, 5))
    elevation[2, 2] = np.nan
    minutes, predecessors = _solve(_graph(elevation, "walk_out"), elevation, (2, 2))
    assert np.isinf(minutes).all()
    assert (predecessors == -9999).all()


@pytest.mark.parametrize("origin", [(-1, 0), (0, -1), (5, 0), (0, 5)])
def test_an_origin_off_the_grid_is_refused_not_wrapped(origin):
    elevation = np.zeros((5, 5))
    with pytest.raises(ValueError, match="outside"):
        _solve(_graph(elevation, "walk_out"), elevation, origin)


def test_no_edge_touches_a_void_and_every_edge_is_finite():
    elevation = np.zeros((6, 6))
    elevation[2, 3] = np.nan
    elevation[4, 1] = 1.0e4                        # a spike too steep to have a speed
    graph = _graph(elevation, "walk_out")
    void = 2 * 6 + 3
    assert graph[void].nnz == 0 and graph[:, void].nnz == 0
    assert np.isfinite(graph.data).all() and (graph.data > 0).all()


def test_solve_from_returns_predecessors_that_lead_home():
    elevation = np.zeros((4, 4))
    minutes, predecessors = _solve(_graph(elevation, "walk_out"), elevation, (0, 0))
    cell = 3 * 4 + 3
    steps = 0
    while cell != 0:
        cell = int(predecessors[cell])
        steps += 1
    assert steps == 3                              # three diagonal steps


# --- Task 4: exits on staged ground ------------------------------------------


def xy(row, col):
    """The EPSG:3857 centre of a cell."""
    return ORIGIN[0] + (col + 0.5) * PIXEL, ORIGIN[1] - (row + 0.5) * PIXEL


def cell_of(point_wgs84):
    """The (row, col) a WGS84 GeoJSON point falls in."""
    x, y = TO3857.transform(*point_wgs84["coordinates"])
    return int((ORIGIN[1] - y) // PIXEL), int((x - ORIGIN[0]) // PIXEL)


def way(route_type, *cells, highway, surface=None, name=None):
    """A route record as query version 5 writes one, through cell centres."""
    return {"id": f"route_{name}", "kind": "route",
            "geometry": {"type": "LineString", "coordinates": [list(xy(*c)) for c in cells]},
            "properties": {"route_type": route_type, "name": name, "osm_id": 1,
                           "surface": surface, "power": None, "highway": highway,
                           "access": None, "tracktype": None}}


def lot(r0, c0, r1, c1, kind="parking"):
    """A parking row whose polygon covers cells r0..r1 x c0..c1, inset a
    quarter cell so all_touched burns no neighbour."""
    x0, y0 = ORIGIN[0] + (c0 + 0.25) * PIXEL, ORIGIN[1] - (r0 + 0.25) * PIXEL
    x1, y1 = ORIGIN[0] + (c1 + 0.75) * PIXEL, ORIGIN[1] - (r1 + 0.75) * PIXEL
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {"id": f"{kind}-w{r0}{c0}", "kind": kind,
            "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}}


def parking_node(row, col):
    return {"id": f"parking-n{row}{col}", "kind": "parking",
            "geometry": {"type": "Point", "coordinates": list(xy(row, col))},
            "properties": {"flags": ["node_only"]}}


def gate(row, col, barrier="gate", access=None):
    """A gate record as query version 5 writes one (rows and cols may be fractional)."""
    return {"id": f"gate-n{row}-{col}", "kind": "gate",
            "geometry": {"type": "Point", "coordinates": list(xy(row, col))},
            "properties": {"osm_id": 1, "barrier": barrier, "access": access}}


def version_4(record):
    """A route record as query version 4 wrote it: no highway, access or tracktype."""
    for key in ("highway", "access", "tracktype"):
        record["properties"].pop(key, None)
    return record


def stage(tmp_path, elevation, routes=(), parking=(), *, osm=True, gates=(),
          query_version=OSM_QUERY_VERSION):
    """`gates=None` or `parking=None` writes no such key at all."""
    rows, cols = elevation.shape
    transform = from_origin(ORIGIN[0], ORIGIN[1], PIXEL, PIXEL)
    west, north = TO4326.transform(*ORIGIN)
    east, south = TO4326.transform(ORIGIN[0] + cols * PIXEL, ORIGIN[1] - rows * PIXEL)
    write_fetch_dir(tmp_path, elevation, transform,
                    make_wgs84_square(west, south, east, north), latitude=LATITUDE)
    if osm:
        payload = {"routes": list(routes), "query_version": query_version}
        if parking is not None:
            payload["parking"] = list(parking)
        if gates is not None:
            payload["gates"] = list(gates)
        (tmp_path / "osm.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def run(directory, row, col, **kw):
    lon, lat = TO4326.transform(*xy(row, col))
    return egress_routes(directory, center_latitude=lat, center_longitude=lon, **kw)


def north_south(route_type, col, rows, highway="residential", **kw):
    return way(route_type, (0, col), (rows - 1, col), highway=highway, **kw)


def test_a_ridge_makes_a_farther_flatter_exit_win(tmp_path):
    cols = np.arange(200, dtype=float)
    elevation = np.tile(60.0 * np.exp(-((cols - 130.0) / 8.0) ** 2), (21, 1))
    routes = [north_south("ROAD", 160, 21, name="East"),     # 60 cells east, over the ridge
              north_south("ROAD", 10, 21, name="West")]      # 90 cells west, on the flat
    two_wd = run(stage(tmp_path, elevation, routes), 10, 100)["exits"][0]
    assert two_wd["kind"] == "2wd" and two_wd["reachable"] is True
    assert cell_of(two_wd["exit_point"]) == (10, 10)
    assert two_wd["gain_m"] < 5


def test_carry_out_avoids_a_grade_walk_out_takes(tmp_path):
    elevation = np.zeros((70, 60))
    elevation[:50, 30:] = 7.0                     # a 70 % step (49 % diagonally) at col 30
    elevation[50:, :] = np.clip((np.arange(60) - 20) * 0.35, 0.0, 7.0)   # a 3.5 % ramp south
    directory = stage(tmp_path, elevation, [north_south("ROAD", 50, 70)])
    walk = run(directory, 5, 25)["exits"][0]
    carry = run(directory, 5, 25, mode="carry_out")["exits"][0]
    assert walk["max_grade_pct"] > 40                           # straight over the step
    assert carry["reachable"] is True and carry["max_grade_pct"] <= 40
    assert carry["distance_m"] > walk["distance_m"] + 300      # round by the ramp


def test_a_trail_is_used_when_it_is_quicker(tmp_path):
    # Off trail straight east is 50 cells at 0.6; the trail doglegs 10 north
    # then 50 east at 1.0 - 60 cells, but quicker.
    routes = [way("TRAIL", (20, 10), (10, 10), (10, 60), highway="path"),
              north_south("ROAD", 60, 30)]
    two_wd = run(stage(tmp_path, np.zeros((30, 70)), routes), 22, 10)["exits"][0]
    assert two_wd["on_trail_share"] > 0.8
    assert cell_of(two_wd["exit_point"])[0] <= 11


def test_both_kinds_always_come_back(tmp_path):
    exits = run(stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)]),
                10, 10)["exits"]
    assert [e["kind"] for e in exits] == ["2wd", "4x4"]
    assert exits[0]["reachable"] is True
    assert set(exits[1]) == {"kind", "reachable", "reason"}
    assert exits[1]["reachable"] is False
    assert exits[1]["reason"] == ("no track or dirt road a 4x4 can reach is mapped in the "
                                  "staged area; a wider radius may reach one")


def test_ways_that_are_no_exit_are_not_called_no_road(tmp_path):
    # Roads are mapped here - a pedestrian street and a busway - but neither
    # is an exit, so the reason may not say none is mapped at all.
    routes = [north_south("ROAD", 30, 21, highway="pedestrian"),
              north_south("ROAD_OTHER", 40, 21, highway="busway")]
    exits = run(stage(tmp_path, np.zeros((21, 60)), routes), 10, 10)["exits"]
    assert exits[0]["reason"] == ("no road or parking lot a 2WD vehicle can reach is mapped "
                                  "in the staged area; a wider radius may reach one")
    assert exits[1]["reason"].startswith("no track or dirt road a 4x4 can reach is mapped")


def _masks(tmp_path, routes, parking=()):
    directory = stage(tmp_path, np.zeros((20, 20)), routes, parking)
    return _way_masks(load_assignment_grid(directory), list(routes), list(parking))


def test_a_stacked_surface_is_dirt_when_any_part_is(tmp_path):
    # OSM stacks surfaces with ";". Read cautiously: any dirt part may need a 4x4.
    masks = _masks(tmp_path, [
        way("ROAD", (2, 0), (2, 19), highway="unclassified", surface="dirt;gravel"),
        way("ROAD", (6, 0), (6, 19), highway="unclassified", surface=" Ground ; gravel"),
        way("ROAD", (10, 0), (10, 19), highway="unclassified", surface="asphalt;gravel"),
        way("ROAD", (14, 0), (14, 19), highway="unclassified", surface="gravel;dirt")])
    for row in (2, 6, 14):
        assert masks["four_wd"][row].all() and not masks["two_wd"][row].any()
    assert masks["two_wd"][10].all() and not masks["four_wd"][10].any()


def test_a_dirt_road_and_a_track_are_4x4_and_a_sealed_or_untagged_road_is_2wd(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (2, 0), (2, 19), highway="unclassified", surface="Dirt"),
        way("TRACK", (6, 0), (6, 19), highway="track"),
        way("ROAD", (10, 0), (10, 19), highway="residential"),
        way("ROAD", (14, 0), (14, 19), highway="tertiary", surface="gravel")])
    assert masks["four_wd"][2].all() and not masks["two_wd"][2].any()
    assert masks["four_wd"][6].all() and not masks["two_wd"][6].any()
    assert masks["two_wd"][10].all() and not masks["four_wd"][10].any()
    assert masks["two_wd"][14].all() and not masks["four_wd"][14].any()
    assert all(masks["on_way"][r].all() for r in (2, 6, 10, 14))
    assert not masks["on_way"][0].any() and not masks["impassable"].any()


def test_only_a_road_is_a_2wd_exit_but_other_highways_are_still_walked(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD_OTHER", (2, 0), (2, 19), highway="busway"),
        way("ROAD_OTHER", (5, 0), (5, 19), highway="rest_area"),
        way("ROAD", (8, 0), (8, 19), highway="pedestrian"),
        way("ROAD", (11, 0), (11, 19), highway="pedestrian", surface="dirt"),
        way("ROAD", (14, 0), (14, 19), highway="residential")])
    for row in (2, 5, 8, 11):
        assert masks["on_way"][row].all()                       # faster ground to walk
        assert not masks["two_wd"][row].any() and not masks["four_wd"][row].any()
    assert masks["two_wd"][14].all()


def test_a_pedestrian_street_nearer_than_a_road_is_walked_past_not_exited_at(tmp_path):
    routes = [north_south("ROAD", 20, 21, highway="pedestrian"),   # 10 cells east
              north_south("ROAD", 40, 21)]                         # 30 cells east
    two_wd = run(stage(tmp_path, np.zeros((21, 60)), routes), 10, 10)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 40)


def test_the_assumptions_say_what_is_an_exit_and_what_is_closed(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    assumptions = run(directory, 10, 10)["assumptions"]
    for words in ("ROAD", "pedestrian", "motorway", "parking lot", "hardstand", "ROAD_OTHER"):
        assert words in assumptions["two_wd"]
    assert "bridge" in assumptions["closed"]
    assert "ramp" in assumptions["closed"] and "at grade" in assumptions["closed"]
    for words in ("proposed", "via_ferrata", "bus_guideway"):
        assert words in assumptions["ignored_ways"]


def test_a_parking_lot_and_a_parking_node_are_2wd_but_a_hardstand_is_not(tmp_path):
    masks = _masks(tmp_path, [], [lot(3, 3, 6, 6), parking_node(15, 15),
                                  lot(10, 10, 12, 12, kind="hardstand")])
    assert masks["two_wd"][3:7, 3:7].all() and masks["two_wd"].sum() == 16 + 1
    assert masks["two_wd"][15, 15]
    assert not masks["two_wd"][10:13, 10:13].any()
    assert not masks["on_way"].any()                    # a lot is an exit, not a way


def test_a_motorway_is_neither_kind_and_closed_but_a_road_across_it_is_open(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (0, 10), (19, 10), highway="motorway"),
        way("ROAD", (0, 15), (19, 15), highway="motorway_link"),
        way("ROAD", (5, 0), (5, 19), highway="residential")])     # a bridge over both
    for col in (10, 15):
        # Only the crossing's own neighbourhood opens: one cell round the point.
        assert not masks["impassable"][5, col]
        assert masks["impassable"][:4, col].all() and masks["impassable"][7:, col].all()
        assert not masks["two_wd"][:, col].any() and not masks["four_wd"][:, col].any()
        assert not masks["on_way"][0, col] and masks["on_way"][5, col]


def test_a_trail_that_ends_on_a_motorway_or_runs_beside_it_opens_nothing(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (0, 10), (19, 10), highway="motorway"),
        way("TRAIL", (5, 0), (5, 10), highway="path"),            # dead-ends on its line
        way("TRAIL", (0, 10.3), (19, 10.3), highway="path")])     # beside it, in its cells
    assert masks["impassable"][:, 10].all()


def test_a_trail_that_only_touches_a_motorway_does_not_lead_across_it(tmp_path):
    # Across the motorway the road is 10 cells off, half of them on the trail;
    # the road behind is 25 cells off trail. Only a real crossing would win.
    routes = [north_south("ROAD", 30, 21, highway="motorway"),
              way("TRAIL", (10, 20), (10, 30), highway="path"),   # dead-ends on the motorway
              north_south("ROAD", 35, 21),                         # beyond it
              north_south("ROAD", 0, 21)]                          # 25 cells back west
    two_wd = run(stage(tmp_path, np.zeros((21, 60)), routes), 10, 25)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 0)


def _diamond(ramps):
    """A motorway down col 30, a crossroad along row 20 bridging it, and the
    chosen ramps of a diamond: each runs from the motorway to a terminal on
    the crossroad at col 26 (west) or col 34 (east). The only 4x4 exit is a
    track far to the east."""
    ends = {"nw": ((5, 30), (20, 26)), "sw": ((35, 30), (20, 26)),
            "ne": ((5, 30), (20, 34)), "se": ((35, 30), (20, 34))}
    return ([north_south("ROAD", 30, 41, highway="motorway"),
             way("ROAD", (20, 0), (20, 50), highway="secondary", name="Crossroad"),
             north_south("TRACK", 55, 41, highway="track")]
            + [way("ROAD", *ends[ramp], highway="motorway_link") for ramp in ramps])


@pytest.mark.parametrize("ramps", [("nw", "sw", "ne", "se"), ("nw",)])
def test_a_diamond_interchange_can_be_walked_across(tmp_path, ramps):
    # The ramps' terminals sit on the crossroad at grade; without them open,
    # four ramps wall the crossroad off at cols 26 and 34.
    directory = stage(tmp_path, np.zeros((41, 60)), _diamond(ramps))
    four_wd = run(directory, 20, 5)["exits"][1]
    assert four_wd["reachable"] is True and cell_of(four_wd["exit_point"]) == (20, 55)


def test_a_ramp_ending_a_hair_off_its_road_still_meets_it(tmp_path):
    # A shared OSM node, reprojected: the ramp's end lands 5e-7 units from
    # the crossroad's line rather than exactly on it.
    ramp = way("ROAD", (0, 15), (10, 15), highway="motorway_link")
    ramp["geometry"]["coordinates"][-1][1] += 5e-7
    masks = _masks(tmp_path, [ramp, way("ROAD", (10, 0), (10, 19), highway="secondary")])
    assert not masks["impassable"][10, 15] and masks["impassable"][:8, 15].all()


def test_a_trail_that_dead_ends_on_a_ramp_does_not_open_it(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (0, 10), (19, 10), highway="motorway_link"),  # its ends touch nothing
        way("TRAIL", (5, 0), (5, 10), highway="path")])           # ends on its middle
    assert masks["impassable"][:, 10].all()


def test_only_a_ramp_end_opens_where_it_meets_a_road_not_a_main_line_end(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (0, 5), (10, 5), highway="motorway"),         # ends on the road below
        way("ROAD", (0, 15), (10, 15), highway="motorway_link"),  # so does this ramp
        way("ROAD", (10, 0), (10, 19), highway="secondary")])
    assert masks["impassable"][10, 5] and masks["impassable"][9, 5]
    assert not masks["impassable"][10, 15] and not masks["impassable"][9, 15]
    assert masks["impassable"][:8, 15].all()


def test_a_track_over_a_motorway_band_is_no_4x4_exit_there(tmp_path):
    masks = _masks(tmp_path, [
        way("ROAD", (0, 10), (19, 10), highway="motorway"),
        way("TRACK", (0, 10.3), (19, 10.3), highway="track"),     # burned over the band
        way("TRACK", (8, 0), (8, 19), highway="track")])          # and across it
    assert not masks["four_wd"][:, 10].any()
    assert masks["four_wd"][8, :10].all() and masks["four_wd"][8, 11:].all()


def test_a_via_ferrata_and_a_guideway_are_plain_ground(tmp_path):
    masks = _masks(tmp_path, [way("ROAD_OTHER", (5, 0), (5, 19), highway="via_ferrata"),
                              way("ROAD_OTHER", (9, 0), (9, 19), highway="bus_guideway")])
    assert not any(mask.any() for mask in masks.values())


def test_a_way_marks_every_cell_it_crosses(tmp_path):
    from shapely.geometry import LineString, box
    # An oblique line that misses most cell centres, in cell units (row, col).
    start, end = (3.3, 0.2), (9.6, 19.7)
    record = way("TRAIL", (0, 0), (0, 1), highway="path")
    record["geometry"]["coordinates"] = [
        [ORIGIN[0] + c * PIXEL, ORIGIN[1] - r * PIXEL] for r, c in (start, end)]
    masks = _masks(tmp_path, [record])
    line = LineString([(c, r) for r, c in (start, end)])
    crossed = np.array([[line.intersects(box(c, r, c + 1, r + 1)) for c in range(20)]
                        for r in range(20)])
    assert (masks["on_way"] == crossed).all()


def test_a_road_that_is_not_built_is_no_way_at_all(tmp_path):
    masks = _masks(tmp_path, [way("ROAD_OTHER", (5, 0), (5, 19), highway="proposed"),
                              way("ROAD_OTHER", (9, 0), (9, 19), highway="construction")])
    assert not any(mask.any() for mask in masks.values())


def test_a_motorway_is_never_an_exit_and_never_crossed(tmp_path):
    routes = [north_south("ROAD", 40, 21, highway="motorway"),
              north_south("ROAD", 45, 21),                     # 10 cells east, beyond it
              north_south("ROAD", 5, 21)]                      # 30 cells west
    two_wd = run(stage(tmp_path, np.zeros((21, 80)), routes), 10, 35)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 5)


def test_a_location_on_a_road_is_its_own_exit(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 40)), [north_south("ROAD", 20, 21)])
    two_wd = run(directory, 10, 20)["exits"][0]
    assert two_wd["minutes"] == 0 and two_wd["distance_m"] == 0
    assert two_wd["route"]["type"] == "Point" and two_wd["route"] == two_wd["exit_point"]
    assert two_wd["gain_m"] == 0 and two_wd["loss_m"] == 0


def test_the_route_runs_from_the_location_to_the_exit_and_measures_its_line(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    two_wd = run(directory, 10, 10)["exits"][0]
    line = two_wd["route"]
    assert line["type"] == "LineString" and len(line["coordinates"]) == 2   # simplified
    start = {"type": "Point", "coordinates": line["coordinates"][0]}
    assert cell_of(start) == (10, 10)
    assert line["coordinates"][-1] == pytest.approx(two_wd["exit_point"]["coordinates"])
    assert cell_of(two_wd["exit_point"]) == (10, 50)
    assert two_wd["distance_m"] == pytest.approx(400.0, abs=CELL_M)
    # Forty flat off-trail steps into a road cell: 39 at 0.6, the last at 1.0.
    flat_kmh = 6 * math.exp(-0.175)
    assert two_wd["minutes"] == pytest.approx(
        39 * 0.6 / (flat_kmh * 0.6) + 0.6 / flat_kmh, abs=0.05)
    assert set(two_wd) >= {"road", "parking_lot", "gates_crossed", "warnings"}
    assert two_wd["road"]["route_type"] == "ROAD" and two_wd["parking_lot"] is None
    assert two_wd["gates_crossed"] == []


GRID = SimpleNamespace(transform=from_origin(0.0, 0.0, 10.0, 10.0), pixel_size_m=10.0)


def test_gain_and_loss_ignore_a_half_metre_ripple_but_count_a_twenty_metre_rise():
    ripple = [0.0, 0.5] * 5                              # cols 0..9
    rise = [2.0 * k for k in range(1, 11)]               # cols 10..19: 2 .. 20
    top = [20.0, 20.5] * 5                               # cols 20..29
    elevation = np.array([ripple + rise + top])
    no = np.zeros(elevation.shape, dtype=bool)
    up = _measure(GRID, list(range(30)), no, elevation)
    assert up["gain_m"] == pytest.approx(20.0) and up["loss_m"] == 0
    down = _measure(GRID, list(range(29, -1, -1)), no, elevation)
    assert down["gain_m"] == 0 and down["loss_m"] == pytest.approx(20.0)
    flat = _measure(GRID, list(range(10)), no, elevation)
    assert flat["gain_m"] == 0 and flat["loss_m"] == 0


def test_off_trail_and_on_trail_lengths_add_up_to_the_path():
    elevation = np.zeros((1, 41))
    on = np.zeros(elevation.shape, dtype=bool)
    on[0, 20:] = True
    measured = _measure(GRID, list(range(41)), on, elevation)
    # 40 steps of 10 m; their destinations are cols 1..40, and 20..40 (21) are on the way.
    assert measured["off_trail_m"] == pytest.approx(190.0)
    assert measured["on_trail_share"] == pytest.approx(210.0 / 400.0, abs=0.001)
    assert measured["off_trail_m"] + measured["on_trail_share"] * 400.0 == pytest.approx(400.0, abs=0.5)
    assert measured["distance_m"] == pytest.approx(400.0)


def test_max_grade_is_the_steepest_step_as_a_percentage():
    elevation = np.zeros((3, 5))
    elevation[0, 1] = 3.0            # (0,0) -> (0,1): 30 %
    elevation[1, 2] = 8.0            # (0,1) -> (1,2): 5 m over a diagonal, 35.4 %
    elevation[1, 3] = 8.0
    elevation[2, 4] = 1.0            # (1,3) -> (2,4): DOWN 7 m over a diagonal, 49.5 %
    path = [0, 1, 1 * 5 + 2, 1 * 5 + 3, 2 * 5 + 4]
    measured = _measure(GRID, path, np.zeros(elevation.shape, dtype=bool), elevation)
    assert measured["max_grade_pct"] == pytest.approx(100 * 7 / (10 * math.sqrt(2)), abs=0.05)
    climb = _measure(GRID, path[:4], np.zeros(elevation.shape, dtype=bool), elevation)
    assert climb["max_grade_pct"] == pytest.approx(100 * 5 / (10 * math.sqrt(2)), abs=0.05)


def test_distance_is_measured_on_the_simplified_line_not_the_cell_path():
    # A staircase of straight steps east and south: 20 steps, 200 m of cell
    # path, but every centre lies within a cell of the diagonal, so the
    # simplified line is the diagonal - 10 x 14.1 m.
    elevation = np.zeros((11, 11))
    path = [0]
    for k in range(10):
        path += [k * 11 + k + 1, (k + 1) * 11 + k + 1]
    measured = _measure(GRID, path, np.zeros(elevation.shape, dtype=bool), elevation)
    assert measured["distance_m"] == pytest.approx(100 * math.sqrt(2), abs=CELL_M)
    assert measured["off_trail_m"] == pytest.approx(200.0)


def test_a_rise_of_exactly_one_metre_counts():
    elevation = np.array([[0.0, 1.0, 0.0]])
    no = np.zeros(elevation.shape, dtype=bool)
    measured = _measure(GRID, [0, 1, 2], no, elevation)
    assert measured["gain_m"] == 1.0 and measured["loss_m"] == 1.0


def test_a_litter_location_on_a_trail_across_steep_ground_is_carried_along_it(tmp_path):
    elevation = np.tile((np.arange(21) * 6.0)[:, None], (1, 60))     # 60 % sidehill
    routes = [way("TRAIL", (10, 0), (10, 50), highway="path"),
              north_south("ROAD", 50, 21)]
    two_wd = run(stage(tmp_path, elevation, routes), 10, 10, mode="carry_out")["exits"][0]
    assert two_wd["reachable"] is True and cell_of(two_wd["exit_point"]) == (10, 50)


def test_a_cliff_band_cuts_a_litter_off_but_not_a_walker(tmp_path):
    elevation = np.zeros((21, 60))
    elevation[:, 30:] = 10.0
    directory = stage(tmp_path, elevation, [north_south("ROAD", 50, 21)])
    carry = run(directory, 10, 10, mode="carry_out")["exits"][0]
    assert carry["reachable"] is False
    assert "litter" in carry["reason"] and "40% grade limit" in carry["reason"]
    assert run(directory, 10, 10)["exits"][0]["reachable"] is True


def test_a_walk_out_with_a_grade_limit_is_not_told_about_a_litter(tmp_path):
    elevation = np.zeros((21, 60))
    elevation[:, 30:] = 10.0
    directory = stage(tmp_path, elevation, [north_south("ROAD", 50, 21)])
    walk = run(directory, 10, 10, max_grade=0.3)["exits"][0]
    assert walk["reachable"] is False and "litter" not in walk["reason"]
    assert walk["reason"] == ("none can be reached within the 30% grade limit set for this "
                              "search; without the limit one can")


def test_a_location_on_a_void_says_so_for_both_exits(tmp_path):
    elevation = np.zeros((21, 60))
    elevation[10, 10] = np.nan
    routes = [north_south("ROAD", 50, 21), north_south("TRACK", 40, 21, highway="track")]
    directory = stage(tmp_path, elevation, routes)
    for mode in MODES:
        exits = run(directory, 10, 10, mode=mode)["exits"]
        assert [e["reason"] for e in exits] == [VOID_ORIGIN_REASON] * 2
    assert "cut off" not in VOID_ORIGIN_REASON


def test_a_litter_location_on_steep_ground_is_carried_from_the_nearest_ground_it_can(tmp_path):
    elevation = np.tile(np.minimum(np.arange(60), 20) * 6.0, (21, 1))   # 60 % west of col 20
    routes = [north_south("ROAD", 50, 21), north_south("TRACK", 40, 21, highway="track")]
    directory = stage(tmp_path, elevation, routes)
    # Ringed by steep ground: every neighbour is closed to a litter too. The
    # carry starts where the face meets the flat, the nearest ground a litter
    # can leave - nearest in time, so a diagonal up the face's gentler line
    # can beat the straight one - and says the rope evacuation to get there.
    exits = run(directory, 10, 10, mode="carry_out")["exits"]
    assert [e["reachable"] for e in exits] == [True, True]
    for exit_ in exits:
        start = exit_["litter_start"]
        assert 19 <= cell_of(start["point"])[1] <= 21
        assert 95 <= start["distance_m"] <= 150 and 0 < start["bearing_deg"] < 180
        assert start["drop_m"] < -50                    # the flat is up on top of the face
        assert any("rope evacuation" in warning and f"{start['distance_m']:.0f} m away"
                   in warning for warning in exit_["warnings"])
        # The route, and its minutes, start at the litter start, not the location.
        assert cell_of({"coordinates": exit_["route"]["coordinates"][0]}) == cell_of(
            start["point"])
    assert cell_of(exits[0]["exit_point"])[1] == 50
    assert all(e["reachable"] and e["litter_start"] is None
               for e in run(directory, 10, 10)["exits"])
    # A walk-out with its own grade limit is told about that limit, not a litter.
    limited = run(directory, 10, 10, max_grade=0.4)["exits"][0]
    assert limited["reason"].startswith("none can be reached within the 40% grade limit")
    # With no track mapped, "none is mapped" is the truer reason for the 4x4.
    (tmp_path / "no_track").mkdir()
    no_track = stage(tmp_path / "no_track", elevation, routes[:1])
    exits = run(no_track, 10, 10, mode="carry_out")["exits"]
    assert exits[0]["reachable"] is True and exits[0]["litter_start"] is not None
    assert exits[1]["reason"].startswith("no track or dirt road")


def test_a_litter_that_no_ground_near_it_can_carry_out_needs_a_rope_or_a_helicopter(tmp_path):
    # The same face, with the road beyond a gap in the elevation model: no
    # cell the location can reach lets a litter reach the road either.
    elevation = np.tile(np.minimum(np.arange(60), 20) * 6.0, (21, 1))
    elevation[:, 30] = np.nan
    directory = stage(tmp_path, elevation, [north_south("ROAD", 50, 21)])
    two_wd = run(directory, 10, 10, mode="carry_out")["exits"][0]
    assert two_wd["reason"] == STEEP_ORIGIN_REASON


def test_a_litter_at_the_foot_of_a_bank_is_carried_off_it(tmp_path):
    # A 9 m bank rises east of col 20: the location's own cell reads
    # (9 - 0) / 20 m = 45 % and is closed to a litter, but a closed cell has
    # steps OUT, and one cell west the ground is flat.
    elevation = np.zeros((21, 60))
    elevation[:, 21:] = 9.0
    routes = [north_south("ROAD", 5, 21), north_south("TRACK", 2, 21, highway="track")]
    exits = run(stage(tmp_path, elevation, routes), 10, 20, mode="carry_out")["exits"]
    assert [e["reachable"] for e in exits] == [True, True]
    assert cell_of(exits[0]["exit_point"]) == (10, 5)
    assert cell_of(exits[1]["exit_point"]) == (10, 2)


def test_a_litter_that_can_leave_a_steep_cell_is_told_the_real_reason(tmp_path):
    # The same bank, with the only track up on top of it: the litter leaves
    # the steep cell westward and reaches the road, so the track's reason is
    # the grade limit - never a rope evacuation it has just been carried out of.
    elevation = np.zeros((21, 60))
    elevation[:, 21:] = 9.0
    routes = [north_south("ROAD", 5, 21), north_south("TRACK", 50, 21, highway="track")]
    exits = run(stage(tmp_path, elevation, routes), 10, 20, mode="carry_out")["exits"]
    assert exits[0]["reachable"] is True
    assert exits[1]["reason"].startswith("a litter cannot reach one within the 40% grade limit")


def test_ground_cut_off_by_a_void_says_so(tmp_path):
    elevation = np.zeros((21, 60))
    elevation[:, 30] = np.nan
    # The road runs down column 0, so it holds cell 0 - where argmin lands
    # when every cell is infinite.
    directory = stage(tmp_path, elevation, [north_south("ROAD", 0, 21)])
    for mode in MODES:
        two_wd = run(directory, 10, 50, mode=mode)["exits"][0]
        assert two_wd["reachable"] is False and "cut off" in two_wd["reason"]


def test_the_ground_factors_can_be_overridden(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    default = run(directory, 10, 10)["exits"][0]["minutes"]
    slower = run(directory, 10, 10, off_way_factor=0.3)["exits"][0]["minutes"]
    # 39 of the 40 steps are off the way; the last is onto the road.
    assert slower - default == pytest.approx(39 * 0.6 / (6 * math.exp(-0.175) * 0.6), abs=0.1)


def test_no_osm_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), osm=False)
    with pytest.raises(ValueError, match="no OpenStreetMap data is staged") as refused:
        run(directory, 10, 10)
    assert str(refused.value) == NO_OSM_ERROR


@pytest.mark.parametrize("bad, words", [
    ({"on_way_factor": 0.0}, "on_way_factor"),
    ({"off_way_factor": -0.1}, "off_way_factor"),
    ({"max_grade": -0.01}, "max_grade"),
    ({"mode": "drag_out"}, "walk_out"),
])
def test_bad_settings_are_refused(tmp_path, bad, words):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    with pytest.raises(ValueError, match=words):
        run(directory, 10, 10, **bad)


def test_a_location_off_the_staged_ground_is_refused_not_snapped(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    with pytest.raises(ValueError, match="outside the staged ground"):
        run(directory, 10, -30)
    assert run(directory, 10, 0)["exits"][0]["reachable"] is True     # the edge is fine


def test_the_envelope(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    result = run(directory, 10, 10, mode="carry_out")
    assert set(result) >= {"exits", "mode", "center", "resolution_m", "assumptions", "warnings"}
    assert result["mode"] == "carry_out" and result["resolution_m"] == pytest.approx(10.0)
    assert result["assumptions"]["max_grade"] == 0.40
    assert set(result["assumptions"]) >= {"tobler", "signed_grade", "ground_factors",
                                          "max_grade", "cell_m", "two_wd", "four_wd",
                                          "not_a_promise", "gates_and_access"}
    assert result["assumptions"]["gates_and_access"] == (
        "said on each exit, never excluded or avoided: a gate or a private road is a "
        "call the planner can make - for the combination, or to the ranger. A gate is "
        "listed as crossed when it stands on a way the route runs through for two or "
        "more cells. A trail the route crosses at a shallow angle (under about 45 "
        "degrees) can meet that rule, so a gate on such a trail near the crossing may "
        "be listed although the team does not pass through it")
    assert result["warnings"] == []                    # current ground, gates staged


# --- Task 5: what an exit ends on --------------------------------------------


def test_the_road_is_the_nearest_way_of_the_exits_own_kind(tmp_path):
    skyline = way("ROAD", (0, 40), (20, 40), highway="secondary", surface="asphalt",
                  name="Skyline Boulevard")
    skyline["properties"]["access"] = "destination"
    routes = [skyline,
              way("ROAD", (0, 41), (20, 41), highway="service", name="Frontage Road"),
              way("TRACK", (10, 40), (10, 59), highway="track", name="Ridge Fire Road")]
    exits = run(stage(tmp_path, np.zeros((21, 60)), routes), 10, 10)["exits"]
    # Both exits are the same cell, where the track leaves the road.
    assert cell_of(exits[0]["exit_point"]) == cell_of(exits[1]["exit_point"]) == (10, 40)
    assert exits[0]["road"] == {"name": "Skyline Boulevard", "route_type": "ROAD",
                                "highway": "secondary", "surface": "asphalt",
                                "access": "destination", "gate_on_road": False}
    assert exits[1]["road"]["name"] == "Ridge Fire Road"
    assert exits[1]["road"]["route_type"] == "TRACK"


def test_two_roads_equally_near_the_exit_are_told_apart_by_name(tmp_path):
    routes = [way("ROAD", (10, 40), (10, 59), highway="residential", name="Alpha Road"),
              north_south("ROAD", 40, 21, name="Zeta Road")]
    two_wd = run(stage(tmp_path, np.zeros((21, 60)), routes), 10, 10)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 40)       # where the two meet
    assert two_wd["road"]["name"] == "Alpha Road"


def test_a_gate_anywhere_on_the_exit_road_is_said(tmp_path):
    routes = [north_south("ROAD", 40, 21, name="Gated Road"),
              north_south("TRACK", 50, 21, highway="track", name="Open Track")]
    gates = [gate(0.4, 40, access="private"),      # ON the road, far from the exit
             gate(10, 45),                         # on neither way
             gate(3, 50.1)]                        # 1 m off the track's line: not on it
    exits = run(stage(tmp_path, np.zeros((21, 60)), routes, gates=gates), 10, 10)["exits"]
    assert exits[0]["road"]["gate_on_road"] is True
    assert exits[1]["road"]["gate_on_road"] is False


def _driveway_ground(tmp_path):
    """Skyline down col 40; a service driveway leaving it one row below the
    exit and running east; a gate 7 m up the driveway. The route comes in
    from the west along row 10 and ends on Skyline at (10, 40)."""
    routes = [north_south("ROAD", 40, 21, name="Skyline Boulevard"),
              way("ROAD", (11, 40), (11, 50), highway="service", name="Driveway")]
    return stage(tmp_path, np.zeros((21, 60)), routes, gates=[gate(11, 40.7)])


def test_a_gate_up_a_side_driveway_is_not_on_the_exit_road(tmp_path):
    two_wd = run(_driveway_ground(tmp_path), 10, 10)["exits"][0]
    assert two_wd["road"]["name"] == "Skyline Boulevard"
    assert two_wd["road"]["gate_on_road"] is False


def test_a_gate_up_a_side_driveway_is_not_crossed(tmp_path):
    # The gate is 1.2 cells from the route's line, inside the 1.5-cell reach,
    # but it bars the driveway, and the route never sets foot on the driveway.
    two_wd = run(_driveway_ground(tmp_path), 10, 10)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 40)
    assert two_wd["gates_crossed"] == []


def test_the_gates_on_a_way_the_route_walks_along_are_listed(tmp_path):
    # The route follows a trail along row 10 to the road. A lift gate stands
    # on the trail; another stands on a two-cell stub of path the route also
    # walks (cols 34 and 35); a third stands on no way at all.
    routes = [north_south("ROAD", 50, 21),
              way("TRAIL", (10, 10), (10, 30), highway="path"),
              way("TRAIL", (10, 34), (10, 35), highway="footway")]
    gates = [gate(10, 20.2, barrier="lift_gate", access="private"),
             gate(10, 34.7, barrier="swing_gate"),
             gate(11.2, 25)]
    directory = stage(tmp_path, np.zeros((21, 60)), routes, gates=gates)
    crossed = run(directory, 10, 10)["exits"][0]["gates_crossed"]
    assert [g["barrier"] for g in crossed] == ["lift_gate", "swing_gate"]
    assert crossed[0]["access"] == "private" and crossed[0]["geometry"]["type"] == "Point"
    lon, lat = TO4326.transform(*xy(10, 20.2))
    assert crossed[0]["geometry"]["coordinates"] == pytest.approx([lon, lat])


def test_a_gate_up_a_driveway_off_the_route_is_not_crossed(tmp_path):
    # The route follows a trail along row 10 to a track at col 50; a service
    # driveway leaves the trail at col 30 and runs south, gated 4 m, 7 m and
    # 12 m in. Walking past the junction is not walking the driveway.
    routes = [way("TRAIL", (10, 10), (10, 50), highway="path"),
              north_south("TRACK", 50, 21, highway="track"),
              way("ROAD", (10, 30), (20, 30), highway="service", name="Driveway")]
    gates = [gate(10.4, 30), gate(10.7, 30), gate(11.2, 30)]
    exits = run(stage(tmp_path, np.zeros((21, 60)), routes, gates=gates), 10, 10)["exits"]
    assert cell_of(exits[1]["exit_point"]) == (10, 50)       # the 4x4 walks past it
    assert exits[1]["gates_crossed"] == []
    assert exits[0]["gates_crossed"] == []                   # the 2WD ends at its mouth


def test_a_gate_on_a_trail_the_route_only_crosses_is_not_crossed(tmp_path):
    # A trail runs north-south across the route's line; its gate is 12 m
    # from the crossing. The route steps over the trail and never follows it.
    routes = [north_south("ROAD", 50, 21), north_south("TRAIL", 30, 21, highway="path")]
    directory = stage(tmp_path, np.zeros((21, 60)), routes,
                      gates=[gate(11.2, 30, barrier="lift_gate")])
    assert run(directory, 10, 10)["exits"][0]["gates_crossed"] == []


def test_an_exit_in_a_parking_lot_names_the_lot_and_the_road_it_opens_onto(tmp_path):
    routes = [north_south("ROAD", 55, 21, name="Skyline Boulevard")]
    two_wd = run(stage(tmp_path, np.zeros((21, 60)), routes, [lot(8, 40, 12, 44)]),
                 10, 10)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 40)
    assert two_wd["parking_lot"] == {"id": "parking-w840", "off": "Skyline Boulevard"}
    assert two_wd["road"] is None                   # no road within a cell of the lot's edge


def _rect_lot(lot_id, r0, c0, r1, c1):
    """A parking polygon from (r0, c0) to (r1, c1) in fractional cell units."""
    x0, x1 = ORIGIN[0] + c0 * PIXEL, ORIGIN[0] + c1 * PIXEL
    y0, y1 = ORIGIN[1] - r0 * PIXEL, ORIGIN[1] - r1 * PIXEL
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {"id": lot_id, "kind": "parking",
            "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}}


def test_of_two_lots_touching_the_exit_cell_the_nearest_its_centre_wins(tmp_path):
    # The exit is cell (10, 40), centred at (10.5, 40.5). The lot listed
    # first only clips its south-east corner; the second holds its centre.
    corner = _rect_lot("parking-w1", 10.8, 40.8, 14.0, 45.0)
    holding = _rect_lot("parking-w2", 8.25, 40.25, 10.6, 44.75)
    routes = [north_south("ROAD", 55, 21, name="Skyline Boulevard")]
    two_wd = run(stage(tmp_path, np.zeros((21, 60)), routes, [corner, holding]),
                 10, 10)["exits"][0]
    assert cell_of(two_wd["exit_point"]) == (10, 40)
    assert two_wd["parking_lot"]["id"] == "parking-w2"


def test_version_4_ground_leaves_road_class_access_and_gates_unknown(tmp_path):
    road = version_4(north_south("ROAD", 40, 21, name="Old Road"))
    result = run(stage(tmp_path, np.zeros((21, 60)), [road], gates=None, query_version=4),
                 10, 10)
    two_wd = result["exits"][0]
    assert two_wd["road"]["name"] == "Old Road"
    assert two_wd["road"]["highway"] is None and two_wd["road"]["access"] is None
    assert two_wd["road"]["gate_on_road"] is None and two_wd["gates_crossed"] == []
    [warning] = [w for w in result["warnings"] if w != NO_EVT_WARNING]
    assert NO_EVT_WARNING in result["warnings"]          # walk-out reads vegetation
    assert "query version 4" in warning and "road class, access and gates" in warning
    assert "unknown" in warning and "motorway" in warning


def test_a_version_5_file_without_a_gate_list_leaves_gates_unknown_not_absent(tmp_path):
    routes = [north_south("ROAD", 40, 21, name="New Road")]
    result = run(stage(tmp_path, np.zeros((21, 60)), routes, gates=None), 10, 10)
    two_wd = result["exits"][0]
    assert two_wd["road"]["highway"] == "residential"          # road class is known
    assert two_wd["road"]["gate_on_road"] is None
    [warning] = [w for w in result["warnings"] if w != NO_EVT_WARNING]
    assert NO_EVT_WARNING in result["warnings"]          # walk-out reads vegetation
    assert "gates are unknown" in warning and "query version" not in warning
    # A plain reuse keeps a version-5 file as it is; only a forced re-stage
    # fetches it again with gates.
    assert "force=True" in warning


def test_version_2_ground_says_parking_is_unknown_not_absent(tmp_path):
    track = version_4(north_south("TRACK", 40, 21, highway="track"))
    result = run(stage(tmp_path, np.zeros((21, 60)), [track], parking=None, gates=None,
                       query_version=2), 10, 10)
    two_wd, four_wd = result["exits"]
    assert two_wd["reachable"] is False
    assert "parking lots are unknown" in two_wd["reason"]
    assert not two_wd["reason"].startswith("no road or parking lot")
    assert four_wd["reachable"] is True and four_wd["parking_lot"] == "unknown"
    assert any("parking" in w and "unknown" in w and "query version 2" in w
               for w in result["warnings"])


def test_version_3_ground_reads_lots_as_known(tmp_path):
    road = version_4(north_south("ROAD", 40, 21))
    result = run(stage(tmp_path, np.zeros((21, 60)), [road], gates=None, query_version=3),
                 10, 10)
    assert result["exits"][0]["parking_lot"] is None
    assert not any("parking" in w for w in result["warnings"])


def test_a_route_more_than_half_off_trail_says_so(tmp_path):
    (tmp_path / "a").mkdir()
    off = run(stage(tmp_path / "a", np.zeros((21, 60)), [north_south("ROAD", 50, 21)]),
              10, 10)["exits"][0]
    [warning] = off["warnings"]          # 39 of its 40 steps are off any way: 390 m
    assert "off any mapped way" in warning and "390 m" in warning
    routes = [way("TRAIL", (20, 10), (10, 10), (10, 60), highway="path"),
              north_south("ROAD", 60, 30)]
    (tmp_path / "b").mkdir()
    trail = run(stage(tmp_path / "b", np.zeros((30, 70)), routes), 22, 10)["exits"][0]
    assert trail["warnings"] == []


@pytest.mark.parametrize("share, warned", [(0.49, True), (0.5, False), (0.51, False)])
def test_the_off_trail_warning_starts_past_half(share, warned):
    from shapely.geometry import Point
    measured = {"line": Point(0, 0), "end": Point(0, 0), "distance_m": 100.0,
                "gain_m": 0.0, "loss_m": 0.0, "on_trail_share": share,
                "off_trail_m": round(100.0 * (1 - share), 1), "max_grade_pct": 0.0}
    exit_ = _exit("2wd", 5.0, measured, road=None, parking_lot=None, gates_crossed=[])
    assert bool(exit_["warnings"]) is warned


def test_ground_staged_too_fine_for_egress_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    with pytest.raises(ValueError, match="too fine") as refused:
        run(directory, 10, 10, max_cells=21 * 60 - 1)
    assert "resolution_m=10.0" in str(refused.value)
    assert run(directory, 10, 10, max_cells=21 * 60)["exits"][0]["reachable"] is True
    assert EGRESS_MAX_CELLS == 1800 * 1800


def test_an_800_cell_square_is_solved_well_inside_twenty_seconds(tmp_path):
    # Guards the default 3,000 m radius plus the 1,000 m planning buffer at
    # 10 m cells: 800 cells a side, the ground egress stages by default.
    directory = stage(tmp_path, np.zeros((800, 800)), [north_south("ROAD", 700, 800)])
    started = time.perf_counter()
    result = run(directory, 400, 100)
    elapsed = time.perf_counter() - started
    assert result["exits"][0]["reachable"] is True
    assert elapsed < 20.0, f"{elapsed:.1f} s"


def test_an_800_cell_carry_out_that_needs_a_second_graph_is_inside_twenty_seconds(tmp_path):
    # The same 800-cell ground, carried out, with the 4x4 exit behind a cliff
    # band: the grade-limit reason builds and solves a second, unlimited graph.
    elevation = np.zeros((800, 800))
    elevation[:, 740:] = 30.0
    routes = [north_south("ROAD", 700, 800), north_south("TRACK", 780, 800, highway="track")]
    directory = stage(tmp_path, elevation, routes)
    started = time.perf_counter()
    result = run(directory, 400, 100, mode="carry_out")
    elapsed = time.perf_counter() - started
    assert result["exits"][0]["reachable"] is True
    assert "litter" in result["exits"][1]["reason"]
    assert elapsed < 20.0, f"{elapsed:.1f} s"


@pytest.mark.parametrize("broken", ["dem missing", "subject missing", "dem corrupt"])
def test_a_broken_staged_directory_reaches_the_caller_without_its_path(tmp_path, broken):
    directory = stage(tmp_path, np.zeros((21, 60)), [north_south("ROAD", 50, 21)])
    if broken == "dem missing":
        (directory / "dem.tif").unlink()
    elif broken == "subject missing":
        (directory / "assignment.json").unlink()
    else:
        (directory / "dem.tif").write_bytes(b"this is not a GeoTIFF")
    with pytest.raises(OSError) as refused:
        run(directory, 10, 10)
    message = str(refused.value)
    assert str(directory) not in message and directory.name not in message, message
    assert refused.value.__cause__ is None and refused.value.__context__ is None


# --- walk-out walks the shared walking ground -----------------------------------


def _sub(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def evt(directory, codes):
    """An evt.tif at 30 m whose pixels cover 3 x 3 of the 10 m cells."""
    import rasterio

    codes = np.asarray(codes, dtype="int16")[::3, ::3]
    with rasterio.open(directory / "evt.tif", "w", driver="GTiff", height=codes.shape[0],
                       width=codes.shape[1], count=1, dtype="int16", crs="EPSG:3857",
                       transform=from_origin(ORIGIN[0], ORIGIN[1], 3 * PIXEL, 3 * PIXEL)) as dst:
        dst.write(codes, 1)


def waterway_line(kind, col, rows):
    record = way("WATERWAY", (0, col), (rows - 1, col), highway=None, name=f"{kind} {col}")
    record["properties"].update(waterway=kind, tunnel=None)
    return record


def test_walk_out_is_slowed_by_brush_and_carry_out_is_not(tmp_path):
    flat = np.zeros((21, 60))
    road = north_south("ROAD", 40, 21, name="Valley Road")
    grass = stage(_sub(tmp_path, "a"), flat, [road])
    evt(grass, np.full(flat.shape, 4407))                    # Herb
    brush = stage(_sub(tmp_path, "b"), flat, [road])
    evt(brush, np.full(flat.shape, 4401))                    # Shrub
    walk_grass = run(grass, 10, 10)["exits"][0]["minutes"]
    walk_brush = run(brush, 10, 10)["exits"][0]["minutes"]
    assert walk_brush == pytest.approx(walk_grass * 0.9 / 0.4, rel=0.05)
    carry_grass = run(grass, 10, 10, mode="carry_out")["exits"][0]["minutes"]
    carry_brush = run(brush, 10, 10, mode="carry_out")["exits"][0]["minutes"]
    assert carry_brush == pytest.approx(carry_grass)


def test_walk_out_crosses_a_river_only_at_a_bridge(tmp_path):
    flat = np.zeros((41, 60))
    road = north_south("ROAD", 50, 41, name="Far Road")
    river = waterway_line("river", 30, 41)
    shut = run(stage(_sub(tmp_path, "a"), flat, [road, river]), 20, 10)
    two_wd = shut["exits"][0]
    assert two_wd["reachable"] is False
    assert "river, lake, open water" in two_wd["reason"]
    bridge = way("TRAIL", (35, 10), (35, 55), highway="path", name="Bridge Path")
    crossed = run(stage(_sub(tmp_path, "b"), flat, [road, river, bridge]), 20, 10)["exits"][0]
    assert crossed["reachable"] is True
    # Carry-out does not see water: it walks straight across.
    assert run(stage(_sub(tmp_path, "c"), flat, [road, river]), 20, 10,
               mode="carry_out")["exits"][0]["reachable"] is True


def test_walk_out_pays_the_creek_crossing(tmp_path):
    flat = np.zeros((21, 60))
    road = north_south("ROAD", 40, 21, name="Creek Road")
    dry = run(stage(_sub(tmp_path, "a"), flat, [road]), 10, 10)["exits"][0]["minutes"]
    wet = run(stage(_sub(tmp_path, "b"), flat, [road, waterway_line("stream", 25, 21)]),
              10, 10)["exits"][0]["minutes"]
    assert wet == pytest.approx(dry + 5.0, abs=0.2)


def test_walk_out_closes_a_cliff_off_trail(tmp_path):
    ground = np.zeros((21, 60))
    ground[:, 25:] = 40.0                                    # a 40 m step, far over 45 degrees
    road = north_south("ROAD", 40, 21, name="Top Road")
    result = run(stage(tmp_path, ground, [road]), 10, 10)
    assert result["exits"][0]["reachable"] is False
    assert "too steep" in result["exits"][0]["reason"]


def test_a_lakeside_lot_stays_an_exit_on_coarse_open_water(tmp_path):
    flat = np.zeros((21, 60))
    directory = stage(tmp_path, flat, [], parking=[lot(9, 39, 11, 41)])
    codes = np.full(flat.shape, 4407)
    codes[:, 39:] = 7292                 # a lake to the east whose 30 m pixels cover the lot
    evt(directory, codes)
    two_wd = run(directory, 10, 10)["exits"][0]
    assert two_wd["reachable"] is True
    assert two_wd["parking_lot"] is not None


def test_a_walker_the_maps_put_in_a_lake_walks_out_by_the_shortest_way_ashore(tmp_path):
    # Found in a live run: a walker 5 m from the shore read as open
    # water on the 30 m land cover, and so did every cell round them, so
    # the walk-out said the ground was cut off while a litter walked out.
    flat = np.zeros((21, 60))
    directory = stage(tmp_path, flat, [north_south("ROAD", 40, 21, name="Shore Road")])
    codes = np.full(flat.shape, 4407)
    codes[:, :15] = 7292                     # a lake west of col 15; the patient at col 10
    evt(directory, codes)
    result = run(directory, 10, 10)
    two_wd = result["exits"][0]
    assert two_wd["reachable"] is True
    start = result["closed_start"]
    assert start["distance_m"] == pytest.approx(50.0)          # five cells east to col 15
    assert start["direction"].startswith("E")
    assert any("reads as ground the walk never enters" in w for w in result["warnings"])
    # Only that shortest line is opened, so the route heads straight ashore.
    assert cell_of({"coordinates": two_wd["route"]["coordinates"][1]})[0] == 10
    # A litter never saw the water, and nothing changes for it.
    assert run(directory, 10, 10, mode="carry_out")["closed_start"] is None


def test_a_walker_on_open_ground_has_no_closed_start(tmp_path):
    road = north_south("ROAD", 40, 21, name="Any Road")
    result = run(stage(tmp_path, np.zeros((21, 60)), [road]), 10, 10)
    assert result["closed_start"] is None
    assert not any("reads as ground the walk never enters" in w for w in result["warnings"])


def test_walk_out_states_the_shared_model(tmp_path):
    road = north_south("ROAD", 40, 21, name="Any Road")
    walk = run(stage(_sub(tmp_path, "a"), np.zeros((21, 60)), [road]), 10, 10)
    assert walk["assumptions"]["walking"]["movement"]["source"] == "sar_geo_mcp.terrain.movement"
    carry = run(stage(_sub(tmp_path, "b"), np.zeros((21, 60)), [road]), 10, 10, mode="carry_out")
    assert "walking" not in carry["assumptions"]
    assert carry["assumptions"]["ground_factors"] == {"on_way": 0.25, "off_way": 0.08}


# --- Staying on the ways once joined ----------------------------------------


def _l_trail_ground(tmp_path):
    """Flat ground, a road along row 0, and a trail from (30, 10) east to
    (30, 40) and north to the road at (0, 40). Straight north from the
    trail's start is 30 cells off any way; along the trail it is 60 on it -
    quicker off the trail at Tobler's 0.6, the shortcut a medevac must not take."""
    elevation = np.zeros((41, 60))
    routes = [way("ROAD", (0, 0), (0, 59), highway="residential", name="Road"),
              way("TRAIL", (30, 10), (30, 40), (0, 40), highway="path", name="Trail")]
    return stage(tmp_path, elevation, routes)


@pytest.mark.parametrize("mode", list(MODES))
def test_a_route_on_a_trail_stays_on_it_rather_than_cut_across(tmp_path, mode):
    directory = _l_trail_ground(tmp_path)
    # A litter is so slow off a trail that its own factors keep it on this
    # one; 0.2 off it (not its 0.08) makes the shortcut quicker for it too.
    factors = {"off_way_factor": 0.2} if mode == "carry_out" else {}
    shortcut = run(directory, 30, 10, mode=mode, leave_way_factor=1.0, **factors)["exits"][0]
    assert shortcut["off_trail_m"] > 200                  # the old way: straight north
    on_trail = run(directory, 30, 10, mode=mode, **factors)["exits"][0]
    assert on_trail["off_trail_m"] == 0 and on_trail["on_trail_share"] == 1.0
    assert cell_of(on_trail["exit_point"])[1] >= 39
    # The minutes are walking minutes along the trail, never counted over:
    # 60 cells of flat way at the mode's on-way speed, give or take the corner.
    factor = MODES[mode]["on_way_factor"]
    per_cell = 10 * 0.06 / (6 * math.exp(-0.175) * factor)
    assert on_trail["minutes"] == pytest.approx(60 * per_cell, rel=0.05)
    assert on_trail["minutes"] > shortcut["minutes"]


def test_a_litter_follows_a_trail_down_a_steep_pitch_and_is_told_to_belay_it(tmp_path):
    # Found in a live run: the carry-out left the trail for 150 m round
    # a pitch steeper than the litter's 40 %. A trail runs east along row 10
    # to a road at col 55; between cols 21 and 22 it drops 15 m in one step,
    # a bank the open ground beside it shares as far south as row 34, so the
    # only way round is a detour south to the gentle ramp below the bank.
    ground = np.zeros((41, 60))
    ground[:, 22:] = -15.0
    ground[35:, :] = np.linspace(0.0, -15.0, 60)
    routes = [way("TRAIL", (10, 5), (10, 50), highway="path", name="Pitch Trail"),
              north_south("ROAD", 55, 41, name="Valley Road")]
    exit_ = run(stage(tmp_path, ground, routes), 10, 5, mode="carry_out")["exits"][0]
    assert exit_["reachable"] is True
    assert exit_["off_trail_after_joining_m"] <= 60              # only the last steps to the road
    assert exit_["steep_pitch_m"] > 0
    assert any("belay or lower the litter" in w for w in exit_["warnings"])
    # A walker has no grade limit, so no pitch to belay.
    walk = run(tmp_path, 10, 5)["exits"][0]
    assert walk["steep_pitch_m"] is None


def test_off_trail_after_joining_counts_only_what_follows_the_first_way():
    grid = SimpleNamespace(transform=from_origin(0, 0, 10, 10), pixel_size_m=10.0)
    elevation = np.zeros((1, 6))
    on_way = np.array([[False, False, True, True, False, False]])
    measured = _measure(grid, [0, 1, 2, 3, 4, 5], on_way, elevation)
    assert measured["off_trail_m"] == pytest.approx(30.0)        # into 1, 4 and 5
    assert measured["off_trail_after_joining_m"] == pytest.approx(20.0)   # 4 and 5
    assert measured["steep_pitch_m"] is None


def _trail_off_the_edge_ground(tmp_path):
    """A live run in miniature: the trail from the location
    runs east off the ground; the only road inside it is a spur 150 m north
    of the trail, so every exit here leaves the trail to reach it."""
    routes = [way("TRAIL", (30, 10), (30, 59), highway="path", name="Lake Trail"),
              way("ROAD", (0, 30), (15, 30), highway="residential", name="Spur Road")]
    return stage(tmp_path, np.zeros((41, 60)), routes)


@pytest.mark.parametrize("mode", list(MODES))
def test_a_trail_that_runs_off_the_ground_is_said_with_where_to_restage(tmp_path, mode):
    directory = _trail_off_the_edge_ground(tmp_path)
    result = run(directory, 30, 10, mode=mode)
    exit_ = result["exits"][0]
    assert exit_["reachable"] is True
    assert exit_["off_trail_after_joining_m"] > 100
    beyond = result["trail_beyond"]
    assert beyond is not None
    assert cell_of(beyond["point"]) == (30, 59)
    assert beyond["way_name"] == "Lake Trail"
    assert beyond["bearing_deg"] == pytest.approx(90.0)
    assert any("leaves the trail" in problem for problem in beyond["problems"])
    # Re-centred east along the trail, the location still well inside.
    lon0, lat0 = TO4326.transform(*xy(30, 10))
    assert beyond["recentre"]["longitude"] > lon0
    assert beyond["recentre"]["latitude"] == pytest.approx(lat0, abs=1e-4)
    assert any(w.startswith("Lake Trail leaves the staged ground") for w in result["warnings"])


def test_a_trail_that_runs_off_the_ground_is_not_said_when_a_trail_reaches_a_road(tmp_path):
    # The L trail reaches its road on the ground; a branch also runs off the
    # east edge, but nothing here leaves a trail, so nothing is said.
    elevation = np.zeros((41, 60))
    routes = [way("ROAD", (0, 0), (0, 59), highway="residential", name="Road"),
              way("TRAIL", (30, 10), (30, 40), (0, 40), highway="path", name="Trail"),
              way("TRAIL", (30, 40), (30, 59), highway="path", name="Branch")]
    result = run(stage(tmp_path, elevation, routes), 30, 10)
    assert result["exits"][0]["off_trail_after_joining_m"] == 0
    assert result["trail_beyond"] is None


def test_the_walk_to_the_first_way_is_never_counted_over(tmp_path):
    # From off every way, straight north to the road never touches the
    # trail: nothing is counted over, and the leave-way factor changes nothing.
    directory = _l_trail_ground(tmp_path)
    default = run(directory, 30, 5)["exits"][0]
    free = run(directory, 30, 5, leave_way_factor=1.0)["exits"][0]
    assert default["minutes"] == pytest.approx(free["minutes"])
    assert cell_of(default["exit_point"]) == cell_of(free["exit_point"]) == (0, 5)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_leave_way_factor_outside_zero_to_one_is_refused(tmp_path, bad):
    directory = _l_trail_ground(tmp_path)
    with pytest.raises(ValueError, match="leave_way_factor"):
        run(directory, 30, 10, leave_way_factor=bad)
    with pytest.raises(ValueError, match="leave_way_factor"):
        check_params(leave_way_factor=bad)


def test_with_no_penalty_the_two_stage_solve_is_the_plain_one():
    rng = np.random.default_rng(3)
    elevation = rng.uniform(0, 4, (15, 15))
    on_way = np.zeros(elevation.shape, dtype=bool)
    on_way[7, :] = True
    on_way[:, 3] = True
    graph = _graph(elevation, "walk_out", on_way=on_way)
    plain, _ = _solve(graph, elevation, (14, 14))
    solve = solve_staying_on_ways(graph, elevation.shape, (14, 14),
                                  void=np.isnan(elevation), on_way=on_way,
                                  leave_way_factor=1.0)
    assert np.allclose(solve.cost, plain)
    for cell in (0, 7 * 15 + 1, 3 * 15 + 12):
        path = solve.trace(cell)
        assert path[0] == 14 * 15 + 14 and path[-1] == cell
        assert solve.walk_minutes(path) == pytest.approx(plain.flat[cell])
