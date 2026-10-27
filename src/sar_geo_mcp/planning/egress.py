"""Egress: the quickest 2WD and 4x4 exit from a location, on foot or by litter.

This module reads the staged ground - the elevation model and OpenStreetMap -
marks which cells are ways, exits and closed, solves the walking graph in
`egress_graph` once from the location, and traces, simplifies and measures the
route to the least-time cell of each kind, or says why none is reached. What
a step costs, and what the minutes are and are not - a rank between two
exits, never a surveyed route - is in `egress_graph`'s docstring.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..geo import to_web_mercator, to_wgs84_geojson
from ..terrain.grid import load_assignment_grid
from ..terrain.slope import slope_gradient
from .egress_graph import LEAVE_WAY_FACTOR, MODES, build_graph, solve_from, solve_staying_on_ways
from .egress_roads import (GATE_ON_ROUTE_CELLS, GATE_ON_WAY_M, _classify, _exit_road,
                           _gates_crossed, _lots, _osm, _osm_warnings, _parking_at)
from .parking import _named_ways, _nearest_way_index

#: The two kinds of exit: (kind, mask name, what the reason calls it).
KINDS = (("2wd", "two_wd", "road or parking lot a 2WD vehicle can reach"),
         ("4x4", "four_wd", "track or dirt road a 4x4 can reach"))
#: Gain and loss ignore rises under this, so the model's noise does not add up.
CLIMB_HYSTERESIS_M = 1.0
#: The most cells egress searches: 1,800 a side, the 8,000 m maximum radius
#: plus its buffer at 10 m. Finer ground is refused before a graph is built -
#: a 2 m landing-zone handle at the default radius is 4,000 cells a side and
#: about 4 GB; at 1 m, about 17 GB.
EGRESS_MAX_CELLS = 1800 * 1800
#: A ramp's end this close (map units) to a way touches it - a shared OSM
#: node, allowing for float noise in the reprojection.
TOUCH_TOLERANCE_UNITS = 1e-6

TOO_FINE_ERROR = ("this ground is {rows:,} x {cols:,} cells of {cell_m:.1f} m, more than "
                  "the {limit:,} egress searches: it was staged too fine for egress. "
                  "Stage the area again with prepare_area at resolution_m=10.0")
OUTSIDE_ERROR = ("the location ({latitude}, {longitude}) is outside the staged ground; "
                 "stage an area centred on it with prepare_area first")
VOID_ORIGIN_REASON = "the location falls on a gap in the elevation model"
STEEP_ORIGIN_REASON = ("the location itself is steeper than a litter can be carried on; "
                       "this needs a technical or low-angle rope evacuation, or a landing zone")
NO_WAY_REASON = "no {what} is mapped in the staged area; a wider radius may reach one"
NO_ROAD_LOTS_UNKNOWN_REASON = (
    "no road a 2WD vehicle can reach is mapped in the staged area, and parking lots are "
    "unknown here - the ground was staged before they were asked for, so one may be there "
    "unseen; a wider radius or a fresh staging may reach one")
LITTER_LIMIT_REASON = ("a litter cannot reach one within the {limit:.0%} grade limit; "
                       "walk-out reaches it, or consider a landing zone")
GRADE_LIMIT_REASON = ("none can be reached within the {limit:.0%} grade limit set for this "
                      "search; without the limit one can")
CUT_OFF_REASON = ("the ground between is cut off (a gap in the elevation model, or a "
                  "motorway nobody walks across)")
WALK_CUT_OFF_REASON = ("the ground between is cut off (a gap in the elevation model, a "
                       "motorway nobody walks across, or a river, lake, open water or ground "
                       "too steep to walk with no mapped way across it)")
OFF_TRAIL_WARNING = ("{off_m:.0f} m of this route - more than half of it - is off any "
                     "mapped way, over ground the model has never seen walked")
LITTER_START_WARNING = (
    "The location is too steep for a litter: this carry starts {distance_m:.0f} m away "
    "({direction}), the nearest ground from which a litter can reach this exit. Getting the "
    "patient there needs a technical or low-angle rope evacuation, which the minutes and "
    "distance do not count")
STEEP_PITCH_WARNING = (
    "{pitch_m:.0f} m of this carry follows the trail where it is steeper than the "
    "litter's {limit:.0%} grade limit, as the elevation model reads it: belay or lower "
    "the litter there, which the minutes do not count")
#: An exit reached by leaving a trail for more than this, after the route
#: first joined one, is one a medevac does not take while a trail runs on.
LEFT_TRAIL_M = 100.0
#: A route to the edge of the ground counts as staying on its trail with up
#: to this much off it after joining (a junction burned a cell apart).
TRAIL_EDGE_SLACK_CELLS = 3
#: How many of the quickest way cells on the edge are traced, looking for
#: one the route reaches without leaving its trail.
TRAIL_EDGE_TRIES = 25
#: The location is kept at least this far inside a re-centred ground.
RECENTRE_MARGIN_M = 500.0
TRAIL_BEYOND_WARNING = (
    "{way} leaves the staged ground {minutes:.0f} min ({distance_m:,.0f} m) along it from "
    "the location, at {latitude:.5f}, {longitude:.5f} ({compass}), and here {problems}: the "
    "road it leads to may lie beyond the edge. Stage ground centred on {recentre_latitude:.5f}, "
    "{recentre_longitude:.5f} at the same radius and resolution, and ask again from the same "
    "location")
CLOSED_START_WARNING = (
    "The location itself reads as ground the walk never enters - open water, a mapped "
    "shoreline or river, or ground too steep to walk - though the patient is there: a "
    "30 m land-cover pixel or a 10 m elevation cell can cover a shore or a ledge. The walk "
    "leaves it by the shortest way to open ground, {distance_m:.0f} m ({direction}); check "
    "that first stretch on the ground")


# --- Reading the staged ground ------------------------------------------------


def _way_masks(grid, routes: list[dict], parking: list[dict],
               classified: dict | None = None) -> dict[str, np.ndarray]:
    """The staged ways burned onto the grid (all_touched, as lz does), as
    boolean masks:

    - `on_way`: every walkable way (TRAIL, TRACK, ROAD, ROAD_OTHER) but a
      motorway - where the on-way ground factor applies.
    - `impassable`: motorway cells, except within one cell of a point where
      another way properly CROSSES a motorway line. A motorway is
      grade-separated, so a way across one is on a bridge or in a tunnel and
      the crossing stays walkable. A way that only touches the line - a trail
      that dead-ends on it, or runs beside it in its cells - opens nothing.
      The one at-grade exception is a ramp's terminal: within one cell of
      each END of a motorway_link that touches another way is also open, so
      a diamond's four terminals do not wall off the crossroad they join.
    - `two_wd`: ROAD cells whose surface is not dirt, and every mapped
      parking lot (polygon or node); never a motorway cell.
    - `four_wd`: TRACK cells and dirt ROAD cells; never a motorway cell.

    Only a ROAD is an exit (ruled 2026-09-23), and not a pedestrian street.
    ROAD_OTHER - busways, raceways, escape ramps, rest areas - and
    pedestrian streets are walked like any way, but no vehicle is sent there.

    A way whose `highway` is NOT_BUILT or NOT_WALKED is ignored: plain
    ground, neither faster nor an exit. On ground staged before
    query version 5 `highway` is absent, so a motorway cannot be told from
    a road there; the caller says so.
    """
    from rasterio.features import rasterize

    shape = grid.elevation.shape

    def burn(geometries: list) -> np.ndarray:
        if not geometries:
            return np.zeros(shape, dtype=bool)
        return rasterize([(geometry, 1) for geometry in geometries], out_shape=shape,
                         transform=grid.transform, fill=0, all_touched=True,
                         dtype="uint8").astype(bool)

    classified = _classify(routes) if classified is None else classified
    ways, motorways, links, two_wd, four_wd = (
        [geometry for _, geometry in classified[name]]
        for name in ("ways", "motorways", "links", "two_wd", "four_wd"))
    two_wd += [geometry for _, geometry in _lots(parking)]

    on_way = burn(ways)
    motorway = burn(motorways)
    radius = abs(grid.transform.a)                  # one cell
    opened = burn(_crossing_neighbourhoods(ways, motorways, radius=radius)
                  + _ramp_terminals(ways, links, radius=radius))
    return {"on_way": on_way,
            "impassable": motorway & ~opened,
            "two_wd": burn(two_wd) & ~motorway,
            "four_wd": burn(four_wd) & ~motorway}


def _crossing_neighbourhoods(ways: list, motorways: list, *, radius: float) -> list:
    """A disc of `radius` (one cell) round every point where a way properly
    crosses a motorway line - the bridge or tunnel to leave open."""
    if not ways or not motorways:
        return []
    from shapely import STRtree

    tree = STRtree(motorways)
    return [way.intersection(motorways[index]).buffer(radius)
            for way in ways for index in tree.query(way, predicate="crosses")]


def _ramp_terminals(ways: list, links: list, *, radius: float) -> list:
    """A disc of `radius` (one cell) round each END of a motorway_link that
    touches a non-motorway way: the ramp's terminal, where it meets the
    surface road at grade. A way ending on a ramp's middle opens nothing,
    and neither does a main motorway line's end."""
    if not ways or not links:
        return []
    from shapely import STRtree

    tree = STRtree(ways)
    return [end.buffer(radius) for link in links for end in link.boundary.geoms
            if len(tree.query(end, predicate="dwithin", distance=TOUCH_TOLERANCE_UNITS))]


def _origin_cell(grid, latitude: float, longitude: float) -> tuple[int, int]:
    """The (row, col) the location falls in. A location on the raster's
    edge is a rounding and is clipped onto it, as `travel_surface` does; one
    more than a cell outside is on other ground altogether and is refused."""
    from rasterio.transform import rowcol

    x, y = to_web_mercator((longitude, latitude, longitude, latitude))[:2]
    rows, cols = rowcol(grid.transform, [x], [y])
    row = int(np.asarray(rows).ravel()[0])
    col = int(np.asarray(cols).ravel()[0])
    height, width = grid.elevation.shape
    if not (-1 <= row <= height and -1 <= col <= width):
        raise ValueError(OUTSIDE_ERROR.format(latitude=latitude, longitude=longitude))
    return int(np.clip(row, 0, height - 1)), int(np.clip(col, 0, width - 1))


# --- The winning cell, its route and its measurements ------------------------


def _best(minutes: np.ndarray, mask: np.ndarray) -> int | None:
    """The flat index of the least-time reachable cell inside `mask`, or None."""
    candidates = np.where(mask, minutes, np.inf)
    best = int(np.argmin(candidates))
    return best if np.isfinite(candidates.flat[best]) else None


def _trace(predecessors: np.ndarray, origin_index: int, cell_index: int) -> list[int]:
    """The flat cell indices from the origin to `cell_index`, in walking order."""
    path = [int(cell_index)]
    while path[-1] != origin_index:
        previous = int(predecessors[path[-1]])
        if previous < 0 or len(path) > len(predecessors):
            raise RuntimeError("the predecessors do not lead back to the location")
        path.append(previous)
    path.reverse()
    return path


def _climb(heights: np.ndarray) -> tuple[float, float]:
    """Gain and loss along `heights`, with a CLIMB_HYSTERESIS_M hysteresis: a
    change counts once it reaches the threshold from the last counted height."""
    gain = loss = 0.0
    reference = float(heights[0])
    for height in heights[1:]:
        change = float(height) - reference
        if change >= CLIMB_HYSTERESIS_M:
            gain += change
            reference = float(height)
        elif change <= -CLIMB_HYSTERESIS_M:
            loss -= change
            reference = float(height)
    return gain, loss


def _measure(grid, path_cells: list[int], on_way: np.ndarray, elevation: np.ndarray,
             *, pitch_grade: float | None = None) -> dict:
    """A cell path's route line (EPSG:3857, simplified to one cell) and its
    measurements. `distance_m` is along the simplified line; the on- and
    off-trail split and the steepest grade are per step, a step counting as
    on a way when the cell it enters is on one. `off_trail_after_joining_m`
    is the off-trail metres after the path first enters a way - the walk to
    the first trail is what off-trail walking is for; a metre off after it
    is a trail left. With `pitch_grade` (a litter's limit),
    `steep_pitch_m` is the metres of steps along a way (both ends on it)
    steeper than it either way: pitches the team belays; otherwise null."""
    from shapely.geometry import LineString, Point

    cells = np.asarray(path_cells, dtype=np.int64)
    rows, cols = np.divmod(cells, elevation.shape[1])
    t = grid.transform                              # cell centres, written out: affine's
    xs = t.c + t.a * (cols + 0.5) + t.b * (rows + 0.5)   # `*` on arrays is deprecated
    ys = t.f + t.d * (cols + 0.5) + t.e * (rows + 0.5)
    cell_m = float(grid.pixel_size_m)
    end = Point(xs[-1], ys[-1])
    no_pitch = None if pitch_grade is None else 0.0
    if len(cells) == 1:
        on = bool(np.asarray(on_way).ravel()[cells[0]])
        return {"line": end, "end": end, "distance_m": 0.0, "gain_m": 0.0, "loss_m": 0.0,
                "on_trail_share": 1.0 if on else 0.0, "off_trail_m": 0.0,
                "off_trail_after_joining_m": 0.0, "steep_pitch_m": no_pitch,
                "max_grade_pct": 0.0}
    diagonal = (np.diff(rows) != 0) & (np.diff(cols) != 0)
    step_m = np.where(diagonal, cell_m * math.sqrt(2.0), cell_m)
    heights = np.asarray(elevation, dtype="float64").ravel()[cells]
    on_cells = np.asarray(on_way, dtype=bool).ravel()[cells]
    on = on_cells[1:]
    path_m = float(step_m.sum())
    off_m = float(step_m[~on].sum())
    joined = np.flatnonzero(on_cells)
    # Steps after the first way cell: step i enters cell i + 1.
    after = np.arange(len(step_m)) >= (joined[0] if joined.size else len(step_m))
    off_after_m = float(step_m[~on & after].sum())
    steep_pitch_m = no_pitch
    if pitch_grade is not None:
        along = on_cells[:-1] & on
        too_steep = np.abs(np.diff(heights)) / step_m > pitch_grade
        steep_pitch_m = round(float(step_m[along & too_steep].sum()), 1)
    gain, loss = _climb(heights)
    line = LineString(np.column_stack([xs, ys])).simplify(abs(grid.transform.a))
    scale = cell_m / abs(grid.transform.a)          # ground metres per map unit
    return {"line": line, "end": end,
            "distance_m": round(line.length * scale, 1),
            "gain_m": round(gain, 1), "loss_m": round(loss, 1),
            "on_trail_share": round((path_m - off_m) / path_m, 3),
            "off_trail_m": round(off_m, 1),
            "off_trail_after_joining_m": round(off_after_m, 1),
            "steep_pitch_m": steep_pitch_m,
            "max_grade_pct": round(float(np.max(np.abs(np.diff(heights)) / step_m)) * 100, 1)}


# --- What an exit ends on: its road, its lot, the gates on the way -----------


def _cell_box(grid, cell_index: int, width: int):
    """The footprint of a cell, EPSG:3857."""
    from shapely.geometry import box

    row, col = divmod(int(cell_index), width)
    t = grid.transform
    xs = (t.c + t.a * col, t.c + t.a * (col + 1))
    ys = (t.f + t.e * row, t.f + t.e * (row + 1))
    return box(min(xs), min(ys), max(xs), max(ys))


def _exit(kind: str, minutes: float, measured: dict, *, road: dict | None,
          parking_lot: dict | None, gates_crossed: list[dict],
          litter_start: dict | None = None, pitch_grade: float | None = None) -> dict:
    """A reachable exit, its geometry in WGS84. A route more than half off
    trail says so on the exit, and so does a carry that starts away from the
    location (`litter_start`), with what that start needs, and a carry over
    trail steeper than the litter's limit (`pitch_grade`)."""
    facts = {key: value for key, value in measured.items() if key not in ("line", "end")}
    warnings = []
    if litter_start is not None:
        warnings.append(LITTER_START_WARNING.format(**litter_start))
    if measured.get("steep_pitch_m"):
        warnings.append(STEEP_PITCH_WARNING.format(pitch_m=measured["steep_pitch_m"],
                                                   limit=pitch_grade))
    if measured["off_trail_m"] > 0 and measured["on_trail_share"] < 0.5:
        warnings.append(OFF_TRAIL_WARNING.format(off_m=measured["off_trail_m"]))
    return {"kind": kind, "reachable": True,
            "exit_point": to_wgs84_geojson(measured["end"]),
            "route": to_wgs84_geojson(measured["line"]),
            "minutes": round(float(minutes), 1), **facts,
            "road": road, "parking_lot": parking_lot, "gates_crossed": gates_crossed,
            "litter_start": litter_start, "warnings": warnings}


# --- A litter that cannot leave the location ----------------------------------


def _from_litter_start(grid, elevation: np.ndarray, masks: dict, graph_ground: dict,
                       graph_settings: dict, unlimited: np.ndarray | None, *,
                       origin: tuple[int, int], staying_on_ways, reached) -> dict[str, dict]:
    """For a location too steep for a litter to leave, each kind's carry
    from the nearest ground it can start from: of the cells from which a
    litter can reach an exit of that kind, the one the location reaches
    soonest over the elevation model without the grade limit (`unlimited`,
    a stand-in for the rope team's way down). The carry is then solved from
    there as from the location. A kind with no such cell is left out."""
    from scipy.sparse.csgraph import dijkstra

    if unlimited is None:
        return {}
    shape = elevation.shape
    rope = unlimited.ravel()
    graph = build_graph(elevation, cell_m=float(grid.pixel_size_m), **graph_ground,
                        **graph_settings)
    reverse = graph.T.tocsr()
    starts: dict[str, int] = {}
    for kind, mask_name, _ in KINDS:
        targets = np.flatnonzero(masks[mask_name])
        if not targets.size:
            continue
        to_exit = dijkstra(reverse, directed=True, indices=targets, min_only=True)
        candidates = np.where(np.isfinite(to_exit), rope, np.inf)
        start = int(np.argmin(candidates))
        if np.isfinite(candidates[start]):
            starts[kind] = start
    del reverse
    found: dict[str, dict] = {}
    for kind, mask_name, _ in KINDS:
        if kind not in starts:
            continue
        start = divmod(starts[kind], shape[1])
        solve = staying_on_ways(graph, start)
        cell = _best(solve.cost, masks[mask_name])
        if cell is not None:
            found[kind] = reached(solve, mask_name, kind, cell,
                                  litter_start=_litter_start(grid, elevation, origin, start))
        del solve
    return found


def _litter_start(grid, elevation: np.ndarray, origin: tuple[int, int],
                  start: tuple[int, int]) -> dict:
    """Where a carry starts that cannot start at the location: the point,
    its straight-line ground distance and bearing from the location, and the
    height between, as the elevation model reads them."""
    from shapely.geometry import Point

    t = grid.transform
    cell_m = float(grid.pixel_size_m)
    (row0, col0), (row1, col1) = origin, start
    east = (col1 - col0) * cell_m
    north = (row0 - row1) * cell_m
    distance = math.hypot(east, north)
    bearing = math.degrees(math.atan2(east, north)) % 360
    drop = float(elevation[row0, col0] - elevation[row1, col1])
    compass = "N NE E SE S SW W NW".split()[int((bearing + 22.5) // 45) % 8]
    height = (f"{abs(drop):.0f} m lower" if drop >= 1 else
              f"{abs(drop):.0f} m higher" if drop <= -1 else "level")
    point = Point(t.c + t.a * (col1 + 0.5), t.f + t.e * (row1 + 0.5))
    return {"point": to_wgs84_geojson(point), "distance_m": round(distance, 1),
            "bearing_deg": round(bearing, 1), "drop_m": round(drop, 1),
            "direction": f"{compass}, {height}"}


# --- A walker whose own cell reads as closed ---------------------------------


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The cells on the straight line from `start` to `end`, both included,
    one per step of the longer axis, so consecutive cells are neighbours."""
    (row0, col0), (row1, col1) = start, end
    steps = max(abs(row1 - row0), abs(col1 - col0))
    if not steps:
        return np.array([row0]), np.array([col0])
    fraction = np.linspace(0.0, 1.0, steps + 1)
    rows = np.rint(row0 + (row1 - row0) * fraction).astype(np.int64)
    cols = np.rint(col0 + (col1 - col0) * fraction).astype(np.int64)
    return rows, cols


def _open_start(closed: np.ndarray, origin: tuple[int, int], *, keep_closed: np.ndarray,
                finite: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """For a walker whose own cell is `closed`, the cells to open so the walk
    can leave it: the straight line to the nearest cell that is open, on
    measured ground, and not in `keep_closed` (a motorway stays closed).
    Returns (the line's cells as a mask, the open cell it reaches), or None
    when the location is not closed or no open cell exists."""
    from scipy import ndimage

    row, col = origin
    if not closed[row, col]:
        return None
    blocked = closed | ~finite
    if blocked.all():
        return None
    _, (near_rows, near_cols) = ndimage.distance_transform_edt(blocked, return_indices=True)
    target = (int(near_rows[row, col]), int(near_cols[row, col]))
    rows, cols = _line_cells(origin, target)
    line = np.zeros(closed.shape, dtype=bool)
    line[rows, cols] = True
    # Never through a motorway or a void: those stay as they are, and a line
    # that would cross one is cut short there (the solve then finds the way
    # round, or says the ground is cut off).
    return line & ~keep_closed & finite, target


# --- A trail that runs off the staged ground ----------------------------------


def _edge_mask(shape: tuple[int, int]) -> np.ndarray:
    edge = np.zeros(shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge


def _trail_to_edge(solve, grid, masks: dict, elevation: np.ndarray, exits_mask: np.ndarray,
                   ) -> tuple[list[int], dict] | None:
    """The quickest way cell on the edge of the ground (not an exit) that the
    route reaches without leaving its trail once on one - within
    TRAIL_EDGE_SLACK_CELLS of off-way ground after joining - as (its cell
    path, its measurements), or None when no trail runs off the ground."""
    on_way = masks["on_way"]
    candidates = _edge_mask(on_way.shape) & on_way & ~exits_mask
    cost = np.where(candidates, solve.cost, np.inf).ravel()
    order = np.argsort(cost)[:TRAIL_EDGE_TRIES]
    slack_m = TRAIL_EDGE_SLACK_CELLS * float(grid.pixel_size_m)
    for cell in order:
        if not np.isfinite(cost[cell]):
            break
        path = solve.trace(int(cell))
        measured = _measure(grid, path, on_way, elevation)
        if measured["off_trail_after_joining_m"] <= slack_m:
            return path, measured
    return None


def _recentre(grid, origin: tuple[int, int], toward: tuple[int, int]) -> tuple[float, float]:
    """A centre for ground of this size moved from the location toward
    `toward` as far as it goes while the location stays RECENTRE_MARGIN_M
    inside it, as (latitude, longitude). The ground staged there at the same
    radius reaches that much farther the trail's way."""
    from pyproj import Transformer

    cell_m = float(grid.pixel_size_m)
    rows, cols = grid.elevation.shape
    half_east, half_north = cols * cell_m / 2.0, rows * cell_m / 2.0
    east = (toward[1] - origin[1]) * cell_m
    north = (origin[0] - toward[0]) * cell_m
    length = math.hypot(east, north) or 1.0
    unit_east, unit_north = east / length, north / length
    shifts = []
    for half, unit in ((half_east, unit_east), (half_north, unit_north)):
        margin = min(RECENTRE_MARGIN_M, half / 2.0)
        if abs(unit) > 1e-9:
            shifts.append((half - margin) / abs(unit))
    shift = min(shifts) if shifts else 0.0
    t = grid.transform
    scale = cell_m / abs(t.a)                       # ground metres per map unit
    x = t.c + t.a * (origin[1] + 0.5) + shift * unit_east / scale
    y = t.f + t.e * (origin[0] + 0.5) + shift * unit_north / scale
    longitude, latitude = Transformer.from_crs("EPSG:3857", "EPSG:4326",
                                               always_xy=True).transform(x, y)
    return round(latitude, 6), round(longitude, 6)


def _way_name_at(point, ways: list, tree) -> str | None:
    """The name of the mapped way nearest `point` (EPSG:3857), or None."""
    if tree is None:
        return None
    index = tree.nearest(point)
    return (ways[int(index)][0].get("properties") or {}).get("name") if index is not None else None


def _trail_beyond(grid, path: list[int], measured: dict, *, minutes: float,
                  origin: tuple[int, int], ways: list, tree, problems: list[str]) -> dict:
    """What a caller is told of a trail that runs off the ground while an
    exit here is unreachable or reached only by leaving a trail."""
    width = grid.elevation.shape[1]
    edge = divmod(int(path[-1]), width)
    point = to_wgs84_geojson(measured["end"])
    bearing = _litter_start(grid, grid.elevation, origin, edge)
    latitude, longitude = _recentre(grid, origin, edge)
    name = _way_name_at(measured["end"], ways, tree)
    return {"point": point, "way_name": name,
            "minutes": round(float(minutes), 1), "distance_m": measured["distance_m"],
            "bearing_deg": bearing["bearing_deg"], "problems": problems,
            "recentre": {"latitude": latitude, "longitude": longitude}}


def _trail_beyond_warning(beyond: dict) -> str:
    longitude, latitude = beyond["point"]["coordinates"][:2]
    bearing = beyond["bearing_deg"]
    compass = "N NE E SE S SW W NW".split()[int((bearing + 22.5) // 45) % 8]
    return TRAIL_BEYOND_WARNING.format(
        way=beyond["way_name"] or "The trail", minutes=beyond["minutes"],
        distance_m=beyond["distance_m"], latitude=latitude, longitude=longitude,
        compass=compass, problems="; ".join(beyond["problems"]),
        recentre_latitude=beyond["recentre"]["latitude"],
        recentre_longitude=beyond["recentre"]["longitude"])


def _exit_problems(found: dict[str, dict]) -> list[str]:
    """Why no exit here is what a medevac takes while a trail runs on: each
    kind unreachable, or reached by leaving a trail for more than
    LEFT_TRAIL_M after joining one. Empty when any kind is reached without
    leaving one - a 2WD road on the trail serves a 4x4 as well."""
    problems = []
    for kind, _mask, _what in KINDS:
        label = "2WD" if kind == "2wd" else "4x4"
        exit_ = found.get(kind)
        if exit_ is None:
            problems.append(f"no {label} exit is reached")
        elif exit_.get("off_trail_after_joining_m", 0.0) > LEFT_TRAIL_M:
            problems.append(f"the {label} exit leaves the trail for "
                            f"{exit_['off_trail_after_joining_m']:,.0f} m")
        else:
            return []
    return problems


# --- Why an exit cannot be reached -------------------------------------------


def _origin_too_steep(elevation: np.ndarray, origin: tuple[int, int], on_way: np.ndarray, *,
                      cell_m: float, mode: str, max_grade: float | None) -> bool:
    """Whether - carrying a litter - the location's own cell is closed by
    the ground-slope rule: steeper than the limit and off any way. Only its
    3x3 is sloped, not the whole grid. It is not a verdict on its own: a
    closed cell has steps out, so the solve still runs, and this only names
    why an exit it could not reach was not reached."""
    row, col = origin
    if mode != "carry_out" or max_grade is None or on_way[row, col]:
        return False
    row0, col0 = max(0, row - 1), max(0, col - 1)
    window = np.asarray(elevation[row0:row + 2, col0:col + 2], dtype="float64")
    return bool(slope_gradient(window, cell_m)[row - row0, col - col0] > max_grade)


def _unreachable_reason(what: str, mask: np.ndarray, *, mode: str, max_grade: float | None,
                        stuck: bool, unlimited_minutes: np.ndarray | None,
                        lots_unknown: bool = False) -> str:
    """Why no cell of a kind was reached: none is mapped (for a 2WD exit on
    ground whose lots were never asked for, none mapped that it can see);
    the litter is stuck on the location itself, too steep to be carried off
    at all; the grade limit (when a solve without it reaches one); or the
    ground is cut off."""
    if not mask.any():
        return NO_ROAD_LOTS_UNKNOWN_REASON if lots_unknown else NO_WAY_REASON.format(what=what)
    if stuck:
        return STEEP_ORIGIN_REASON
    if unlimited_minutes is not None and np.isfinite(unlimited_minutes[mask]).any():
        template = LITTER_LIMIT_REASON if mode == "carry_out" else GRADE_LIMIT_REASON
        return template.format(limit=max_grade)
    return WALK_CUT_OFF_REASON if mode == "walk_out" else CUT_OFF_REASON


# --- The entry point -----------------------------------------------------------


def _settings(mode: str, on_way_factor: float | None, off_way_factor: float | None,
              max_grade: float | None, leave_way_factor: float | None = None) -> dict:
    """The mode's defaults with the caller's overrides, validated."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    settings = {**MODES[mode], "leave_way_factor": LEAVE_WAY_FACTOR}
    for name, value in (("on_way_factor", on_way_factor), ("off_way_factor", off_way_factor),
                        ("max_grade", max_grade), ("leave_way_factor", leave_way_factor)):
        if value is not None:
            settings[name] = float(value)
    for name in ("on_way_factor", "off_way_factor"):
        if not settings[name] > 0:
            raise ValueError(f"{name} ({settings[name]}) must be above zero; it scales a "
                             "walking speed, and at zero nobody ever arrives")
    if not 0 < settings["leave_way_factor"] <= 1:
        raise ValueError(f"leave_way_factor ({settings['leave_way_factor']}) must be above "
                         "zero and at most 1; it divides a step off a way once the route has "
                         "joined one, and above 1 would draw routes off the trails")
    if settings["max_grade"] is not None and not settings["max_grade"] >= 0:
        raise ValueError(f"max_grade ({settings['max_grade']}) is below zero; it is the "
                         "steepest grade a step may take, either way")
    return settings


def check_params(*, on_way_factor: float | None = None, off_way_factor: float | None = None,
                 max_grade: float | None = None,
                 leave_way_factor: float | None = None) -> None:
    """The checks `egress_routes` makes of its walking overrides before it
    reads anything, in its own words, for every mode: what is refused does
    not depend on the mode, only the defaults do. Public, and needing no
    ground, so a scenario refuses a bad override before it stages any."""
    for mode in MODES:
        _settings(mode, on_way_factor, off_way_factor, max_grade, leave_way_factor)


def egress_routes(directory: Path, *, center_latitude: float, center_longitude: float,
                  mode: str = "walk_out", on_way_factor: float | None = None,
                  off_way_factor: float | None = None,
                  max_grade: float | None = None,
                  leave_way_factor: float | None = None,
                  max_cells: int = EGRESS_MAX_CELLS) -> dict:
    """The quickest 2WD exit and the quickest 4x4 exit from a location, on
    foot (`walk_out`) or by litter (`carry_out`), over staged ground. Both
    kinds always come back; one that cannot be reached says why. Ground of
    more than `max_cells` cells is refused as staged too fine."""
    from shapely import STRtree
    from shapely.geometry import shape as shapely_shape

    settings = _settings(mode, on_way_factor, off_way_factor, max_grade, leave_way_factor)
    grid = load_assignment_grid(directory)
    rows, cols = grid.elevation.shape
    if rows * cols > max_cells:
        raise ValueError(TOO_FINE_ERROR.format(rows=rows, cols=cols, limit=max_cells,
                                               cell_m=float(grid.pixel_size_m)))
    osm = _osm(directory)
    classified = _classify(osm["routes"])
    masks = _way_masks(grid, osm["routes"], osm["parking"], classified=classified)
    elevation = np.asarray(grid.elevation, dtype="float64")
    shape = elevation.shape
    cell_m = float(grid.pixel_size_m)
    origin = _origin_cell(grid, center_latitude, center_longitude)
    void = np.isnan(elevation)

    # The step graph's ground. Walk-out walks the shared walking ground
    # (`planning.walking`, `terrain.movement`) - vegetation, creeks, rivers,
    # open water and steep ground - as the rings and hasty minutes do.
    # Carry-out keeps its own: the litter factors, its grade limit, and the
    # ways and motorways alone.
    graph_ground = {"on_way": masks["on_way"], "impassable": masks["impassable"]}
    walk = None
    closed_start = None
    trail_beyond = None
    if mode == "walk_out":
        from ..terrain.movement import DEFAULT as MOVEMENT
        from .walking import assumptions as walking_assumptions
        from .walking import load_walking_ground

        walk = load_walking_ground(directory, MOVEMENT, grid=grid, max_cells=max_cells)
        # An exit cell stays open whatever the 30 m vegetation raster says of
        # it - a lakeside lot is still a lot - and a motorway stays closed.
        exits_mask = masks["two_wd"] | masks["four_wd"]
        closed = walk.closed & ~exits_mask
        # The patient is where they are: a location the maps read as water
        # or a cliff is a shore or a ledge the rasters are too coarse for.
        # Its own cell has steps out, but every neighbour may be closed too,
        # so the shortest line to open ground is opened and said.
        if not void[origin]:
            opened = _open_start(closed, origin, keep_closed=masks["impassable"],
                                 finite=~void)
            if opened is not None:
                line, target = opened
                closed = closed & ~line
                closed_start = _litter_start(grid, elevation, origin, target)
        graph_ground.update(impassable=closed | masks["impassable"],
                            off_way_scale=walk.vegetation_scale, crossing=walk.creek,
                            crossing_minutes=MOVEMENT.creek_crossing_minutes)

    # What an exit ends on, in map units: a cell, and the ground metres in one.
    cell_units = abs(grid.transform.a)
    scale = cell_m / cell_units
    trees = {name: STRtree([geometry for _, geometry in classified[name]])
             if classified[name] else None for name in ("two_wd", "four_wd")}
    gate_points = ([(gate, shapely_shape(gate["geometry"])) for gate in osm["gates"]
                    if gate.get("geometry")] if osm["gates_known"] else None)
    ways_tree = (STRtree([geometry for _, geometry in classified["ways"]])
                 if classified["ways"] else None)
    lots = _lots(osm["parking"])
    named = _named_ways(osm["routes"])
    named_tree = _nearest_way_index(named)

    def ends_on(path: list[int], measured: dict, mask_name: str) -> dict:
        on_way = GATE_ON_WAY_M / scale
        crossed: list[dict] = []
        if gate_points is not None:
            path_area = [_cell_box(grid, cell, shape[1]) for cell in path]
            crossed = _gates_crossed(measured["line"], path_area, gate_points,
                                     classified["ways"], ways_tree,
                                     reach=GATE_ON_ROUTE_CELLS * cell_units, on_way=on_way)
        return {"road": _exit_road(measured["end"], classified[mask_name], trees[mask_name],
                                   reach=cell_units * math.sqrt(2.0),
                                   gate_points=gate_points, gate_reach=on_way),
                "parking_lot": (_parking_at(_cell_box(grid, path[-1], shape[1]), lots, named,
                                            named_tree, scale)
                                if osm["parking_known"] else "unknown"),
                "gates_crossed": crossed}

    exits: list[dict] = []
    graph_settings = {name: settings[name]
                      for name in ("on_way_factor", "off_way_factor", "max_grade")}
    exits_mask = masks["two_wd"] | masks["four_wd"]

    def staying_on_ways(graph, start: tuple[int, int]):
        return solve_staying_on_ways(graph, shape, start, void=void, on_way=masks["on_way"],
                                     leave_way_factor=settings["leave_way_factor"],
                                     keep_open=exits_mask)

    def reached(solve, mask_name: str, kind: str, cell: int,
                litter_start: dict | None = None) -> dict:
        path = solve.trace(cell)
        pitch_grade = settings["max_grade"]
        measured = _measure(grid, path, masks["on_way"], elevation, pitch_grade=pitch_grade)
        return _exit(kind, solve.walk_minutes(path), measured, litter_start=litter_start,
                     pitch_grade=pitch_grade, **ends_on(path, measured, mask_name))

    if void[origin]:
        exits = [{"kind": kind, "reachable": False, "reason": VOID_ORIGIN_REASON}
                 for kind, _, _ in KINDS]
    else:
        steep_origin = _origin_too_steep(elevation, origin, masks["on_way"], cell_m=cell_m,
                                         mode=mode, max_grade=settings["max_grade"])
        graph = build_graph(elevation, cell_m=cell_m, **graph_ground, **graph_settings)
        solve = staying_on_ways(graph, origin)
        del graph
        # Rope is named only when the litter cannot leave at all: the cell is
        # closed for its slope AND the solve reached nothing but itself. A
        # closed cell has steps out, so a litter at the foot of a bank leaves
        # it, and an exit it then misses has the grade limit's reason.
        stuck = steep_origin and np.count_nonzero(np.isfinite(solve.cost)) == 1
        found: dict[str, dict] = {}
        for kind, mask_name, _ in KINDS:
            cell = _best(solve.cost, masks[mask_name])
            if cell is not None:
                found[kind] = reached(solve, mask_name, kind, cell)
        # Where a trail the route keeps to runs off the ground: said only
        # below, when an exit here is missing or reached by leaving a trail.
        trail_edge = _trail_to_edge(solve, grid, masks, elevation, exits_mask)
        edge_minutes = solve.walk_minutes(trail_edge[0]) if trail_edge else None
        del solve
        unlimited = None
        if settings["max_grade"] is not None and (stuck or len(found) < len(KINDS)):
            # Built only now, after the first graph is dropped, so the
            # memory peak stays near one graph's.
            free = build_graph(elevation, cell_m=cell_m, **graph_ground,
                               **{**graph_settings, "max_grade": None})
            unlimited, _ = solve_from(free, shape, origin, void=void)
            del free
        if stuck:
            found.update(_from_litter_start(
                grid, elevation, masks, graph_ground, graph_settings, unlimited,
                origin=origin, staying_on_ways=staying_on_ways, reached=reached))
        for kind, mask_name, what in KINDS:
            if kind in found:
                exits.append(found[kind])
                continue
            mask = masks[mask_name]
            exits.append({"kind": kind, "reachable": False,
                          "reason": _unreachable_reason(
                              what, mask, mode=mode, max_grade=settings["max_grade"],
                              stuck=stuck and not found, unlimited_minutes=unlimited,
                              lots_unknown=(mask_name == "two_wd"
                                            and not osm["parking_known"]))})
        problems = _exit_problems(found)
        if problems and trail_edge is not None:
            trail_beyond = _trail_beyond(grid, *trail_edge, minutes=edge_minutes,
                                         origin=origin, ways=classified["ways"],
                                         tree=ways_tree, problems=problems)

    warnings = _osm_warnings(osm) + (list(walk.warnings) if walk is not None else [])
    if closed_start is not None:
        warnings.append(CLOSED_START_WARNING.format(**closed_start))
    if trail_beyond is not None:
        warnings.append(_trail_beyond_warning(trail_beyond))
    return {
        "exits": exits,
        "mode": mode,
        "center": {"latitude": center_latitude, "longitude": center_longitude},
        "closed_start": closed_start,
        "trail_beyond": trail_beyond,
        "resolution_m": round(cell_m, 3),
        "assumptions": {
            "tobler": "6 * exp(-3.5 * |grade + 0.05|) km/h",
            "signed_grade": True,
            "ground_factors": {"on_way": settings["on_way_factor"],
                               "off_way": settings["off_way_factor"]},
            "leave_way_factor": settings["leave_way_factor"],
            "staying_on_ways": (
                "a route walks freely to its first mapped way; after it, a step off "
                f"every way counts {1 / settings['leave_way_factor']:.3g} times its minutes "
                "in choosing the route, so it leaves a trail only where the way round "
                "is that many times longer. An exit's own cells are never counted over. "
                "`minutes` are walking minutes, never counted over"),
            "litter_start": (
                "when the location is too steep for a litter to leave at all, the carry "
                "starts from the nearest ground (by the elevation model, without the grade "
                "limit) from which a litter can reach an exit; getting the patient there "
                "needs a rope evacuation and is not in the minutes"),
            "max_grade": settings["max_grade"],
            "cell_m": round(cell_m, 3),
            "two_wd": ("a ROAD - not a motorway, motorway_link or pedestrian street - "
                       "not surfaced dirt, or a mapped parking lot (amenity=parking; a "
                       "hardstand is not one). ROAD_OTHER ways (busways, raceways, escape "
                       "ramps, rest areas) and pedestrian streets are walked, never exits"),
            "four_wd": ("a TRACK, or a ROAD surfaced dirt that is not a pedestrian street; a "
                        "surface stacked with \";\" is dirt when any part of it is"),
            "closed": ("motorway and motorway_link cells, except within a cell of where "
                       "another mapped way crosses one: that way is on a bridge or in a "
                       "tunnel, so it stays open. Also open: within a cell of each end of a "
                       "ramp (motorway_link) that meets another way - its terminal, where it "
                       "joins the surface road at grade. Otherwise a way that only touches "
                       "a motorway opens nothing"),
            "ignored_ways": ("a way whose highway is proposed, construction, planned, "
                             "abandoned, disused or razed is not on the ground, and a "
                             "via_ferrata or bus_guideway is not walked: all are plain "
                             "ground, neither faster nor an exit"),
            "gates_and_access": ("said on each exit, never excluded or avoided: a gate or a "
                                 "private road is a call the planner can make - for the "
                                 "combination, or to the ranger. A gate is listed as "
                                 "crossed when it stands on a way the route runs through "
                                 "for two or more cells. A trail the route crosses at a "
                                 "shallow angle (under about 45 degrees) can meet that "
                                 "rule, so a gate on such a trail near the crossing may be "
                                 "listed although the team does not pass through it"),
            "ground": (
                "walk_out walks the shared walking ground: vegetation by LANDFIRE type off a "
                "way, a creek's crossing minutes, rivers, shorelines, open water and ground "
                "over the impassable slope closed except at a mapped way across them; an "
                "exit cell is never closed. carry_out keeps the litter's own factors and "
                "grade limit over the ways and motorways alone, and does not see vegetation "
                "or water" if mode == "walk_out" else
                "carry_out: the litter's own factors and grade limit over the ways and "
                "motorways alone; it does not see vegetation or water"),
            "not_a_promise": ("walking over the elevation model, OpenStreetMap"
                              + (" and LANDFIRE" if walk is not None else "")
                              + "; it does not see fences, private land"
                              + ("" if walk is not None else ", water crossings, brush")
                              + " or trails OSM lacks. Read the minutes as a rank."),
            **({"walking": walking_assumptions(MOVEMENT)} if walk is not None else {}),
        },
        "warnings": warnings,
    }
