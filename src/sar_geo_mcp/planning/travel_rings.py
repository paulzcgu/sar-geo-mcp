"""Travel-time rings: how far a subject can walk from a place last seen in a
given number of minutes, drawn as the loop round the ground they can reach.

WHAT THIS IS
------------
A radius circle says "everything within 2 km". A travel-time ring says
"everything within 60 minutes' walk", and so bulges out along a trail, pulls
in at a cliff band or a river with no bridge, and shrinks in thick brush.
Every point on its outline is, to this model, the same walking time from the
location.

It is solved on the shared walk (`planning.walking`, with the numbers of
`terrain.movement` - the same walk hasty candidates' minutes come from),
outward from the location, once, and cut at each time asked for:

- each step costs its ground length over Tobler's hiking speed on the step's
  SIGNED grade, so climbing out of a canyon is slower than walking down it;
- on a mapped trail, track or road the speed is Tobler's (factor 1.0); off
  one it is Tobler's off-path 0.6, times a VEGETATION factor for what
  LANDFIRE says grows there (brush slows a walker most, grass least);
- a CREEK costs a fixed number of minutes to get into, for finding a place
  to cross and crossing it;
- a RIVER, a canal, a lake or reservoir shore, and open water are closed,
  except within a cell of where a mapped way crosses them - a bridge, a
  ford, a dam path;
- ground steeper than the cliff slope (45 degrees by default) is closed
  off a mapped way - on a 10 m elevation model that is a cliff or near it;
- a motorway is closed as egress closes it, open at its bridges and ramps.

Everything above is an ASSUMPTION, returned in `assumptions` beside the
rings, and every factor can be overridden per call.

WHAT THIS IS NOT
----------------
A prediction of where the subject is. The minutes are WALKING minutes at a
steady pace; a lost person stops, rests, backtracks and sleeps, so the
planner turns elapsed time into walking time, not this tool. It knows no
lost-person behaviour and no likelihood (that stays with the planner), and
it does not see fences, private land, brush OpenStreetMap and LANDFIRE do not
map, or trails OSM lacks. Read the rings as the shape of the reachable
ground - where it stretches and where it is pinched - not as a boundary a
subject cannot be outside.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..geo import acres, to_wgs84_geojson
from ..terrain.grid import load_assignment_grid
from ..terrain.movement import DEFAULT, TOBLER_MAX_KMH
from .area import MAX_EXTENT_M, PLANNING_AREA_BUFFER_M
from .egress import _origin_cell, _trace
from .walking import (NO_EVT_WARNING, NO_OSM_WARNING,  # noqa: F401  (re-exported)
                      WALKING_MAX_CELLS, WATERWAY_KINDS_UNKNOWN_WARNING, load_walking_ground,
                      solve)
from .walking import assumptions as walking_assumptions

#: The times drawn when none are asked for, in walking minutes.
DEFAULT_MINUTES = (15.0, 30.0, 60.0)
#: The most rings one call draws.
MAX_RINGS = 8
#: A hole in a ring smaller than this (ground square metres) is the model's
#: noise - one steep cell - not ground worth drawing round.
MIN_HOLE_M2 = 2500.0
#: The eight directions `reach_by_direction_m` reports, clockwise from north.
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

VOID_ORIGIN_ERROR = ("the location falls on a gap in the elevation model, so there is no "
                     "measured ground to walk from; move it a few metres onto measured ground")
EDGE_WARNING = (
    "the {minutes:g}-minute ring reaches the edge of the staged ground, {reach:,.0f} m from "
    "the location, so it is cut there and the ground it encloses is a lower bound. {advice}")
EDGE_ADVICE_RADIUS = ("No one walks farther than {bound:,.0f} m in {minutes:g} minutes "
                      "here; prepare_area centred on the location with radius_m={radius:,.0f} "
                      "stages enough ground")
EDGE_ADVICE_TOO_FAR = ("The largest area prepare_area stages ({radius:,.0f} m radius) "
                       "reaches about {reach:,.0f} m from its centre, less than the "
                       "{bound:,.0f} m a walker can cover in {minutes:g} minutes here, so this "
                       "ring cannot be drawn whole; read it as cut")


# --- Settings -----------------------------------------------------------------


def minutes_list(minutes) -> list[float]:
    """The ring times, ascending and once each; every one above zero and
    finite, and no more than MAX_RINGS of them. Public because a scenario
    checks the times it was given with this, before it stages anything."""
    values = DEFAULT_MINUTES if minutes is None else minutes
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError("minutes must be a list of numbers of walking minutes") from None
    if not values:
        raise ValueError("minutes is empty; give at least one ring time")
    for value in values:
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"a ring time ({value:g}) must be a number of minutes above zero")
    values = sorted(set(values))
    if len(values) > MAX_RINGS:
        raise ValueError(f"{len(values)} ring times asked for; the most one call draws is "
                         f"{MAX_RINGS}")
    return values


# --- Cutting the surface into rings -------------------------------------------


def _polygon(mask: np.ndarray, grid, cell_units: float, scale: float):
    """The cells of `mask` as one (Multi)Polygon in EPSG:3857, its stairs
    simplified to about a cell and its holes under MIN_HOLE_M2 filled."""
    from rasterio.features import shapes
    from shapely.geometry import MultiPolygon, Polygon, shape as shapely_shape
    from shapely.ops import unary_union

    pieces = [shapely_shape(geometry) for geometry, _ in
              shapes(mask.astype("uint8"), mask=mask, transform=grid.transform,
                     connectivity=8)]
    merged = unary_union(pieces).simplify(cell_units, preserve_topology=True)
    min_hole = MIN_HOLE_M2 / (scale * scale)                    # in map units squared
    parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
    kept = [Polygon(part.exterior, [hole for hole in part.interiors
                                    if Polygon(hole).area >= min_hole])
            for part in parts if isinstance(part, Polygon) and not part.is_empty]
    kept.sort(key=lambda part: -part.area)
    return kept[0] if len(kept) == 1 else MultiPolygon(kept)


#: The largest radius `prepare_area` stages round a point: it refuses an
#: area over MAX_EXTENT_M across, measured on its own circle polygon, which
#: puts a radius of exactly half of that a hair over - so the advice names
#: the largest whole metre under it, which it stages at any latitude.
LARGEST_RADIUS_M = MAX_EXTENT_M / 2.0 - 1.0


def rings_model(*, speed_factor: float = 1.0, off_way_factor: float | None = None,
                creek_crossing_minutes: float | None = None,
                cliff_slope_deg: float | None = None,
                vegetation_factors: dict[str, float] | None = None):
    """The movement model a ring is walked on: the shared default with the
    caller's overrides, validated (`MovementModel.with_overrides` raises
    ValueError for a bad one). Public so a scenario can refuse a bad
    override, and size the ground, before it stages anything."""
    return DEFAULT.with_overrides(speed_factor=speed_factor, off_way_factor=off_way_factor,
                                  creek_crossing_minutes=creek_crossing_minutes,
                                  impassable_slope_deg=cliff_slope_deg,
                                  vegetation_factors=vegetation_factors)


def fastest_factor(model) -> float:
    """How many times Tobler's peak the quickest ground on `model` is
    walked at: its speed factor times the quicker of a mapped way and open
    ground - the off-way factor times the quickest vegetation factor, and
    never under 1.0, since a lifeform with no data slows no one. On the
    default model that is a trail (1.0); an off-way or vegetation override
    above it makes open ground the quickest, and the reach bound must
    follow, or the edge advice names too small a radius."""
    open_ground = model.off_way_factor * max(1.0, *model.vegetation_factors.values())
    return model.speed_factor * max(model.on_way_factor, open_ground)


def reach_bound_m(minutes: float, speed_factor: float = 1.0) -> float:
    """The farthest anyone gets in `minutes` on this model: the top speed
    (Tobler's peak, slightly downhill) the whole way, on the quickest
    ground - pass `fastest_factor(model)` as `speed_factor`."""
    return TOBLER_MAX_KMH * 1000.0 * speed_factor * minutes / 60.0


def radius_to_fit(minutes: float, speed_factor: float = 1.0) -> float:
    """The radius `prepare_area`, centred on the location, must stage for a
    ring of `minutes` to fit whole: the reach bound less the planning
    buffer, rounded up to 250 m (never under 250 m). It may be more than the
    largest area `prepare_area` stages, and is infinite when the bound
    overflows (an absurd time or factor), which no ground fits. Public
    because a scenario sizes the ground it stages for the rings with it, by
    the same rule the edge warning names a radius by."""
    bound = reach_bound_m(minutes, speed_factor)
    if not math.isfinite(bound):
        return math.inf
    return math.ceil(max(bound - PLANNING_AREA_BUFFER_M, 250.0) / 250.0) * 250.0


def _edge_advice(minutes: float, speed_factor: float) -> str:
    """What to stage so the ring fits: a radius from the model's top speed
    (`speed_factor` is `fastest_factor`), or, past the largest area, that it
    cannot be drawn whole."""
    bound = reach_bound_m(minutes, speed_factor)
    radius = radius_to_fit(minutes, speed_factor)
    if radius > LARGEST_RADIUS_M:
        reach = LARGEST_RADIUS_M + PLANNING_AREA_BUFFER_M
        if bound > reach:
            return EDGE_ADVICE_TOO_FAR.format(radius=LARGEST_RADIUS_M, bound=bound,
                                              minutes=minutes, reach=reach)
        # Rounding up to 250 m went past what prepare_area stages, but the
        # largest it does stage still holds the bound: name that, never a
        # radius it would refuse.
        radius = LARGEST_RADIUS_M
    return EDGE_ADVICE_RADIUS.format(bound=bound, minutes=minutes, radius=radius)


def _cell_centres(grid, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:3857 centres of flat cell indices."""
    rows, cols = np.divmod(np.asarray(cells, dtype=np.int64), grid.elevation.shape[1])
    t = grid.transform
    return t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e


def _route(cell: int, minutes_surface: np.ndarray, grid, *, predecessors: np.ndarray,
           origin_index: int, on_way: np.ndarray, name_way, distance_m: np.ndarray,
           bearing_deg: np.ndarray) -> dict:
    """The quickest route from the location to one cell, split into trail
    and off-trail minutes, and where it last leaves a mapped way.

    A step counts as on a way when the cell it enters is on one, as egress
    measures its routes; a creek's crossing minutes fall on the step into
    the creek, which is off a way. `leaves_trail_at` is the last way cell
    before the route goes off for good - null when the route never touches
    a way, or ends on one. `trail` is the way that cell is on."""
    from shapely.geometry import LineString, Point

    path = np.asarray(_trace(predecessors, origin_index, cell), dtype=np.int64)
    flat_minutes = minutes_surface.ravel()
    flat_on_way = np.asarray(on_way, dtype=bool).ravel()
    steps = np.diff(flat_minutes[path])
    entered_way = flat_on_way[path[1:]]
    trail_minutes = float(steps[entered_way].sum())
    off_minutes = float(steps[~entered_way].sum())
    on_path = np.flatnonzero(flat_on_way[path])
    last_on = int(on_path[-1]) if on_path.size else None
    xs, ys = _cell_centres(grid, path)
    leaves = None
    trail = None
    if last_on is not None:
        departure = Point(xs[last_on], ys[last_on])
        trail = name_way(departure)
        if last_on < len(path) - 1:
            leaves = to_wgs84_geojson(departure)
    line = (LineString(np.column_stack([xs, ys])).simplify(abs(grid.transform.a))
            if len(path) > 1 else Point(xs[0], ys[0]))
    return {
        "reach_m": round(float(distance_m.flat[cell]), 0),
        "bearing_deg": round(float(bearing_deg.flat[cell]), 0),
        "minutes": round(float(flat_minutes[cell]), 1),
        "trail_minutes": round(trail_minutes, 1),
        "off_trail_minutes": round(off_minutes, 1),
        "leaves_trail_at": leaves,
        "trail": trail,
        "point": to_wgs84_geojson(Point(xs[-1], ys[-1])),
        "route": to_wgs84_geojson(line),
    }


def _ring(minutes_surface: np.ndarray, limit: float, grid, *, distance_m: np.ndarray,
          bearing_deg: np.ndarray, valid: np.ndarray, top_factor: float,
          predecessors: np.ndarray, origin_index: int, on_way: np.ndarray,
          name_way) -> tuple[dict, str | None]:
    """One ring: the ground within `limit` walking minutes, its outline, and
    its measurements, with the quickest route to its farthest ground in each
    direction. Returns the ring and its edge warning, or None."""
    from shapely.geometry import LineString, Point

    cell_units = abs(grid.transform.a)
    scale = float(grid.pixel_size_m) / cell_units                # ground metres per map unit
    inside = minutes_surface <= limit
    geometry = _polygon(inside, grid, cell_units, scale)
    largest = geometry if geometry.geom_type == "Polygon" else geometry.geoms[0]

    reach = np.where(inside, distance_m, -1.0)
    far = int(np.argmax(reach))
    far_row, far_col = divmod(far, inside.shape[1])
    t = grid.transform
    far_point = Point(t.c + (far_col + 0.5) * t.a, t.f + (far_row + 0.5) * t.e)
    sectors = (np.floor(((bearing_deg + 22.5) % 360.0) / 45.0).astype(np.int64)) % 8
    by_direction = {}
    routes = {}
    for index, name in enumerate(DIRECTIONS):
        in_sector = inside & (sectors == index)
        if not in_sector.any():
            by_direction[name], routes[name] = 0.0, None
            continue
        cell = int(np.argmax(np.where(in_sector, distance_m, -1.0)))
        routes[name] = _route(cell, minutes_surface, grid, predecessors=predecessors,
                              origin_index=origin_index, on_way=on_way, name_way=name_way,
                              distance_m=distance_m, bearing_deg=bearing_deg)
        by_direction[name] = routes[name]["reach_m"]

    border = np.zeros(inside.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    # A ring reaching measured ground on the raster's edge, or ground beside
    # a void on it, is cut; ground merely near the edge is not.
    reaches_edge = bool((inside & border & valid).any())
    area_m2 = geometry.area * scale * scale
    ring = {
        "minutes": limit,
        "geometry": to_wgs84_geojson(geometry),
        "outline": to_wgs84_geojson(LineString(largest.exterior.coords)),
        "parts": 1 if geometry.geom_type == "Polygon" else len(geometry.geoms),
        "area_acres": round(acres(area_m2), 1),
        "equivalent_radius_m": round(math.sqrt(area_m2 / math.pi), 0),
        "max_reach_m": round(float(distance_m.flat[far]), 0),
        "max_reach_bearing_deg": round(float(bearing_deg.flat[far]), 0),
        "farthest_point": to_wgs84_geojson(far_point),
        "reach_by_direction_m": by_direction,
        "routes_by_direction": routes,
        "reaches_edge": reaches_edge,
    }
    warning = None
    if reaches_edge:
        warning = EDGE_WARNING.format(minutes=limit, reach=float(distance_m.flat[far]),
                                      advice=_edge_advice(limit, top_factor))
    return ring, warning


def _way_namer(ways: list[tuple[dict, object]], *, reach: float):
    """A function naming the mapped way nearest a point, within `reach` (a
    cell's diagonal, map units), as {name, route_type, highway}, or None.
    Ties go to the name, so two runs agree."""
    if not ways:
        return lambda point: None
    from shapely import STRtree

    tree = STRtree([geometry for _, geometry in ways])

    def name(point) -> dict | None:
        indices, distances = tree.query_nearest(point, max_distance=reach,
                                                return_distance=True, all_matches=True)
        if len(indices) == 0:
            return None
        _, index = min(zip(distances, indices), key=lambda pair: (
            float(pair[0]), str((ways[int(pair[1])][0].get("properties") or {}).get("name")
                                or "")))
        properties = ways[int(index)][0].get("properties") or {}
        return {"name": properties.get("name"), "route_type": properties.get("route_type"),
                "highway": properties.get("highway")}

    return name


# --- The entry point -----------------------------------------------------------


def travel_time_rings(directory: Path, *, center_latitude: float, center_longitude: float,
                      minutes=None, speed_factor: float = 1.0,
                      off_way_factor: float | None = None,
                      creek_crossing_minutes: float | None = None,
                      cliff_slope_deg: float | None = None,
                      vegetation_factors: dict[str, float] | None = None,
                      max_cells: int = WALKING_MAX_CELLS) -> dict:
    """Rings of equal walking time from a location over staged ground, one
    per entry in `minutes` (walking minutes, default 15, 30 and 60), each
    the ground a walker reaches within that time and its outline. Ground of
    more than `max_cells` cells is refused as staged too fine. The walk is
    the shared one (`planning.walking`, `terrain.movement`), with the
    caller's overrides."""
    limits = minutes_list(minutes)
    model = rings_model(speed_factor=speed_factor, off_way_factor=off_way_factor,
                        creek_crossing_minutes=creek_crossing_minutes,
                        cliff_slope_deg=cliff_slope_deg, vegetation_factors=vegetation_factors)
    grid = load_assignment_grid(directory)
    ground = load_walking_ground(directory, model, grid=grid, max_cells=max_cells)
    origin = _origin_cell(grid, center_latitude, center_longitude)
    valid = ground.valid
    if not valid[origin]:
        raise ValueError(VOID_ORIGIN_ERROR)
    height, width = ground.elevation.shape
    cell_m = ground.cell_m
    warnings = list(ground.warnings)
    speed = model.speed_factor
    on_way = ground.on_way
    # Searched only as far as the longest ring: everything past it is inf,
    # and on a big area that is most of the graph.
    origin_index = origin[0] * width + origin[1]
    surface, predecessors = solve(ground, model, origin, limit=limits[-1], predecessors=True)

    # Ground distance and bearing from the location to every cell centre.
    from ..geo import to_web_mercator

    x0, y0 = to_web_mercator((center_longitude, center_latitude,
                              center_longitude, center_latitude))[:2]
    t = grid.transform
    cols = np.arange(width)
    rows = np.arange(height)
    dx = (t.c + (cols + 0.5) * t.a - x0)[None, :] * (cell_m / abs(t.a))
    dy = (t.f + (rows + 0.5) * t.e - y0)[:, None] * (cell_m / abs(t.a))
    distance_m = np.hypot(dx, dy)
    bearing_deg = np.degrees(np.arctan2(dx, dy)) % 360.0

    name_way = _way_namer(ground.classified["ways"], reach=abs(t.a) * math.sqrt(2.0))
    rings = []
    for limit in limits:
        ring, warning = _ring(surface, limit, grid, distance_m=distance_m,
                              bearing_deg=bearing_deg, valid=valid,
                              top_factor=fastest_factor(model),
                              predecessors=predecessors, origin_index=origin_index,
                              on_way=on_way, name_way=name_way)
        rings.append(ring)
        if warning:
            warnings.append(warning)

    return {
        "rings": rings,
        "center": {"latitude": center_latitude, "longitude": center_longitude},
        "resolution_m": round(cell_m, 3),
        "speed_factor": speed,
        "ground": {
            "vegetation": ground.lifeform_shares,
            "water": ground.water_counts,
            "closed_share": ground.closed_share(),
        },
        "assumptions": {
            **walking_assumptions(model),
            "rings": ("each ring is the ground within its minutes of the location, its "
                      "outline simplified to about a cell"),
            "not_a_prediction": ("walking minutes at a steady pace, not elapsed time: a lost "
                                 "person stops, rests and backtracks, and no lost-person "
                                 "behaviour or likelihood is modelled. Fences, private land, "
                                 "unmapped trails and brush LANDFIRE does not show are unseen. "
                                 "Read the rings as the shape of the reachable ground"),
        },
        "warnings": warnings,
    }
