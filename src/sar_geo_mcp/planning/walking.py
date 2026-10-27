"""The walking ground: staged terrain read once into what a walk needs, and
the least walking time over it.

Every tool that walks a person across staged ground builds it here - the
travel-time rings, and hasty candidates' minutes from the centre - so the
same LKP gives the same minutes to the same cell in both. The numbers are
the shared movement model's (`terrain.movement`); this module only reads
the ground they apply to:

- mapped trails, tracks and roads (`egress._way_masks`: a motorway is
  closed except at its bridges and ramp terminals, as egress closes it);
- vegetation by LANDFIRE lifeform at each cell (`evt.codes_at_cells`), and
  its open water, closed off a way;
- mapped water: a creek costs minutes to get into; a river, canal or tidal
  channel, and a lake, reservoir or coast shoreline, are closed except
  within a cell of a mapped way across them;
- slope (`slope.slope_gradient`): ground off a way steeper than the
  model's impassable slope is closed.

The walk itself is egress's directed graph (`egress_graph.build_graph`):
Tobler on each step's signed grade, the ground factor of the cell stepped
into, and a creek's minutes on the step into it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..sources.landfire import lifeform_of
from ..terrain.evt import codes_at_cells
from ..terrain.movement import CLOSED_LIFEFORMS, DEFAULT, MovementModel
from ..terrain.slope import slope_gradient
from .egress import EGRESS_MAX_CELLS, _crossing_neighbourhoods, _way_masks
from .egress_graph import build_graph
from .egress_roads import _classify

#: The most cells a walking-time search covers: egress's limit, 1,800 a side.
WALKING_MAX_CELLS = EGRESS_MAX_CELLS

#: `waterway` values that are closed but at a crossing, and those crossed
#: slowly. Any other value (a dam, a weir, a waterfall, a lock gate) is not
#: water to be crossed and is left as ground.
RIVER_WATERWAYS = frozenset({"river", "canal", "tidal_channel", "riverbank"})
CREEK_WATERWAYS = frozenset({"stream", "brook", "ditch", "drain"})
#: A waterway with a `tunnel` tag other than these runs underground - a
#: culvert - and there is nothing to cross.
NOT_A_TUNNEL = frozenset({"no"})

TOO_FINE_ERROR = ("this ground is {rows:,} x {cols:,} cells of {cell_m:.1f} m, more than "
                  "the {limit:,} a walking-time search covers: it was staged too fine. Stage "
                  "the area again with prepare_area at resolution_m=10.0")
NO_OSM_WARNING = (
    "no OpenStreetMap data is staged for this ground: no trail, road, river, creek or "
    "lakeshore is seen, so trails speed no one, no creek slows anyone, and only the open "
    "water LANDFIRE maps stops anyone - a narrow river it misses does not; walking times "
    "follow the elevation model and vegetation alone. Stage the area again once Overpass "
    "answers")
NO_EVT_WARNING = (
    "no LANDFIRE vegetation can be read for this ground (evt.tif is missing or unreadable), "
    "so brush slows no one and open water is not seen: walking times are "
    "as if over open ground, too quick where the vegetation is thick. Stage the area "
    "again with prepare_area and force=True to read it")
WATERWAY_KINDS_UNKNOWN_WARNING = (
    "this ground was staged before waterway kinds were kept, so a river cannot be told "
    "from a creek: every mapped waterway is read as a creek - crossed, slowly - and a "
    "river that should stop a walker except at a bridge does not, and a creek in a "
    "culvert is still crossed. Stage the area again with prepare_area and force=True to "
    "read them")

ASSUMPTIONS = {
    "method": ("least walking time on a directed 8-neighbour graph over the elevation "
               "model, Dijkstra; a step costs its ground length over Tobler on its signed "
               "grade times the ground factor of the cell it enters"),
    "creeks": ("waterway stream, brook, ditch or drain: getting into one costs the "
               "crossing minutes, walking along it nothing more; within a cell of a "
               "mapped way across it, and on a way's own cells, it costs nothing"),
    "closed_water": ("waterway river, canal, tidal_channel or riverbank, a lake, "
                     "reservoir or coast shoreline, and LANDFIRE open water off a way: "
                     "never entered, except within a cell of a mapped way across it "
                     "(a bridge, a ford, a dam path). A waterway in a tunnel (a culvert) "
                     "is not water to cross"),
    "steep_ground": ("ground off a mapped way whose steepest gradient is over the "
                     "impassable slope is never entered; below it Tobler makes steep "
                     "ground slow. A way's cells are exempt: the model does not see a "
                     "benched tread"),
    "motorways": "closed, except at a bridge or a ramp's terminal, as egress_routes",
    "dem_void": "a cell with no elevation is never entered: ground nobody measured is "
                "not a shortcut",
    "speed_factor": "multiplies every speed, on a way and off; divides the creek crossing",
}


@dataclass
class WalkingGround:
    """What a walk over one staged directory needs, read once."""

    grid: object
    elevation: np.ndarray            # float64, NaN where nodata
    on_way: np.ndarray               # bool: a mapped trail, track or road (not a motorway)
    closed: np.ndarray               # bool: never stepped into
    creek: np.ndarray                # bool: costs the crossing minutes to step into
    vegetation_scale: np.ndarray | None   # off-way vegetation factor per cell, or None
    classified: dict                 # egress_roads._classify of the staged routes
    lifeform_shares: dict | None     # share of cells per lifeform, or None without EVT
    water_counts: dict
    warnings: list[str] = field(default_factory=list)
    osm_known: bool = True

    @property
    def cell_m(self) -> float:
        return float(self.grid.pixel_size_m)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.elevation)

    def closed_share(self) -> float:
        valid = self.valid
        return round(float((self.closed & valid).sum()) / max(int(valid.sum()), 1), 3)


def _osm_routes(directory: Path) -> tuple[list[dict], list[dict], bool]:
    """The staged routes and parking, and whether an osm.json was there at
    all. No file is not refused: a walk can still be costed on the elevation
    model and vegetation, and a warning says what is unseen."""
    path = Path(directory) / "osm.json"
    if not path.is_file():
        return [], [], False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("routes") or [], payload.get("parking") or [], True


def _vegetation(directory: Path, grid, model: MovementModel):
    """The per-cell off-way vegetation factor, the open-water mask, and the
    share of cells in each lifeform - or (None, None, None) when no evt.tif
    is staged or it cannot be read. A cell off the EVT raster is unknown:
    factor 1.0, not water."""
    from rasterio.errors import RasterioError

    evt_path = Path(directory) / "evt.tif"
    if not evt_path.is_file():
        return None, None, None
    height, width = grid.elevation.shape
    rows, cols = np.mgrid[0:height, 0:width]
    try:
        codes = codes_at_cells(evt_path, grid.transform, rows.ravel(), cols.ravel())
    except (OSError, RasterioError):
        # LANDFIRE is the optional layer: an unreadable file costs the walk
        # its vegetation, with the caller's warning, never the walk itself.
        return None, None, None
    unique, inverse = np.unique(codes, return_inverse=True)
    lifeforms = [lifeform_of(int(code)) for code in unique]
    factor_of = np.array([model.vegetation_factor(form) for form in lifeforms], dtype="float64")
    water_of = np.array([form in CLOSED_LIFEFORMS for form in lifeforms], dtype=bool)
    counts = np.bincount(inverse, minlength=len(unique))
    shares: dict[str, float] = {}
    for form, count in zip(lifeforms, counts):
        shares[form] = shares.get(form, 0.0) + int(count) / inverse.size
    inverse = inverse.reshape(height, width)
    return (factor_of[inverse], water_of[inverse],
            {form: round(share, 3) for form, share in sorted(shares.items(),
                                                             key=lambda item: -item[1])})


def _water(grid, routes: list[dict], ways: list) -> dict:
    """The mapped water burned onto the grid (all_touched, as ways are):

    - `creek`: creeks, brooks, ditches and drains, crossed at a cost;
    - `closed`: rivers, canals, tidal channels and lake, reservoir or coast
      shorelines, not entered at all;

    both open within a cell of every point where a mapped way crosses them
    (`egress._crossing_neighbourhoods`, which opens a motorway at a bridge
    the same way). A waterway with a `tunnel` tag runs underground and is
    left out. A record without a `waterway` key was staged before the kind
    was kept, and is read as a creek: the gentler reading, which the caller
    warns of."""
    from rasterio.features import rasterize
    from shapely.geometry import shape as shapely_shape

    shape = grid.elevation.shape
    creeks, closed = [], []
    counts = {"rivers": 0, "creeks": 0, "shorelines": 0, "culverts": 0}
    kinds_known = True
    for route in routes:
        properties = route.get("properties") or {}
        route_type = properties.get("route_type")
        if route_type == "WATER_EDGE":
            closed.append(shapely_shape(route["geometry"]))
            counts["shorelines"] += 1
            continue
        if route_type != "WATERWAY":
            continue
        if "waterway" not in properties:
            kinds_known = False
            creeks.append(shapely_shape(route["geometry"]))
            counts["creeks"] += 1
            continue
        tunnel = str(properties.get("tunnel") or "no").strip().lower()
        if tunnel not in NOT_A_TUNNEL:
            counts["culverts"] += 1
            continue
        kind = str(properties.get("waterway") or "").strip().lower()
        if kind in RIVER_WATERWAYS:
            closed.append(shapely_shape(route["geometry"]))
            counts["rivers"] += 1
        elif kind in CREEK_WATERWAYS:
            creeks.append(shapely_shape(route["geometry"]))
            counts["creeks"] += 1

    def burn(geometries: list) -> np.ndarray:
        if not geometries:
            return np.zeros(shape, dtype=bool)
        return rasterize([(geometry, 1) for geometry in geometries], out_shape=shape,
                         transform=grid.transform, fill=0, all_touched=True,
                         dtype="uint8").astype(bool)

    opened = burn(_crossing_neighbourhoods(ways, creeks + closed,
                                           radius=abs(grid.transform.a)))
    return {"creek": burn(creeks) & ~opened, "closed": burn(closed) & ~opened,
            "kinds_known": kinds_known, "counts": counts}


def load_walking_ground(directory: Path, model: MovementModel = DEFAULT, *, grid=None,
                        max_cells: int = WALKING_MAX_CELLS) -> WalkingGround:
    """Read a staged directory into a WalkingGround. `grid` may be passed
    when the caller already loaded it. Ground of more than `max_cells` cells
    is refused as staged too fine (ValueError, TOO_FINE_ERROR)."""
    from ..terrain.grid import load_assignment_grid

    grid = load_assignment_grid(directory) if grid is None else grid
    height, width = grid.elevation.shape
    if height * width > max_cells:
        raise ValueError(TOO_FINE_ERROR.format(rows=height, cols=width, limit=max_cells,
                                               cell_m=float(grid.pixel_size_m)))
    elevation = np.asarray(grid.elevation, dtype="float64")
    warnings: list[str] = []

    routes, parking, osm_known = _osm_routes(directory)
    if not osm_known:
        warnings.append(NO_OSM_WARNING)
    classified = _classify(routes)
    masks = _way_masks(grid, routes, parking, classified=classified)
    on_way = masks["on_way"]
    water = _water(grid, routes, [geometry for _, geometry in classified["ways"]])
    if osm_known and not water["kinds_known"]:
        warnings.append(WATERWAY_KINDS_UNKNOWN_WARNING)

    vegetation_scale, open_water, lifeform_shares = _vegetation(directory, grid, model)
    if vegetation_scale is None:
        warnings.append(NO_EVT_WARNING)
        open_water = np.zeros(elevation.shape, dtype=bool)

    with np.errstate(invalid="ignore"):
        steep = (slope_gradient(elevation, float(grid.pixel_size_m))
                 > math.tan(math.radians(model.impassable_slope_deg))) & ~on_way
    # Open water from LANDFIRE is 30 m and can spill onto a lakeside trail,
    # so a way's cells stay open through it; a mapped river line is exact,
    # and only a crossing opens it.
    closed = masks["impassable"] | water["closed"] | steep | (open_water & ~on_way)
    return WalkingGround(grid=grid, elevation=elevation, on_way=on_way, closed=closed,
                         creek=water["creek"] & ~on_way, vegetation_scale=vegetation_scale,
                         classified=classified, lifeform_shares=lifeform_shares,
                         water_counts=water["counts"], warnings=warnings,
                         osm_known=osm_known)


def solve(ground: WalkingGround, model: MovementModel, origin_cell: tuple[int, int], *,
          limit: float | None = None, predecessors: bool = False):
    """Minutes from `origin_cell` (row, col) to every cell, `inf` where
    unreachable or past `limit`; with `predecessors`, also each cell's
    predecessor on its quickest path (-9999 where none). An origin on a
    void has no measured ground to start from: every cell is `inf`."""
    from scipy.sparse.csgraph import dijkstra

    height, width = ground.elevation.shape
    row, col = int(origin_cell[0]), int(origin_cell[1])
    if not ground.valid[row, col]:
        empty = np.full((height, width), np.inf)
        return (empty, np.full(height * width, -9999, dtype=np.int32)) if predecessors else empty
    speed = model.speed_factor
    graph = build_graph(ground.elevation, cell_m=ground.cell_m, on_way=ground.on_way,
                        impassable=ground.closed, on_way_factor=model.on_way_factor * speed,
                        off_way_factor=model.off_way_factor * speed, max_grade=None,
                        off_way_scale=ground.vegetation_scale, crossing=ground.creek,
                        crossing_minutes=model.creek_crossing_minutes / speed)
    kwargs = {"directed": True, "indices": row * width + col,
              "return_predecessors": predecessors}
    if limit is not None:
        kwargs["limit"] = limit * (1 + 1e-9)
    result = dijkstra(graph, **kwargs)
    del graph
    if predecessors:
        minutes, previous = result
        return np.asarray(minutes, dtype="float64").reshape(height, width), previous
    return np.asarray(result, dtype="float64").reshape(height, width)


def assumptions(model: MovementModel = DEFAULT) -> dict:
    """The walking rules and the movement model's numbers, for an answer."""
    return {**ASSUMPTIONS, "movement": model.assumptions()}
