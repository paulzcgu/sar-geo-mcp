"""The anatomy of a boundary: what it follows, how it was drawn, what it holds.

THE METHOD IS THE REPORT'S
--------------------------
The anatomy report walked each real boundary in 10 m steps and asked, at
every step, which mapped feature is nearest within tolerance: road or
trail within 20 m, stream within 25 m, elevation-model drainage or ridge
within 30 m, fence or power line within 20 m, otherwise nothing. Here each
class's lines are collected once - OpenStreetMap routes as staged, drainages
and ridges traced over the staged raster - and every sample is measured
against them exactly. Consecutive samples of one class form a run; runs
under 30 m are absorbed so the list reads as edges, not chatter. A straight
drawn edge is a single vertex-to-vertex segment of 150 m or more; a shared
edge is one another assignment on the same map runs alongside, within 15 m
measured ACROSS the edge.

The tolerances are deliberately loose: a planner drawing on a phone at
1:24,000 is 10 to 20 m off the line they mean, and OpenStreetMap trails are
themselves offset by as much. What the numbers say is "this edge was meant
to be that feature", not "this edge is on it". Loose tolerances are also why
the distances are exact rather than read off a rasterised distance field:
a field quantises to the pixel, and on a 10 m grid a sample 21 m from a
trail reads back as 16 m - the difference between "this edge follows the
trail" and "it does not", decided by rounding. There is one sample per 10 m
of boundary, so measuring properly costs nothing worth saving.

WHAT IS PER DIRECTORY AND WHAT IS PER TARGET
--------------------------------------------
The class lines, the flow routing behind the drainages and ridges, the slope
and the landform classes are facts about the staged raster, not about any one
polygon: they are a `Ground`, built once per directory and measured against
by every target. Only the walk, the distances to those lines, and the mask
statistics are per target - phase 2 describes hundreds of candidates against
one staging, and rebuilding the Ground per candidate would be the whole cost
of the run. The map is the exception: it can change between two calls, so
map features are read fresh per call and passed in.

A polygon's boundary is a RING, and a ring has no beginning. The sampler
starts where the polygon happens to have been drawn from, so the run that
crosses that seam is folded back together and reported once, with `wraps`
set; forcing the first run to start at zero and the last to end at the
perimeter would report the drawing rather than the ground.

WHAT IT NEVER SAYS
------------------
Nothing here ranks or scores. A block whose edges follow nothing mapped is
reported as following nothing mapped - OpenStreetMap lacks social trails
and ranch roads, and the report attributes most "no mapped feature" length
to that gap. Norm flags say "unusual on these maps", not "wrong".
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ..geo import ground_meters_per_mercator_unit
from ..terrain.features.base import DEM_VOID_WARNING
from ..terrain.features.context import TerrainContext
from ..terrain.grid import project_geometry_to_3857
from ..terrain.hydro import HYDRO_ASSUMPTIONS, extract_channels, upstream_is_truncated
from ..sources.osm import OSM_QUERY_VERSION_BARRIERS, OSM_QUERY_VERSION_MINOR_LINES
from ..geo import acres
from .norms import load_norms, norm_flags, resource_median_ha, target_area_range_acres

#: Ground metres between boundary samples.
SAMPLE_SPACING_M = 10.0
#: Runs shorter than this are absorbed into a neighbour.
MIN_RUN_M = 30.0
#: A single vertex-to-vertex segment at least this long is a drawn straight edge.
STRAIGHT_EDGE_MIN_M = 150.0
#: Within this of another assignment's boundary is a shared seam.
SHARED_EDGE_M = 15.0
#: Corridor classifier, from the report.
CORRIDOR_MIN_ASPECT = 3.0
CORRIDOR_MAX_MEAN_WIDTH_M = 250.0
CORRIDOR_AXIS_MIN_RATIO = 0.5
#: A feature within this of the target counts as inside it. A trail drawn as
#: the edge of an assignment is the assignment's trail; whether the polygon
#: was snapped a hair inside or outside it is not a fact about the ground.
INTERIOR_BUFFER_M = 0.5

#: Matching tolerance per class, ground metres. Mapped classes come first,
#: because ties are broken by this order: on a tie the feature someone
#: surveyed wins over one the DEM inferred.
TOLERANCE_M = {"road": 20.0, "trail": 20.0, "waterway": 25.0, "barrier": 20.0,
               "water_edge": 25.0, "drainage": 30.0, "ridge": 30.0}
CLASSES = tuple(TOLERANCE_M)

#: OSM route_type -> boundary class. Since OSM query version 4, a
#: distribution line (`power=minor_line`) is a barrier too, so a boundary
#: re-staged at version 4 beside one reports it in `barrier`; the
#: composition moves because the data grew, not because the method changed.
ROUTE_CLASS = {"TRAIL": "trail", "TRACK": "road", "ROAD": "road", "ROAD_OTHER": "road",
               "WATERWAY": "waterway", "BARRIER": "barrier", "WATER_EDGE": "water_edge"}

MISSING_OSM_WARNING = "osm.json missing; trails, roads, waterways and barriers unknown"

#: A directory staged by an older Overpass query holds fewer features than
#: one staged today, and nothing in the file says so unless the version is
#: read: "no fence on this edge" and "fences were never fetched" are
#: different facts. The version is known when the warning is written, so it
#: says exactly what is missing: before version 2, every barrier, every
#: water edge and some road types; at 2 or 3, distribution power lines only
#: (version 3 added parking, which a boundary does not read).
STALE_OSM_QUERY_WARNING = (
    "this directory was staged with an older OpenStreetMap query (v{version}); "
    "barriers and water edges were not fetched and some road types are missing. "
    "Re-stage (prepare_terrain force=True) to include them.")

# `OSM_QUERY_VERSION_BARRIERS` (2), the version that first fetched barriers
# and water edges, is `terrain.linear`'s, imported above and re-exported
# here under the name this module has always had.

STALE_OSM_MINOR_LINES_WARNING = (
    "this directory was staged with an older OpenStreetMap query (v{version}); "
    "distribution power lines (power=minor_line) were not fetched, though every "
    "other barrier was. Re-stage (prepare_terrain force=True) to include them.")


class Size(BaseModel):
    area_acres: float | None
    perimeter_m: float | None
    length_m: float
    width_m: float | None
    mean_width_m: float | None


class Boundary(BaseModel):
    composition_pct: dict[str, float]
    on_feature_pct: float
    straight_pct: float
    straight_freehand_pct: float
    shared_pct: float | None
    shared_with: list[str]
    shared_reason: str | None
    runs: list[dict]


class BoundaryAnatomy(BaseModel):
    shape: str                       # corridor | block | line
    axis_feature: str | None
    axis_ratio: float | None
    size: Size
    boundary: Boundary
    interior: dict
    norm_flags: list[dict]
    assumptions: dict
    warnings: list[str]


# --- lines by class ----------------------------------------------------------------


def class_lines(ctx: TerrainContext) -> tuple[dict[str, list[dict]], list[str]]:
    """Every line the anatomy matches to, by class, as Web Mercator shapely
    geometries with the id and name an agent can cite. Drainages and ridges
    are traced over the whole staged raster, not clipped to the subject,
    because a boundary sits on a feature whichever side of it the polygon
    lies."""
    from shapely.geometry import shape as shapely_shape

    lines: dict[str, list[dict]] = {cls: [] for cls in CLASSES}
    warnings: list[str] = []

    payload = ctx.osm_payload
    if payload is None:
        warnings.append(MISSING_OSM_WARNING)
    else:
        # Absent means version 1: the key was added with version 2, so every
        # file without it was written by the query before it.
        version = int(payload.get("query_version", 1))
        # Against the minor-lines version, not the current one: a boundary
        # reads nothing a later query added (5: road class, access, gates),
        # so a version-4 file is complete for it.
        if version < OSM_QUERY_VERSION_MINOR_LINES:
            template = (STALE_OSM_QUERY_WARNING if version < OSM_QUERY_VERSION_BARRIERS
                        else STALE_OSM_MINOR_LINES_WARNING)
            warnings.append(template.format(version=version))
        for route in payload.get("routes", []):
            cls = ROUTE_CLASS.get(route["properties"].get("route_type"))
            if cls is None:
                continue
            lines[cls].append({"id": route["id"], "name": route["properties"].get("name"),
                               "geom": shapely_shape(route["geometry"])})

    grid = ctx.grid
    min_area = HYDRO_ASSUMPTIONS["min_contributing_area_m2"]
    for cls, flow in (("drainage", ctx.flow), ("ridge", ctx.inverted_flow)):
        for channel in extract_channels(flow.accumulation, flow.network, grid.transform,
                                        pixel_size_m=grid.pixel_size_m, min_area_m2=min_area,
                                        clip_to=None, kind=cls, id_prefix=cls):
            lines[cls].append({"id": channel["id"], "name": None,
                               "geom": shapely_shape(channel["geometry"])})
    return lines, warnings


# --- the ground under one staged directory -------------------------------------------


#: The ground's warning that a drainage or ridgeline runs off the raster, by
#: the axis it is about. hasty_candidates drops both: it lists neither axis,
#: and the warnings speak only of those axes' ranking.
EDGE_WARNINGS: dict[str, str] = {
    axis: (f"{kind} reaches the raster edge; the ground draining into it from "
           "outside cannot be seen, so its rank among the others is understated")
    for axis, kind in (("drainage", "a drainage"), ("ridge", "a ridgeline"))}


@dataclass
class Ground:
    """Everything about a staged directory that every describe shares.

    The grid, the class lines with their spatial index, the slope and the
    landform classes are facts about the ground, not about any one target,
    and the two pysheds passes behind the drainage and ridge lines are by far
    the most expensive thing here. Built once per directory and reused for
    every target: phase 2 describes hundreds of candidates against one
    staging, and rebuilding this per candidate would dominate the whole run.

    Map features are deliberately NOT here. They are the state of a map that
    can change between two calls a minute apart, so they are fetched fresh
    per call, with forced revalidation, and passed to `describe_boundary`.
    """

    data_dir: Path
    grid: object
    scale: float                      # ground metres per Web Mercator unit
    lines: dict[str, list[dict]]
    trees: dict[str, object]          # class -> shapely.STRtree over its geoms (or None)
    slope_deg: np.ndarray
    landform_classes: np.ndarray
    warnings: list[str] = field(default_factory=list)
    #: osm.json was not staged, so trails, roads, waterways and barriers are
    #: unknown here - an absence of mapped lines, not a fact that there are none.
    osm_missing: bool = False

    @property
    def staged_bounds(self):
        """The staged raster's own extent as a Web Mercator polygon.

        This is the MEASURED GROUND: the subject plus the buffer it was staged
        with, and the only ground on which slope, climb, effort and travel
        time exist at all. It is therefore the honest limit of a LINE
        candidate - a team is handed a trail rather than the arc where a
        circle happened to cut it, but it cannot be handed terrain nobody
        measured. Read from the raster's own transform and shape, not from
        the manifest, which carries placeholders in older directories and in
        test fixtures.
        """
        from shapely.geometry import box

        return box(*raster_extent_3857(self.grid))


def load_ground(data_dir: Path) -> Ground:
    """Read the staged directory and measure everything a target is measured
    against."""
    import shapely

    from ..terrain.landform import classify_landforms

    ctx = TerrainContext(data_dir)
    lines, warnings = class_lines(ctx)
    osm_missing = MISSING_OSM_WARNING in warnings
    grid = ctx.grid

    trees: dict[str, object] = {}
    for cls, entries in lines.items():
        trees[cls] = (shapely.STRtree(np.array([e["geom"] for e in entries], dtype=object))
                      if entries else None)

    min_area = HYDRO_ASSUMPTIONS["min_contributing_area_m2"]
    for axis, flow in (("drainage", ctx.flow), ("ridge", ctx.inverted_flow)):
        if upstream_is_truncated(flow.accumulation, grid.pixel_size_m, min_area):
            warnings.append(EDGE_WARNINGS[axis])
    if ctx.has_voids:
        warnings.append(DEM_VOID_WARNING)

    slope_deg = ctx.slope_aspect.slope_deg
    return Ground(
        data_dir=Path(data_dir), grid=grid,
        scale=ground_meters_per_mercator_unit(grid.latitude_deg),
        lines=lines, trees=trees, slope_deg=slope_deg,
        landform_classes=classify_landforms(grid.elevation, slope_deg),
        warnings=warnings, osm_missing=osm_missing,
    )


#: fastmcp runs sync tools in a threadpool, so two concurrent first describes
#: on one directory would otherwise both build the Ground - the expensive
#: thing this cache exists to do once. One lock PER directory: a single lock
#: serialised first describes on unrelated stagings behind each other's
#: pysheds passes, which is seconds of waiting for no shared work. The meta
#: lock below guards the dict alone and is held for a dictionary lookup, never
#: across a load. The dict is never pruned; its entries are a path string and
#: a lock, and a process sees a handful of staged directories.
_GROUND_LOCKS: dict[str, threading.Lock] = {}
_GROUND_LOCKS_LOCK = threading.Lock()


def _ground_lock(key: str) -> threading.Lock:
    """The lock for one resolved directory, created on first ask."""
    with _GROUND_LOCKS_LOCK:
        return _GROUND_LOCKS.setdefault(key, threading.Lock())


@lru_cache(maxsize=4)
def _ground_cached(data_dir: str, stamp: tuple | None) -> Ground:
    return load_ground(Path(data_dir))


def _manifest_stamp(data_dir: Path) -> tuple | None:
    try:
        st = (Path(data_dir) / "manifest.json").stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def ground_for(data_dir: str) -> Ground:
    """`load_ground`, cached by directory AND by its manifest's stamp, so a
    re-staged directory (same path, new files - what `prepare_terrain` does
    when a boundary moved) is loaded again rather than served stale.

    `stage_terrain` writes `manifest.json` last, which is what makes the
    stamp a sound barrier: a manifest with a new mtime and size means every
    other file in the directory is already the new one. Small cache, keyed by
    a string because `lru_cache` needs something hashable - the same
    reasoning as `terrain/shapes._grid_for`.

    The lock is per directory, so a first describe on one staging does not
    wait behind a first describe on another.
    """
    key = str(Path(data_dir).resolve())
    with _ground_lock(key):
        return _ground_cached(key, _manifest_stamp(Path(data_dir)))


# Tests and long-lived callers reach for these on `ground_for` itself.
ground_for.cache_clear = _ground_cached.cache_clear
ground_for.cache_info = _ground_cached.cache_info


def class_distances(samples, lines: dict[str, list[dict]], scale: float,
                    trees: dict[str, object] | None = None
                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Ground metres from every boundary sample to the nearest line of each
    class, and which line that was; infinity where a class has no lines.

    Measured against the line geometries themselves rather than by reading a
    rasterised Euclidean distance field: a field is quantised to the pixel,
    and at 10 m pixels that error is the same size as the difference between
    two of these tolerance classes.

    `trees` are the Ground's prebuilt indexes; without them one is built
    here, which is what a bare call in a test wants.
    """
    import shapely

    points = shapely.points(np.array([(s[0], s[1]) for s in samples], dtype="float64"))
    distances: dict[str, np.ndarray] = {}
    nearest: dict[str, np.ndarray] = {}
    for cls, entries in lines.items():
        if not entries:
            distances[cls] = np.full(len(samples), np.inf, dtype="float64")
            nearest[cls] = np.zeros(len(samples), dtype="int64")
            continue
        geoms = np.array([e["geom"] for e in entries], dtype=object)
        tree = (trees or {}).get(cls) or shapely.STRtree(geoms)
        index = tree.nearest(points)
        distances[cls] = shapely.distance(points, geoms[index]) * scale
        nearest[cls] = np.asarray(index, dtype="int64")
    return distances, nearest


# --- sampling the boundary ----------------------------------------------------------


def sample_boundary(line_3857, spacing_units: float) -> list[tuple[float, float, float]]:
    """(x, y, distance along in Mercator units) every `spacing_units`,
    including both ends."""
    total = line_3857.length
    if total == 0:
        # Read through the parts: a multi-part walk has no `.coords` of its own.
        first = walk_parts(line_3857)[0]
        return [(first.coords[0][0], first.coords[0][1], 0.0)]
    count = max(2, int(math.ceil(total / spacing_units)) + 1)
    out = []
    for i in range(count):
        d = min(total, i * spacing_units)
        p = line_3857.interpolate(d)
        out.append((p.x, p.y, d))
    return out


def classify_samples(distances: dict[str, np.ndarray]) -> list[str]:
    """Nearest class within its tolerance per sample, ranked by
    distance/tolerance so a close trail beats a marginal drainage; 'none'
    when nothing qualifies. Ties go to the first class in `TOLERANCE_M`,
    which is why the mapped classes are listed before the inferred ones."""
    count = len(next(iter(distances.values())))
    classes: list[str] = []
    for i in range(count):
        best, best_ratio = "none", None
        for cls, tol in TOLERANCE_M.items():
            ratio = distances[cls][i] / tol
            if ratio <= 1.0 and (best_ratio is None or ratio < best_ratio):
                best, best_ratio = cls, ratio
        classes.append(best)
    return classes


def runs_from_samples(samples_m, classes, spacing_m: float, *,
                      total_m: float | None = None) -> list[dict]:
    """Consecutive samples of one class -> [{follows, start_m, end_m}], with
    `samples_m` carrying distances along in ground metres.

    `total_m` says the samples walk a closed ring and do NOT repeat its start
    point: the seam between the last run and the first is put half way round
    the gap between the last sample and the start, so the runs cover the ring
    exactly once. Without it the walk is an open line, and the first run
    starts at zero while the last ends on the final sample.
    """
    runs: list[dict] = []
    for (x, y, pos_m), cls in zip(samples_m, classes):
        if runs and runs[-1]["follows"] == cls:
            runs[-1]["end_m"] = pos_m
        else:
            runs.append({"follows": cls, "start_m": pos_m, "end_m": pos_m})
    if not runs:
        return runs
    # Each run extends half a spacing either side of its samples.
    for run in runs:
        run["start_m"] = run["start_m"] - spacing_m / 2
        run["end_m"] = run["end_m"] + spacing_m / 2
    if total_m is None:
        runs[0]["start_m"] = 0.0
        runs[-1]["end_m"] = samples_m[-1][2]
    else:
        seam = (samples_m[-1][2] + total_m) / 2.0
        runs[-1]["end_m"] = seam
        runs[0]["start_m"] = seam - total_m
    return runs


def fold_ring_seam(runs: list[dict], total_m: float) -> list[dict]:
    """Fold a ring's last run into its first when they are the same class.

    A ring has no first or last edge. Where the sampler began is an accident
    of how the polygon was drawn, and reporting one trail-following edge as
    two runs because the drawing started half way along it describes the
    drawing, not the ground. The folded run keeps its length by starting
    before zero; `ring_run_order` puts that start back in range.
    """
    if len(runs) < 2 or runs[0]["follows"] != runs[-1]["follows"]:
        return [dict(run) for run in runs]
    folded = [{**runs[0], "start_m": runs[-1]["start_m"] - total_m}]
    folded += [dict(run) for run in runs[1:-1]]
    return folded


def ring_run_order(runs: list[dict], total_m: float) -> list[dict]:
    """Every run of a ring reported from a start in [0, total_m), in order
    around it. The run that crosses the seam carries `wraps` and an `end_m`
    past `total_m`, which is how a reader tells "starts at 615 m and runs
    through the start point" from "starts at 615 m and stops"."""
    if not runs:
        return runs
    if len(runs) == 1:
        return [{**runs[0], "start_m": 0.0, "end_m": total_m, "wraps": False}]
    out = []
    for run in runs:
        length = run["end_m"] - run["start_m"]
        start = run["start_m"] % total_m
        out.append({**run, "start_m": start, "end_m": start + length,
                    "wraps": start + length > total_m + 1e-9})
    out.sort(key=lambda run: run["start_m"])
    return out


def merge_short_runs(runs: list[dict], min_m: float, *, ring: bool = False) -> list[dict]:
    """Absorb runs shorter than `min_m` into the longer neighbour, then join
    neighbours of one class, until nothing short remains or one run is left.

    `ring` says the runs close on themselves, as a polygon's do: the last run
    is the first run's neighbour, and the first the last's. Without it the two
    ends of the list are treated as ends of a line, so a short run that
    happens to land next to the seam is absorbed by the only neighbour it is
    allowed rather than by the longer of its two real ones - and where the
    polygon was drawn from then decides which class gets the credit. On a real
    boundary that moved 20 m between classes for a rotation of the start
    vertex, which is a fact about the drawing, not the ground.
    """
    runs = [dict(r) for r in runs]
    total_m = sum(r["end_m"] - r["start_m"] for r in runs)
    while len(runs) > 1:
        lengths = [r["end_m"] - r["start_m"] for r in runs]
        i = min(range(len(runs)), key=lambda k: lengths[k])
        if lengths[i] >= min_m:
            break
        count = len(runs)
        if ring:
            left, right = runs[(i - 1) % count], runs[(i + 1) % count]
        else:
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < count else None
        if left is None or (right is not None and
                            (right["end_m"] - right["start_m"]) > (left["end_m"] - left["start_m"])):
            # The absorbing run reaches BACK over this one; across the seam
            # that is a lap short of where the run sits, hence the -total.
            right["start_m"] = runs[i]["start_m"] - (total_m if ring and i == count - 1 else 0.0)
        else:
            left["end_m"] = runs[i]["end_m"] + (total_m if ring and i == 0 else 0.0)
        del runs[i]
        runs = _join_same_class(runs)
        if ring:
            # Put the starts back in [0, total_m) and re-fold the seam, so the
            # next pass sees the ring in order with its lengths still summing
            # to the perimeter.
            runs = ring_run_order(fold_ring_seam(runs, total_m), total_m)
    return runs


def _join_same_class(runs: list[dict]) -> list[dict]:
    joined: list[dict] = []
    for run in runs:
        if joined and joined[-1]["follows"] == run["follows"]:
            joined[-1]["end_m"] = run["end_m"]
        else:
            joined.append(dict(run))
    return joined


def straight_segments(line_3857, scale: float, min_m: float) -> list[tuple[float, float]]:
    """(start_m, end_m) along the line of every single vertex-to-vertex
    segment at least `min_m` ground metres long.

    Walks PARTS, not one coordinate list: a line candidate's axis is a
    MultiLineString whenever its named way forks or the area cuts it into
    pieces, and `.coords` on a multi-part geometry raises NotImplementedError.
    The running position carries across parts in their own order, which is the
    frame `sample_boundary` interpolates in, so the two still line up.
    """
    out = []
    pos = 0.0
    for part in walk_parts(line_3857):
        coords = list(part.coords)
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            seg = math.hypot(x1 - x0, y1 - y0) * scale
            if seg >= min_m:
                out.append((pos, pos + seg))
            pos += seg
    return out


def walk_parts(line_3857) -> list:
    """A walk as its LineString parts: itself, or the members of a multi."""
    if line_3857.is_empty:
        return []
    if line_3857.geom_type in ("LineString", "LinearRing"):
        return [line_3857]
    return [part for part in getattr(line_3857, "geoms", [])
            if part.geom_type in ("LineString", "LinearRing") and not part.is_empty]


def crossing_probe(walk, position_units: float, half_units: float):
    """A short segment laid ACROSS the boundary at `position_units` along it.

    Two blocks that meet at a corner are not sharing an edge, but every
    sample near that corner is within tolerance of the other's boundary, so
    a plain distance test counts the corner as a seam and inflates the
    shared share by a few points on every polygon. Probing perpendicular to
    the edge asks the question that was meant: does the neighbour run
    alongside this edge, on one side or the other of it?
    """
    from shapely.geometry import LineString

    total = walk.length
    if total == 0:
        return None
    here_units = min(position_units, total)
    direction = _segment_direction(walk, here_units)
    if direction is None:
        step = min(max(half_units * 0.25, 1e-6), total / 2.0)
        ahead = walk.interpolate(min(position_units + step, total))
        behind = walk.interpolate(max(position_units - step, 0.0))
        direction = (ahead.x - behind.x, ahead.y - behind.y)
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    nx, ny = -dy / norm, dx / norm
    here = walk.interpolate(here_units)
    return LineString([(here.x - nx * half_units, here.y - ny * half_units),
                       (here.x + nx * half_units, here.y + ny * half_units)])


def _segment_direction(walk, position_units: float):
    """The direction of the vertex-to-vertex segment the position lies on, or
    None for a walk with no single coordinate list.

    Taken from the segment rather than from a chord spanning the position:
    near a corner a chord straddles the vertex and points diagonally into it,
    which swings the probe into a neighbour that only meets the corner - the
    exact error the probe exists to avoid.
    """
    try:
        coords = list(walk.coords)
    except (AttributeError, NotImplementedError):
        return None
    travelled = 0.0
    fallback = None
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        fallback = (x1 - x0, y1 - y0)
        if position_units <= travelled + length:
            return fallback
        travelled += length
    return fallback


# --- the description ----------------------------------------------------------------


def _target_3857(geometry_wgs84: dict):
    """The target in Web Mercator. Lines are NOT buffered into corridors
    here (project_geometry_to_3857 does that for assignment grids); a line
    target is described as a line."""
    from pyproj import Transformer
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform

    if geometry_wgs84.get("type") in ("LineString", "MultiLineString"):
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        return shapely_transform(to_3857.transform, shapely_shape(geometry_wgs84))
    return project_geometry_to_3857(geometry_wgs84)


def raster_extent_3857(grid) -> tuple[float, float, float, float]:
    """The staged raster's own extent from its transform and shape - the
    manifest's dem_bbox_3857 is not trusted here because test fixtures and
    older directories carry placeholders."""
    t = grid.transform
    rows, cols = grid.elevation.shape
    xs = (t.c, t.c + t.a * cols)
    ys = (t.f, t.f + t.e * rows)
    return min(xs), min(ys), max(xs), max(ys)


def _extent_wgs84(minx: float, miny: float, maxx: float, maxy: float
                  ) -> tuple[float, float, float, float]:
    """(west, south, east, north) for a Web Mercator extent. The projection is
    axis-aligned, so transforming the two opposite corners is the whole
    rectangle."""
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = to_wgs84.transform(minx, miny)
    east, north = to_wgs84.transform(maxx, maxy)
    return west, south, east, north


def _walk_of(target):
    """The line the sampler walks: a line target itself, a polygon's exterior."""
    if target.geom_type in ("LineString", "MultiLineString"):
        return target
    return target.exterior


def _neighbour_line(geometry_wgs84: dict):
    """The line another assignment occupies on the map.

    A polygon neighbour is its own boundary; a LINE neighbour already IS its
    boundary and must not be buffered into a 100 m corridor first - that puts
    its edge 50 m off the line it was drawn on, and a line running half a
    corridor's width away then reads as a shared seam.
    """
    projected = _target_3857(geometry_wgs84)
    if projected.geom_type in ("Polygon", "MultiPolygon"):
        return projected.boundary
    return projected


def describe_boundary(data_dir: Path, geometry_wgs84: dict, *, norms: dict | None = None,
                      resource_type: str = "", map_features: list[dict] | None = None,
                      exclude_label: str | None = None,
                      ground: Ground | None = None) -> BoundaryAnatomy:
    """Describe one polygon or line against the staged ground.

    `map_features` are the features of the map the subject was staged with,
    already read by the caller; `exclude_label` names the target itself when
    it is one of those assignments so it is not its own neighbour.

    `resource_type` is the resource this would be assigned to. It bands the
    area flag against that resource's corpus range instead of the whole
    corpus, and `assumptions.area_band` says which band was used either way.

    `ground` is the staged directory's shared measurements. A caller
    describing many targets against one staging - phase 2's candidates -
    builds it once with `load_ground` and hands it in; a single call leaves
    it out and gets the small per-directory cache.
    """
    norms = norms if norms is not None else load_norms()
    ground = ground if ground is not None else ground_for(str(Path(data_dir).resolve()))
    grid, scale, lines = ground.grid, ground.scale, ground.lines
    warnings = list(ground.warnings)
    # Only the map is read per call; everything else is the Ground's.
    ctx = TerrainContext(data_dir, map_features=map_features)

    target = _target_3857(geometry_wgs84)
    if target.geom_type == "MultiPolygon":
        # A multipart assignment is usually one real block plus a sliver left
        # by an editing accident. Describing the largest part and saying so is
        # a more useful answer than refusing: the alternative was the caller
        # getting nothing at all about a polygon that is 97% one shape.
        parts = sorted(target.geoms, key=lambda part: part.area, reverse=True)
        whole_area = sum(part.area for part in parts)
        share = 100.0 * parts[0].area / whole_area if whole_area else 0.0
        warnings.append(
            f"multipolygon target: described the largest of {len(parts)} parts "
            f"({share:.0f}% of the area); the others are not described")
        target = parts[0]
    minx, miny, maxx, maxy = raster_extent_3857(grid)
    tb = target.bounds
    if tb[0] < minx or tb[1] < miny or tb[2] > maxx or tb[3] > maxy:
        # The caller passed WGS84 and thinks in WGS84; refusing in Web
        # Mercator metres alone leaves them nothing to act on, so the same
        # extent is named in the coordinates their target was written in.
        west, south, east, north = _extent_wgs84(minx, miny, maxx, maxy)
        raise ValueError(
            f"the target lies outside the staged extent {[round(v) for v in (minx, miny, maxx, maxy)]} "
            f"(Web Mercator), which is longitude {west:.5f} to {east:.5f}, latitude "
            f"{south:.5f} to {north:.5f} (WGS84); stage a larger area or a closer one")

    is_line = target.geom_type in ("LineString", "MultiLineString")
    walk = _walk_of(target)
    total_m = walk.length * scale
    if total_m == 0:
        raise ValueError("the target is degenerate: it has no length")
    rect = None
    if not is_line:
        # A line has no enclosing rectangle to speak of, and asking for one
        # says nothing about it; only a polygon is measured this way.
        rect = target.minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            raise ValueError(
                f"the target is degenerate: its minimum rotated rectangle is a "
                f"{rect.geom_type.lower()}, so the polygon encloses no ground")

    spacing_units = SAMPLE_SPACING_M / scale
    samples = sample_boundary(walk, spacing_units)
    # A ring is a polygon's exterior, or a line drawn back to its own start.
    # A MultiLineString reports `is_closed` when each of its parts happens to
    # close, and walking several disjoint loops as one ring would put a seam
    # in the middle of the walk.
    is_ring = (walk.geom_type == "LinearRing"
               or (walk.geom_type == "LineString" and walk.is_closed))
    if is_ring and len(samples) > 1:
        # The closing sample repeats the first: on a ring it is one point,
        # not two, and counting it twice puts a false seam at the start.
        samples = samples[:-1]
    samples_m = [(x, y, d * scale) for x, y, d in samples]
    distances, nearest = class_distances(samples, lines, scale, ground.trees)
    classes = classify_samples(distances)

    runs = runs_from_samples(samples_m, classes, SAMPLE_SPACING_M,
                             total_m=total_m if is_ring else None)
    if is_ring:
        runs = fold_ring_seam(runs, total_m)
    runs = merge_short_runs(runs, MIN_RUN_M, ring=is_ring)
    if is_ring:
        runs = ring_run_order(fold_ring_seam(runs, total_m), total_m)
    positions = [pos_m for _x, _y, pos_m in samples_m]
    for run in runs:
        run["feature_id"], run["feature_name"] = _nearest_feature(
            run, walk, scale, total_m, lines.get(run["follows"], []),
            nearest.get(run["follows"]), positions)

    composition = {cls: 0.0 for cls in (*CLASSES, "none")}
    for run in runs:
        composition[run["follows"]] += run["end_m"] - run["start_m"]
    composition_pct = {cls: (100.0 * v / total_m if total_m else 0.0) for cls, v in composition.items()}
    on_feature_pct = 100.0 - composition_pct["none"]

    straight = straight_segments(walk, scale, STRAIGHT_EDGE_MIN_M)
    straight_m = sum(e - s for s, e in straight)
    freehand_m = 0.0
    for s, e in straight:
        inside = [c for (x, y, d), c in zip(samples_m, classes) if s <= d <= e]
        if inside and sum(1 for c in inside if c == "none") * 2 > len(inside):
            freehand_m += e - s
    straight_pct = 100.0 * straight_m / total_m if total_m else 0.0
    straight_freehand_pct = 100.0 * freehand_m / total_m if total_m else 0.0

    shared_pct, shared_with, shared_reason = None, [], None
    assignments = ctx.map_assignments
    if assignments is None:
        shared_reason = "no map was staged with this area, or the map could not be read"
    else:
        neighbours = []
        for feature in assignments:
            props = feature.get("properties") or {}
            label = f"{props.get('letter') or ''}{props.get('number') or ''}".strip() or props.get("title", "")
            if exclude_label and label.upper() == exclude_label.upper():
                continue
            try:
                neighbours.append((label, _neighbour_line(feature["geometry"])))
            except Exception:
                continue
        if neighbours:
            tol_units = SHARED_EDGE_M / scale
            hits = 0
            for _x, _y, d in samples:
                probe = crossing_probe(walk, d, tol_units)
                if probe is None:
                    continue
                near = [label for label, boundary in neighbours if probe.intersects(boundary)]
                if near:
                    hits += 1
                    for label in near:
                        if label not in shared_with:
                            shared_with.append(label)
            shared_pct = 100.0 * hits / len(samples)
        else:
            shared_pct = 0.0

    if is_line:
        size = Size(area_acres=None, perimeter_m=None, length_m=total_m, width_m=None, mean_width_m=None)
        shape_kind, axis, axis_ratio = "line", None, None
        interior: dict = {}
    else:
        area_m2 = target.area * scale * scale
        rc = list(rect.exterior.coords)
        e1 = math.hypot(rc[1][0] - rc[0][0], rc[1][1] - rc[0][1]) * scale
        e2 = math.hypot(rc[2][0] - rc[1][0], rc[2][1] - rc[1][1]) * scale
        length_m, width_m = max(e1, e2), min(e1, e2)
        mean_width_m = area_m2 / length_m if length_m else 0.0
        size = Size(area_acres=acres(area_m2), perimeter_m=total_m, length_m=length_m,
                    width_m=width_m, mean_width_m=mean_width_m)
        interior, axis_m = _interior(ground, target, warnings)
        aspect = length_m / width_m if width_m else float("inf")
        if aspect >= CORRIDOR_MIN_ASPECT and mean_width_m < CORRIDOR_MAX_MEAN_WIDTH_M:
            shape_kind = "corridor"
            # The axis is chosen from metres inside the BARE polygon, not
            # inside the half-metre skin the interior report uses: a trail
            # drawn along each long edge lies in the skin twice over and
            # would be named the axis of a corridor it never runs down. The
            # ratio is capped at 1 for the same reason - an axis cannot be
            # longer than the corridor it is the axis of.
            # `max` keeps the first maximal element, so a tie between two
            # classes holding the same length inside goes to whichever comes
            # first in CLASSES - the mapped classes before the inferred ones.
            best_cls = max(CLASSES, key=lambda c: axis_m[c])
            axis_ratio = min(1.0, axis_m[best_cls] / length_m) if length_m else 0.0
            axis = best_cls if axis_ratio >= CORRIDOR_AXIS_MIN_RATIO else None
            if axis is None:
                axis_ratio = None
        else:
            shape_kind, axis, axis_ratio = "block", None, None

    facts: dict = {"on_feature_pct": on_feature_pct}
    if shape_kind == "line":
        facts["length_m"] = total_m
    else:
        facts["area_acres"] = size.area_acres
        if shape_kind == "corridor":
            facts["width_m"] = size.mean_width_m
    flags = norm_flags(facts, norms, resource_type)

    return BoundaryAnatomy(
        shape=shape_kind, axis_feature=axis, axis_ratio=axis_ratio, size=size,
        boundary=Boundary(composition_pct=composition_pct, on_feature_pct=on_feature_pct,
                          straight_pct=straight_pct, straight_freehand_pct=straight_freehand_pct,
                          shared_pct=shared_pct, shared_with=shared_with,
                          shared_reason=shared_reason, runs=runs),
        interior=interior, norm_flags=flags,
        assumptions={"tolerances_m": TOLERANCE_M, "sample_spacing_m": SAMPLE_SPACING_M,
                     "min_run_m": MIN_RUN_M, "straight_edge_min_m": STRAIGHT_EDGE_MIN_M,
                     "shared_edge_m": SHARED_EDGE_M,
                     "interior_buffer_m": INTERIOR_BUFFER_M,
                     "interior_buffer_note": "a feature lying on the edge counts as interior",
                     "interior_note": ("interior is empty for a line target; axis_ratio is "
                                       "null for a block and for a corridor with no axis"),
                     "area_band": _area_band_note(norms, resource_type),
                     "runs_note": ("on a closed boundary runs are ordered from the first "
                                   "vertex and one run may carry wraps: true with end_m past "
                                   "the perimeter; on a line the first run starts at 0"),
                     "corridor": {"min_aspect": CORRIDOR_MIN_ASPECT,
                                  "max_mean_width_m": CORRIDOR_MAX_MEAN_WIDTH_M,
                                  "axis_min_ratio": CORRIDOR_AXIS_MIN_RATIO,
                                  "axis_ratio": ("the axis class's interior metres as a "
                                                 "fraction of the corridor's length, at "
                                                 "most 1")},
                     "drainage_and_ridge_threshold_m2": HYDRO_ASSUMPTIONS["min_contributing_area_m2"],
                     "feature_ids": ("drainage and ridge ids are ranked over the whole staged "
                                     "raster; they are not the subject-clipped ids in "
                                     "terrain_metrics or identify_features"),
                     "note": ("drainage and waterway are separate classes here; the corpus "
                              "combined them under 'drainage'")},
        warnings=warnings,
    )


def _area_band_note(norms: dict, resource_type: str) -> str:
    """Which band the area flag was read against, in words.

    A flag that says "below the band" is only actionable if the reader knows
    WHICH band, and a describe that raises no flag at all says nothing at
    all - so the band is stated whether or not anything was flagged.
    """
    key = (resource_type or "").strip().upper()
    if key and resource_median_ha(key, norms) is not None:
        spread = target_area_range_acres(key, norms)
        return (f"area_acres, where the target has one, was banded against the {key} target "
                f"range ({spread['p25']:.3g} to {spread['p75']:.3g} acres, p25-p75 derived "
                f"from that resource's corpus median of {spread['p50']:.3g} acres); the other "
                "metrics "
                "are banded against the whole corpus, which has no per-resource figures "
                "for them")
    reason = (f"the corpus has no median for resource_type {key}" if key
              else "no resource_type was given")
    return (f"area_acres, where the target has one, was banded against the whole corpus "
            f"p10-p90 because {reason}; pass a resource_type the corpus knows to band it "
            "against that resource instead")


def _nearest_feature(run: dict, walk, scale: float, total_m: float, entries: list[dict],
                     nearest: np.ndarray | None, positions: list[float]):
    """Which line of the run's class the run follows.

    The feature most of the run's OWN samples were nearest to, not the one
    nearest its midpoint: a run that follows a trail to a junction and
    continues on the trail beyond it is named by the trail it mostly is,
    and a midpoint that happens to fall on the junction cannot decide it.
    The midpoint is the fallback for a run holding no sample at all.
    """
    if not entries or run["follows"] == "none":
        return None, None
    if nearest is not None and len(nearest):
        indices = [nearest[i] for i in _run_sample_indices(run, positions, total_m)]
        if indices:
            counts = Counter(int(i) for i in indices)
            best = entries[min(counts, key=lambda index: (-counts[index], index))]
            return best["id"], best.get("name")
    mid = walk.interpolate((((run["start_m"] + run["end_m"]) / 2.0) % total_m) / scale)
    best = min(entries, key=lambda e: e["geom"].distance(mid))
    return best["id"], best.get("name")


def _run_sample_indices(run: dict, positions: list[float], total_m: float) -> list[int]:
    """The samples that fall inside a run, wrapping past the seam when the run
    does."""
    start, end = run["start_m"], run["end_m"]
    if end > total_m:
        wrapped = end - total_m
        return [i for i, pos in enumerate(positions) if pos >= start or pos <= wrapped]
    return [i for i, pos in enumerate(positions) if start <= pos <= end]


def _interior(ground: Ground, target, warnings: list[str]) -> tuple[dict, dict]:
    """The interior report, and the same class metres measured against the
    BARE polygon.

    The report counts a feature lying on the edge as held, which is what a
    planner means by "this block has the trail in it". The second set is for
    the corridor axis, where an edge-drawn feature is not the axis at all and
    counting it once per edge would say a two-edge trail runs twice the
    corridor's length.
    """
    from rasterio.features import rasterize

    from ..terrain.landform import landform_fractions

    grid, scale = ground.grid, ground.scale
    out: dict = {}
    axis_m: dict = {}
    # A line drawn as the target's own edge lies ON it, and whether the
    # polygon was snapped a hair inside or outside is not a fact about the
    # ground: measure against a half-metre skin, not the bare polygon.
    held = target.buffer(INTERIOR_BUFFER_M / scale)
    for cls in CLASSES:
        metres = 0.0
        bare = 0.0
        for entry in ground.lines[cls]:
            inter = entry["geom"].intersection(held)
            if not inter.is_empty:
                metres += inter.length * scale
            core = entry["geom"].intersection(target)
            if not core.is_empty:
                bare += core.length * scale
        out[f"{cls}_m"] = metres
        axis_m[cls] = bare
    mask = rasterize([(target, 1)], out_shape=grid.elevation.shape,
                     transform=grid.transform, fill=0, all_touched=False).astype(bool)
    if not mask.any():
        # A corridor thinner than a pixel contains no pixel centre. Touched
        # pixels are wider than the target, so say so rather than report
        # nothing.
        mask = rasterize([(target, 1)], out_shape=grid.elevation.shape,
                         transform=grid.transform, fill=0, all_touched=True).astype(bool)
        if mask.any():
            warnings.append("interior statistics use touched pixels; "
                            "the target is thinner than a pixel")
    slope = ground.slope_deg[mask]
    slope = slope[~np.isnan(slope)]
    out["slope_mean_deg"] = float(slope.mean()) if slope.size else None
    out["slope_p90_deg"] = float(np.percentile(slope, 90)) if slope.size else None
    out["landform_fractions"] = (landform_fractions(ground.landform_classes, mask)
                                 if mask.any() else {})
    return out, axis_m
