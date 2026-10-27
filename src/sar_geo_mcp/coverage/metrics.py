"""The coverage contract: what a team's tracks actually covered.

Second input to the Phase 8 debrief agent. Everything here is computed;
no model authors any of it. `effort_coverage` is the Koopman coverage
number (swept area over zone area) and is reported BESIDE a searcher's
claimed POD, never instead of it - the pair is the point.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyproj
from pydantic import BaseModel, Field
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from ..geo import ground_meters_per_mercator_unit
from ..terrain.grid import read_subject
from ..terrain.metrics import TerrainMetrics
from .corridor import (
    DEFAULT_CORRIDOR_HALF_WIDTH_M,
    GAP_CLOSE_MAX_LATERAL_FACTOR,
    compute_corridor_coverage,
)
from .swath import CELL_GROUND_M, build_grid, burn_swath, close_gaps, find_gaps
from .tracks import TrackSelectionError, select_tracks

SCHEMA_VERSION = "1.1.0"

# The schema_version the entered/exited/searching-hours timestamp fields
# (Phase 8b) first appeared in - pinned separately from SCHEMA_VERSION (not
# read from it) so a future, unrelated version bump does not silently change
# what `is_stale` below considers "predates the timestamp feature". A real
# operator-reported bug (phase9d): out/QA103/coverage_metrics.json was
# computed before this feature landed but still validated fine (the fields
# just default to None), and the debrief silently carried those nulls onto
# SAR-110 fields 11-13 instead of flagging the stale input.
TIMESTAMPS_SINCE = (1, 1, 0)

# A track whose in-zone length is at or below this fraction of its total
# looks like a transit recording rather than a search of this assignment.
# Geometry alone cannot tell the two apart (a searcher may genuinely have
# only grazed the boundary), so this is surfaced as an informational note
# for the debrief agent to ask about - never an exclusion (operator
# decision, spec SS5.2 amended 2026-08-17: no transit classification).
MOSTLY_OUTSIDE_FRACTION = 0.10


class TrackSummary(BaseModel):
    name: str
    length_in_zone_m: float
    length_total_m: float
    in_zone_fraction: float = Field(ge=0.0, le=1.0)
    # ISO-8601 UTC, from the track's own GPS timestamps (see
    # tracks.SelectedTrack.time_entered_ms / time_exited_ms). None when the
    # track carries no timestamps at all - never a guess.
    time_entered: str | None = None
    time_exited: str | None = None


class ShapeOverlap(BaseModel):
    id: str
    #: The registry kind of this shape ("drainage", "steep_band", ...).
    #:
    #: Carried because a consumer holding only the id has to parse it -
    #: `steep_35_0` is a steep band, `cliff_2` a cliff band - and an id
    #: format is not an interface. The feature registry owns what kinds
    #: exist; `_attribute_gaps` has the kind in hand while building this,
    #: so it says it rather than leaving every reader to guess.
    #:
    #: Defaulted, not required: coverage_metrics.json files written before
    #: this existed must still load, and there are real ones on disk. An
    #: empty kind means "written by an older version", never "no kind".
    kind: str = ""
    # Fraction of the SHAPE's own extent (area for a polygon, length for a
    # line) that falls inside the gap - not the fraction of the gap the
    # shape covers. "intersects" cannot distinguish a shape that is 90%
    # unsearched from one merely grazed at an edge; this can.
    fraction_in_gap: float = Field(ge=0.0, le=1.0)


class GapSummary(BaseModel):
    id: str
    # Exactly one of area_m2 / length_m is populated, matching
    # assignment_kind ("area" gaps are polygons; "line" gaps are corridor
    # segments with no area at all - a null area_m2 there is honest, a 0.0
    # would lie).
    area_m2: float | None = None
    length_m: float | None = None
    geometry: dict
    dominant_vegetation: str | None = None
    mean_slope_deg: float | None = None
    # Registry shapes with a non-trivial fraction of their own extent inside
    # this gap, largest fraction first. Not "contains": most real shapes are
    # only partly in a gap (e.g. 36% of steep_35_0), and a bare containment
    # list would assert the whole shape went unsearched, which is false.
    unsearched_shapes: list[ShapeOverlap] = []


class ClosedGap(BaseModel):
    """A gap suppressed as GPS noise (Phase 9i) - NOT in `gaps`, does not
    count as unsearched, and is never asked about in the debrief, but never
    silently dropped either: this is the record a reviewer checks to see
    what got waved through and why. See corridor.ClosedGap (line) and
    swath.ClosedArea (area), whose fields this mirrors."""
    id: str
    # Exactly one of area_m2 / length_m is populated, matching
    # assignment_kind - same area/length duality as GapSummary above.
    area_m2: float | None = None
    length_m: float | None = None
    # The track's worst (maximum) distance from the route across the gap -
    # what corridor.py's lateral-consistency test measured. Line-only: an
    # area closure has no single equivalent number (a raster region has no
    # one "route" to measure sideways from), so this stays null there -
    # honest, not an invented 0.0.
    max_lateral_m: float | None = None
    geometry: dict


class CoverageMetrics(BaseModel):
    schema_version: str = SCHEMA_VERSION
    assignment_label: str
    # "area" (Koopman swept-area/zone-area math) or "line" (corridor math) -
    # lets the Phase 8 debrief agent phrase questions correctly ("the middle
    # 400 m of the trail" vs "2 acres of the northeast corner").
    assignment_kind: Literal["area", "line"] = "area"
    teams: list[str]
    effective_sweep_width_m: float
    covered_fraction: float = Field(ge=0.0, le=1.0)
    # Koopman coverage number C = effort / zone area - area math, so it is
    # meaningless for a line assignment (no area to divide by) and null
    # there rather than a misleading 0.0 or an invented value.
    effort_coverage: float | None
    zone_area_m2: float | None
    # Populated only for assignment_kind == "line" - null area_m2 above
    # already tells the reader a line document carries no zone area; this is
    # the corresponding "carries no line length" signal for area documents.
    line_length_m: float | None = None
    tracks: list[TrackSummary]
    gaps: list[GapSummary]
    # Gaps suppressed as GPS noise (Phase 9i) - disjoint from `gaps` above,
    # not a subset filtered back out of it. See ClosedGap.
    closed_gaps: list[ClosedGap] = []
    # What to check before trusting the numbers above, each a sentence: the
    # names the tracks were chosen by (a drone flight or probability ring
    # counted when nothing was plainly named for the team, one set aside
    # beside a plainly named track, a name with no team number), a track
    # mostly outside, gaps closed as GPS noise, a default corridor width.
    # Empty means none of THESE was found - not that every team named has a
    # track here: a team with no line reaching the assignment is not warned
    # of, so `tracks` is where to see whose lines were counted.
    warnings: list[str] = []
    # Earliest time_entered / latest time_exited across every track (not
    # necessarily the same track at each end) - the debrief's computed "you
    # entered about HH:MM, exited HH:MM" claim. time_searching_hours is the
    # elapsed span between them; all three are null together when no track
    # carries a timestamp.
    time_entered: str | None = None
    time_exited: str | None = None
    time_searching_hours: float | None = None


def _parse_version(value: str) -> tuple[int, int, int]:
    """"1.1.0" -> (1, 1, 0). Anything that doesn't parse (a foreign or
    corrupted string) sorts as (0, 0, 0) - the safe "treat as ancient"
    default, so a malformed schema_version warns instead of crashing."""
    try:
        major, minor, patch = (int(part) for part in value.split("."))
        return (major, minor, patch)
    except (ValueError, TypeError, AttributeError):
        return (0, 0, 0)


def is_stale(metrics: CoverageMetrics) -> bool:
    """True when this coverage document cannot be trusted to carry a real
    entered/exited/searching-hours window: either its schema predates the
    timestamp feature outright, or (defensively) it claims a current schema
    but the times are null anyway despite having selected tracks - both are
    the same failure from a debrief's point of view (operator bug report,
    phase9d: a stale coverage_metrics.json was consumed silently, leaving
    SAR-110 fields 11-13 blank when a re-run would have filled them)."""
    old_schema = _parse_version(metrics.schema_version) < TIMESTAMPS_SINCE
    times_missing_despite_tracks = (
        metrics.time_entered is None and metrics.time_exited is None
        and bool(metrics.tracks)
    )
    return old_schema or times_missing_despite_tracks


@dataclass(frozen=True)
class CoverageResult:
    """`compute_coverage`'s full output: the LLM-facing digest plus geometry.

    The digest (`metrics`) is read by a model every debrief turn, so it must
    stay small - track geometry (a dense GPS polyline, easily 1000+ vertices)
    does not belong in it. `track_geojson` carries that geometry for the one
    consumer that needs it, the renderer (see render.annotate_coverage),
    packaged as a FeatureCollection ready to write straight to a sidecar
    file. Splitting the return this way, rather than having
    `compute_coverage` write the sidecar itself, keeps the function a pure
    computation - it takes data in and returns data out, with no I/O beyond
    the read-only terrain/assignment lookups it already does. The caller
    (cli.run_coverage) decides what gets written where.
    """
    metrics: CoverageMetrics
    track_geojson: dict


def compute_coverage(data_dir: Path, teams: list[str], features: list[dict],
                     corridor_width_m: float | None = None) -> CoverageResult:
    """Coverage for one prepared assignment directory.

    `corridor_width_m` is the FULL corridor width (ground metres) for a line
    assignment, overriding both the terrain-derived ESW and the no-terrain
    default; it is ignored for area assignments.
    """
    data_dir = Path(data_dir)
    assignment = read_subject(data_dir)

    to_mercator = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform
    zone = shapely_transform(to_mercator, shape(assignment["geometry"]))
    is_line = zone.geom_type in ("LineString", "MultiLineString")
    is_area = zone.geom_type in ("Polygon", "MultiPolygon")
    if not is_line and not is_area:
        raise ValueError(
            f"assignment geometry type {zone.geom_type!r} is not coverage-able "
            "(expected a polygon or a line assignment)"
        )

    metrics_path = data_dir / "terrain_metrics.json"
    terrain = None
    if metrics_path.exists():
        terrain = TerrainMetrics.model_validate_json(
            metrics_path.read_text(encoding="utf-8")
        )
    elif is_area:
        # Line assignments have a terrain-optional fallback (see
        # _compute_line_coverage); area coverage has no substitute for the
        # vegetation-adjusted sweep width, so this stays a hard requirement.
        raise FileNotFoundError(
            f"{metrics_path} not found. Call the `terrain_metrics` tool on "
            f"this handle first - coverage needs the computed sweep width."
        )

    # CalTopo features are WGS84; select_tracks requires `zone` and the
    # features to already share a coordinate system (see tracks.py), and
    # `zone` above is Web Mercator. Without this, a real track's coordinates
    # (lon/lat degrees) can never spatially intersect the projected zone
    # (Mercator metres) - select_tracks would raise on every real map.
    mercator_features = _project_features_to_mercator(features, to_mercator)

    if is_area:
        return _compute_area_coverage(teams, zone, terrain, mercator_features)
    return _compute_line_coverage(data_dir, teams, zone, terrain, mercator_features,
                                  corridor_width_m)


def _compute_area_coverage(teams: list[str], zone, terrain: TerrainMetrics,
                           features: list[dict]) -> CoverageResult:
    factor = ground_meters_per_mercator_unit(terrain.data_quality.latitude_deg)
    sweep_width = terrain.traverse.effective_sweep_width_m

    # clip_margin is half the sweep width, converted from ground metres to
    # the zone's own units (Mercator) via ground_factor - a searcher walking
    # just outside the assignment boundary genuinely sweeps into it
    # (operator decision, spec SS5.2 amended 2026-08-17: no transit
    # classification; clip everything and use what remains).
    clip_margin = (sweep_width / 2) / factor
    # What the operator must check about the names the tracks were chosen
    # by - a drone flight counted for want of a plainly named track, a line
    # set aside - first: whether these are the team's lines decides whether
    # any number below is the team's coverage.
    name_warnings: list[str] = []
    selected = select_tracks(features, zone, teams, ground_factor=factor,
                             clip_margin=clip_margin, warnings=name_warnings)

    # A narrow ESW (e.g. a ~4 m brush corridor) can be quantised away on the
    # default 2 m grid - half the sweep width would collapse to less than
    # one cell. Use a finer cell so the swath still resolves; floor of 0.5 m
    # keeps the raster from ballooning for a vanishingly small ESW.
    cell_ground_m = CELL_GROUND_M
    if sweep_width / 3 < CELL_GROUND_M:
        cell_ground_m = max(0.5, sweep_width / 3)

    grid = build_grid(zone, cell_ground_m=cell_ground_m, ground_factor=factor)
    swath = burn_swath(grid, [t.geometry for t in selected], sweep_width)
    # Fold in cells lost to raster-scale GPS jitter between adjacent sweep
    # lanes before anything downstream (covered_fraction, gaps) sees the
    # swath - see swath.close_gaps for the closing-radius derivation and
    # the guarantee that it only ADDS cells, never removes real coverage.
    closed_swath, closed_regions = close_gaps(grid, swath, sweep_width)
    zone_area = zone.area * factor * factor

    covered = (
        float(closed_swath.sum() / grid.zone_mask.sum()) if grid.zone_mask.any() else 0.0
    )
    # effort_coverage (Koopman C = effort / zone area) is computed purely
    # from track length and sweep width - it measures search EFFORT
    # invested, not what the raster resolved as covered, so closing (which
    # only changes which cells the raster counts as covered) leaves it
    # unchanged by construction. covered_fraction is the one that rises.
    effort = (sum(t.length_in_zone_m for t in selected) * sweep_width / zone_area
              if zone_area else 0.0)

    closed_gaps = [ClosedGap(id=r.id, area_m2=r.area_m2, geometry=r.geometry)
                   for r in closed_regions]
    warnings = (name_warnings + _mostly_outside_warnings(selected)
                + _closed_area_warning(closed_gaps))

    gaps = [GapSummary(id=g.id, area_m2=g.area_m2, geometry=g.geometry)
            for g in find_gaps(grid, closed_swath)]
    _attribute_gaps(gaps, terrain)

    time_entered, time_exited, time_searching_hours = _time_window(selected)

    metrics = CoverageMetrics(
        assignment_label=terrain.assignment_label,
        assignment_kind="area",
        teams=list(teams),
        effective_sweep_width_m=sweep_width,
        covered_fraction=min(1.0, covered),
        effort_coverage=effort,
        zone_area_m2=zone_area,
        tracks=_track_summaries(selected),
        gaps=gaps,
        closed_gaps=closed_gaps,
        warnings=warnings,
        time_entered=time_entered,
        time_exited=time_exited,
        time_searching_hours=time_searching_hours,
    )
    return CoverageResult(metrics=metrics, track_geojson=_track_geojson(selected))


def _compute_line_coverage(data_dir: Path, teams: list[str], zone,
                           terrain: TerrainMetrics | None, features: list[dict],
                           corridor_width_m: float | None) -> CoverageResult:
    warnings: list[str] = []

    if corridor_width_m is not None:
        full_width = corridor_width_m
    elif terrain is not None:
        full_width = terrain.traverse.effective_sweep_width_m
    else:
        full_width = DEFAULT_CORRIDOR_HALF_WIDTH_M * 2
        warnings.append(
            f"no terrain_metrics.json for this line assignment; using the "
            f"default {full_width:.0f} m corridor "
            f"({DEFAULT_CORRIDOR_HALF_WIDTH_M:.0f} m half-width). Call the "
            f"`terrain_metrics` tool first for a vegetation-adjusted width."
        )

    if terrain is not None:
        latitude = terrain.data_quality.latitude_deg
        assignment_label = terrain.assignment_label
    else:
        # No terrain_metrics.json to carry these - manifest.json is the same
        # provenance compute_terrain_metrics itself would have read them
        # from (see terrain.metrics.compute_terrain_metrics), and `fetch`
        # always writes it before terrain or coverage ever run.
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        latitude = float(manifest["latitude_deg"])
        assignment_label = manifest.get("assignment", "")

    factor = ground_meters_per_mercator_unit(latitude)
    half_width = full_width / 2

    # Select against the corridor polygon itself (the line buffered by the
    # corridor half-width), not `zone` + clip_margin: select_tracks measures
    # `length_in_zone_m` against the UNBUFFERED zone when a clip_margin is
    # given (see tracks.select_tracks docstring), which is the right area
    # denominator for polygon assignments but a degenerate, near-always-zero
    # one for a zero-width line - it would flag almost every real track as
    # "mostly outside" regardless of how well it actually follows the trail.
    # Passing the pre-buffered corridor as `zone` with clip_margin=0 makes
    # length_in_zone_m mean "length inside the corridor", which is the
    # quantity that is actually meaningful here - while still clipping
    # candidates to exactly the same region the literal reading would have.
    corridor_zone = zone.buffer(half_width / factor)
    # The names the tracks were chosen by are said for this selection only,
    # the one the coverage counts (see `_compute_area_coverage`). The wider
    # one below decides on the same names map-wide, so its fallback would be
    # the same fact said twice, and a line it sets aside that reaches only
    # its wider band would never have counted as covered ground. Kept apart
    # so they go first, as on an area: whether these are the team's lines
    # decides whether any number below - the default corridor's included -
    # is the team's coverage.
    name_warnings: list[str] = []
    selected = select_tracks(features, corridor_zone, teams, ground_factor=factor,
                             clip_margin=0.0, warnings=name_warnings)

    # A second, more generously clipped selection FOR THE CLOSING DECISION
    # ONLY (see compute_corridor_coverage's lateral_track_geometries
    # docstring): `selected` above is clipped tight to the corridor itself
    # (so length_in_zone_m means "length inside the corridor", per the
    # comment above) - which also cuts off any genuine drift the closing
    # rule needs to see to judge a gap fairly. GAP_CLOSE_MAX_LATERAL_FACTOR
    # is the exact boundary beyond which the lateral test could never pass
    # anyway, so clipping to it loses nothing the closing decision could
    # have used. Verified against a real line assignment: without this second selection,
    # only 2 of 4 real GPS-noise gaps closed.
    lateral_zone = zone.buffer((GAP_CLOSE_MAX_LATERAL_FACTOR * half_width) / factor)
    lateral_selected = select_tracks(features, lateral_zone, teams, ground_factor=factor,
                                     clip_margin=0.0)

    coverage = compute_corridor_coverage(
        zone, [t.geometry for t in selected],
        half_width_ground_m=half_width, ground_factor=factor,
        lateral_track_geometries=[t.geometry for t in lateral_selected],
    )

    gaps = [GapSummary(id=g.id, length_m=g.length_m, geometry=g.geometry)
            for g in coverage.gaps]
    closed_gaps = [ClosedGap(id=c.id, length_m=c.length_m, max_lateral_m=c.max_lateral_m,
                             geometry=c.geometry)
                   for c in coverage.closed_gaps]

    warnings = (name_warnings + warnings + _mostly_outside_warnings(selected)
                + _closed_gaps_warning(closed_gaps))

    time_entered, time_exited, time_searching_hours = _time_window(selected)

    metrics = CoverageMetrics(
        assignment_label=assignment_label,
        assignment_kind="line",
        teams=list(teams),
        effective_sweep_width_m=full_width,
        covered_fraction=min(1.0, max(0.0, coverage.covered_fraction)),
        effort_coverage=None,
        zone_area_m2=None,
        line_length_m=coverage.total_length_m,
        tracks=_track_summaries(selected),
        gaps=gaps,
        closed_gaps=closed_gaps,
        warnings=warnings,
        time_entered=time_entered,
        time_exited=time_exited,
        time_searching_hours=time_searching_hours,
    )
    return CoverageResult(metrics=metrics, track_geojson=_track_geojson(selected))


def _track_summaries(selected) -> list[TrackSummary]:
    return [TrackSummary(name=t.name,
                         length_in_zone_m=t.length_in_zone_m,
                         length_total_m=t.length_total_m,
                         in_zone_fraction=min(1.0, t.in_zone_fraction),
                         time_entered=_iso(t.time_entered_ms),
                         time_exited=_iso(t.time_exited_ms))
            for t in selected]


def _closed_gaps_warning(closed_gaps: list[ClosedGap]) -> list[str]:
    """Line-assignment `warnings` entry summarising gaps suppressed as GPS
    drift (see corridor.compute_corridor_coverage) - one sentence, not one
    per gap, so it does not compete with the real gaps for attention; the
    detail (which gap, exact length/lateral) lives in `closed_gaps` itself,
    per "nothing is deleted silently"."""
    if not closed_gaps:
        return []
    total_length = sum(g.length_m for g in closed_gaps)
    max_lateral = max(g.max_lateral_m for g in closed_gaps)
    plural = "" if len(closed_gaps) == 1 else "s"
    return [
        f"closed {len(closed_gaps)} short interior gap{plural} totalling "
        f"{total_length:.0f} m as GPS drift (max lateral {max_lateral:.0f} m); "
        "see closed_gaps"
    ]


def _closed_area_warning(closed_gaps: list[ClosedGap]) -> list[str]:
    """Area-assignment analogue of `_closed_gaps_warning`, for raster
    slivers folded into "covered" by swath.close_gaps - see that
    function's docstring for the closing-radius derivation."""
    if not closed_gaps:
        return []
    total_area = sum(g.area_m2 for g in closed_gaps)
    plural = "" if len(closed_gaps) == 1 else "s"
    return [
        f"closed {len(closed_gaps)} short GPS-noise sliver{plural} totalling "
        f"{total_area:.0f} m2 as raster closing; see closed_gaps"
    ]


def _iso(epoch_ms: int | None) -> str | None:
    """Epoch milliseconds -> ISO-8601 UTC, seconds precision."""
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _time_window(selected) -> tuple[str | None, str | None, float | None]:
    """Earliest entered / latest exited across every SelectedTrack, and the
    elapsed hours between them. A track missing timestamps simply does not
    contribute; if none of them have any, all three come back None rather
    than a fabricated 0.0."""
    entered_ms = [t.time_entered_ms for t in selected if t.time_entered_ms is not None]
    exited_ms = [t.time_exited_ms for t in selected if t.time_exited_ms is not None]
    if not entered_ms or not exited_ms:
        return None, None, None
    earliest_ms, latest_ms = min(entered_ms), max(exited_ms)
    hours = (latest_ms - earliest_ms) / 3_600_000.0
    return _iso(earliest_ms), _iso(latest_ms), hours


def _mostly_outside_warnings(selected) -> list[str]:
    # Nothing is excluded, so this is information for the debrief agent, not
    # a warning about lost coverage - a small in-zone fraction just means
    # the track looks like a transit recording and is worth asking about.
    return [
        f"track {t.name} is mostly outside the assignment "
        f"({t.in_zone_fraction:.1%} of {t.length_total_m:.0f} m inside) - "
        "transit recording?"
        for t in selected if t.in_zone_fraction <= MOSTLY_OUTSIDE_FRACTION
    ]


def _track_geojson(selected) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": t.name},
                "geometry": mapping(t.geometry),
            }
            for t in selected
        ],
    }


def _project_features_to_mercator(features: list[dict], transform) -> list[dict]:
    """Reproject each LineString feature's x/y from WGS84 to Web Mercator.

    Elevation and timestamp ordinates (if present) pass through untouched;
    only the first two positions of each coordinate are geographic.
    """
    projected = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            projected.append(feature)
            continue
        coordinates = geometry.get("coordinates") or []
        new_coordinates = []
        for coordinate in coordinates:
            x, y = transform(coordinate[0], coordinate[1])
            new_coordinates.append([x, y, *coordinate[2:]])
        projected.append({
            **feature,
            "geometry": {**geometry, "coordinates": new_coordinates},
        })
    return projected


# Below this a gap merely grazes a shape's edge - these registry polygons
# have thousands of stair-step vertices (rasterized then polygonized), so an
# adjacency hit that touches without meaningfully overlapping is likely.
MIN_SHAPE_OVERLAP_FRACTION = 0.10


def _shape_overlap_fraction(shape_geometry, gap_geometry) -> float:
    """Fraction of `shape_geometry`'s own extent that falls inside the gap.

    Uses whichever measure the shape actually has - area for a polygon,
    length for a line - so a steep band that is 36% inside a gap reads as
    0.36, not as a bare "yes, it intersects". A shape with neither (a point,
    or a degenerate empty geometry) falls back to a binary in/out test.
    """
    intersection = shape_geometry.intersection(gap_geometry)
    if shape_geometry.area > 0:
        return min(1.0, intersection.area / shape_geometry.area)
    if shape_geometry.length > 0:
        return min(1.0, intersection.length / shape_geometry.length)
    return 0.0 if intersection.is_empty else 1.0


def _attribute_gaps(gaps: list[GapSummary], terrain: TerrainMetrics) -> None:
    """Report how much of each terrain shape a gap leaves unsearched.

    A gap the debriefer can see is "2 acres of dense scrub, 39% of the
    steep band unsearched" produces a much better - and truer - question
    than "gap_0 containing steep_35_3", which would claim the whole band
    went unsearched even when most of it was actually covered.
    """
    registry = [(s.id, s.kind, shape(s.geometry)) for s in terrain.shapes]
    for gap in gaps:
        geometry = shape(gap.geometry)
        overlaps = [
            ShapeOverlap(id=sid, kind=kind, fraction_in_gap=fraction)
            for sid, kind, g in registry
            if (fraction := _shape_overlap_fraction(g, geometry)) >= MIN_SHAPE_OVERLAP_FRACTION
        ]
        overlaps.sort(key=lambda overlap: -overlap.fraction_in_gap)
        gap.unsearched_shapes = overlaps
