"""Corridor coverage for line assignments: pure line math, no CalTopo knowledge.

Mirrors how swath.py is pure raster math for polygon assignments. A line
assignment (a trail segment, a drainage) has no area, so "covered" cannot
mean "swept area over zone area" - instead a corridor of `half_width_ground_m`
is drawn around the *track*, and a section of the *assignment line* counts as
covered when it falls inside that corridor.

Approach: sample the assignment line densely (every SAMPLE_SPACING_GROUND_M
of ground length), test each sample's distance to the nearest track segment,
mark it covered or not, then merge consecutive uncovered samples into gap
segments via `shapely.ops.substring` of the assignment line. Assignment
lines run 1-6 km, so this is a few thousand samples - a `shapely.STRtree`
over individual track segments (not whole tracks) keeps the nearest-distance
query fast; a per-sample `line.distance(point)` against a whole multi-
thousand-vertex track would be O(samples x vertices).
"""

import math
from dataclasses import dataclass

import numpy as np
from shapely import STRtree, points
from shapely.geometry import LineString, mapping
from shapely.ops import substring

# Ground metres between samples along the assignment line. Fine enough to
# localise a gap boundary to within a couple of metres; coarse enough that a
# 6 km assignment line is a few thousand samples, not tens of thousands.
SAMPLE_SPACING_GROUND_M = 2.0

# Default corridor HALF-width in ground metres. Used only when terrain
# metrics (and therefore a vegetation-adjusted ESW) are unavailable and no
# explicit --corridor-width was given - see coverage.metrics.
DEFAULT_CORRIDOR_HALF_WIDTH_M = 5.0

# Below this length a gap is GPS wobble, not a real unsearched stretch: a
# consumer-grade GPS can drift several metres from the true line even while
# the corridor is being genuinely walked, briefly pushing a sample outside
# the corridor and back in. 25 m is comfortably above that noise floor.
MIN_GAP_LENGTH_M = 25.0

# --- GPS-noise gap closing (Phase 9i) --------------------------------------
#
# MIN_GAP_LENGTH_M above already hides sub-25 m runs from the report, but a
# real GPS track can drift far enough, for long enough, to open a REPORTED
# gap that is still just noise. Measured on QB102 (6856 m route, corridor
# half-width 5.3 m, ESW 10.55 m): five reported gaps, of which four (26 m,
# 26 m, 40 m, 28 m) were GPS wobble - the track never left the corridor by
# more than about 15 m across any of them - and one (3952 m) was genuinely
# never entered. A gap only CLOSES (moves to closed_gaps, counts as
# covered) when it is interior (route covered on both sides - see
# compute_corridor_coverage), short, and laterally consistent with drift;
# all three below are cheap per-gap checks against quantities already
# computed while sampling, so closing costs nothing extra.

# A closed gap's length must fall under this multiple of the corridor's own
# FULL width (never a fixed metre constant, so a 4 m brush ESW and a 30 m
# open-country ESW each get a threshold that scales with them). At QB102's
# 10.55 m ESW that is a ~53 m ceiling - the four noise gaps (<=40 m) clear
# it with room to spare, while gap_0 (3952 m) is not remotely close.
GAP_CLOSE_MAX_LENGTH_FACTOR = 5.0

# A closed gap's track must never wander farther from the route than this
# multiple of the corridor HALF-width - the same quantity a sample is
# already tested against for "covered", so "how far past it, at worst" is
# the natural scale. At QB102's 5.3 m half-width that is a ~16 m ceiling:
# comfortably above the noise on all four real noise gaps (largest closest
# approach recorded was 15.2 m, gap_2) yet well below what a genuine
# multi-metre detour around an obstacle would produce.
GAP_CLOSE_MAX_LATERAL_FACTOR = 3.0


@dataclass(frozen=True)
class CorridorSegment:
    id: str
    length_m: float
    geometry: dict  # GeoJSON mapping, in the assignment line's own coordinates


@dataclass(frozen=True)
class ClosedGap:
    """A gap suppressed as GPS noise - not in `gaps`, not unsearched, never
    asked about in the debrief, but never silently dropped either: every
    field a reviewer needs to see why this stretch was waved through lives
    here, alongside the length/lateral thresholds it passed."""
    id: str
    length_m: float
    max_lateral_m: float  # the track's worst (maximum) distance from the route across this gap
    geometry: dict  # GeoJSON mapping, in the assignment line's own coordinates


@dataclass(frozen=True)
class CorridorCoverage:
    covered_fraction: float
    total_length_m: float
    gaps: list[CorridorSegment]  # largest first, GPS-noise gaps excluded
    closed_gaps: list[ClosedGap]  # largest first; see ClosedGap


def _line_parts(geometry) -> list[LineString]:
    """Flatten a LineString/MultiLineString into non-empty LineString parts."""
    parts = getattr(geometry, "geoms", [geometry])
    return [part for part in parts if not part.is_empty and part.geom_type == "LineString"]


def _track_segments(track_geometries: list) -> list[LineString]:
    """Every track geometry broken into elementary 2-point segments.

    A STRtree of whole (dense, self-crossing) track LineStrings would still
    make each nearest-distance query walk the whole geometry internally;
    indexing individual segments instead gives the tree real spatial
    buckets to prune with.
    """
    segments: list[LineString] = []
    for geometry in track_geometries:
        for part in _line_parts(geometry):
            coords = list(part.coords)
            for a, b in zip(coords, coords[1:]):
                segments.append(LineString([a, b]))
    return segments


def _sample_ground_distances(line: LineString, tree: STRtree | None,
                             ground_factor: float,
                             spacing_ground_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along `line` and each one's ground distance to `tree`.

    Returns (distances_along_line, ground_distance_to_nearest_track) - both
    arrays in the same order and length, sample positions in the line's own
    (Mercator) units, distances already converted to ground metres via
    `ground_factor`.
    """
    length = line.length
    spacing = spacing_ground_m / ground_factor if ground_factor else spacing_ground_m
    n_samples = max(2, int(math.ceil(length / spacing)) + 1)
    distances = np.linspace(0.0, length, n_samples)

    if tree is None or len(tree.geometries) == 0:
        return distances, np.full(n_samples, np.inf)

    sample_points = [line.interpolate(d) for d in distances]
    xs = np.fromiter((p.x for p in sample_points), dtype=float, count=n_samples)
    ys = np.fromiter((p.y for p in sample_points), dtype=float, count=n_samples)
    query_points = points(xs, ys)

    # query_nearest on an array of query geometries returns a (2, n) index
    # array - row 0 is the QUERY point index, row 1 the TREE geometry index
    # (see shapely.STRtree.query_nearest) - plus a distances array aligned
    # to the same columns. Any query point with no match at all (only
    # possible if it were None/empty, which a shapely Point never is) would
    # be dropped from the output entirely, so `pair_index` can be shorter
    # than n_samples but never longer, and never contains a value >=
    # n_samples: indexing `ground_distance` with row 1 (tree indices, which
    # can run up to len(segments) - 1) instead of row 0 previously raised an
    # IndexError on real multi-thousand-segment track data, where the tree
    # is larger than the sample count.
    (pair_index, _tree_index), mercator_distance = tree.query_nearest(
        query_points, return_distance=True, all_matches=False
    )
    ground_distance = np.full(n_samples, np.inf)
    ground_distance[pair_index] = mercator_distance * ground_factor
    return distances, ground_distance


def _uncovered_runs(covered: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, end_index) pairs of consecutive uncovered samples, inclusive."""
    runs: list[tuple[int, int]] = []
    run_start = None
    for i, is_covered in enumerate(covered):
        if not is_covered and run_start is None:
            run_start = i
        elif is_covered and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(covered) - 1))
    return runs


def compute_corridor_coverage(
    assignment_line, track_geometries: list, half_width_ground_m: float,
    ground_factor: float = 1.0,
    sample_spacing_ground_m: float = SAMPLE_SPACING_GROUND_M,
    min_gap_length_m: float = MIN_GAP_LENGTH_M,
    lateral_track_geometries: list | None = None,
) -> CorridorCoverage:
    """Corridor coverage of `assignment_line` by `track_geometries`.

    `assignment_line` and `track_geometries` must already share a
    coordinate system; `ground_factor` converts that system's units to
    ground metres (the same convention as swath.py and tracks.py).
    `half_width_ground_m` is the corridor HALF-width - a sample is covered
    when a track comes within this distance of it.

    `lateral_track_geometries`, if given, is used ONLY to measure
    max_lateral_m for the closing decision - `track_geometries` still
    decides what is covered/uncovered (so gap boundaries and lengths are
    unaffected). This matters because a caller may need to pass CLIPPED
    track geometry for `track_geometries` (a caller with its own reasons to
    clip tight to the corridor - e.g. metrics.py clips to exactly
    half_width_ground_m so `length_in_zone_m` measures "length inside the
    corridor"), and that same tight clip would starve the lateral test of
    the evidence it needs: a sample whose nearest surviving (clipped) track
    point is far away in ALONG-ROUTE distance reports a large max_lateral
    that reflects the clip boundary, not how far the track actually
    wandered - see metrics._compute_line_coverage, which passes a track set
    clipped instead to GAP_CLOSE_MAX_LATERAL_FACTOR * half_width (verified
    against QB102: without this, only 2 of 4 real GPS-noise gaps closed,
    because the two longer ones' nearest surviving clipped-track point was
    tens of metres away along-route even though the real track wandered
    only ~10-15 m).
    """
    parts = _line_parts(assignment_line)
    segments = _track_segments(track_geometries)
    tree = STRtree(segments) if segments else None

    lateral_geometries = (
        track_geometries if lateral_track_geometries is None else lateral_track_geometries
    )
    lateral_segments = _track_segments(lateral_geometries)
    lateral_tree = STRtree(lateral_segments) if lateral_segments else None

    total_length_mercator = sum(part.length for part in parts)
    total_uncovered_mercator = 0.0
    closed_length_mercator = 0.0
    gap_candidates: list[tuple[float, LineString]] = []
    closed_candidates: list[tuple[float, float, LineString]] = []

    for part in parts:
        if part.length == 0:
            continue
        distances, ground_distance = _sample_ground_distances(
            part, tree, ground_factor, sample_spacing_ground_m
        )
        # Same sample positions (same part, spacing, ground_factor) as
        # above, queried against the (possibly wider) lateral tree - see
        # the docstring above for why these can legitimately differ.
        _, lateral_distance = _sample_ground_distances(
            part, lateral_tree, ground_factor, sample_spacing_ground_m
        )
        covered = ground_distance <= half_width_ground_m
        runs = _uncovered_runs(covered)

        for start_idx, end_idx in runs:
            start_dist = distances[start_idx]
            end_dist = distances[end_idx]
            total_uncovered_mercator += end_dist - start_dist

            length_m = (end_dist - start_dist) * ground_factor
            # Runs below the noise floor still count toward the uncovered
            # length above (covered_fraction is honest about them) but are
            # not surfaced as their own gap - mirrors swath.MIN_GAP_M2,
            # which filters what gets reported, not what counts as covered.
            if length_m < min_gap_length_m:
                continue

            # Interior test: a run touching either end of THIS part's
            # sample array has no covered sample beyond it on that side -
            # i.e. it touches an end of the route (or, for a MultiLineString,
            # an end of this physical piece of it) - and is never closed.
            # This is what protects a genuinely unsearched tail from being
            # waved through as "just drift".
            is_interior = start_idx != 0 and end_idx != len(covered) - 1
            max_lateral_m = float(np.max(lateral_distance[start_idx:end_idx + 1]))
            geometry = substring(part, start_dist, end_dist, normalized=False)

            if is_interior and _is_gps_noise(length_m, max_lateral_m, half_width_ground_m):
                closed_length_mercator += end_dist - start_dist
                closed_candidates.append((length_m, max_lateral_m, geometry))
            else:
                gap_candidates.append((length_m, geometry))

    # Closed gaps WERE walked (the track just drifted while doing it), so
    # they count as covered here even though the sample-level distance test
    # above still saw them as briefly outside the corridor.
    covered_length_mercator = max(
        0.0, total_length_mercator - total_uncovered_mercator + closed_length_mercator
    )
    covered_fraction = (
        min(1.0, covered_length_mercator / total_length_mercator)
        if total_length_mercator else 0.0
    )

    gap_candidates.sort(key=lambda pair: -pair[0])
    gaps = [
        CorridorSegment(id=f"gap_{position}", length_m=length_m, geometry=mapping(geometry))
        for position, (length_m, geometry) in enumerate(gap_candidates)
    ]

    closed_candidates.sort(key=lambda triple: -triple[0])
    closed_gaps = [
        ClosedGap(id=f"closed_gap_{position}", length_m=length_m,
                  max_lateral_m=max_lateral_m, geometry=mapping(geometry))
        for position, (length_m, max_lateral_m, geometry) in enumerate(closed_candidates)
    ]

    return CorridorCoverage(
        covered_fraction=covered_fraction,
        total_length_m=total_length_mercator * ground_factor,
        gaps=gaps,
        closed_gaps=closed_gaps,
    )


def _is_gps_noise(length_m: float, max_lateral_m: float, half_width_ground_m: float) -> bool:
    """Short-and-laterally-consistent test for gap closing (rules 2 and 3 -
    "interior" (rule 1) is a property of the run's position, checked by the
    caller before this is even reached).

    `sweep_width_m` is the corridor's FULL width - GAP_CLOSE_MAX_LENGTH_FACTOR
    is expressed against the full width because that is the physical quantity
    a searcher's path is compared to ("5x the swept strip"); the lateral
    ceiling instead scales off the HALF-width because that is the exact
    distance a sample is already tested against for "covered" (see
    `covered = ground_distance <= half_width_ground_m` above) - "how far past
    the corridor edge, at worst" is the natural quantity for GPS drift.
    """
    sweep_width_m = 2 * half_width_ground_m
    is_short = length_m < GAP_CLOSE_MAX_LENGTH_FACTOR * sweep_width_m
    is_lateral_consistent = max_lateral_m < GAP_CLOSE_MAX_LATERAL_FACTOR * half_width_ground_m
    return is_short and is_lateral_consistent
