"""Fusing several sightings of linear ground into one list of unmarked paths.

THE QUESTION
------------
"Is there a road or trail here that the map does not show?" Several sources
each give a partial answer: the land agencies' own road and trail records,
and the elevation itself (a cut road is a bench - `lineaments.py`). None is
right alone. An agency record can be decades stale or drawn 40 m off; a
terrain bench can be a fault scarp or plain noise.

So nothing here decides. Each candidate says WHICH INDEPENDENT SOURCES saw
it, and the list is ranked by that count. Two sources agreeing on one line
is worth more than either alone - but only when they really are two. The
USGS trail layer republishes the Forest Service's lines, so each sighting
carries a `lineage` (whose survey it is), and agreement is counted in
lineages, never in layers.

THE STEPS, ALL IN EPSG:3857
---------------------------
1. `unmapped_parts`: the part of each line farther than the tolerance from
   every mapped way, cut into pieces at least the minimum length.
2. `merge_same_paths`: pieces from different sources lying along the same
   ground become ONE candidate. Pieces are taken in source priority (the
   richest attributes first, terrain last) and then longest first; a piece
   with at least `SAME_PATH_FRACTION` of its length within the tolerance of
   a candidate already accepted is folded into it as a sighting, with how
   much of that candidate it covers.
3. `cross_support`: two candidates that both survived, but one of which
   lies mostly along the other (a long terrain bench over a short agency
   segment), are noted on each other - without merging, because neither is
   a part of the other.
4. `tier_of`: "corroborated" (two or more lineages), "agency_record" (one
   agency only) or "terrain_only".

All of this is pure geometry: no files, no network.
"""

from __future__ import annotations

import statistics

from shapely import STRtree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

#: Two independent positional errors stack here - OSM's (about 15 m, the
#: `lineaments.DEFAULT_TOLERANCE_M` figure) and an agency inventory's, which
#: for a road digitised off a 1:24,000 quad is itself about 12 m at the
#: National Map Accuracy Standard. 25 m covers both without swallowing a
#: parallel road on the next bench down.
AGENCY_TOLERANCE_M = 25.0

#: A piece shorter than this is a stub where two datasets disagree at a
#: junction, not a path. The same floor `lineaments.MIN_LENGTH_M` uses.
MIN_LENGTH_M = 50.0

#: Share of a piece's length that must lie along a candidate for the two to
#: be the same path.
SAME_PATH_FRACTION = 0.5

#: A candidate whose MEDIAN distance to a mapped way is under this may be
#: that mapped way, drawn in a different place by a different survey.
CLOSE_TO_MAPPED_M = 60.0

#: Spacing of the points the median above is measured at.
SAMPLE_STEP_M = 10.0

#: Which piece leads when several sources saw the same ground: the one whose
#: attributes say most (a road number and maintenance level beat a bare
#: line), terrain last because it carries no name at all.
SOURCE_PRIORITY = {"usfs_roads": 0, "usfs_trails": 1, "blm": 2,
                   "usgs_roads": 3, "usgs_trails": 4, "terrain": 5}

TIERS = ("corroborated", "agency_record", "terrain_only")

#: The fields of a piece that travel with it into a sighting.
SIGHTING_FIELDS = ("source", "lineage", "layer", "kind", "name", "ref", "status")


def line_parts(geometry) -> list[LineString]:
    """Every LineString in a geometry, joining runs that meet end to end."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    lines = [part for part in getattr(geometry, "geoms", [])
             if part.geom_type == "LineString" and not part.is_empty]
    lines += [line for part in getattr(geometry, "geoms", [])
              if part.geom_type == "MultiLineString" for line in part.geoms]
    if not lines:
        return []
    merged = linemerge(MultiLineString(lines))
    return list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]


def unmapped_parts(geometry, known_buffer, min_length_units: float) -> list[LineString]:
    """The pieces of `geometry` outside `known_buffer` (None: nothing is
    mapped, so all of it), each at least `min_length_units` long."""
    residue = geometry if known_buffer is None else geometry.difference(known_buffer)
    return [part for part in line_parts(residue) if part.length >= min_length_units]


def fraction_within(line, buffered) -> float:
    """Share of `line`'s length inside `buffered`."""
    if line.length <= 0:
        return 0.0
    return line.intersection(buffered).length / line.length


def flat_buffer(line, distance: float):
    """A buffer with square-cut ends, so the share of another line it covers
    stops where `line` stops. Round ends would credit a 100 m bench with
    140 m of a road at a 20 m tolerance."""
    return line.buffer(distance, cap_style="flat")


def median_distance(line, other, step_units: float) -> float | None:
    """Median distance from points along `line` (every `step_units`) to
    `other`, in the same units; None when there is no `other`."""
    if other is None or other.is_empty:
        return None
    count = max(2, int(line.length / step_units) + 1)
    distances = [line.interpolate(index / (count - 1), normalized=True).distance(other)
                 for index in range(count)]
    return statistics.median(distances)


def _sighting(piece: dict, coverage: float) -> dict:
    sighting = {key: piece.get(key) for key in SIGHTING_FIELDS}
    sighting["coverage"] = round(coverage, 3)
    return sighting


def merge_same_paths(pieces: list[dict], tolerance_units: float) -> list[dict]:
    """Pieces (dicts with `geom` and the SIGHTING_FIELDS) folded into
    candidates. Each candidate is its leading piece plus `sightings`: one
    entry for itself (coverage 1.0) and one for every piece folded in, with
    the share of the candidate that piece covers."""
    order = sorted(pieces, key=lambda piece: (SOURCE_PRIORITY.get(piece["source"], 99),
                                              -piece["geom"].length))
    if not order:
        return []
    tree = STRtree([piece["geom"] for piece in order])
    accepted: dict[int, dict] = {}
    buffers: dict[int, object] = {}

    for index, piece in enumerate(order):
        best = None
        for other in tree.query(piece["geom"], predicate="dwithin", distance=tolerance_units):
            other = int(other)
            if other >= index or other not in accepted:
                continue
            if other not in buffers:
                buffers[other] = accepted[other]["geom"].buffer(tolerance_units)
            share = fraction_within(piece["geom"], buffers[other])
            if share >= SAME_PATH_FRACTION and (best is None or share > best[0]):
                best = (share, other)
        if best is None:
            accepted[index] = {**piece, "sightings": [_sighting(piece, 1.0)]}
            continue
        candidate = accepted[best[1]]
        coverage = fraction_within(candidate["geom"], flat_buffer(piece["geom"], tolerance_units))
        candidate["sightings"].append(_sighting(piece, coverage))

    return [accepted[index] for index in sorted(accepted)]


def cross_support(candidates: list[dict], tolerance_units: float) -> None:
    """Note, on each candidate, any OTHER candidate of a different lineage
    lying along at least `SAME_PATH_FRACTION` of it. In place; adds
    `supported_by`, a list of indices into `candidates`."""
    for candidate in candidates:
        candidate.setdefault("supported_by", [])
    if len(candidates) < 2:
        return
    tree = STRtree([candidate["geom"] for candidate in candidates])
    buffers: dict[int, object] = {}
    for index, candidate in enumerate(candidates):
        for other in tree.query(candidate["geom"], predicate="dwithin",
                                distance=tolerance_units):
            other = int(other)
            if other == index or candidates[other]["lineage"] == candidate["lineage"]:
                continue
            if other not in buffers:
                buffers[other] = flat_buffer(candidates[other]["geom"], tolerance_units)
            share = fraction_within(candidate["geom"], buffers[other])
            if share >= SAME_PATH_FRACTION:
                candidate["supported_by"].append(other)
                candidate["sightings"].append(_sighting(candidates[other], share))


def lineages_of(candidate: dict) -> list[str]:
    return sorted({sighting["lineage"] for sighting in candidate["sightings"]})


def tier_of(candidate: dict) -> str:
    lineages = lineages_of(candidate)
    if len(lineages) >= 2:
        return "corroborated"
    return "terrain_only" if lineages == ["terrain"] else "agency_record"
