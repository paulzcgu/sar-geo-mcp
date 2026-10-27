"""Hasty candidates: the lines a team can run on day one.

THE CORPUS RULE
---------------
Start from the lines, not the area. First-period hasty assignments are
trails and roads drawn as lines end to end (77% of line assignments sit on
one). So: every trail and road in the area is a line candidate, run end
to end and never split at junctions; a mapped waterway is a corridor
candidate at its corpus width, listed when `include` names it
(`DEFAULT_INCLUDE` is trails and roads). Drainages and ridges traced off
the elevation model are not offered (0.37.0): on real ground they came back
as hundreds of short, unstable pieces that buried the walkable lines.

"Never split at junctions" is a rule about geometry, not a sentiment. A
NAMED way entering the area is ONE candidate, drawn to its ends: the ways of
one class and name are grouped into connected components, and a component is
a candidate however many three-way junctions it holds. A fork is acceptable -
it is what a planner would draw by hand - and `facts.branches` says so. What
is never done is a `unary_union` first, which NODES the input and turns a
crossing into four stubs; two UNNAMED tracks that merely touch likewise stay
two candidates, because without a name there is nothing to say they are one
way.

"Drawn to its ends" means to the edge of the MEASURED GROUND - the area plus
the planning buffer - not to the edge of the search area. A planner hands a
team a trail, not the arc where a circle happened to cut it. Corridors are
the other way about: a waterway corridor is clipped to the subject, because
it has no meaning past it.

WHAT THIS NEVER DOES
--------------------
It never ranks. The list is ordered by distance from the centre, which is
a fact; the agent reorders by what it knows about the incident. Every
number here is a measurement, and every departure from the corpus is a
flag, not a score.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer

from ..geo import acres, to_wgs84_geojson
from ..terrain.grid import CORRIDOR_HALF_WIDTH_M, project_geometry_to_3857
from .boundary import EDGE_WARNINGS, Ground
from .candidates import Candidate, CandidateSet, candidate_id, make_set_id
from .effort import (
    ASSUMPTIONS as TRAVERSE_ASSUMPTIONS,
    AXIS_SAMPLE_M,
    DEFAULT_DENSITY,
    DENSITY_NOTE,
    candidate_effort,
    sample_along,
)
from .norms import load_norms, norm_flags, norms_digest
from ..terrain.movement import DEFAULT as MOVEMENT
from .travel import TRAVEL_ASSUMPTIONS, travel_minutes_to, travel_surface

#: Built once. A pyproj Transformer costs milliseconds to construct and
#: nothing to reuse, and a set of a hundred candidates converts a point or a
#: geometry several times each; these two are stateless and thread-safe.
_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

#: `include` values and the Ground classes each covers.
INCLUDE_CLASSES: dict[str, tuple[str, ...]] = {
    "trail": ("trail",),
    "road": ("road",),          # TRACK, ROAD and ROAD_OTHER all landed in "road"
    "waterway": ("waterway",),
}
#: Every kind `include` can name.
ALL_INCLUDE = ("trail", "road", "waterway")
#: The kinds listed when `include` is None: the trails and roads a team walks
#: end to end. Waterway corridors only when named.
DEFAULT_INCLUDE = ("trail", "road")
LINE_CLASSES = ("trail", "road")
CORRIDOR_CLASSES = ("waterway",)

#: Ways sharing a class and a name merge where their ends touch within this.
MERGE_TOUCH_M = 5.0
#: A way is one of a merged run's own ways when this much of its length lies
#: within MERGE_TOUCH_M of the run; a way that merely crosses it is not.
AXIS_ID_FRACTION = 0.5
#: How close to the raster's edge counts as ON it, and how much length a clip
#: may remove and still count as removing nothing. One metre: the staged OSM
#: and the raster are cut from the same Web Mercator box, so a way the fetch
#: clipped lands exactly on the edge, and the slack here is for floating point
#: rather than for nearness.
EDGE_CLIP_TOLERANCE_M = 1.0
#: A clipped LINE run shorter than this is not a candidate. `min_length_m`
#: replaces it, for LINE classes only, when a caller asks for runs an
#: assignment could be drawn on instead of every stub the map holds.
MIN_CANDIDATE_M = 30.0
#: The corpus p10 for a LINE assignment, ground metres: the shortest tenth of
#: the real first-period line assignments were at least this long. The figure
#: a planner reaches for when they want assignment-sized runs rather than
#: trail stubs, quoted in the `hasty_candidates` tool docstring and named here
#: so a consumer can import the corpus number instead of hardcoding it.
LINE_ASSIGNMENT_P10_M = 593.0
#: A clipped CORRIDOR axis shorter than this is not a candidate: a 60 m piece
#: of creek inside the area is not a line a team can be sent down.
MIN_CORRIDOR_AXIS_M = 100.0
#: How many candidates a caller who names no limit gets. The list is ordered by
#: distance from the centre, so the default is the nearest fifty - a working
#: shortlist rather than every line the ground holds. `limit=0` is every one.
DEFAULT_LIMIT = 50

#: A line candidate's effort assumptions are all shared with the set, so the
#: candidate points at the set's copy rather than carrying an empty object.
SHARED_EFFORT_NOTE = "see the set's assumptions.effort"

#: The subject's own shape decides whether it has an area at all.
LINE_SUBJECT_NOTE = ("the subject is a line; the area is the "
                     f"{CORRIDOR_HALF_WIDTH_M:.0f} m corridor either side "
                     "(terrain.grid.CORRIDOR_HALF_WIDTH_M)")
POINT_SUBJECT_ERROR = "a point subject has no area; stage an area with prepare_area"

#: No readable map means overlap with existing assignments was never measured.
#: Reported as None with this reason rather than 0.0, which would be
#: indistinguishable from a map that was read and carries no assignment at
#: all - boundary.py's `shared_pct`/`shared_reason` convention.
NO_MAP_OVERLAP_REASON = ("no map was staged with this area, or the map could not be "
                         "read, so overlap was not computed")

# Decimal places by fact name, applied to every float in a candidate's facts.
# Precision beyond what the measurement supports is bytes in every payload and
# a claim about the ground nobody made: a 10 m DEM does not know a line's
# length to the picometre. Six decimals of a degree is about 0.1 m.

#: A name that IS the unit: `latitude`, `longitude`.
_PLACES_BY_NAME: dict[str, int] = {"latitude": 6, "longitude": 6}
#: A name that CARRIES the unit as a word: `hours_one_way`,
#: `sweep_time_hours_per_pass`, `travel_minutes_from_center`. Six seconds is
#: already finer than a walking-time surface can honestly claim.
_PLACES_BY_WORD: dict[str, int] = {"hours": 3, "minutes": 1}
#: A name that ENDS in the unit. Longest suffix first: `area_m2` is metres
#: squared, not metres.
_PLACES_BY_SUFFIX: tuple[tuple[str, int], ...] = (
    ("_pct", 1), ("_acres", 2), ("_deg", 1), ("_kmh", 2), ("_m2", 1), ("_m", 1),
)


def hasty_params(*, include: list[str] | tuple[str, ...] | None = None,
                 corridor_widths_m: dict | None = None,
                 max_distance_m: float | None = None, limit: int | None = None,
                 min_length_m: float | None = None, norms: dict | None = None) -> dict:
    """The parameters the set id is hashed over, normalised and checked.

    A separate function because a caller has to know a set's id BEFORE
    generating it - that is how `hasty_candidates` the tool can hand back a
    set it already made instead of measuring the same ground twice - and the
    id must come from exactly the normalisation the generator uses, not from
    a second copy of it that can drift.

    `limit`: None means `DEFAULT_LIMIT`, 0 means every candidate; the
    normalised value (None for unlimited) is what the id is hashed over, so
    "the default" and "all of them" are two different sets.

    `min_length_m`: None keeps `MIN_CANDIDATE_M`, and is hashed as None so
    today's sets keep today's ids. A number replaces the minimum for LINE
    classes only, and is hashed too: a set of assignment-sized runs is its
    own set, not a shorter version of the default one under its name.

    `norms_digest` (`norms.norms_digest`): every candidate's flags are read
    against the norms' bands, which the corridor widths, hashed already,
    did not cover. Without it a server restarted with another
    agency's `--norms` over the same cache read back the old corpus's flags
    as current, a line unflagged that the agency's own length band flags
    (PR #4 review B23). Every set id changed once with it (0.31.11):
    a set saved before is measured again on its first call, and the ids a
    planner already holds still name the sets they named.
    """
    norms = norms if norms is not None else load_norms()
    include_list = sorted(dict.fromkeys(include or DEFAULT_INCLUDE))
    unknown = [i for i in include_list if i not in INCLUDE_CLASSES]
    if unknown:
        raise ValueError(f"unknown include value(s) {unknown}; choose from "
                         f"{sorted(INCLUDE_CLASSES)}")
    widths = dict(norms["corridor_width_m"]["by_axis"])
    widths.update(_checked_widths(corridor_widths_m))
    return {"include": include_list,
            "corridor_widths_m": {k: widths[k] for k in CORRIDOR_CLASSES},
            "max_distance_m": float(max_distance_m) if max_distance_m is not None else None,
            "limit": _checked_limit(limit),
            "min_length_m": _checked_min_length(min_length_m),
            "norms_digest": norms_digest(norms)}


def _checked_min_length(min_length_m: float | None) -> float | None:
    """None (today's `MIN_CANDIDATE_M`) or a positive distance.

    Zero is refused rather than read as "no minimum": a run of no length is
    not a candidate, and a caller who wants every stub already gets one by
    leaving this out.
    """
    if min_length_m is None:
        return None
    min_length_m = float(min_length_m)
    if not min_length_m > 0:
        raise ValueError("min_length_m must be greater than 0 ground metres, or None "
                         f"for the default of {MIN_CANDIDATE_M:g}; got {min_length_m:g}")
    return min_length_m


def _checked_limit(limit: int | None) -> int | None:
    """`DEFAULT_LIMIT` for None, None (unlimited) for 0, the number otherwise.

    An unbounded default is the failure this exists to prevent: a bare call on
    a 1.5 km area returned 1,369 candidates and about 350k tokens. A caller who
    wants all of them says so with 0.
    """
    if limit is None:
        return DEFAULT_LIMIT
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be a positive count, 0 for every candidate, "
                         f"or None for the default of {DEFAULT_LIMIT}; got {limit}")
    return limit or None


def hasty_candidates(ground: Ground, subject: dict, *, include: list[str] | tuple[str, ...] | None = None,
                     corridor_widths_m: dict | None = None, max_distance_m: float | None = None,
                     limit: int | None = None, min_length_m: float | None = None,
                     norms: dict | None = None,
                     map_features: list[dict] | None = None,
                     handle: str = "", map_version: str | None = None) -> CandidateSet:
    """Generate and measure the hasty candidates for a staged subject.

    `subject` is the staged `subject.json` (an area from `prepare_area`, or
    an assignment); the centre is the area's centre or the geometry's
    centroid and is stated in `assumptions`. `map_features` are the map's
    features read fresh by the caller, for overlap with existing
    assignments; None means no map, and overlap is then reported as unknown
    with its reason rather than as a zero nobody measured.

    `limit` keeps the first candidates after the ordering and says in the
    summary how many there were; None is `DEFAULT_LIMIT` (50) and 0 is every
    candidate. It is one of the parameters the set id is hashed over, so a
    limited set is its own set rather than a silent truncation of another one.

    `min_length_m` replaces `MIN_CANDIDATE_M` for LINE classes only, so a
    caller can ask for assignment-sized runs rather than 36 m trail stubs;
    corridor axes keep `MIN_CORRIDOR_AXIS_M`, which is a fact about what the
    elevation model can see rather than about how long an assignment should be.
    It is judged against a line candidate's TOTAL length on the staged ground -
    see `_line_runs`, where a named way entering the area is one candidate
    drawn to its ends - not against one branch of it.

    `map_version` is only stored on the returned set, not measured here: the
    caller identifies the map state `map_features` was read from, so the same
    parameters generated against a later map state are recognised as a fresh
    measurement rather than reused unchanged.
    """
    norms = norms if norms is not None else load_norms()
    params = hasty_params(include=include, corridor_widths_m=corridor_widths_m,
                          max_distance_m=max_distance_m, limit=limit,
                          min_length_m=min_length_m, norms=norms)
    include_list = params["include"]
    widths = params["corridor_widths_m"]
    limit = params["limit"]
    min_line_m = params["min_length_m"] if params["min_length_m"] is not None else MIN_CANDIDATE_M

    geometry_type = (subject.get("geometry") or {}).get("type")
    if geometry_type in ("Point", "MultiPoint"):
        raise ValueError(POINT_SUBJECT_ERROR)
    area = project_geometry_to_3857(subject["geometry"])
    scale = ground.scale
    centre_latlon = subject.get("center")
    if centre_latlon:
        longitude, latitude = _centre_lon_lat(centre_latlon)
        centre = _to_3857(longitude, latitude)
        centre_note = "the area's centre as supplied"
    else:
        centre = area.centroid
        centre_note = "the polygon's centroid"
    max_distance = params["max_distance_m"]

    # A drainage or ridgeline running off the raster speaks of those axes'
    # ranking alone, and the hasty list holds neither.
    edge_only = set(EDGE_WARNINGS.values())
    warnings: list[str] = [w for w in ground.warnings if w not in edge_only]
    have_map = map_features is not None
    if not have_map:
        warnings.append(NO_MAP_OVERLAP_REASON)
    existing, unprojectable = _existing_assignments(map_features)
    if unprojectable:
        warnings.append(
            "these existing assignments could not be projected and are not in the "
            f"overlap figures: {', '.join(unprojectable)}")

    raw: list[dict] = []
    if any(i in LINE_CLASSES for i in include_list):
        if ground.osm_missing:
            warnings.append("no line candidates: osm.json missing, so trails and roads are unknown")
        for cls in LINE_CLASSES:
            if cls in include_list:
                raw.extend(_line_runs(ground, cls, area, min_line_m))
    if any(i in CORRIDOR_CLASSES for i in include_list):
        raw.extend(_corridor_axes(ground, area))

    candidates: list[dict] = []          # working entries, not yet Candidate records
    dropped = 0
    for entry in raw:
        axis = entry["axis"]
        nearest = _nearest_point(axis, centre)
        distance_m = axis.distance(centre) * scale
        if max_distance is not None and distance_m > max_distance:
            dropped += 1
            continue
        entry["distance_m"] = distance_m
        entry["nearest"] = nearest
        candidates.append(entry)
    candidates.sort(key=lambda e: (e["distance_m"], e["cls"], e.get("name") or "", e["axis_ids"][0]))
    total_before_limit = len(candidates)
    if limit is not None and total_before_limit > limit:
        candidates = candidates[:limit]
        warnings.append(
            f"only the first {limit} of {total_before_limit} candidates are listed "
            "(limit); the rest are farther from the centre")

    set_id = make_set_id("hasty", params)
    evt_path = ground.data_dir / "evt.tif"
    # One union of the existing assignments for the whole set: it does not
    # change between candidates, and unioning it per candidate is the most
    # expensive thing in the loop on a map carrying a hundred of them.
    existing_union = _union_of(existing)
    # ONE walking-time surface for the whole set. Every candidate is measured
    # from the same origin, so the Dijkstra solve over the raster is done once
    # here and read per candidate; per candidate it would be the graph build
    # and the solve a hundred times over for one identical answer.
    try:
        surface = travel_surface(ground, centre, warnings)
    except ValueError as error:
        # Ground too fine to walk (more cells than a walking-time search
        # covers): every candidate's minutes are unknown, never zero.
        surface = None
        warnings.append(f"travel_minutes_from_center is null on every candidate: {error}")

    records: list[Candidate] = []
    for index, entry in enumerate(candidates, start=1):
        cls, axis = entry["cls"], entry["axis"]
        kind = "line" if cls in LINE_CLASSES else "corridor"
        if kind == "line":
            geometry_3857 = axis
        else:
            geometry_3857 = axis.buffer(widths[cls] / 2.0 / scale, cap_style="flat")
        facts = _rounded(_facts(ground, entry, geometry_3857, kind, centre, existing,
                                existing_union, widths, evt_path, area, have_map,
                                surface))
        band_facts = {"length_m": facts["length_m"]} if kind == "line" else {"width_m": facts["width_m"]}
        records.append(Candidate(
            id=candidate_id(set_id, index), kind=kind, axis=cls, axis_ids=entry["axis_ids"],
            sources=entry["sources"], name=entry.get("name"),
            geometry=to_wgs84_geojson(geometry_3857), facts=facts,
            norm_flags=norm_flags(band_facts, norms), warnings=entry.get("warnings", []),
        ))

    by_kind: dict[str, int] = {}
    by_axis: dict[str, int] = {}
    for record in records:
        by_kind[record.kind] = by_kind.get(record.kind, 0) + 1
        by_axis[record.axis] = by_axis.get(record.axis, 0) + 1
    if not records:
        warnings.append("no hasty candidates in this area with these settings")

    assumptions = {
        # `center` is a point and nothing else: the same {latitude, longitude}
        # shape as every other point in the payload. Where it came from is
        # its own assumption, next to it.
        "center": {"latitude": _to_wgs84(centre)[1], "longitude": _to_wgs84(centre)[0]},
        "center_note": centre_note,
        "ordered_by": "distance_from_center_m ascending - a fact, not a ranking",
        "merge_touch_m": MERGE_TOUCH_M,
        "axis_id_fraction": AXIS_ID_FRACTION,
        # The minimum actually applied to LINE runs on this call: today's
        # default, or the `min_length_m` the caller asked for.
        "min_candidate_m": min_line_m,
        "min_corridor_axis_m": MIN_CORRIDOR_AXIS_M,
        "axis_sample_m": AXIS_SAMPLE_M,
        "corridor_widths_m": {k: widths[k] for k in CORRIDOR_CLASSES},
        "lines_not_split_at_junctions": "the corpus's hasty lines run a trail end to end",
        # The rule the line candidates were built by, stated where a planner
        # reading a forked candidate's `branches` can find out why it is one
        # candidate rather than several.
        "line_candidate_rule": (
            "one line candidate per class, name and connected component: a named way "
            "entering the area is one candidate drawn to its ends, length_m is its total "
            "length on the staged ground, and facts.branches counts the pieces. Unnamed "
            "ways are merged only where one way's end meets another's"),
        # Where a candidate's geometry stops, and why the two kinds differ.
        # A planner comparing a trail's length_m against the map is entitled
        # to know which limit produced it.
        "line_extent": (
            "a line candidate runs to the edge of the STAGED GROUND - the area plus the "
            "planning buffer, which is the ground slope, climb, effort and travel time "
            "were measured on - not to the edge of the subject area; a way that carries "
            "on past it has facts.clipped_at_edge true and a warning saying so"),
        "corridor_extent": (
            "a waterway corridor's axis is clipped to the SUBJECT AREA: a corridor "
            "past the subject is not a line anybody asked for"),
        # The density class and the traverse table are the same for every
        # candidate in the set, so they are stated once here; each candidate's
        # `facts.effort.assumptions` carries only what varies with it.
        "effort": {"density_class": DEFAULT_DENSITY, "density_note": DENSITY_NOTE,
                   "traverse": TRAVERSE_ASSUMPTIONS,
                   "movement": MOVEMENT.assumptions()},
        # The cost model behind `travel_minutes_from_center`, stated once for
        # the set beside `effort` - the two read the same Tobler curve, and a
        # planner comparing them is entitled to see that they agree.
        "travel": TRAVEL_ASSUMPTIONS,
    }
    if geometry_type in ("LineString", "MultiLineString"):
        assumptions["area_note"] = LINE_SUBJECT_NOTE

    return CandidateSet(
        set_id=set_id, tool="hasty_candidates", params=params, handle=handle,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        candidates=records,
        summary={"count": len(records), "by_kind": by_kind, "by_axis": by_axis,
                 "dropped_beyond_max_distance": dropped,
                 "truncated_to": limit, "total_before_limit": total_before_limit,
                 "existing_assignments": len(existing)},
        assumptions=assumptions,
        warnings=warnings,
        map_version=map_version,
    )


# --- lines ---------------------------------------------------------------------------


def _line_runs(ground: Ground, cls: str, area, min_length_m: float) -> list[dict]:
    """One candidate per (class, name, CONNECTED COMPONENT), drawn to its ends.

    THE RULE
    --------
    A named way that enters the subject area is ONE candidate. A fork in it is
    acceptable - it is what a planner would draw by hand - so the candidate is
    the whole connected component of that name and its `length_m` is its total
    length on the STAGED GROUND, which is what `min_length_m` is judged
    against.

    HOW FAR IT RUNS
    ---------------
    To the edge of the measured ground - the area plus the planning buffer -
    not to the edge of the subject area. A SAR planner hands a team a trail,
    not the arc where a circle happened to cut it. Measured at Windy Hill on a
    1,500 m area: Spring Ridge Trail is 4,584 m and the circle left 1,796 of
    them; Razorback Ridge 2,309 of 4,402; Hamms Gulch 3,491 of 4,470. Between
    four trails the circle cut off 3,073 m of walkable, mapped trail, and the
    operator asked three times for them whole.

    The staged raster is the limit because it is where the measurements are.
    Past its edge there is no slope, no climb, no vegetation and no travel
    time, so a candidate drawn past it would carry numbers about ground
    nobody looked at. A candidate the raster cuts says so twice: in
    `facts.clipped_at_edge` and in a warning naming what it means. "Cuts"
    includes a way that merely ENDS on the edge, because the staged osm.json
    is itself cut to the staging box before it ever gets here - Skyline
    Boulevard runs for tens of kilometres and arrives as a way that stops
    dead on the boundary. A team cannot be told which of the two it is, and
    it makes no difference to them: the assignment stops where the data does.

    The AREA still decides whether there is a candidate at all. A way that
    never enters it is on measured ground and is still nobody's assignment -
    without that rule every trail in the buffer would join the list.

    WHY ONE CANDIDATE PER COMPONENT, MEASURED
    -----------------------------------------
    `linemerge` chains lines only through a node where exactly TWO ends meet.
    A trail that loops or carries a spur meets itself at three-way junctions,
    so it was chopped there: Anniversary Trail at Windy Hill is 8 OSM ways and
    1,698 m entirely inside a 1,500 m area, and came out as seven runs of 428,
    387, 267, 216, 164, 124 and 112 m with every junction between them
    measuring 0.00 m - they touch. Under a 593 m floor (the corpus p10 for a
    line assignment) every run was short, so a 1.7 km trail a planner would
    certainly assign produced no candidate at all. And keeping only
    `_longest_piece` of a clip threw the rest away even when it was on staged
    ground, which is why the whole clip is kept here instead.

    What this does NOT do is unite a name across ground it does not run on:
    Rapley Ranch Road on the same area is three ways, and two of them are a
    420 m stretch 1,968 m from the rest. They are a second component, not part
    of the first, and they are not silently added to it - nor is that second
    component a candidate at all unless it enters the area itself.

    UNNAMED WAYS ARE UNCHANGED
    --------------------------
    They keep the old merge - joined only where one way's END meets another's,
    each merged chain its own candidate. It is the NAME that says two chains
    meeting at a junction are one mapped way; without one there is nothing to
    justify uniting two unrelated tracks that happen to touch, and doing so
    would hand a planner a single assignment made of somebody else's trail.
    """
    entries = ground.lines.get(cls, [])
    by_name: dict[str | None, list[dict]] = {}
    for entry in entries:
        by_name.setdefault(entry.get("name"), []).append(entry)

    runs: list[dict] = []
    touch = MERGE_TOUCH_M / ground.scale
    staged = ground.staged_bounds
    for name, group in by_name.items():
        chains = _merge_lines([e["geom"] for e in group])
        # A gap under `touch` between two chains of one name is still one way.
        chains, bridges = _join_close_parts(chains, touch, ground.scale)
        components = _components(chains) if name is not None else [[c] for c in chains]
        for component in components:
            whole = _as_line(component)
            # The area decides whether this way is a candidate; the staged
            # ground decides how far it runs.
            if not whole.intersects(area):
                continue
            pieces: list = []
            reentered = False
            for chain in component:
                inside = _line_pieces(chain.intersection(staged))
                reentered = reentered or len(inside) > 1
                pieces.extend(inside)
            if not pieces:
                continue
            # The floor is judged against the WHOLE component on staged ground.
            # That is the point of the fix: a 1.7 km trail is not seven stubs.
            on_ground = sum(piece.length for piece in pieces)
            length_m = on_ground * ground.scale
            if length_m < min_length_m:
                continue
            # What the clip here removed, and whether the candidate ENDS on
            # the raster's edge. Both are asked because the staged osm.json is
            # itself cut to the staging box (`terrain.linear.routes_for`), so
            # in a real staging a way the fetch already truncated arrives with
            # nothing left for the clip above to remove - and an end lying on
            # the edge is then the only evidence that the way carried on.
            cut_m = max(0.0, (whole.length - on_ground) * ground.scale)
            clipped_at_edge = (cut_m > EDGE_CLIP_TOLERANCE_M
                               or _ends_on_edge(pieces, staged, ground.scale))
            warnings = []
            if clipped_at_edge:
                how_much = (f"about {round(cut_m)} m of it lies beyond the raster"
                            if cut_m > EDGE_CLIP_TOLERANCE_M
                            else "how much further it runs cannot be seen from here")
                warnings.append(
                    f"this way reaches the edge of the staged ground ({how_much}): the "
                    "assignment stops where the terrain data stops, not where the way "
                    "does. Stage a larger area to be offered the rest")
            if reentered:
                warnings.append("the run leaves and re-enters the staged ground; every "
                                "piece on measured ground is part of this candidate")
            if len(pieces) > 1:
                warnings.append(_branch_warning(len(pieces)))
            # A bridge that was merged INTO this component lies on it exactly;
            # the tolerance is for floating point, not for nearness.
            warnings.extend(f"joined a gap of {round(gap_m)} m between ways"
                            for bridge, gap_m in bridges
                            if whole.distance(bridge.interpolate(0.5, normalized=True)) < 1e-6)
            runs.append({"cls": cls, "axis": _as_line(pieces), "branches": len(pieces),
                         "axis_ids": _ways_of(group, whole, touch),
                         "clipped_at_edge": clipped_at_edge,
                         "sources": [cls], "name": name, "warnings": warnings})
    return runs


def _ends_on_edge(pieces: list, staged, scale: float) -> bool:
    """Does an end of any piece lie on the staged raster's boundary?

    A way that ends ON the edge of the measured ground is a way whose real end
    nobody here can see: it may stop there, or it may run for another ten
    kilometres. Both mean the same thing to the team holding the assignment,
    so both are reported. Ends only - a way that merely RUNS along the edge
    for a stretch has not been cut by it.
    """
    from shapely.geometry import Point

    edge = staged.exterior
    tolerance = EDGE_CLIP_TOLERANCE_M / scale
    for piece in pieces:
        coords = list(piece.coords)
        for end in (coords[0], coords[-1]):
            if edge.distance(Point(end)) <= tolerance:
                return True
    return False


def _branch_warning(branches: int) -> str:
    """What more than one branch means for the team handed the assignment.

    A fork is an acceptable assignment - it is what a planner would draw by
    hand - but it is not one walk, and a planner must not be handed one
    without being told. Stated as a warning rather than left to be discovered
    on the ground, beside the `branches` count it explains.
    """
    return (f"{branches} branches: this candidate forks, or the area cuts it into "
            "pieces, so it is not a single walk end to end - a team covering all of "
            "it has to double back to a junction")


def _components(chains: list) -> list[list]:
    """Chains grouped into connected components: two chains that touch at ANY
    node - a three-way junction as much as an end-to-end one - are one
    component, and so are two that touch only through a third.

    Union-find over `intersects`, never a `unary_union`: a union NODES the
    input, splitting both ways at a crossing, which is the corpus's opposite
    and would turn one component into a handful of stubs before anything was
    measured. Components come back in the order of their first chain, so the
    candidate list stays deterministic.
    """
    parent = list(range(len(chains)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            if find(i) != find(j) and chains[i].intersects(chains[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for index, chain in enumerate(chains):
        groups.setdefault(find(index), []).append(chain)
    return list(groups.values())


def _line_pieces(geometry) -> list:
    """The LineString pieces of a clip.

    A clip against the area can also produce POINTS, where a line only grazes
    the ring, and a point is not a piece of an assignment; a GeometryCollection
    of both is what shapely hands back when it happens.
    """
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [part for part in getattr(geometry, "geoms", [])
            if part.geom_type == "LineString" and not part.is_empty]


def _as_line(pieces: list):
    """One LineString when a component happens to be simple, a
    MultiLineString otherwise. Never a collection of one part: a simple trail
    is a simple trail, and its geometry on the wire should say so."""
    from shapely.geometry import MultiLineString

    return pieces[0] if len(pieces) == 1 else MultiLineString(pieces)


def _merge_lines(geoms: list) -> list:
    """Join lines that share an END POINT, as a list of parts.

    `linemerge` only, never a union first: a union NODES the input, splitting
    both ways at a crossing, and four stubs where a planner sees two trails is
    exactly what the corpus rule forbids - "run a trail end to end, never
    split at junctions". `linemerge` joins only where one line's end meets
    another's, which is the same thing as "these are one way", and leaves two
    ways that merely cross each other alone. It takes LineStrings, so any
    MultiLineString among the inputs is exploded first.
    """
    from shapely.ops import linemerge

    lines = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            lines.append(geom)
        else:
            lines.extend(g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString")
    if not lines:
        return []
    if len(lines) == 1:
        return lines
    merged = linemerge(lines)
    if merged.geom_type == "LineString":
        return [merged]
    return [g for g in getattr(merged, "geoms", []) if g.geom_type == "LineString"]


def _join_close_parts(parts: list, touch: float, scale: float) -> tuple[list, list]:
    """Merge line parts whose ends are within `touch` of each other.

    Returns the merged parts and the bridges that were laid, each with the
    ground gap it spans, so a candidate can say a gap was joined rather than
    presenting a run that is longer than anything actually mapped.
    """
    from shapely.geometry import LineString

    if len(parts) < 2:
        return parts, []
    bridges = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            ends_a = [a.coords[0], a.coords[-1]]
            ends_b = [b.coords[0], b.coords[-1]]
            for ea in ends_a:
                for eb in ends_b:
                    gap = math.dist(ea, eb)
                    if 0 < gap <= touch:
                        bridges.append((LineString([ea, eb]), gap * scale))
    if not bridges:
        return parts, []
    return _merge_lines(parts + [bridge for bridge, _gap in bridges]), bridges


def _ways_of(group: list[dict], part, touch: float) -> list[str]:
    """The ids of the ways a merged run is made of.

    Provenance is kept through the merge rather than recovered geometrically
    afterwards: a way BELONGS to the run when most of it lies along the run,
    not when it happens to touch it. A trail crossing this one shares a point
    with it and would otherwise be credited as one of its ways.
    """
    ids = []
    corridor = part.buffer(touch)
    for entry in group:
        geom = entry["geom"]
        if not geom.length:
            continue
        shared = geom.intersection(corridor)
        if shared.geom_type not in ("LineString", "MultiLineString"):
            continue
        if shared.length / geom.length > AXIS_ID_FRACTION:
            ids.append(entry["id"])
    return ids or [group[0]["id"]]


def _longest_piece(geometry):
    if geometry.geom_type == "LineString":
        return geometry
    pieces = [g for g in getattr(geometry, "geoms", []) if g.geom_type == "LineString"]
    return max(pieces, key=lambda g: g.length) if pieces else None


# --- corridors -----------------------------------------------------------------------


def _corridor_axes(ground: Ground, area) -> list[dict]:
    """Mapped waterway axes clipped to the area, each the longest piece
    inside it and at least MIN_CORRIDOR_AXIS_M long."""
    out: list[dict] = []
    for w in ground.lines.get("waterway", []):
        clipped = w["geom"].intersection(area)
        if clipped.is_empty:
            continue
        piece = _longest_piece(clipped)
        if piece is None or piece.length * ground.scale < MIN_CORRIDOR_AXIS_M:
            continue
        out.append({"cls": "waterway", "geom": w["geom"], "name": w.get("name"),
                    "axis_ids": [w["id"]], "sources": ["waterway"], "axis": piece,
                    "warnings": ([] if clipped.geom_type == "LineString" else
                                 ["the axis leaves and re-enters the area; the longest piece inside is the candidate"])})
    return out


# --- facts ----------------------------------------------------------------------------


def _checked_widths(corridor_widths_m: dict | None) -> dict[str, float]:
    """The caller's width overrides, refused rather than ignored when they
    name a class that has no corridor or a width that is not a width.

    A typo silently dropped is the worst of the three outcomes: the planner
    reads the corpus width back and believes it is the one they asked for.
    """
    overrides = corridor_widths_m or {}
    unknown = sorted(k for k in overrides if k not in CORRIDOR_CLASSES)
    if unknown:
        raise ValueError(f"unknown corridor width class(es) {unknown}; choose from "
                         f"{sorted(CORRIDOR_CLASSES)}")
    checked: dict[str, float] = {}
    for key, value in overrides.items():
        try:
            width = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"corridor width for {key!r} must be a number in ground "
                             f"metres; got {value!r}") from None
        if not width > 0:
            raise ValueError(f"corridor width for {key!r} must be greater than 0 "
                             f"ground metres; got {width:g}")
        checked[key] = width
    return checked


def _rounded(facts):
    """Every float in a candidate's facts at the precision the measurement
    supports, by what the name says it is.

    One pass over the whole facts tree rather than a round() at each site:
    the rule is about the units, not about the caller, and a fact added later
    is rounded because of what it is called. Ints and strings pass through
    untouched, so a junction count stays a count.
    """
    if isinstance(facts, dict):
        return {key: _rounded_value(key, value) for key, value in facts.items()}
    return facts


def _rounded_value(key: str, value):
    if isinstance(value, dict):
        return _rounded(value)
    # None is "not measured" - a slope over a DEM void, a bearing from the
    # centre a candidate runs through - and rounding it is not a question.
    if value is None:
        return None
    # bool is an int; neither is a measurement with decimals to lose.
    if isinstance(value, float):
        places = _places_for(key)
        if places is not None:
            # `+ 0.0` normalises -0.0, which an empty sum of losses produces
            # and which reads on the wire as a negative measurement.
            return round(value, places) + 0.0
    return value


def _places_for(key: str) -> int | None:
    if key in _PLACES_BY_NAME:
        return _PLACES_BY_NAME[key]
    for word in key.split("_"):
        if word in _PLACES_BY_WORD:
            return _PLACES_BY_WORD[word]
    for suffix, digits in _PLACES_BY_SUFFIX:
        if key.endswith(suffix):
            return digits
    return None


def _facts(ground: Ground, entry: dict, geometry_3857, kind: str, centre, existing: list,
           existing_union, widths: dict, evt_path, area, have_map: bool,
           surface) -> dict:
    scale = ground.scale
    axis = entry["axis"]
    slopes, elevations = _along_axis(ground, axis)
    # Difference FIRST, then drop the NaN deltas. Dropping the NaN elevations
    # first joins the samples either side of a DEM void into one step and
    # reports a climb across ground nobody measured; a delta that spans a void
    # is unknown, and unknown is not zero and not a gain.
    deltas = np.diff(elevations) if elevations.size > 1 else np.array([])
    deltas = deltas[~np.isnan(deltas)]
    nearest_lon, nearest_lat = _to_wgs84(entry["nearest"])
    distance_m = entry["distance_m"]
    # An axis over a DEM void has no measured slope and no measured climb.
    # None says that; 0.0 would read as flat ground and no ascent, which is a
    # measurement nobody made - and `nanmean` over all-NaN is a RuntimeWarning
    # and a NaN, which is not JSON either.
    measured_slope = slopes.size and not bool(np.isnan(slopes).all())
    facts: dict = {"length_m": axis.length * scale}
    if kind == "line":
        # How many separate pieces the axis is in: 1 for a simple line, more
        # when the named way forks or the staged ground cuts it up. A team
        # cannot walk a fork end to end, so the count is a fact about the
        # assignment and not an artefact of the geometry - `_branch_warning`
        # says what it means. Only lines have it; a corridor axis is one axis.
        facts["branches"] = int(entry.get("branches", 1))
        # The way carries on past the edge of the measured ground, so this
        # candidate is shorter than the way it is drawn from. A consumer can
        # render that without parsing the warning that says the same thing.
        # Only lines have it: a corridor is traced FROM the raster and is
        # clipped to the subject area, which is not an edge of the data.
        facts["clipped_at_edge"] = bool(entry.get("clipped_at_edge", False))
    facts.update({
        "distance_from_center_m": distance_m,
        # Point to point is not how a team gets there. A creek 200 m away
        # across a cliff band is farther, in the sense a planner acts on, than
        # a trail 800 m away along a contour. None when the walk from the
        # centre never reaches it - see `travel.TRAVEL_ASSUMPTIONS`.
        "travel_minutes_from_center": (travel_minutes_to(surface, ground, geometry_3857)
                                       if surface is not None else None),
        # A candidate that passes through the centre has no direction from it;
        # 0 would read as due north.
        "bearing_from_center_deg": (_bearing(centre, entry["nearest"]) if distance_m else None),
        "nearest_point": {"latitude": nearest_lat, "longitude": nearest_lon},
        "slope_mean_deg": float(np.nanmean(slopes)) if measured_slope else None,
        "gain_m": float(deltas[deltas > 0].sum()) if deltas.size else None,
        "loss_m": float(-deltas[deltas < 0].sum()) if deltas.size else None,
        "ingress": _ingress(ground, axis, entry["cls"], area),
    })
    if kind == "corridor":
        facts["width_m"] = float(widths[entry["cls"]])
        facts["area_acres"] = acres(geometry_3857.area * scale * scale)
    # The density class and the traverse table are the same for every
    # candidate; the set states them once. A line's assumptions are ALL of
    # that kind, so it points at the set's copy rather than carrying `{}`,
    # which reads as "this candidate assumed nothing".
    effort = candidate_effort(ground, geometry_3857, kind, evt_path=evt_path,
                              include_shared_assumptions=False)
    if not effort["assumptions"]:
        effort["assumptions"] = SHARED_EFFORT_NOTE
    facts["effort"] = effort
    (facts["overlap_existing_pct"], facts["overlap_labels"],
     facts["overlap_reason"]) = _overlap(geometry_3857, kind, existing, existing_union, have_map)
    return facts


def _along_axis(ground: Ground, axis) -> tuple[np.ndarray, np.ndarray]:
    """Slope and elevation along an axis, at `effort.sample_along`'s points."""
    rows, cols = sample_along(ground, axis)
    return ground.slope_deg[rows, cols], ground.grid.elevation[rows, cols]


def _ingress(ground: Ground, axis, cls: str, area) -> dict:
    """Junctions with walkable lines (other than the axis's own ways) and
    crossings of the area ring are where a team gets onto this candidate.

    A line axis now runs past the area to the edge of the staged ground, so
    the crossings are where it passes THROUGH the subject's ring rather than
    where the area cut it: a trail entering and leaving the circle still
    counts 2. `area.boundary`, not `area.exterior`: a subject can be a
    MultiPolygon (two lobes of one search area), which has no single exterior
    at all, and the edge of a hole is as much a way in as the outer ring is.
    """
    junctions = 0
    for other_cls in ("trail", "road"):
        entries = ground.lines.get(other_cls, [])
        tree = ground.trees.get(other_cls)
        nearby = tree.query(axis) if tree is not None else range(len(entries))
        for index in nearby:
            entry = entries[int(index)]
            if entry["geom"].intersects(axis):
                crossing = entry["geom"].intersection(axis)
                if crossing.geom_type == "Point":
                    junctions += 1
                elif crossing.geom_type == "MultiPoint":
                    junctions += len(crossing.geoms)
                # A shared segment (the axis's own way) is not a junction.
    return {"junctions": junctions, "area_crossings": _point_count(axis.intersection(area.boundary))}


def _point_count(geometry) -> int:
    """Points in an intersection. A ring with a hole is a MultiLineString, so
    the result can be a GeometryCollection when the axis also runs ALONG part
    of the ring; only the points count as crossings."""
    if geometry.geom_type == "Point":
        return 1
    if geometry.geom_type == "MultiPoint":
        return len(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return sum(_point_count(g) for g in geometry.geoms)
    return 0


def _overlap(geometry_3857, kind: str, existing: list, existing_union,
             have_map: bool) -> tuple[float | None, list[str], str | None]:
    """How much of this candidate an existing assignment already covers.

    None with a reason when no map was readable, 0.0 when a map WAS read and
    holds nothing this candidate touches. The two are different facts - one
    is "not measured", the other is "measured, and clear" - and a planner
    deciding whether a line is already someone's job needs to be able to tell
    them apart. boundary.py's `shared_pct`/`shared_reason` says the same
    thing about a boundary's shared edges.
    """
    if not have_map:
        return None, [], NO_MAP_OVERLAP_REASON
    if not existing or existing_union is None:
        return 0.0, [], None
    labels = [label for label, geom in existing if geom.intersects(geometry_3857)]
    inside = geometry_3857.intersection(existing_union)
    if kind == "line":
        pct = 100.0 * inside.length / geometry_3857.length if geometry_3857.length else 0.0
    else:
        pct = 100.0 * inside.area / geometry_3857.area if geometry_3857.area else 0.0
    return pct, labels, None


def _union_of(existing: list):
    """One geometry for every existing assignment, or None when there are
    none. Built once per set: it is the same for every candidate."""
    from shapely.ops import unary_union

    if not existing:
        return None
    return unary_union([geom for _label, geom in existing])


def _existing_assignments(map_features: list[dict] | None) -> tuple[list[tuple[str, object]], list[str]]:
    """The map's assignments as Web Mercator geometries, and the labels of the
    ones that would not project.

    A malformed geometry is a fact about the map: the overlap figures are
    computed without it, so the caller has to be told which labels are not in
    them rather than being handed a number that quietly means less than it says.
    """
    if not map_features:
        return [], []
    out: list[tuple[str, object]] = []
    failed: list[str] = []
    for feature in map_features:
        props = feature.get("properties") or {}
        if props.get("class") != "Assignment" or not feature.get("geometry"):
            continue
        label = f"{props.get('letter') or ''}{props.get('number') or ''}".strip() or props.get("title", "")
        try:
            geom = project_geometry_to_3857(feature["geometry"])
        except Exception:
            geom = None
        if geom is None or geom.is_empty or not geom.is_valid:
            failed.append(label or "(unlabelled)")
            continue
        out.append((label, geom))
    return out, failed


# --- geometry helpers ----------------------------------------------------------------


def _centre_lon_lat(centre) -> tuple[float, float]:
    """`(longitude, latitude)` from either shape a staged centre comes in.

    `prepare_area` writes `[latitude, longitude]`; every point the tools hand
    back is `{"latitude", "longitude"}`, so a caller round-tripping one of
    those into a subject is reading the same fact, written the other way.
    """
    if isinstance(centre, dict):
        return float(centre["longitude"]), float(centre["latitude"])
    latitude, longitude = centre
    return float(longitude), float(latitude)


def _to_3857(longitude: float, latitude: float):
    from shapely.geometry import Point

    x, y = _TO_3857.transform(longitude, latitude)
    return Point(x, y)


def _to_wgs84(point) -> tuple[float, float]:
    return _TO_WGS84.transform(point.x, point.y)


def _nearest_point(axis, centre):
    from shapely.ops import nearest_points

    on_axis, _on_centre = nearest_points(axis, centre)
    return on_axis


def _bearing(origin, target) -> float:
    """Initial bearing in Web Mercator, which at these scales is within a
    degree of geodesic; 0 north, 90 east."""
    dx, dy = target.x - origin.x, target.y - origin.y
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
