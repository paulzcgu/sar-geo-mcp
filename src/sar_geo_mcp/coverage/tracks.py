"""Selecting the tracks that belong to one assignment.

Track labels are operator-entered and inconsistent: on one real map a single
assignment's tracks were labelled "1211.2 QQ", "1215.2XY", "1215-2ZZ",
"1215ZZ" and "1215.2 xq" - dot, hyphen, space, nothing, and mixed case; on
another, "QX106" puts the team number after the searcher's initials. Teams
provide only their number, so a team id selects a CANDIDATE set by matching
as a maximal digit run in the normalised label (any separator, either
order - "1024" and "3102" do not match team 102, because the digit run
continues), and geometry decides membership (see select_tracks). Neither
test is a fallback for the other.

Tracks live in two CalTopo containers - class "Shape" (Lines & Polygons) and
"AppTrack" (App Tracks) - and the same sortie can be recorded into both and
exported as two features with the same label and near-identical geometry.
Counted twice, effort_coverage (which sums track length) would double for
that sortie. select_tracks collapses such pairs (see
_collapse_duplicate_sorties); two genuinely different tracks that merely
share a label - a different route, or a real second sortie - are not
touched.
"""

import logging
import re
from dataclasses import dataclass

from shapely.geometry import LineString, Point

_NOISE = re.compile(r"[^a-z0-9]")


def normalise_label(label: str) -> str:
    """Lowercase, with every separator removed."""
    return _NOISE.sub("", label.lower())


from .teams import STRONG, WEAK, has_team_number, match_strength, matches_team  # noqa: F401


# Classes CalTopo stores line features under. "Shape" is the Lines & Polygons
# folder; "AppTrack" is App Tracks; the other two appear in the map schema.
logger = logging.getLogger(__name__)

TRACK_CLASSES = ("Shape", "AppTrack", "FieldTrack", "LiveTrack")

# The same sortie can be recorded into two containers (class Shape and
# AppTrack) and exported as two features with identical label and geometry -
# see the module docstring and _collapse_duplicate_sorties. Both thresholds
# are in ground metres, via `ground_factor`:
#
# - Hausdorff distance <= 10 m: comfortably above the rounding/simplification
#   noise between two exports of the *same* recorded points (observed on a
#   real duplicate pair: 0.0 m exactly) and comfortably below the distance
#   between two tracks that took a genuinely different route.
# - Length ratio <= 1.1 (10%): tolerates that noise in the length figure
#   too, while still rejecting a same-named track that is a materially
#   different length (observed on a real non-duplicate pair: 72x) or a
#   different route of coincidentally similar length (a real second
#   sortie - a different route fails the Hausdorff test instead).
#
# Both must hold; either alone is not enough (a 72x-longer track can still
# start at a nearby point, and two dissimilar routes can have similar
# length).
DUPLICATE_HAUSDORFF_M = 10.0
DUPLICATE_LENGTH_RATIO = 1.1


class TrackSelectionError(RuntimeError):
    """No usable tracks - the message tells the operator which case it is."""


@dataclass(frozen=True)
class SelectedTrack:
    name: str
    geometry: object          # clipped to the assignment (plus clip_margin, if given)
    length_in_zone_m: float
    length_total_m: float
    # Epoch milliseconds, from the track's own recorded points - see
    # _timestamps_in_zone. None when the track carries no timestamp ordinate
    # at all, or none of its points land inside the (possibly buffered) zone;
    # never guessed.
    time_entered_ms: int | None = None
    time_exited_ms: int | None = None

    @property
    def in_zone_fraction(self) -> float:
        if not self.length_total_m:
            return 0.0
        return self.length_in_zone_m / self.length_total_m


def line_from_feature(feature: dict):
    """2-D LineString from a CalTopo feature, or None if unusable.

    Track coordinates are [lon, lat, elevation_m, timestamp_ms]; shapely
    rejects 4-D points, so only x and y survive here.
    """
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        return None
    coordinates = geometry.get("coordinates") or []
    if len(coordinates) < 2:
        return None
    return LineString([(c[0], c[1]) for c in coordinates])


def _is_duplicate_sortie(a: LineString, b: LineString, ground_factor: float) -> bool:
    """True when `a` and `b` are the same recorded track, exported twice."""
    if a.length == 0 or b.length == 0:
        return False
    hausdorff_m = a.hausdorff_distance(b) * ground_factor
    length_ratio = max(a.length, b.length) / min(a.length, b.length)
    return hausdorff_m <= DUPLICATE_HAUSDORFF_M and length_ratio <= DUPLICATE_LENGTH_RATIO


def _collapse_duplicate_sorties(
    named: list[tuple[str, LineString, list]], ground_factor: float
) -> list[tuple[str, LineString, list]]:
    """Collapse same-label candidates whose geometry is near-identical.

    Dedup is geometric, not by name: two candidates only collapse when both
    the label matches AND the geometry is a near-identical match (see
    DUPLICATE_HAUSDORFF_M / DUPLICATE_LENGTH_RATIO). A same-named pair that
    fails the geometry test is kept as two tracks - it is either a
    genuinely different route, or a second sortie under the same label,
    both of which are real coverage that must not be dropped. The raw
    (4-D) coordinate list rides along unexamined - dedup never looks at it,
    it is just carried through to the survivor for the time-bounds step.
    """
    kept: list[tuple[str, LineString, list]] = []
    for title, line, coordinates in named:
        if any(existing_title == title
              and _is_duplicate_sortie(line, existing_line, ground_factor)
              for existing_title, existing_line, _ in kept):
            continue
        kept.append((title, line, coordinates))
    return kept


def _timestamps_in_zone(coordinates: list, clip_zone) -> list[int]:
    """Epoch-ms timestamps of a track's own recorded points that land inside
    `clip_zone` (already buffered by clip_margin, if any).

    Tested point-by-point against the ORIGINAL coordinates, not the vertices
    of the clipped geometry (`line.intersection(clip_zone)`): shapely invents
    new vertices exactly on the clip boundary when it clips a line, and those
    carry no timestamp at all. A point missing its fourth (timestamp)
    ordinate contributes nothing - a track can be partly timed and still get
    a best-effort answer from the points that are.
    """
    timestamps: list[int] = []
    for coordinate in coordinates:
        if len(coordinate) < 4 or coordinate[3] is None:
            continue
        if clip_zone.intersects(Point(coordinate[0], coordinate[1])):
            timestamps.append(int(coordinate[3]))
    return timestamps


#: Why a weak match is weak, in words a planner can check against the title
#: (see teams._adverse_context). It must be true of every title demoted,
#: not only the drone flights and rings it was written for: a power line
#: ("Distribution"), a map import ("OSM"), and a surname holding a month's
#: abbreviation ("Mar" in "104 Mark") are demoted too, and a reason they
#: cannot find in the title would send the planner looking for the wrong
#: thing. "Such as" because the word list is longer than a sentence.
_READS_AS_SOMETHING_ELSE = ("a unit, ordinal or clock time after the number, or a word such "
                            "as FLIGHT, DRONE, OSM, Distribution or a month or its "
                            "abbreviation, even inside a longer word")
#: How many titles a warning names before it only counts the rest: a map's
#: worth of probability rings must not bury the sentence that matters.
_TITLES_NAMED = 8


def _titles(titles: list[str]) -> str:
    """Each title once, in the order met, the first `_TITLES_NAMED` by name
    and the rest counted."""
    distinct = list(dict.fromkeys(titles))
    named = ", ".join(distinct[:_TITLES_NAMED])
    if len(distinct) > _TITLES_NAMED:
        named += f" and {len(distinct) - _TITLES_NAMED} more"
    return named


def _teams(teams: list[str]) -> str:
    return ("team " if len(teams) == 1 else "teams ") + ", ".join(teams)


def _owners(teams: list[str], one: bool) -> str:
    """Whose lines the planner is asked to check them against. With several
    teams, "the team's" leaves them guessing which team is meant - and
    which of them a line carries is not said either (see
    `_set_aside_warning`)."""
    if len(teams) == 1:
        return "the team's"
    return "one of those teams'" if one else "those teams'"


def _fallback_warning(teams: list[str], used: list[str]) -> str:
    """Nothing plainly named for the team: the coverage counts lines whose
    names read as something else, and only the operator can say whether
    they are the team's."""
    one = len(set(used)) == 1
    return (f"no track is plainly named for {_teams(teams)}, so this coverage counts "
            f"{_titles(used)}, whose {'name reads' if one else 'names read'} as something "
            f"else ({_READS_AS_SOMETHING_ELSE}) - check "
            f"{'it is' if one else 'they are'} {_owners(teams, one)} own before taking the "
            f"ground {'it covers' if one else 'they cover'} as searched")


def _set_aside_warning(teams: list[str], set_aside: list[str]) -> str:
    """Lines reaching the assignment that were set aside beside plainly
    named ones: if they were the team's, their ground is missing. Which of
    several teams each one carries is not said - "carries teams 3, 4"
    would be untrue of a line carrying one of them."""
    one = len(set(set_aside)) == 1
    return (f"set aside {_titles(set_aside)}, which {'reaches' if one else 'reach'} this "
            f"assignment but whose {'name reads' if one else 'names read'} as something "
            f"else ({_READS_AS_SOMETHING_ELSE}): this coverage counts only the tracks plainly "
            f"named for {_teams(teams)} - if {'it is' if one else 'they are'} "
            f"{_owners(teams, one)} own, the ground {'it' if one else 'they'} covered is "
            "missing here")


def _set_aside_error(set_aside: list[str]) -> str:
    """What "none intersect" leaves out when a set-aside line does reach
    the assignment."""
    one = len(set(set_aside)) == 1
    return (f" {'A line whose name reads' if one else 'Lines whose names read'} as something "
            f"else ({_READS_AS_SOMETHING_ELSE}) {'does' if one else 'do'} reach it, and "
            f"{'was' if one else 'were'} set aside beside the plainly named: "
            f"{_titles(set_aside)}")


def _word_name_warning(team: str, titles: list[str]) -> str:
    """A name with no team number selected these by a word alone. "Can":
    a call sign is such a word too, and finds one team's lines only."""
    one = len(set(titles)) == 1
    return (f"{team!r} has no team number in it, so it selected {_titles(titles)} by that "
            f"word alone - check {'it is the team' if one else 'they are the team'}'s own: "
            "an assignment's label can name every team's lines on it")


def _say(warnings: list[str] | None, text: str) -> None:
    if warnings is not None and text not in warnings:
        warnings.append(text)


def select_tracks(features: list[dict], zone, teams: list[str],
                  ground_factor: float = 1.0,
                  clip_margin: float = 0.0,
                  warnings: list[str] | None = None) -> list[SelectedTrack]:
    """Tracks belonging to `teams` that actually enter `zone` (or its margin).

    Name selects candidates; geometry decides membership. Both stages are
    required. `zone` and the features must already share a coordinate system;
    `ground_factor` converts that system's units to ground metres.

    There is no transit classification (operator decision, spec SS5.2
    amended 2026-08-17): every candidate that intersects the (possibly
    buffered) zone is kept and used in full - a track is not excluded, or
    partly excluded, for having little length inside the assignment.

    `clip_margin` (same units as `zone`) buffers the zone outward before a
    track is clipped and tested for intersection: a searcher walking just
    outside the boundary genuinely sweeps into it, so the geometry used for
    the swath should extend that far too. `length_in_zone_m` is still
    measured against the UNBUFFERED zone, because it feeds the Koopman
    effort number alongside `zone_area_m2` (the unbuffered zone's area) -
    pairing a buffered numerator with an unbuffered denominator would
    overstate effort coverage relative to the area it is judged against.

    `warnings`, when given, gains what the operator must check about the
    names the selection went by, each once and naming the lines: that
    nothing was plainly named for the team, so the lines counted are ones
    whose names read as something else (a drone flight, a probability
    ring); that such lines reaching the assignment were set aside beside
    plainly named ones, so whatever they covered is missing if they were
    the team's; and that a name with no team number in it selected lines
    by a word alone. Only lines that reach the (possibly buffered) zone are
    named - one across the map would not have counted either way. The
    server log says the first two as well, but the log never reaches the
    caller: it is not where an operator reads.
    """
    strong: list[tuple[str, LineString, list]] = []
    weak: list[tuple[str, LineString, list]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("class") not in TRACK_CLASSES:
            continue
        title = (properties.get("title") or "").strip()
        strength = None
        for team in teams:
            found = match_strength(title, team)
            if found == STRONG:
                strength = STRONG
                break
            if found == WEAK:
                strength = WEAK
        if strength is None:
            continue
        line = line_from_feature(feature)
        if line is not None:
            coordinates = (feature.get("geometry") or {}).get("coordinates") or []
            (strong if strength == STRONG else weak).append(
                (title, line, coordinates))

    # A map's Shape layer carries more than searcher tracks: drone sorties,
    # flight logs and probability rings all contain digits that can read as
    # a team number, and the spatial filter cannot tell one from somebody
    # walking. When anything matched plainly, use only those - counting a
    # drone line as team coverage reports ground as searched that nobody
    # swept.
    named = strong or weak
    set_aside = weak if strong else []
    falling_back = bool(weak) and not strong
    if falling_back:
        # Falling back is right - a real team CAN be oddly named, and
        # refusing would lose their work. But this is exactly when somebody
        # needs telling: on a real map the only "team 3" matches were
        # probability rings like "75% (2.3mi) - Hiker", and counting
        # those as swept ground would overstate coverage badly. Telling the
        # log alone told nobody (PR #4 review B2): `warnings`, below, is
        # what reaches the planner.
        logger.warning(
            "team match: no track is plainly named for team(s) %s; falling "
            "back to %d whose names read as something else (%s). Check these "
            "are really that team's tracks.",
            ", ".join(teams), len(weak),
            ", ".join(title for title, _line, _c in weak[:5]),
        )

    named = _collapse_duplicate_sorties(named, ground_factor)

    if not named:
        titles = sorted({
            (f.get("properties") or {}).get("title", "").strip()
            for f in features
            if (f.get("properties") or {}).get("class") in TRACK_CLASSES
        } - {""})
        sample = ", ".join(titles[:8])
        if len(titles) > 8:
            sample += f" ... ({len(titles)} named lines in total)"
        raise TrackSelectionError(
            f"no tracks on this map are named for team(s) {', '.join(teams)}. "
            f"Named lines here look like: {sample or '(none)'}. If this "
            f"agency labels tracks a different way, pass the number as it "
            f"appears in those names."
        )

    clip_zone = zone.buffer(clip_margin) if clip_margin else zone
    # The set-aside lines whose setting aside changes this coverage: those
    # that reach the zone. One elsewhere on the map would not have counted
    # anyway, and naming every ring on it would bury the one that matters.
    set_aside_here = [title for title, line, _c in set_aside if line.intersects(clip_zone)]
    selected: list[SelectedTrack] = []
    for title, line, coordinates in named:
        if not line.intersects(clip_zone):
            continue
        clipped = line.intersection(clip_zone)
        length_in_zone_m = (
            clipped.length if not clip_margin else line.intersection(zone).length
        ) * ground_factor
        # Time bounds are tested against clip_zone (the same, possibly
        # buffered, region used for membership and for `clipped` above) -
        # not the unbuffered `zone` used for length_in_zone_m - so a
        # searcher's genuine sweep just outside the boundary still counts
        # toward when they were working the assignment.
        timestamps = _timestamps_in_zone(coordinates, clip_zone)
        selected.append(SelectedTrack(
            name=title, geometry=clipped,
            length_in_zone_m=length_in_zone_m,
            length_total_m=line.length * ground_factor,
            time_entered_ms=min(timestamps) if timestamps else None,
            time_exited_ms=max(timestamps) if timestamps else None,
        ))

    if not selected:
        # A line set aside that does reach the assignment is the one thing
        # here an operator could act on; "none intersect" alone would send
        # them looking for the wrong data.
        aside = _set_aside_error(set_aside_here) if set_aside_here else ""
        raise TrackSelectionError(
            f"found {len(named)} track(s) for team(s) {', '.join(teams)}, but "
            f"none intersect this assignment - check the team number, and that "
            f"the track is from this assignment and on this map.{aside}"
        )
    if set_aside:
        # Reported, never silent. A narrowed track list is how coverage ends
        # up computed from part of a team's work, and the operator is the
        # only one who can tell whether "FLIGHT 3" is their team 3.
        logger.warning(
            "team match: using %d plainly-named track(s); ignored %d whose "
            "names read as something else (%s)",
            len(selected), len(set_aside),
            ", ".join(title for title, _line, _c in set_aside[:5]),
        )
    if falling_back:
        _say(warnings, _fallback_warning(teams, [track.name for track in selected]))
    if set_aside_here:
        _say(warnings, _set_aside_warning(teams, set_aside_here))
    # A name with no number selects by a word, which is also how an
    # assignment's label - an easy slip for a track name - would select
    # every team's line on it. Named for the lines only such a name chose:
    # a line a team number also chose needs no second look on its account.
    numbered = [team for team in teams if has_team_number(team)]
    for team in teams:
        if has_team_number(team):
            continue
        chosen = [track.name for track in selected
                  if match_strength(track.name, team) is not None
                  and not any(match_strength(track.name, other) for other in numbered)]
        if chosen:
            _say(warnings, _word_name_warning(team, chosen))
    return selected
