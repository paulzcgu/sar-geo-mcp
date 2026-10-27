"""What every scenario shares: sections and their status, the refusals made
before any work, staging, the export file, and one helper per feature a
scenario calls. Not tools.

A SECTION IS ONE FEATURE'S ANSWER, SUMMARISED
---------------------------------------------
A scenario answers in sections, one per feature it calls. Each is a dict
with `status` ("ok", "failed" or "skipped"), `error` (None when ok, else
"<ExceptionType>: <message>"), `warnings` (the feature's own) and the
section's compact facts. A section that raises is caught here and recorded
as failed - `Exception`, never `BaseException`, so a Ctrl-C or a
`SystemExit` still stops the call - because a medical lookup with no network
must not take the landing zones down with it. The scenario's overall
`status` is "ok" when every section asked for is ok, "failed" when every one
failed with nothing to show, and "partial" otherwise; a skipped section
counts neither way, and nor does an `export` section, which only draws what
the others found and whose every problem is a warning: an unwritable file
must not turn "ok" into "partial", nor an export with nothing to draw turn
"failed" into "partial". A section made of several calls fails when any of
them fails but keeps the ones that answered - egress, one call per mode,
under `modes`; urban, the blocks and then their grouping, the blocks under
`blocks` - so it never makes the whole answer read "failed" while it still
holds a way out, or the blocks of the town.

A CALLER'S MISTAKE IS NOT A SECTION FAILURE
-------------------------------------------
Bad coordinates, a radius `prepare_area` would refuse, an unknown section in
`include`, an `overrides` key no section has, a keyword its feature does
not take, and an override value the feature tool would refuse are refused
with a ValueError before anything is staged or fetched. Reported as a failed
section they would cost the caller the staging and the network first, and
then read as though the ground or a service had let them down. An override
value is read as the feature tool's own schema reads it - "6" a number,
"false" false - because `overrides` is an untyped mapping that no schema
converts on the way in: read as given, "false" is truthy and would turn on
the very thing the caller turned off. A number that is not finite ("nan",
"inf") is refused too, though the tool's schema would take it: NaN compares
false with every limit, so it passes each range check and the section
answers "ok" with nothing found - silence that reads as clear.

COMPACT, NEVER GEOMETRY
-----------------------
A section carries points (a touchdown, an exit, a hospital) but no
polygon, route, ring or outline: a scenario's answer is read by a planner or
a model first, and a patch outline costs thousands of tokens and says
nothing a number does not. Each call a section makes carries `detail_call`,
`{"tool": ..., "arguments": {...}}`, the exact feature call that returns the
full answer, geometry included - a call that raised keeps it too, so the
caller can make it again. What the export file needs of the geometry
travels beside the section in `Outcome.features` and never reaches the
caller.

UNKNOWN IS NULL, NEVER ZERO
---------------------------
A feature whose input was not staged answers an empty list and a warning:
parking and trail or road hasty lines on ground staged during an
OpenStreetMap outage, a clearing with no vegetation panel. The section
keeps the warning, but the summary - the facts read first - carries none
beside its counts, and a 0 there, or an axis left out of a count by axis,
reads exactly as ground where the input was read and holds nothing: "no
parking here", "no trails". So each count the summary copies is null when
its input was missing, in the section and the summary alike, decided by
the helper that sees the feature's whole answer: parking's total
(`_lots_unknown`); the hasty axes read from OpenStreetMap (`_unknown_axes`),
a hasty kind none of whose axes asked for could be read, and the hasty
total when no axis asked for could be; a feature kind whose detector
declares the warning it gave (`terrain.features.input_missing`). A count
that read some of its input counts what it read, the nulls beside it
saying what it could not; a count whose input was read stays a number, 0
included: that is a real "none".

THE HELPERS CALL FEATURE AND COMPONENT FUNCTIONS
------------------------------------------------
Through the module (`lz_feature.lz_candidates(ctx, ...)`), never MCP and
never `planning` directly, so a section sees the envelope - warnings,
staleness - an MCP caller of the tool sees, and a test can stand a stub in
for the function. There is exactly one helper per feature: a second
scenario needing landing zones calls `lz_section`, not a copy of it.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, get_type_hints

from pydantic import ConfigDict, TypeAdapter, ValidationError

from ..cache import atomic_write_text
from ..components import candidates as candidates_component
from ..components import ground as ground_component
from ..components import staging as staging_component
from ..context import ServerContext
from ..features import coverage as coverage_feature
from ..features import egress as egress_feature
from ..features import hasty as hasty_feature
from ..features import lz as lz_feature
from ..features import medical as medical_feature
from ..features import parking as parking_feature
from ..features import travel_rings as rings_feature
from ..features import urban as urban_feature
from ..planning.area import MAX_EXTENT_M, PLANNING_AREA_BUFFER_M, build_area
from ..planning.boundary import MISSING_OSM_WARNING as NO_OSM_LINES_WARNING
from ..planning.boundary import ROUTE_CLASS
from ..planning.corridors import ALL_INCLUDE as HASTY_ALL_INCLUDE
from ..planning.corridors import DEFAULT_INCLUDE as HASTY_DEFAULT_INCLUDE
from ..planning.corridors import CORRIDOR_CLASSES, LINE_CLASSES, hasty_params
from ..planning.egress import check_params as check_egress_params
from ..planning.egress_graph import MODES as EGRESS_DEFAULTS
from ..planning.egress_roads import GATES_UNKNOWN_WARNING, ROAD_FACTS_UNKNOWN_WARNING
from ..planning.export import EXPORTS_DIRNAME, check_label_prefix, folder_title, in_one_folder
from ..planning.lz import check_params as check_lz_params
from ..planning.medical import CATEGORIES, ORIGIN_COLOR, ORIGIN_SYMBOL
from ..planning.medical import check_params as check_medical_params
from ..planning.norms import resource_median_ha
from ..planning.parking import MISSING_OSM_WARNING as NO_OSM_PARKING_WARNING
from ..planning.populated import check_params as check_block_params
from ..planning.populated import marker_colour, panel_pixel_m
from ..planning.populated import populated_areas as populated_areas_impl
from ..planning.travel_rings import (LARGEST_RADIUS_M, fastest_factor, minutes_list,
                                     radius_to_fit, rings_model)
from ..planning.walking import NO_EVT_WARNING, NO_OSM_WARNING, WATERWAY_KINDS_UNKNOWN_WARNING
from ..sources.osm import OSM_QUERY_VERSION_PARKING

OK, FAILED, SKIPPED = "ok", "failed", "skipped"
#: The section that draws what the others found for CalTopo. Its problems
#: are warnings, never a failure (`set_export_section`, `write_export`), so
#: `overall_status` leaves it out, and `requested` refuses it alone.
EXPORT_SECTION = "export"

#: The radius a scenario's ground must stay under: `prepare_area` refuses an
#: area over 9,000 m across, before its 1,000 m planning buffer, measured on
#: its own circle polygon - which puts a 4,500 m circle a hair over the
#: limit everywhere, and a 4,499.9 m one over it near the poles. So this is
#: only the first, cheap bound; `check_radius` then asks `build_area` itself.
MAX_RADIUS_M = MAX_EXTENT_M / 2.0

#: `lz_candidates` will not grade ground coarser than 3 m a pixel, so every
#: scenario stages its landing-zone ground at 2 m.
LZ_RESOLUTION_M = 2.0
#: `egress_routes` and `travel_time_rings` walk every cell and refuse ground
#: over 1,800 x 1,800 cells, so their ground is staged at 10 m: even the
#: widest area `prepare_area` stages (just under 4,500 m round the point, plus
#: the 1,000 m planning buffer) is then 1,100 cells a side.
WALKING_RESOLUTION_M = 10.0


# --- Sections ----------------------------------------------------------------


@dataclass
class Outcome:
    """One section's answer: what the caller reads, and the GeoJSON features
    it contributes to the export file. Kept apart because the features carry
    the geometry the section leaves out."""

    section: dict[str, Any]
    features: list[dict] = field(default_factory=list)
    #: Lines the section adds to the description of the scenario's own
    #: point marker: what the feature's own point marker would have said,
    #: such as the medical tool's "road route starts 640 m away", which the
    #: export must not lose when the scenario draws the point instead.
    point_notes: list[str] = field(default_factory=list)
    #: The `staleness` block the feature returned with its answer, kept off
    #: the section: every call on one assignment handle carries one, and a
    #: scenario says it once (`staleness_warnings`), not once per section.
    staleness: dict[str, Any] | None = None


class StagingFailed(RuntimeError):
    """A section's ground could not be staged, so the section did not run."""


class DependencyFailed(RuntimeError):
    """A section needs what another section computes first, and that section
    failed - or was left out of include with nothing saved by an earlier
    call - so this one did not run: coverage of an area assignment takes its
    sweep width from the metrics `terrain_metrics` saves."""


class FeatureFailed(Exception):
    """A feature call raised: what it raised, and the call itself, so the
    failed section can still say exactly how to make it again."""

    def __init__(self, error: Exception, call: dict[str, Any]):
        super().__init__(f"{type(error).__name__}: {error}")
        self.error, self.call = error, call


def call_feature(ctx: ServerContext, tool: str, function: Callable,
                 arguments: dict[str, Any]) -> dict[str, Any]:
    """`function(ctx, **arguments)`, the feature behind `tool`; whatever it
    raises comes back as FeatureFailed carrying the call, for `guard`."""
    try:
        return function(ctx, **arguments)
    except Exception as error:  # noqa: BLE001 - re-raised with the call attached
        raise FeatureFailed(error, detail_call(tool, arguments)) from error


def section(status: str, *, error: str | None = None, warnings=(), **data) -> dict[str, Any]:
    """A section dict: the three fields every section has, then its data."""
    return {"status": status, "error": error, "warnings": list(warnings), **data}


def skipped() -> Outcome:
    return Outcome(section(SKIPPED))


def error_text(ctx: ServerContext, error: Exception) -> str:
    """"<ExceptionType>: <message>", with every credential the server holds
    blanked out. The domain already says why a panel or chip fetch failed
    without its URL (`net_errors.safe_reason`), but a section's error is
    whatever the feature raised, from anywhere below it, and a CalTopo
    GetMap URL carries the WMS token in its path: one unguarded message
    would put it in an answer an agent logs."""
    text = f"{type(error).__name__}: {error}"
    for secret in (ctx.config.wms_token, ctx.config.mapbox_token,
                   ctx.config.google_maps_key, ctx.config.caltopo_key):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def guard(ctx: ServerContext, work: Callable[[], Outcome]) -> Outcome:
    """`work()`, or a failed section saying why. Catches `Exception` only:
    an interrupt or an exit is the operator's, not a section's. A feature
    call that raised (`call_feature`) leaves its `detail_call` on the failed
    section, so the caller can retry it or make it directly; a section whose
    ground could not be staged never made one, and its error names the
    ground, whose staging record carries the `prepare_area` call."""
    try:
        return work()
    except FeatureFailed as failure:
        return Outcome(section(FAILED, error=error_text(ctx, failure.error),
                               detail_call=failure.call))
    except Exception as error:  # noqa: BLE001 - one section must not sink the rest
        return Outcome(section(FAILED, error=error_text(ctx, error)))


def answered_in_part(entry: dict[str, Any]) -> bool:
    """Whether a section holds an answer although it failed: a section made
    of several calls fails when any of them did, but keeps the ones that
    answered - egress each mode's own status under `modes`, urban the blocks
    populated_areas drew under `blocks` when the grouping after them raised
    (a saved set, which `get_candidate` and its `detail_call` still read)."""
    if any(part.get("status") == OK for part in (entry.get("modes") or {}).values()):
        return True
    return bool((entry.get("blocks") or {}).get("set_id"))


def overall_status(sections: dict[str, dict]) -> str:
    """"ok" when every section asked for is ok, "failed" when every one
    failed with nothing to show, "partial" otherwise. A failed section that
    still holds an answer (`answered_in_part`) makes it partial: an egress
    whose carry-out raised but whose walk-out came back has a way out to
    report, and "failed" would tell the reader to throw it away. Skipped
    sections count neither way, and nor does the export section
    (EXPORT_SECTION): it finds nothing of its own, and a file it could not
    write is a warning, not a finding lost. A scenario refuses an `include`
    that asks for none of the others (`requested`), so there is always one
    to count."""
    asked = [entry for name, entry in sections.items()
             if entry["status"] != SKIPPED and name != EXPORT_SECTION]
    if all(entry["status"] == OK for entry in asked):
        return OK
    if all(entry["status"] == FAILED and not answered_in_part(entry) for entry in asked):
        return FAILED
    return "partial"


def summarise(sections: dict[str, dict], asked: tuple[str, ...],
              facts: dict[str, Callable[[dict], Any]], *,
              in_part: tuple[str, ...] = ()) -> dict[str, Any]:
    """A scenario's `summary`: one block per section asked for that has a
    facts function in `facts` - an export has none, its paths are in
    `exports` - in the order of `facts`, read off the answered section, or
    None when the section did not answer. A section named in `in_part` is
    read whenever it holds an answer, whole or in part (`answered_in_part`):
    an egress whose carry-out raised still has a walk-out to report, and
    `egress_facts` says each mode's own. Facts only, in the features' own
    orders - nearest first, quickest first, largest first; nothing is
    chosen. One function for every scenario, so "null" means the same
    thing in each."""
    return {name: read(sections[name])
            if sections[name]["status"] == OK
            or (name in in_part and answered_in_part(sections[name])) else None
            for name, read in facts.items() if name in asked}


def detail_call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The feature call that returns a section's full answer, exactly as
    the section made it."""
    return {"tool": tool, "arguments": dict(arguments)}


# --- Refusals made before any work -------------------------------------------


def check_point(latitude: float, longitude: float) -> None:
    for name, value, limit in (("latitude", latitude, 90.0), ("longitude", longitude, 180.0)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f"{name} must be a number within -{limit:g}..{limit:g}; "
                             f"got {value!r}")


def check_radius(name: str, value: float, *, latitude: float, longitude: float,
                 resolution_m: float | None = None) -> None:
    """A radius round the point that `prepare_area` will stage at
    `resolution_m`, or a ValueError now. The bound alone is not enough:
    `prepare_area` measures its own circle polygon, which puts a 4,500 m
    circle a hair over 9,000 m across, and more at high latitude. So after
    the cheap bound this runs the very checks `prepare_area` refuses with
    before its first network call - `build_area`, and the pixel count at the
    resolution - which read no file and fetch nothing: what is refused here
    is exactly what staging would refuse. Call `check_point` first."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value) or not 0 < value < MAX_RADIUS_M:
        raise ValueError(f"{name} must be above 0 and under {MAX_RADIUS_M:,.0f} m "
                         f"(prepare_area stages at most {MAX_EXTENT_M:,.0f} m across); "
                         f"got {value!r}")
    # Imported here, as `planning.area.prepare_area` imports it: the staging
    # module is the one that fetches, and only its pixel rule is wanted.
    from ..staging import pixels_for_resolution

    try:
        area = build_area(center_latitude=latitude, center_longitude=longitude,
                          radius_m=float(value), resolution_m=resolution_m)
        if resolution_m is not None:
            pixels_for_resolution(area.extent_m + 2 * PLANNING_AREA_BUFFER_M, resolution_m)
    except ValueError as error:
        raise ValueError(f"{name} of {value:,} m cannot be staged here, so nothing was: "
                         f"prepare_area would refuse it - {error}") from error


#: A search radius under this is most likely miles or feet read as metres:
#: `radius_unit` defaults to "m" in every scenario taking one, as it does in
#: prepare_area, so half a mile asked as `radius=0.5` is half a metre.
#: Warned of, not refused: a caller may mean it, and the unit stays the one
#: the other tools read, so `radius=600` means the same metres everywhere.
SMALL_RADIUS_M = 50.0


def small_radius_warnings(radius_m: float, *, what: str) -> list[str]:
    """A top-level warning when a scenario's search radius is under
    SMALL_RADIUS_M, else none. `what` says what so small a circle is to that
    scenario - a lot or two rather than a neighbourhood - because the answer
    still reads as a search of the area: an empty hasty section would read
    as "no hasty lines here" when it is a circle a metre across. One helper,
    so every scenario taking a `radius_unit` says the same."""
    if radius_m >= SMALL_RADIUS_M:
        return []
    return [f"the search radius is {radius_m:g} m, {what}: radius is read in radius_unit, "
            "which is metres unless given - half a mile is radius=0.5, radius_unit=\"mi\""]


def check_limit(name: str, value, every: str) -> None:
    """A count a section keeps: 0 for every one, or more. A bool is refused
    although Python counts it an int - `True` is nobody's limit."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be 0 (every {every}) or more; got {value!r}")


def check_per_category(value) -> None:
    """`medical_per_category`, in the medical tool's own range, refused here
    rather than as a failed section once the grounds are staged."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError(f"medical_per_category must be between 1 and 10; got {value!r}")


def requested(include, names: tuple[str, ...]) -> tuple[str, ...]:
    """The sections asked for, in the scenario's own order. None is every
    one. A direct caller is not held to the `Literal` in the tool's schema,
    so an unknown name is refused here - a typo would otherwise quietly skip
    the section it meant."""
    if include is None:
        return names
    if isinstance(include, str):
        raise ValueError(f"include must be a list of section names, such as [{include!r}]")
    unknown = sorted({str(name) for name in include} - set(names))
    if unknown:
        raise ValueError(f"include names no section of this scenario: {unknown}; "
                         f"its sections are {list(names)}")
    if not include:
        raise ValueError(f"include is empty, so there would be nothing to do: leave it "
                         f"out for every section, or name some of {list(names)}")
    if set(include) == {EXPORT_SECTION}:
        others = [name for name in names if name != EXPORT_SECTION]
        raise ValueError(f"include names only {EXPORT_SECTION}, which draws what the other "
                         f"sections find, so there would be nothing to draw: name some of "
                         f"{others} beside it")
    return tuple(name for name in names if name in include)


#: Why a section helper sets these keywords itself, in the refusal's words.
_STAGED_HERE = ("the scenario stages this section's ground itself and passes its handle; "
                "change the ground with the scenario's own radius arguments")
_CENTRE_HERE = ("the scenario's own latitude and longitude are the centre, so every "
                "distance, bearing and minute is from the one point")
_ARGUMENT_HERE = ("the scenario sets it from one of its own arguments, and the section "
                  "helper always passes it")
_SET_HERE = ("the scenario exports the candidate set its own section generated; call "
             "caltopo_export directly for another set, or for some of its candidates")
_ASSIGNMENT_HERE = ("the scenario stages the assignment itself with prepare_terrain and "
                    "passes its handle; name another assignment with map_id and label")

#: Per section: the feature tool behind it, and every keyword the helper
#: always sets, which `overrides` may not: the helper spreads the overrides
#: after its own arguments, so one that got through would silently win - an
#: egress `mode` override would run walk-out twice and label one of them
#: carry-out. Held here, by the helper, rather than left to each scenario's
#: `scenario_sets`, which only rewords the refusal.
SECTION_TOOLS: dict[str, tuple[str, Callable, dict[str, str]]] = {
    "lz": ("lz_candidates", lz_feature.lz_candidates,
           {"handle": _STAGED_HERE, "center_latitude": _CENTRE_HERE,
            "center_longitude": _CENTRE_HERE, "limit": _ARGUMENT_HERE}),
    "egress": ("egress_routes", egress_feature.egress_routes,
               {"handle": _STAGED_HERE, "center_latitude": _CENTRE_HERE,
                "center_longitude": _CENTRE_HERE, "mode": _ARGUMENT_HERE}),
    "medical": ("nearest_medical_facilities", medical_feature.nearest_medical_facilities,
                {"latitude": _CENTRE_HERE, "longitude": _CENTRE_HERE,
                 "per_category": _ARGUMENT_HERE}),
    "parking": ("parking_candidates", parking_feature.parking_candidates,
                {"handle": _STAGED_HERE, "limit": _ARGUMENT_HERE}),
    "hasty": ("hasty_candidates", hasty_feature.hasty_candidates,
              {"handle": _STAGED_HERE, "limit": _ARGUMENT_HERE}),
    "travel_rings": ("travel_time_rings", rings_feature.travel_time_rings,
                     {"handle": _STAGED_HERE, "center_latitude": _CENTRE_HERE,
                      "center_longitude": _CENTRE_HERE, "minutes": _ARGUMENT_HERE}),
    # `include_area` is not here: the helper passes True as a default the
    # caller may change, so it goes BEFORE the overrides, which then win.
    "export": ("caltopo_export", candidates_component.caltopo_export,
               {"handle": _STAGED_HERE, "set_id": _SET_HERE, "candidate_ids": _SET_HERE}),
    # On one CalTopo assignment, staged by `stage_assignment`. None of the
    # three takes a keyword the helper does not set, so today every override
    # is refused; the entries hold the refusals' reasons, and a keyword one
    # of the tools gains later reaches it with no change here.
    "metrics": ("terrain_metrics", ground_component.terrain_metrics,
                {"handle": _ASSIGNMENT_HERE}),
    "coverage": ("coverage_for", coverage_feature.coverage_for,
                 {"handle": _ASSIGNMENT_HERE, "track_names": _ARGUMENT_HERE}),
    "features": ("identify_features", ground_component.identify_features,
                 {"handle": _ASSIGNMENT_HERE, "kinds": _ARGUMENT_HERE}),
}

#: Sections that call more than one feature, each part named by the tool it
#: calls, with the keywords the helper always sets in that call. Their
#: overrides are {tool: keyword arguments}, each mapping checked against its
#: own tool as a one-feature section's is. The urban section draws blocks
#: with populated_areas and groups them with urban_assignments, and both take
#: `max_distance_m` and `marker_color`: a flat mapping could not say which
#: call a keyword is meant for.
SECTION_PARTS: dict[str, dict[str, tuple[Callable, dict[str, str]]]] = {
    "urban": {
        "populated_areas": (urban_feature.populated_areas,
                            {"handle": _STAGED_HERE, "building_markers": _ARGUMENT_HERE}),
        "urban_assignments": (urban_feature.urban_assignments,
                              {"handle": _STAGED_HERE, "target_acres": _ARGUMENT_HERE,
                               "resource_type": _ARGUMENT_HERE,
                               "building_markers": _ARGUMENT_HERE}),
    },
}

#: Every feature function a section calls, by tool, as it is at import.
_FEATURE_FUNCTIONS: dict[str, Callable] = {
    **{tool: function for tool, function, _fixed in SECTION_TOOLS.values()},
    **{tool: function for parts in SECTION_PARTS.values()
       for tool, (function, _fixed) in parts.items()},
}

#: Each feature's keywords after `ctx`, by tool, read once from the real
#: functions at import: what `overrides` is checked against is the tool's
#: contract, not whatever a test has stood in for the function since.
FEATURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    tool: tuple(parameter for parameter in inspect.signature(function).parameters
                if parameter != "ctx")
    for tool, function in _FEATURE_FUNCTIONS.items()
}

#: What an override must be beyond the tool's schema: a finite number where
#: a number goes. The schema takes "nan" and "inf" as floats, and NaN
#: compares false with every limit, so `{"lz": {"max_slope_deg": "nan"}}`
#: would pass each range check and grade no ground at all - an "ok" section
#: with no landing zone, silence that reads as "none here". Refused instead,
#: where a section answering nothing could not be told from one finding
#: nothing.
_OVERRIDE_CONFIG = ConfigDict(allow_inf_nan=False)

#: Each feature keyword's type, by tool, as the tool's own schema reads it:
#: a pydantic adapter over the real function's annotation, read once at
#: import as FEATURE_KEYWORDS is. The MCP wrapper has the same annotations
#: (tests/test_tool_registration.py), so an override is converted and refused
#: exactly as the same value given to the tool over MCP would be - but for a
#: number that is not finite (`_OVERRIDE_CONFIG`).
FEATURE_TYPES: dict[str, dict[str, TypeAdapter]] = {
    tool: {keyword: TypeAdapter(hint, config=_OVERRIDE_CONFIG)
           for keyword, hint in get_type_hints(function).items()
           if keyword in FEATURE_KEYWORDS[tool]}
    for tool, function in _FEATURE_FUNCTIONS.items()
}

#: Each feature keyword's default, by tool, read once at import: what a
#: feature's own range checks (FEATURE_CHECKS) compare an override against,
#: such as the marginal slope a `max_slope_deg` override must not pass.
FEATURE_DEFAULTS: dict[str, dict[str, Any]] = {
    tool: {name: parameter.default
           for name, parameter in inspect.signature(function).parameters.items()
           if parameter.default is not inspect.Parameter.empty}
    for tool, function in _FEATURE_FUNCTIONS.items()
}

#: Keywords a scenario's own check reads in the domain's words, so a value
#: the tool's schema refuses is left as given for that check to refuse: the
#: corridor widths (`check_hasty_overrides`, which also refuses a misspelt
#: class the model would silently drop), the walking pace (`rings_ground`),
#: the label prefix (`check_export_overrides`) and a house-marker colour
#: (`check_urban_overrides`, only when markers are drawn - no feature reads a
#: colour otherwise). Text the schema reads as a value is still read so
#: first (`_read_values`): "0.5" is a pace, and refused as "not a number"
#: the refusal would ask for the very value it was given.
DOMAIN_CHECKED: dict[str, tuple[str, ...]] = {
    "hasty_candidates": ("corridor_widths_m",),
    "travel_time_rings": ("speed_factor",),
    "caltopo_export": ("label_prefix",),
    "populated_areas": ("marker_color",),
    "urban_assignments": ("marker_color",),
}

#: Per tool, the feature's own range checks - the ones it makes before it
#: reads any ground or asks any service - run on the overrides over the
#: tool's defaults, so a value it would refuse once the ground is staged is
#: refused before, in its own words: a zero `min_length_m`, a 1,000 km
#: hospital search, a slope limit above the marginal one, a negative grade.
#: The rings, the export and the urban tools are checked by the scenario's
#: own helpers (`rings_ground`, `check_export_overrides`,
#: `check_urban_overrides`), which need the scenario's own arguments.
FEATURE_CHECKS: dict[str, Callable[..., Any]] = {
    "hasty_candidates": hasty_params,
    "lz_candidates": check_lz_params,
    "egress_routes": check_egress_params,
    "nearest_medical_facilities": check_medical_params,
}

#: What a feature tool's schema wants, by pydantic's error type, in a
#: planner's words rather than pydantic's ("Input should be a valid number").
_WANTS = {"float_type": "a number", "float_parsing": "a number",
          "finite_number": "a finite number", "int_type": "a whole number",
          "int_parsing": "a whole number", "int_from_float": "a whole number",
          "bool_type": "true or false", "bool_parsing": "true or false",
          "string_type": "text", "list_type": "a list", "dict_type": "a mapping",
          "model_type": "a mapping"}


def check_overrides(scenario: str, overrides, sections: tuple[str, ...],
                    scenario_sets: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """`overrides` checked against the sections and their features, as
    {section: keyword arguments}, one entry per section (empty when none).

    `scenario_sets` names, per section, which of this scenario's arguments
    sets a keyword - {"lz": {"limit": "lz_limit"}} - so the refusal can say
    which argument to use instead; the keyword is refused without it too.
    Everything is refused before any work: an unknown section, a value that
    is not a mapping (None is no overrides), a keyword the scenario sets,
    a keyword the feature does not take, and a value the feature tool would
    refuse - by its schema or by its own range checks (`_read_values`). A
    value the tool's schema converts comes back converted, so the call and
    its `detail_call` carry what the tool reads: "6" is 6.0, "false" False.

    A section that calls more than one feature (SECTION_PARTS) takes
    {tool: keyword arguments} instead, each checked against its tool as
    above, and its entry holds only the tools given something: `{}` when
    none, so it reads as "no overrides" as any other section's does."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError(f"overrides must be a mapping of section name to keyword "
                         f"arguments, such as {{{sections[0]!r}: {{...}}}}; "
                         f"got {type(overrides).__name__}")
    unknown = sorted(set(overrides) - set(sections))
    if unknown:
        raise ValueError(f"overrides names no section of {scenario}: {unknown}; "
                         f"its sections are {list(sections)}")
    checked: dict[str, dict[str, Any]] = {}
    for name in sections:
        # None only: `or {}` would pass a [] or a 0 as "no overrides".
        given = overrides.get(name)
        given = {} if given is None else given
        sets = scenario_sets.get(name, {})
        if name in SECTION_PARTS:
            checked[name] = _check_parts(scenario, name, given, sets)
            continue
        tool, _function, helper_sets = SECTION_TOOLS[name]
        checked[name] = _check_keywords(scenario, f"overrides[{name!r}]", tool, given,
                                        helper_sets, sets)
    return checked


def _check_keywords(scenario: str, where: str, tool: str, given,
                    helper_sets: dict[str, str], sets: dict[str, str]) -> dict[str, Any]:
    """One feature call's overrides, checked against `tool`: a mapping, no
    keyword the helper or the scenario sets, none the tool does not take.
    `where` is how a refusal names them, such as `overrides['lz']`."""
    if not isinstance(given, dict):
        raise ValueError(f"{where} must be a mapping of {tool}'s keyword arguments; "
                         f"got {type(given).__name__}")
    fixed = dict(helper_sets)
    for keyword, argument in sets.items():
        fixed[keyword] = f"{scenario} sets it from its own argument {argument}; pass that instead"
    refused = sorted(set(given) & set(fixed))
    if refused:
        raise ValueError("; ".join(f"{where} cannot set {keyword!r}: {fixed[keyword]}"
                                   for keyword in refused))
    accepted = [keyword for keyword in FEATURE_KEYWORDS[tool] if keyword not in fixed]
    strangers = sorted(set(given) - set(accepted))
    if strangers and not accepted:
        # "it accepts []" would read as a fault in the refusal itself.
        raise ValueError(f"{where} names {strangers}, but {tool} takes no keyword "
                         f"{scenario} does not set itself, so there is nothing to override")
    if strangers:
        raise ValueError(f"{where} names {strangers}, which {tool} does not take here; "
                         f"it accepts {accepted}")
    return _read_values(where, tool, given)


def _read_values(where: str, tool: str, given: dict[str, Any]) -> dict[str, Any]:
    """`given`, a feature call's overrides, read as `tool` reads them over
    MCP, or a ValueError naming the keyword. Each value goes through the
    tool's own schema (FEATURE_TYPES): converted where it converts - "6" to
    6.0, "false" to False, a tuple to a list - and refused where it
    refuses: text for a number, a null slope, a hasty kind that is not one,
    a bare string where a list goes (read a letter at a time otherwise) -
    and, beyond the schema, a number that is not finite (`_OVERRIDE_CONFIG`).
    Then the feature's own range checks (FEATURE_CHECKS), on the overrides
    over the tool's defaults. The keywords a scenario's own check reads
    (DOMAIN_CHECKED) are left for that check: text the schema reads as a
    value is passed on read ("0.5" as 0.5), anything else as given.

    Without this an override reaches the plain function as given, which no
    schema has read: a bad value fails its section after the grounds are
    staged, often in Python's words rather than the tool's, and "false"
    reads as true - the scenario and the tool answering one request two
    ways."""
    left = DOMAIN_CHECKED.get(tool, ())
    types = FEATURE_TYPES[tool]
    read: dict[str, Any] = {}
    for keyword, value in given.items():
        if keyword not in types:
            read[keyword] = value
            continue
        if keyword in left:
            read[keyword] = _text_read(types[keyword], value)
            continue
        try:
            read[keyword] = _as_given(value, types[keyword].validate_python(value))
        except ValidationError as error:
            raise ValueError(f"{where}: {_type_refusal(keyword, tool, error)}") from error
    check = FEATURE_CHECKS.get(tool)
    if check is not None:
        values = {**FEATURE_DEFAULTS[tool],
                  **{keyword: value for keyword, value in read.items() if keyword not in left}}
        takes = inspect.signature(check).parameters
        try:
            check(**{name: value for name, value in values.items() if name in takes})
        except (TypeError, ValueError) as error:
            raise ValueError(f"{where}: {error}") from error
    return read


def _text_read(adapter: TypeAdapter, value):
    """A DOMAIN_CHECKED override: text as the tool's schema reads it, when
    it reads it, else `value` as given. Only text: the domain check reads
    a number, a mapping or a null in its own words - stricter than the
    schema at times (a flag is no pace) - but would refuse "0.5" as not a
    number. The corridor widths' text is no mapping, so stays as given."""
    if not isinstance(value, str):
        return value
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return value


def _as_given(given, read):
    """`read`, what the tool's schema made of `given`, but the caller's own
    number wherever it is the same number: a whole number where the tool
    takes a float is read as that float, and keeping it keeps the refusals
    that quote it ("pixel_m (3) must be ...") and the `detail_call` as the
    caller wrote them. Anything the schema changed - text to a number or a
    flag, a tuple to a list - is what the tool reads, and so what is kept."""
    if isinstance(read, float) and isinstance(given, int) and not isinstance(given, bool) \
            and read == given:
        return given
    if isinstance(read, dict) and isinstance(given, dict) and read.keys() == given.keys():
        return {key: _as_given(given[key], value) for key, value in read.items()}
    if isinstance(read, list) and isinstance(given, (list, tuple)) and len(read) == len(given):
        return [_as_given(old, new) for old, new in zip(given, read)]
    return read


def _type_refusal(keyword: str, tool: str, error: ValidationError) -> str:
    """The first thing `tool`'s schema refused in a value, in a planner's
    words: where in the value (`include[0]`), what it must be, and what it
    was."""
    first = error.errors()[0]
    place = keyword + "".join(f"[{part!r}]" for part in first["loc"])
    if first["type"] == "literal_error":
        wants = f"one of {first['ctx']['expected']}"
    else:
        wants = _WANTS.get(first["type"])
    said = (f"must be {wants}" if wants
            else str(first["msg"]).replace("Input should", "must", 1))
    if first["type"] == "list_type" and isinstance(first["input"], str):
        said += f", such as [{first['input']!r}]"
    if first["type"] == "finite_number":
        # The scenario's refusal, not the tool's: its schema takes "nan".
        return (f"{place} {said}; got {first['input']!r}, which no range check can be "
                "trusted to catch - NaN passes every one, and the section would answer "
                "as though nothing were there")
    return f"{place} {said}, as {tool} reads it; got {first['input']!r}"


def _check_parts(scenario: str, name: str, given, sets: dict[str, str]
                 ) -> dict[str, dict[str, Any]]:
    """The overrides of a section that calls several features (SECTION_PARTS):
    {tool: keyword arguments}, each tool one the section calls, checked as
    `_check_keywords` checks a one-feature section's. A flat mapping of
    keywords is refused with the shape it needs, because a keyword two of
    the tools take could be meant for either call. `sets` applies to each
    tool that takes the keyword: there it is refused as the scenario's to
    set, and given to a tool that never takes it, it is refused as any
    stranger is - with what that tool accepts, not as though the scenario
    set it there."""
    parts = SECTION_PARTS[name]
    last = list(parts)[-1]
    example = next(keyword for keyword in FEATURE_KEYWORDS[last]
                   if keyword not in parts[last][1] and keyword not in sets)
    shape = (f"the {name} section calls {list(parts)}, so give each its own keyword "
             f"arguments, such as {{{name!r}: {{{last!r}: {{{example!r}: ...}}}}}}")
    if not isinstance(given, dict):
        raise ValueError(f"overrides[{name!r}] must be a mapping by tool - {shape}; "
                         f"got {type(given).__name__}")
    strangers = sorted(set(given) - set(parts))
    if strangers:
        raise ValueError(f"overrides[{name!r}] names {strangers}, which is no tool it "
                         f"calls: {shape}")
    checked = {}
    for tool, (_function, helper_sets) in parts.items():
        # None only, as for a section: a [] or a 0 is refused as no mapping.
        if given.get(tool) is None:
            continue
        keywords = _check_keywords(scenario, f"overrides[{name!r}][{tool!r}]", tool,
                                   given[tool], helper_sets,
                                   {keyword: argument for keyword, argument in sets.items()
                                    if keyword in FEATURE_KEYWORDS[tool]})
        if keywords:
            checked[tool] = keywords
    return checked


def unused_overrides(overrides: dict[str, dict[str, Any]], asked: tuple[str, ...]
                     ) -> list[str]:
    """A top-level warning for each section given overrides (as
    `check_overrides` returned them) but left out of `include`. They were
    accepted - a section left out is not an error - but no call took them,
    and a caller who tuned that section would otherwise never hear it."""
    return [f"overrides[{name!r}] was not used: {name} was left out of include"
            for name, given in overrides.items() if given and name not in asked]


# --- Staging -----------------------------------------------------------------


@dataclass(frozen=True)
class Ground:
    """Ground a section needs: a circle round the scenario's point, at a
    resolution (None is `prepare_area`'s fixed raster). Each feature needs
    its own: a landing zone is graded at 2 m, an exit walked at 10 m."""

    radius_m: float
    resolution_m: float | None = None


def stage_grounds(ctx: ServerContext, grounds: dict[str, Ground], *, latitude: float,
                  longitude: float, map_id: str = "", public: bool = True
                  ) -> dict[str, dict[str, Any]]:
    """Stage each ground once, through the `prepare_area` component, and say
    how it went: {ground name: record}. Two names asking for the same ground
    share one staging and one handle. A ground that cannot be staged is a
    failed record, never a raise: it fails only the sections standing on it
    (`ground_handle`). The area outline `prepare_area` returns is left out;
    the record's `detail_call` returns it."""
    records: dict[str, dict[str, Any]] = {}
    done: dict[Ground, dict[str, Any]] = {}
    for name, ground in grounds.items():
        if ground in done:
            records[name] = dict(done[ground])
            continue
        arguments = {"center_latitude": latitude, "center_longitude": longitude,
                     "radius_m": ground.radius_m, "map_id": map_id, "public": public,
                     "resolution_m": ground.resolution_m}
        facts = {"radius_m": ground.radius_m, "resolution_m": ground.resolution_m}
        try:
            payload = staging_component.prepare_area(ctx, **arguments)
        except Exception as error:  # noqa: BLE001 - recorded; only its sections fail
            record = section(FAILED, error=error_text(ctx, error), handle=None, **facts,
                             detail_call=detail_call("prepare_area", arguments))
        else:
            record = section(OK, warnings=payload.get("warnings") or [],
                             handle=payload["handle"], **facts,
                             dem_resolution_m=payload.get("dem_resolution_m"),
                             staged=payload.get("staged"),
                             osm_available=payload.get("osm_available"),
                             detail_call=detail_call("prepare_area", arguments))
        done[ground] = record
        records[name] = record
    return records


def ground_handle(staging: dict[str, dict[str, Any]], name: str) -> str:
    """The handle of a staged ground, or StagingFailed saying why there is
    none - raised inside a section's `guard`, so that section alone fails."""
    record = staging[name]
    if record["status"] != OK:
        raise StagingFailed(f"the {name} ground ({_ground_words(record)}) could not be "
                            f"staged, so this section did not run - {record['error']}")
    return record["handle"]


def _ground_words(record: dict[str, Any]) -> str:
    """What a staging record staged, as a refusal names it: a radius round
    the point (`stage_grounds`), or an assignment on a map
    (`stage_assignment`), whose record has no radius."""
    if "radius_m" not in record:
        return f"assignment {record['label']!r} on map {record['map_id']}"
    resolution = record["resolution_m"]
    return (f"{record['radius_m']:,.0f} m radius"
            + (f" at {resolution:g} m" if resolution is not None else ""))


# --- Staging one CalTopo assignment -------------------------------------------


def check_assignment(map_id, label) -> None:
    """A map id and an assignment label, each a non-blank string, or a
    ValueError now. A blank one would cost a map read, or name a handle no
    staging can make, before failing every section with the same reason."""
    for name, value, example in (
            ("map_id", map_id, "the CalTopo map id, such as the ABC1234 in caltopo.com/m/ABC1234"),
            ("label", label, "the assignment as CalTopo labels it, such as 'AB' or 'BT 201'")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be {example}; got {value!r}")


def stage_assignment(ctx: ServerContext, *, map_id: str, label: str,
                     public: bool = True) -> dict[str, Any]:
    """Stage one CalTopo assignment through the `prepare_terrain` component
    and say how it went, as `stage_grounds` does for an area: `status`,
    `error`, `handle` (None when it could not be staged), `map_id` and
    `label`, what prepare_terrain said - `staged` (False when the ground on
    disk was already the assignment as it stands), its `reason` and the
    boundary `fingerprint` - its warnings, and the call as `detail_call`.
    Never raises: a map that cannot be read, or a label that names no
    assignment or several, is a failed record, and `ground_handle` then
    fails every section standing on it with that reason."""
    arguments = {"map_id": map_id, "label": label, "public": public}
    facts = {"map_id": map_id, "label": label}
    try:
        payload = staging_component.prepare_terrain(ctx, **arguments)
    except Exception as error:  # noqa: BLE001 - recorded; its sections fail
        return section(FAILED, error=error_text(ctx, error), handle=None, **facts,
                       detail_call=detail_call("prepare_terrain", arguments))
    return section(OK, warnings=payload.get("warnings") or [], handle=payload["handle"],
                   **facts, staged=payload.get("staged"), reason=payload.get("reason"),
                   fingerprint=payload.get("fingerprint"),
                   detail_call=detail_call("prepare_terrain", arguments))


def staleness_warnings(checks: dict[str, dict[str, Any]]) -> list[str]:
    """What the `staleness` blocks of the sections on one assignment handle
    (`{section: block}`, in the order the calls were made) say a reader must
    not miss, as at most one warning naming every section that answered.

    Every section is computed from the same staged ground, so a check says
    whether THAT ground is still the assignment, and what it says holds for
    every section, not only the one whose call made it: a boundary found
    moved at the last check puts all of their numbers on the OLD polygon,
    and one found current at the last check confirms all of them. The latest
    check decides, as it does for the scenario's top-level `staleness`, so
    the two never contradict each other. A boundary found moved at an
    earlier check is still said when the latest could not check it (nothing
    since has shown it back), and when the latest found it as staged again
    the warning says it may be being edited. A boundary that could not be
    checked is a different fact from one checked and current, and is said
    as such. The boundary was current when `prepare_terrain` ran - it
    re-stages one that moved - so a move here happened during the call."""
    if not checks:
        return []
    answered = ", ".join(checks)
    blocks = list(checks.values())
    latest = blocks[-1]
    moved = any(block.get("current") is False for block in blocks)
    if moved and latest.get("current") is not True:
        return [f"the assignment boundary has changed since it was staged: every section here "
                f"({answered}) was computed on the staged ground, so its numbers describe the "
                "OLD polygon - call again, and prepare_terrain re-stages the ground for the "
                "boundary as it now stands, before acting on them"]
    if latest.get("checked") is False:
        reason = latest.get("reason") or "no reason given"
        earlier = any(block.get("current") is True for block in blocks[:-1])
        return [f"the assignment boundary could not be checked against the live map at the "
                f"latest check ({reason})"
                + ("; an earlier check in this call found it as staged" if earlier else "")
                + f": every section here ({answered}) describes the ground as it was staged, "
                  "which may not be the assignment as it now stands"]
    if moved:
        return [f"a check during this call found the assignment boundary changed since it was "
                f"staged, and the latest check found it as staged again, so it may be being "
                f"edited: every section here ({answered}) describes the boundary as it was "
                "staged - call again before acting on them"]
    return []


# --- Landing zones -----------------------------------------------------------


#: What a landing zone keeps in a section: everything but the patch outline.
LZ_FIELDS = ("id", "grade", "touchdown", "clear_diameter_m", "slope_mean_deg",
             "slope_max_deg", "canopy_fraction", "distance_m", "bearing_deg",
             "powerline", "road", "parking_lot", "chips", "chip_path", "warnings")
#: The `lz_candidates` assumptions that say what was and was NOT checked.
LZ_CHECKS = ("structures", "trees", "water_sources_checked", "powerline_kinds_checked",
             "road_class")
#: A landing zone's marker colour, by grade.
LZ_COLOR = {"preferred": "#00C853", "minimal": "#FFD700", "marginal": "#FF8000"}
LZ_SYMBOL = "point"


def lz_section(ctx: ServerContext, handle: str, *, latitude: float, longitude: float,
               limit: int, overrides: dict[str, Any], point_name: str = "the point"
               ) -> Outcome:
    """`lz_candidates` from the point, compact: each zone without its patch
    outline, the grade counts, and what was not checked. `point_name` is
    what the markers call the point distances are measured from."""
    arguments = {"handle": handle, "center_latitude": latitude,
                 "center_longitude": longitude, "limit": limit, **overrides}
    payload = call_feature(ctx, "lz_candidates", lz_feature.lz_candidates, arguments)
    candidates = [{key: candidate.get(key) for key in LZ_FIELDS}
                  for candidate in payload["candidates"]]
    summary = payload.get("summary") or {}
    assumptions = payload.get("assumptions") or {}
    checks = {key: assumptions.get(key) for key in LZ_CHECKS}
    return Outcome(
        section(OK, warnings=payload.get("warnings") or [], handle=handle,
                candidates=candidates, grades=summary.get("grades"),
                total_before_limit=summary.get("total_before_limit"),
                truncated_to=summary.get("truncated_to"),
                resolution_m=payload.get("resolution_m"), checks=checks,
                detail_call=detail_call("lz_candidates", arguments)),
        [lz_marker(candidate, checks, point_name=point_name)
         for candidate in payload["candidates"]])


def lz_marker(candidate: dict, checks: dict[str, Any] | None = None, *,
              point_name: str = "the point") -> dict:
    """A landing zone as a CalTopo Marker at its touchdown, coloured by
    grade, its description the facts a pilot's planner reads off it -
    and, from `checks`, what was NOT checked, because the map is what the
    field team sees and a marker silent about trees reads as clear of them.
    "point" is a symbol CalTopo draws (`planning.medical` checked it in an
    import; "heliport" is not one and draws as an empty circle)."""
    checks = checks or {}
    grade = candidate.get("grade") or "?"
    lines = [f"{grade.capitalize()} landing zone, {candidate['clear_diameter_m']:.0f} m "
             "clear circle."]
    if candidate.get("slope_mean_deg") is not None:
        lines.append(f"Slope {candidate['slope_mean_deg']:g} deg mean, "
                     f"{candidate['slope_max_deg']:g} deg max.")
    if candidate.get("canopy_fraction") is not None:
        lines.append(f"Canopy {candidate['canopy_fraction'] * 100:.0f} %.")
    bearing = candidate.get("bearing_deg")
    lines.append(f"{candidate['distance_m']:.0f} m"
                 + (f" at {bearing:.0f} deg" if bearing is not None else "")
                 + f" from {point_name}.")
    powerline = candidate.get("powerline")
    kinds = checks.get("powerline_kinds_checked") or []
    if isinstance(powerline, dict):
        lines.append(f"Power line {powerline['distance_m']:.0f} m away"
                     + (f" ({powerline['name']})." if powerline.get("name") else "."))
    elif powerline == "unknown":
        lines.append("Power lines unknown here.")
    elif kinds and "minor_line" not in kinds:
        # Ground staged before distribution lines were asked for: only a
        # transmission line would have been found.
        lines.append("No mapped transmission line near; distribution lines were not "
                     "checked on this ground (an unmapped wire is for the imagery).")
    else:
        lines.append("No mapped power line near (an unmapped wire is for the imagery).")
    road = candidate.get("road")
    if isinstance(road, dict):
        what = road.get("name") or road.get("highway") or road.get("route_type") or "a road"
        lines.append(f"{'ON' if road.get('on_road') else 'Beside'} {what}, "
                     f"{road['distance_m']:.0f} m from the touchdown.")
    elif road == "unknown":
        lines.append("Roads unknown here.")
    not_checked = [what for what in ("trees", "structures") if checks.get(what) == "not checked"]
    if checks.get("water_sources_checked") == []:
        not_checked.append("open water")
    if not_checked:
        lines.append(f"Not checked here: {', '.join(not_checked)}.")
    lines.extend(candidate.get("warnings") or [])
    lines.append("Graded from elevation and imagery rasters, not a site survey. "
                 "The pilot decides.")
    return {"type": "Feature", "geometry": candidate["touchdown"],
            "properties": {"class": "Marker", "title": f"{candidate['id']} ({grade})",
                           "description": "\n".join(lines),
                           "marker-symbol": LZ_SYMBOL,
                           "marker-color": LZ_COLOR.get(grade, LZ_COLOR["marginal"]),
                           "marker-size": 1}}


def lz_brief(candidate: dict) -> dict:
    return {key: candidate.get(key) for key in
            ("id", "grade", "distance_m", "bearing_deg", "clear_diameter_m")}


def lz_facts(lz: dict) -> dict[str, Any]:
    """The landing zones' top-line facts, from an answered `lz_section`:
    the nearest zone of any grade, the nearest preferred one, and - when the
    limit left every preferred zone out - how many and the call listing them.

    lz_candidates orders by distance before grade, so the first is the
    nearest of any grade, the limit kept the nearest, and a preferred zone
    among them is the nearest preferred one anywhere. On most ground the
    nearest few are minimal or marginal, so the nearest preferred one is
    often past a limit of 5: rather than a bare null a reader could take for
    "none near", say how many there are and the call that lists them. Not
    made here: it cuts an imagery chip for every zone, a network fetch each,
    and the feature has no way to grade without them."""
    candidates = lz["candidates"]
    preferred = [candidate for candidate in candidates if candidate["grade"] == "preferred"]
    beyond = max(0, ((lz.get("grades") or {}).get("preferred") or 0) - len(preferred))
    find = None
    if not preferred and beyond:
        find = detail_call("lz_candidates", {**lz["detail_call"]["arguments"], "limit": 0})
    return {"nearest_any": lz_brief(candidates[0]) if candidates else None,
            "nearest_preferred": lz_brief(preferred[0]) if preferred else None,
            "preferred_beyond_limit": beyond,
            "preferred_detail_call": find}


# --- Egress ------------------------------------------------------------------


#: A route's line colour in the export, by mode.
EGRESS_STROKE = {"carry_out": "#8000FF", "walk_out": "#0080FF"}
EGRESS_LABEL = {"carry_out": "Carry-out", "walk_out": "Walk-out"}
#: `egress_routes` keywords whose default differs by mode
#: (`planning.egress_graph.MODES`).
EGRESS_PER_MODE = ("on_way_factor", "off_way_factor", "max_grade")


def check_egress_overrides(modes: tuple[str, ...], overrides: dict[str, Any]) -> None:
    """Refuse, before any work, an egress override that would reach more
    than one mode. Every call the section makes gets the same overrides, but
    each of these keywords has a different default per mode: a `max_grade`
    meant to tighten the litter's 40 % would put a grade limit on a walk-out
    that has none, and a litter's off-way factor would make every walk-out
    minute absurd. One value for both is almost always a mistake, so the
    caller names the one mode it is for."""
    given = sorted(set(overrides) & set(EGRESS_PER_MODE))
    if given and len(modes) > 1:
        defaults = "; ".join(
            f"{mode}: " + ", ".join(f"{key} {EGRESS_DEFAULTS[mode][key]}"
                                    for key in EGRESS_PER_MODE)
            for mode in modes)
        raise ValueError(f"overrides['egress'] sets {given}, whose defaults differ by mode "
                         f"({defaults}), so one value cannot serve {list(modes)}: name the "
                         f"one mode it is for in egress_modes, and call again for the other")


def egress_section(ctx: ServerContext, handle: str, *, latitude: float, longitude: float,
                   modes: tuple[str, ...], overrides: dict[str, Any],
                   restage: Callable[[float, float], str] | None = None) -> Outcome:
    """`egress_routes` from the point once per mode, compact: each mode's
    two exits without their routes. A mode that raises is recorded under
    `modes` and fails the section, but the modes that answered stay - the
    walk-out exit is still worth having when the carry-out could not be
    worked out - and `overall_status` reads such a section as partial.
    Each reachable exit is drawn (`route_line`), handed its mode's
    warnings so that its line can say what they say of the vehicle end:
    gates, or road class, access and gates, unknown on this ground. An exit
    in a lot with no road beside it carries nothing that says so; only the
    answer's warnings do. The mode's other warnings stay in the section. An
    exit at the point itself has no route to draw, so what its line would
    say of the vehicle end goes on the scenario's point (`point_notes`).
    Call `check_egress_overrides` first.

    `restage(latitude, longitude)` stages ground of the same size centred
    there and returns its handle. Given, a mode whose answer says its
    trail runs off the ground (`trail_beyond`) while no exit is reached
    without leaving a trail is asked again on ground re-centred where that
    answer says - staged once, for every such mode - and the second answer
    is kept when it is more on the trails (`_more_on_trail`). The mode then
    says so under `recentred`, and its `detail_call` names the new handle."""
    payloads: dict[str, dict[str, Any]] = {}
    calls: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for mode in modes:
        arguments = {"handle": handle, "center_latitude": latitude,
                     "center_longitude": longitude, "mode": mode, **overrides}
        calls[mode] = arguments
        try:
            payloads[mode] = egress_feature.egress_routes(ctx, **arguments)
        except Exception as error:  # noqa: BLE001 - one mode must not sink the other
            errors[mode] = error_text(ctx, error)
    recentred: dict[str, dict[str, Any]] = {}
    notes: dict[str, list[str]] = {}
    beyond_modes = [mode for mode in modes
                    if mode in payloads and payloads[mode].get("trail_beyond")]
    if restage is not None and beyond_modes:
        centre = payloads[beyond_modes[0]]["trail_beyond"]["recentre"]
        try:
            new_handle = restage(centre["latitude"], centre["longitude"])
        except Exception as error:  # noqa: BLE001 - the first answers stand
            for mode in beyond_modes:
                notes[mode] = [RECENTRE_FAILED.format(error=error_text(ctx, error))]
        else:
            for mode in beyond_modes:
                arguments = {**calls[mode], "handle": new_handle}
                try:
                    second = egress_feature.egress_routes(ctx, **arguments)
                except Exception as error:  # noqa: BLE001 - the first answer stands
                    notes[mode] = [RECENTRE_FAILED.format(error=error_text(ctx, error))]
                    continue
                if _more_on_trail(second, payloads[mode]):
                    beyond = payloads[mode]["trail_beyond"]
                    payloads[mode], calls[mode] = second, arguments
                    recentred[mode] = {"handle": new_handle, "center": dict(centre),
                                       "because": _trail_beyond_words(beyond)}
                    notes[mode] = [RECENTRED_NOTE.format(
                        latitude=centre["latitude"], longitude=centre["longitude"],
                        because=recentred[mode]["because"])]
                else:
                    notes[mode] = [RECENTRE_NO_BETTER.format(
                        latitude=centre["latitude"], longitude=centre["longitude"])]

    modes_out: dict[str, dict[str, Any]] = {}
    warnings_by_mode: dict[str, list[str]] = {}
    features: list[dict] = []
    point_notes: list[str] = []
    first_error = None
    for mode in modes:
        if mode in errors:
            first_error = first_error or errors[mode]
            modes_out[mode] = {"status": FAILED, "error": errors[mode],
                               "exits": [],
                               "detail_call": detail_call("egress_routes", calls[mode])}
            continue
        payload = payloads[mode]
        warnings_by_mode[mode] = [*notes.get(mode, []), *(payload.get("warnings") or [])]
        modes_out[mode] = {"status": OK, "error": None,
                           "exits": [compact_exit(exit_) for exit_ in payload["exits"]],
                           "trail_beyond": payload.get("trail_beyond"),
                           "recentred": recentred.get(mode),
                           "detail_call": detail_call("egress_routes", calls[mode])}
        for exit_ in payload["exits"]:
            if not exit_.get("reachable"):
                continue
            if (exit_.get("route") or {}).get("type") == "LineString":
                features.append(route_line(mode, exit_,
                                           ground_warnings=warnings_by_mode[mode]))
            else:
                # The exit is the point's own cell: no line to carry its
                # vehicle end, and a map silent about a gated private road
                # at the point reads as open.
                point_notes.append(
                    f"{EGRESS_LABEL.get(mode, mode)} {_exit_kind(exit_)} exit is here: "
                    + " ".join(_vehicle_end(
                        exit_, _vehicle_end_unknowns(warnings_by_mode[mode]))))
    return Outcome(section(FAILED if first_error else OK, error=first_error,
                           warnings=merged_warnings(warnings_by_mode, modes), handle=handle,
                           modes=modes_out),
                   features, point_notes=point_notes)


#: Said on a mode answered again on re-centred ground, and why.
RECENTRED_NOTE = ("the exits are from ground re-centred on {latitude:.5f}, {longitude:.5f}, "
                  "because {because}")
RECENTRE_NO_BETTER = ("ground re-centred on {latitude:.5f}, {longitude:.5f} was asked too, and "
                      "kept no more to the trails, so these exits are the first ground's")
RECENTRE_FAILED = ("the trail runs off the ground, but ground re-centred along it could not "
                   "be asked ({error}), so these exits are the first ground's")


def _trail_beyond_words(beyond: dict[str, Any]) -> str:
    """Why a mode was asked again, in a few words: its trail and its problems."""
    way = beyond.get("way_name") or "the trail"
    return f"{way} ran off the first ground and there {'; '.join(beyond['problems'])}"


def _on_trail_badness(payload: dict[str, Any]) -> list[float]:
    """Per kind, how far its exit leaves a trail after joining one (inf when
    unreachable), smallest first: a medevac wants one exit on the trail."""
    return sorted(exit_.get("off_trail_after_joining_m", math.inf) if exit_.get("reachable")
                  else math.inf for exit_ in payload["exits"])


def _more_on_trail(second: dict[str, Any], first: dict[str, Any]) -> bool:
    """Whether `second` keeps to the trails better than `first`: its best
    exit leaves a trail for less, then its other exit does."""
    return _on_trail_badness(second) < _on_trail_badness(first)


def compact_exit(exit_: dict) -> dict:
    """One exit without its route: the facts a planner compares two by, and
    the vehicle end's - a gate on the exit road, a private road, the lot and
    the road it opens onto, which names the way in when the exit lies in a
    lot with no road beside it."""
    if not exit_.get("reachable"):
        return {"kind": exit_["kind"], "reachable": False, "reason": exit_.get("reason")}
    road = exit_.get("road") if isinstance(exit_.get("road"), dict) else {}
    return {"kind": exit_["kind"], "reachable": True, "minutes": exit_["minutes"],
            "distance_m": exit_.get("distance_m"), "gain_m": exit_.get("gain_m"),
            "loss_m": exit_.get("loss_m"), "exit_point": exit_.get("exit_point"),
            "road_name": road.get("name"), "road_access": road.get("access"),
            "gate_on_road": road.get("gate_on_road"),
            "parking_lot": exit_.get("parking_lot"),
            "gates_crossed": len(exit_.get("gates_crossed") or []),
            "off_trail_m": exit_.get("off_trail_m"),
            "on_trail_share": exit_.get("on_trail_share"),
            "off_trail_after_joining_m": exit_.get("off_trail_after_joining_m"),
            "steep_pitch_m": exit_.get("steep_pitch_m"),
            "litter_start": exit_.get("litter_start"),
            "warnings": list(exit_.get("warnings") or [])}


def merged_warnings(by_mode: dict[str, list[str]], modes: tuple[str, ...]) -> list[str]:
    """The warnings of the modes that answered (`by_mode`), once each, in
    order. One every mode asked for (`modes`) gave is said plainly; any
    other is prefixed with the names of the modes that gave it - so a
    walk-out-only warning about vegetation is not read as the litter's,
    including when the litter's call raised and said nothing at all."""
    order: list[str] = []
    for warnings in by_mode.values():
        order.extend(warning for warning in warnings if warning not in order)
    merged = []
    for warning in order:
        gave = [mode for mode, warnings in by_mode.items() if warning in warnings]
        merged.append(warning if set(gave) >= set(modes)
                      else f"{', '.join(gave)}: {warning}")
    return merged


#: What the summary keeps of a mode's quickest exit beyond its kind,
#: minutes and distance: the vehicle end, exactly as `compact_exit` gives
#: it, so a null reads the same in the summary as in the section.
EXIT_VEHICLE_END = ("road_name", "road_access", "gate_on_road", "parking_lot")


def egress_facts(egress: dict) -> dict[str, Any]:
    """From an `egress_section` that holds an answer, whole or in part: per
    mode, its quickest reachable exit (kind, minutes, distance and its
    vehicle end, EXIT_VEHICLE_END), the string "none reachable" when the
    mode answered and neither exit could be reached, or None when that mode
    failed and another answered. Quickest is egress_routes' own minutes,
    within one mode; nothing is chosen between modes. The vehicle end is
    kept because the summary is read first, beside no section: a quickest
    exit that gave only its minutes would read as a road an ambulance can
    use, gated or private as it may be. It flags them where the ground can
    tell: its nulls are the section's - `road_access` null is untagged or
    unknown, never public, and `gate_on_road` null is never false.
    `gates_crossed` is not kept: on ground with no gate list it is 0, a
    number that reads as none, beside a warning the summary does not
    carry. `summarise` reads it with egress in `in_part`, so a section none
    of whose modes answered is None, as any section that did not answer is."""
    return {mode: _quickest_exit(entry) for mode, entry in (egress.get("modes") or {}).items()}


def _quickest_exit(mode: dict) -> dict | str | None:
    """One mode's quickest reachable exit; "none reachable" when the mode
    answered and neither exit could be reached; None when it failed."""
    if mode["status"] != OK:
        return None
    reachable = [exit_ for exit_ in mode["exits"] if exit_["reachable"]]
    if not reachable:
        return "none reachable"
    quickest = min(reachable, key=lambda exit_: exit_["minutes"])
    return {"kind": quickest["kind"], "minutes": quickest["minutes"],
            "distance_m": quickest["distance_m"], "off_trail_m": quickest.get("off_trail_m"),
            **({"litter_start": quickest["litter_start"]} if quickest.get("litter_start")
               else {}),
            **{key: quickest[key] for key in EXIT_VEHICLE_END}}


#: ROAD_FACTS_UNKNOWN_WARNING either side of the query version it names,
#: to know the warning in an answer whatever version it gives.
_ROAD_FACTS_HEAD, _, _ROAD_FACTS_TAIL = ROAD_FACTS_UNKNOWN_WARNING.partition("{version}")


def _vehicle_end_unknowns(ground_warnings: Iterable[str]) -> set[str]:
    """What an `egress_routes` answer's own warnings say its staged ground
    cannot tell at the vehicle end: "road_facts" when its OpenStreetMap was
    read before road class, access and gates were asked for, "gates" when
    the file only lacks a gate list. Said on the answer, never on an exit:
    `gate_on_road` is null on such ground when the exit has a road, but an
    exit in a lot with no road beside it has nothing to tell by. Known by
    the egress module's own warnings, as `_lots_unknown` knows parking's."""
    unknown = set()
    for warning in ground_warnings:
        if warning == GATES_UNKNOWN_WARNING:
            unknown.add("gates")
        elif warning.startswith(_ROAD_FACTS_HEAD) and warning.endswith(_ROAD_FACTS_TAIL):
            unknown.add("road_facts")
    return unknown


def _vehicle_end(exit_: dict, unknown: set[str] | None) -> list[str]:
    """What a route line says of where the vehicle meets the team: the
    road (an unnamed one by its kind), or the lot and the road it opens
    onto; the road's access, tagged or not; and gates, always - crossed on
    the way, on the stretch of road the exit lies on or none mapped there,
    what was not checked, or unknown. `unknown` is `_vehicle_end_unknowns`
    of the answer's own warnings, or None when the caller did not give
    them: a lot exit with no road beside it then cannot tell whether gates
    were read at all, and says they are unknown.

    The line is all a field team sees in CalTopo, and one silent about a
    gate or a private road reads as open, as a landing-zone marker silent
    about trees reads as clear (`lz_marker`): an ambulance sent to a gated
    private road loses the minutes the exit was picked for. So an unknown
    is said as one, and "0 gates" never is: on ground with no gate list
    there were none to count. A clear is said only of what was looked at:
    `gate_on_road` is one OpenStreetMap way's, the one the exit lies on -
    OpenStreetMap ends a way at junctions and tag changes, so a named road
    is often several - and only within the ground read. "No gate" is said
    of that stretch, and the road in beyond it is said to be unchecked."""
    road = exit_.get("road") if isinstance(exit_.get("road"), dict) else None
    lot = exit_.get("parking_lot")
    told = unknown is not None
    unknown = unknown or set()
    lines = []
    the_road = None
    if road:
        name = road.get("name")
        kind = "track" if road.get("route_type") == "TRACK" else "road"
        # Many tracks carry no name: the line still says what the vehicle
        # drives to, as `lz_marker` names a road by its class.
        tag = f" (OpenStreetMap highway={road['highway']})" if road.get("highway") else ""
        lines.append(f"Exit on {name}." if name else f"Exit on an unnamed {kind}{tag}.")
        the_road = name or f"the unnamed {kind}"
    if isinstance(lot, dict):
        lines.append(f"Exit in parking lot {lot.get('id')}, off {lot['off']}."
                     if lot.get("off") else
                     f"Exit in parking lot {lot.get('id')}; no named road or trail lies near "
                     "it, so the road in is not named.")
    elif lot == "unknown":
        lines.append("Parking lots unknown on this ground (its OpenStreetMap was read before "
                     "lots were asked for), never none: the exit may lie in one.")
    if road and "road_facts" not in unknown:
        # Untagged is said too: a private driveway is often mapped with no
        # access tag, and a line silent about access reads as public.
        lines.append(f"Access on {the_road}: {road['access']}, as OpenStreetMap tags it."
                     if road.get("access") else
                     f"Access on {the_road}: not tagged in the OpenStreetMap read for this "
                     "plan, which does not make it public.")
    crossed = len(exit_.get("gates_crossed") or [])
    if crossed:
        lines.append(f"{crossed} gate{'s' if crossed > 1 else ''} on the way.")
    # The way a vehicle comes in: the exit's road, or - for an exit in a
    # lot with no road beside it - a road no gate was looked for on.
    road_in = (the_road if road else
               "the road into the lot" if isinstance(lot, dict) else "the road in")
    if "road_facts" in unknown:
        lines.append("Road class, access and gates unknown on this ground (its "
                     "OpenStreetMap was read before they were asked for), never none: "
                     f"{road_in} may be private, gated or a motorway.")
    elif ("gates" in unknown or (road and road.get("gate_on_road") is None)
          # No road to tell by, no gate crossed, and no warnings to say
          # whether gates were read: unknown, never "no gate".
          or (not told and not road and not crossed)):
        why = (" (the OpenStreetMap read for this plan has no gate list)"
               if "gates" in unknown else "")
        lines.append(f"Gates unknown on this ground{why}, never none: the way out and "
                     f"{road_in} may be gated.")
    elif road:
        lines.append(f"Gate mapped somewhere on the stretch of {the_road} the exit lies on: "
                     "a vehicle may need the combination or the ranger to reach the exit."
                     if road.get("gate_on_road") else
                     f"No gate mapped on {'' if crossed else 'the way out or on '}the stretch "
                     f"of {the_road} the exit lies on.")
        lines.append("The road in beyond that stretch, and beyond the ground read for this "
                     "plan, is not checked for gates.")
    else:
        if not crossed:
            lines.append("No gate mapped on the way out.")
        lines.append(f"Gates on {road_in} are not checked.")
    return lines


def _exit_kind(exit_: dict) -> str:
    """An exit's kind as a line says it: 2WD, 4x4."""
    return exit_["kind"].upper() if exit_["kind"] == "2wd" else exit_["kind"]


#: Said on every route line with any ground off a mapped way, unless the
#: exit's own warning (over half off trail) already says it.
OFF_TRAIL_LINE = "{off_m:,.0f} m of this route is off any mapped trail, track or road."
OFF_TRAIL_HEAD = "of this route - more than half of it - is off any"


def route_line(mode: str, exit_: dict, *, ground_warnings: Iterable[str] | None = None
               ) -> dict:
    """A reachable exit's route as a CalTopo line Shape. Its description is
    what the field team reads of it: the minutes, distance and climb, the
    vehicle end (`_vehicle_end`), the exit's own warnings, and that it is
    no surveyed route. `ground_warnings` are the `egress_routes` answer's
    own, which alone say that gates, or road class, access and gates, are
    unknown for an exit in a lot with no road beside it. Left out, such an
    exit's gates are said to be unknown: never "no gate" on a guess."""
    label = EGRESS_LABEL.get(mode, mode)
    kind = _exit_kind(exit_)
    lines = [f"{label} to a {kind} exit: {exit_['minutes']:.0f} min, "
             f"{exit_.get('distance_m') or 0:,.0f} m, "
             f"+{exit_.get('gain_m') or 0:.0f} m / -{exit_.get('loss_m') or 0:.0f} m."]
    lines.extend(_vehicle_end(exit_, None if ground_warnings is None
                              else _vehicle_end_unknowns(ground_warnings)))
    warnings = list(exit_.get("warnings") or [])
    off_m = exit_.get("off_trail_m") or 0
    if off_m >= 1 and not any(OFF_TRAIL_HEAD in warning for warning in warnings):
        lines.append(OFF_TRAIL_LINE.format(off_m=off_m))
    lines.extend(warnings)
    lines.append("Walked over the elevation model and OpenStreetMap, not a surveyed "
                 "route; read the minutes as a rank. The team leader decides.")
    return {"type": "Feature", "geometry": exit_["route"],
            "properties": {"class": "Shape",
                           "title": f"{label} {kind} {exit_['minutes']:.0f} min",
                           "description": "\n".join(lines),
                           "stroke": EGRESS_STROKE.get(mode, EGRESS_STROKE["walk_out"]),
                           "stroke-width": 3, "stroke-opacity": 0.9}}


# --- Medical -----------------------------------------------------------------


def medical_section(ctx: ServerContext, *, latitude: float, longitude: float,
                    per_category: int, overrides: dict[str, Any]) -> Outcome:
    """`nearest_medical_facilities` at the point, compact: per category the
    hospitals by road time (by straight-line distance where no road time is
    known, as the feature's warnings say), and the feature's Markdown table
    verbatim."""
    arguments = {"latitude": latitude, "longitude": longitude,
                 "per_category": per_category, **overrides}
    payload = call_feature(ctx, "nearest_medical_facilities",
                           medical_feature.nearest_medical_facilities, arguments)
    categories: dict[str, list[dict]] = {category: [] for category in CATEGORIES}
    for row in payload["rows"]:
        categories.setdefault(row["category"], []).append({
            "rank": row["rank"], "name": row["name"], "trauma": row["trauma"]["label"],
            "helipad": row["helipad"], "road_minutes": row["road_minutes"],
            "air_minutes": row["air_minutes"], "road_distance_m": row["road_distance_m"],
            "phone": row["phone"], "latitude": row["latitude"],
            "longitude": row["longitude"]})
    point = payload.get("point") or {}
    markers = payload["markers"]["features"]
    # The hospitals only: the scenario draws its own point. But what the
    # medical tool's own point says - how the hospitals were ranked (by road
    # time, or by straight-line distance with the road router down, and
    # nothing when no hospital is drawn), and where the road route really
    # starts when that is over 200 m away - goes onto the scenario's point,
    # or a team reading the map in CalTopo takes every road minute as
    # starting at the patient, or a "#1" as the quickest by road.
    origin = next((feature for feature in markers
                   if feature["properties"].get("role") == "origin"), None)
    notes = (origin["properties"].get("description") or "").splitlines() if origin else []
    return Outcome(
        section(OK, warnings=payload.get("warnings") or [], in_us=point.get("in_us"),
                road_start=payload.get("road_start"), categories=categories,
                table=payload["table"],
                detail_call=detail_call("nearest_medical_facilities", arguments)),
        [feature for feature in markers if feature["properties"].get("role") == "hospital"],
        point_notes=[note for note in notes if note])


def medical_facts(medical: dict) -> dict[str, Any]:
    """The nearest trauma center, from an answered `medical_section` - the
    feature's own first row - or None when it found none. `ranked_by` says
    how it is the nearest: "road_time", or "straight_line" when it has no
    road time. The feature puts every routed hospital before any unrouted
    one, so a first row with no road time means none of the trauma centers
    it chose has one - the road router could not be reached, or found no
    road to any - and the category is in straight-line order. A null
    `road_minutes` alone reads as a road time not reported, of a hospital
    still chosen by road; across a bay or a ridge the nearest as the crow
    flies can be the slowest by road, and the summary carries no warning to
    say so."""
    rows = medical["categories"].get("trauma_center") or []
    first = rows[0] if rows else None
    return {"nearest_trauma_center": None if first is None else {
        "name": first["name"], "trauma": first["trauma"], "helipad": first["helipad"],
        "road_minutes": first["road_minutes"], "air_minutes": first["air_minutes"],
        "ranked_by": "straight_line" if first["road_minutes"] is None else "road_time"}}


# --- Parking -----------------------------------------------------------------


#: Grey, not black: `export_set` draws a search area's centre as a black
#: "point", and a scenario may put both on one map.
PARKING_SYMBOL, PARKING_COLOR = "point", "#808080"


def parking_section(ctx: ServerContext, handle: str, *, limit: int,
                    overrides: dict[str, Any], point_name: str = "the centre") -> Outcome:
    """`parking_candidates` on a staged ground, compact: each lot or
    hardstand without its outline - `point` is guaranteed inside it.
    `point_name` is what the markers call the staged ground's centre, which
    the distances are measured from. `total_before_limit` is null when no
    lots could be read (`_lots_unknown`): unknown, not none."""
    arguments = {"handle": handle, "limit": limit, **overrides}
    payload = call_feature(ctx, "parking_candidates", parking_feature.parking_candidates,
                           arguments)
    candidates = [{key: value for key, value in candidate.items() if key != "geometry"}
                  for candidate in payload["candidates"]]
    summary = payload.get("summary") or {}
    return Outcome(
        section(OK, warnings=payload.get("warnings") or [], handle=handle,
                candidates=candidates,
                total_before_limit=(None if _lots_unknown(payload)
                                    else summary.get("total_before_limit")),
                truncated_to=summary.get("truncated_to"),
                detail_call=detail_call("parking_candidates", arguments)),
        [parking_marker(candidate, point_name=point_name) for candidate in candidates])


def _lots_unknown(payload: dict) -> bool:
    """Whether a `parking_candidates` answer could read no lots at all, so
    its empty list is unknown rather than "no parking here": the staged
    ground has no osm.json (an Overpass outage at staging - the feature's
    own warning says so), or one fetched by a query older than parking
    (`query_version` below 3, as `egress_routes` reads the same ground's
    lots as unknown). The feature reads a missing file as version 1 too,
    but that is its default, not a promise: the warning is checked
    first."""
    if NO_OSM_PARKING_WARNING in (payload.get("warnings") or []):
        return True
    version = payload.get("query_version")
    return version is not None and int(version) < OSM_QUERY_VERSION_PARKING


def parking_facts(parking: dict) -> dict[str, Any]:
    """From an answered `parking_section`: how many sites it kept and how
    many there were, and the nearest - the feature orders by distance, so
    it is the first - or None when there is no site at all. The counts are
    None when the section's total is - no lots could be read - because a 0
    beside a null `nearest` is how "none here" reads, and the summary
    carries no warning to say otherwise."""
    candidates = parking["candidates"]
    first = candidates[0] if candidates else None
    total = parking.get("total_before_limit")
    return {"count": None if total is None else len(candidates), "total_before_limit": total,
            "nearest": None if first is None else {
                key: first.get(key) for key in
                ("id", "kind", "name", "distance_from_center_m", "bearing_from_center_deg",
                 "flags")}}


def parking_marker(candidate: dict, *, point_name: str = "the centre") -> dict:
    """A lot or hardstand as a grey CalTopo Marker at its `point`. Its
    distance is said from `point_name`, as every other marker in the same
    file says its own, so a map reader never wonders which centre."""
    flags = candidate.get("flags") or []
    lines = [f"{candidate.get('kind') or 'site'}, "
             f"{candidate['distance_from_center_m']:,.0f} m from {point_name}."]
    if flags:
        lines.append("Flags: " + ", ".join(flags) + ".")
    lines.append("From OpenStreetMap; gates, hours and the road in are not checked.")
    point = candidate["point"]
    return {"type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [point["longitude"], point["latitude"]]},
            "properties": {"class": "Marker",
                           "title": candidate.get("name") or candidate["id"],
                           "description": "\n".join(lines),
                           "marker-symbol": PARKING_SYMBOL, "marker-color": PARKING_COLOR,
                           "marker-size": 1}}


# --- Hasty lines -------------------------------------------------------------


#: What a hasty candidate keeps of its facts in a section: what a planner
#: bands and compares candidates by. A line has no width and a corridor no
#: branches, so each keeps only those it has. Slope, climb, effort and
#: ingress stay with `get_candidate`, the line or corridor itself too. Not
#: `overlap_reason`: whether a map was read is the set's, so every candidate
#: gives the same one, and the section says it once (`hasty_section`).
HASTY_FACTS = ("length_m", "width_m", "area_acres", "branches", "clipped_at_edge",
               "distance_from_center_m", "bearing_from_center_deg",
               "travel_minutes_from_center", "overlap_existing_pct", "overlap_labels")


def check_hasty_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """`overrides['hasty']` with `corridor_widths_m` checked now and kept as
    the plain mapping JSON brings: the tool takes a `CorridorWidths`, and
    without this a mapping would raise inside the section once the ground
    was staged, while the model itself in a `detail_call` is not JSON. A bad
    width is the caller's mistake, refused before any work."""
    widths = overrides.get("corridor_widths_m")
    if widths is None:
        return dict(overrides)
    if isinstance(widths, hasty_feature.CorridorWidths):
        return {**overrides, "corridor_widths_m": widths.as_overrides()}
    if not isinstance(widths, dict):
        raise ValueError("overrides['hasty']['corridor_widths_m'] must be a mapping of "
                         "corridor class to metres, such as {'waterway': 40}; got "
                         f"{type(widths).__name__}")
    # The model ignores a name it does not know, so a misspelt class would
    # quietly keep the corpus width the caller meant to change.
    classes = tuple(hasty_feature.CorridorWidths.model_fields)
    unknown = sorted(set(widths) - set(classes))
    if unknown:
        raise ValueError(f"overrides['hasty']['corridor_widths_m'] is not a set of corridor "
                         f"widths: {unknown} is no corridor class; the classes are "
                         f"{list(classes)}")
    try:
        model = hasty_feature.CorridorWidths(**widths)
    except (TypeError, ValueError) as error:
        raise ValueError(f"overrides['hasty']['corridor_widths_m'] is not a set of corridor "
                         f"widths (waterway, above 0 m): {error}") from error
    return {**overrides, "corridor_widths_m": model.as_overrides()}


def _hasty_candidates(ctx: ServerContext, **arguments) -> dict[str, Any]:
    """`hasty_candidates` with `corridor_widths_m` made the model the tool
    takes (`check_hasty_overrides` keeps it a mapping, for the
    `detail_call`). Looked up on the module at call time, so a test's stub
    stands in."""
    widths = arguments.get("corridor_widths_m")
    if isinstance(widths, dict):
        arguments = {**arguments, "corridor_widths_m": hasty_feature.CorridorWidths(**widths)}
    return hasty_feature.hasty_candidates(ctx, **arguments)


def compact_hasty(candidate: dict) -> dict:
    """One hasty candidate without its geometry: what it is, its size, where
    it lies from the centre and how long the walk to it is, what the map
    already has on it, and its norm flags as their notes."""
    facts = candidate.get("facts") or {}
    return {"id": candidate["id"], "kind": candidate.get("kind"), "axis": candidate.get("axis"),
            "name": candidate.get("name"),
            **{key: facts[key] for key in HASTY_FACTS if key in facts},
            "flags": [flag.get("note") for flag in candidate.get("norm_flags") or []],
            "warnings": list(candidate.get("warnings") or [])}


def hasty_section(ctx: ServerContext, handle: str, *, limit: int,
                  overrides: dict[str, Any]) -> Outcome:
    """`hasty_candidates` on a staged area, compact: the set id and each
    candidate's facts, never its line or corridor - `get_candidate` returns
    one in full, and `set_export_section` draws the set for CalTopo. Call
    `check_hasty_overrides` first.

    `overlap_reason` is said once, on the section: why every candidate's
    `overlap_existing_pct` is null (no map was read - unknown, not zero),
    or null when the map was read. The feature repeats it on each
    candidate, about 2 KB of one sentence at the default limit of 20.

    `by_axis` and `by_kind` are the feature's counts of the candidates
    kept, plus a null for each axis that could not be seen
    (`_unknown_axes`), since an axis absent from the counts reads as "none
    kept". A kind is null when every axis of it that was asked for could
    not be seen - `line` whenever a line axis was, every line being read
    from OpenStreetMap; `corridor` only when waterway was its one axis
    asked for - and `total_before_limit` is null when no axis asked for
    could be seen. A count with at least one axis read stays a number: it
    counts what could be read, and the nulls beside it say what could
    not."""
    arguments = {"handle": handle, "limit": limit, **overrides}
    payload = call_feature(ctx, "hasty_candidates", _hasty_candidates, arguments)
    summary = payload.get("summary") or {}
    reason = next((reason for candidate in payload["candidates"]
                   if (reason := (candidate.get("facts") or {}).get("overlap_reason"))), None)
    by_axis, by_kind = summary.get("by_axis"), summary.get("by_kind")
    total = summary.get("total_before_limit")
    unknown = _unknown_axes(payload)
    if unknown:
        # Never over a count: an axis the feature kept candidates on was
        # seen, whatever the warnings say.
        by_axis = dict(by_axis or {})
        by_kind = dict(by_kind or {})
        for axis in unknown:
            by_axis.setdefault(axis, None)
        unseen = {axis for axis in unknown if by_axis[axis] is None}
        asked = _asked_axes(payload)
        for kind, classes in (("line", LINE_CLASSES), ("corridor", CORRIDOR_CLASSES)):
            wanted = [axis for axis in asked if axis in classes]
            if wanted and unseen.issuperset(wanted):
                by_kind.setdefault(kind, None)
        if unseen.issuperset(asked):
            # Nothing asked for could be read: the feature's 0 is no count
            # at all, and `hasty_facts` reads this null as its own.
            total = None
    return Outcome(section(
        OK, warnings=payload.get("warnings") or [], handle=handle, set_id=payload["set_id"],
        created_at=payload.get("created_at"), reused=payload.get("reused"),
        candidates=[compact_hasty(candidate) for candidate in payload["candidates"]],
        by_axis=by_axis, by_kind=by_kind,
        total_before_limit=total,
        truncated_to=summary.get("truncated_to"),
        dropped_beyond_max_distance=summary.get("dropped_beyond_max_distance"),
        existing_assignments=summary.get("existing_assignments"), overlap_reason=reason,
        detail_call=detail_call("hasty_candidates", arguments)))


#: The hasty axes drawn from OpenStreetMap - the classes a staged route is
#: read as (`planning.boundary.ROUTE_CLASS`) that hasty_candidates offers -
#: in its own order: trail, road, waterway. On ground with no osm.json they
#: are unknown.
OSM_AXES = tuple(axis for axis in HASTY_ALL_INCLUDE if axis in ROUTE_CLASS.values())


def _asked_axes(payload: dict) -> tuple[str, ...]:
    """The axes a `hasty_candidates` answer was asked for: its `params`'
    include, or the feature's default when it gives none."""
    return tuple((payload.get("params") or {}).get("include") or HASTY_DEFAULT_INCLUDE)


def _unknown_axes(payload: dict) -> list[str]:
    """The axes a `hasty_candidates` answer could not see: those of
    OSM_AXES it was asked for (`_asked_axes`), when the staged ground had
    no osm.json - the ground's own warning, which every set built on it
    carries. An axis not asked for is the caller's choice, not an unknown;
    on ground with OpenStreetMap, none is unknown."""
    if NO_OSM_LINES_WARNING not in (payload.get("warnings") or []):
        return []
    asked = _asked_axes(payload)
    return [axis for axis in OSM_AXES if axis in asked]


def hasty_facts(hasty: dict) -> dict[str, Any]:
    """From an answered `hasty_section`: how many candidates it kept, by
    kind (line, corridor) and by axis (trail, road, waterway), and how many there were before the limit. What could not be
    seen is null in the section's counts and so here: an axis, a kind none
    of whose axes asked for could be seen, and - when no axis asked for
    could be - the total and so `count`, as `parking_facts` does, because
    a 0 there is how "none kept" reads and the summary carries no warning
    to say otherwise."""
    total = hasty.get("total_before_limit")
    return {"count": None if total is None else len(hasty["candidates"]),
            "by_kind": hasty.get("by_kind") or {},
            "by_axis": hasty.get("by_axis") or {},
            "total_before_limit": total}


# --- Walking rings ------------------------------------------------------------


#: The largest radius a scenario stages for the walking rings: the largest
#: whole metre under `prepare_area`'s limit, which it stages at any latitude
#: (its own circle polygon puts 4,500 m a hair over, and 4,499.9 m over near
#: the poles - see `check_radius`). The rings tool's own edge advice names
#: the same radius (`planning.travel_rings.LARGEST_RADIUS_M`), so the two
#: never disagree. With the 1,000 m planning buffer the ground then reaches
#: about 5,500 m from the point: 55 minutes on a good trail at the model's
#: top speed, so a longer ring is cut there, and `reaches_edge` says so.
RINGS_MAX_RADIUS_M = LARGEST_RADIUS_M
#: The `travel_time_rings` keywords that set how fast anyone walks, and so
#: how much ground the rings need: checked by the feature's own model
#: (`rings_model`) before anything is staged.
RINGS_MODEL_KEYWORDS = ("speed_factor", "off_way_factor", "creek_crossing_minutes",
                        "cliff_slope_deg", "vegetation_factors")
#: What a ring keeps in a section: its measurements and two points, never
#: its polygon, outline or the routes to its farthest ground.
RING_FIELDS = ("minutes", "parts", "area_acres", "equivalent_radius_m", "max_reach_m",
               "max_reach_bearing_deg", "farthest_point", "reach_by_direction_m",
               "reaches_edge")
#: Ring outline colours in the export: one cyan, light to dark, the
#: shortest ring lightest. A hue nothing else in these files uses - not the
#: red of an assignment or a hospital, the black of the search area, the
#: blue of the scenario's point, the grey of parking, nor the green, yellow
#: and orange of the landing-zone grades or the violet and blue of egress
#: routes - so a ring is never read as one of those when every file is
#: imported into one map. `ring_line` spreads the rings drawn over the
#: whole ramp, so the default three are light, middle and dark.
RING_STROKES = ("#00E5FF", "#00CDE6", "#00B5CC", "#009DB3", "#008599", "#006D80",
                "#005566", "#003D4D")
#: What a ring outline says on the map when the walk lacked a source, by
#: the warning the walk gives (`planning.walking`). The section carries the
#: warning itself; a team reading the line in CalTopo sees only this, and a
#: ring walked without OpenStreetMap is a different ring - no trail speeds
#: anyone and no river stops anyone.
RING_GROUND_NOTES = {
    NO_OSM_WARNING: ("No OpenStreetMap on this ground: no trail, road, river or creek was "
                     "seen, so trails speed no one and a river stops no one."),
    NO_EVT_WARNING: ("No LANDFIRE vegetation on this ground: brush slows no one, so the "
                     "ring is too large where it is thick."),
    WATERWAY_KINDS_UNKNOWN_WARNING: ("Ground staged before waterway kinds were kept: every "
                                     "river is read as a creek and crossed."),
}


def check_ring_minutes(name: str, minutes) -> list[float]:
    """The ring times as `travel_time_rings` reads them - ascending, once
    each, every one above zero, at most eight - or a ValueError now, naming
    the scenario's argument. A string is refused: it would be read a
    character at a time."""
    if isinstance(minutes, str):
        raise ValueError(f"{name} must be a list of walking minutes, such as [15, 30, 60]")
    try:
        return minutes_list(minutes)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error


def rings_ground(minutes: list[float], overrides: dict[str, Any]) -> Ground:
    """The ground `travel_time_rings` needs round the point for the longest
    of `minutes` to fit whole - the radius the feature's own edge warning
    would name (`radius_to_fit`), on the quickest ground of the model the
    overrides give (`fastest_factor`: a trail, unless an `off_way_factor`
    or `vegetation_factors` override makes open ground quicker) - capped
    at RINGS_MAX_RADIUS_M, at 10 m. Fewer or shorter rings stage less
    ground; an absurd time or factor stages the cap. The overrides that set
    the pace are checked here, by the feature's own model, because the size
    depends on them: a bad one is refused before anything is staged."""
    speed = overrides.get("speed_factor")
    speed = 1.0 if speed is None else speed
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) \
            or not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"overrides['travel_rings']['speed_factor'] must be a number above "
                         f"zero (0.5 for a subject who walks at half pace); got {speed!r}")
    pace = {key: overrides[key] for key in RINGS_MODEL_KEYWORDS
            if overrides.get(key) is not None}
    try:
        model = rings_model(**{**pace, "speed_factor": float(speed)})
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"overrides['travel_rings'] cannot be walked: {error}") from error
    radius = min(radius_to_fit(max(minutes), fastest_factor(model)), RINGS_MAX_RADIUS_M)
    return Ground(radius, WALKING_RESOLUTION_M)


def rings_section(ctx: ServerContext, handle: str, *, latitude: float, longitude: float,
                  minutes: list[float], overrides: dict[str, Any],
                  point_name: str = "the point") -> Outcome:
    """`travel_time_rings` from the point, compact: each ring's area, reach
    and edge, never its polygon, outline or routes. The outlines go to the
    export (`ring_line`)."""
    arguments = {"handle": handle, "center_latitude": latitude, "center_longitude": longitude,
                 "minutes": list(minutes), **overrides}
    payload = call_feature(ctx, "travel_time_rings", rings_feature.travel_time_rings,
                           arguments)
    rings = payload["rings"]
    warnings = payload.get("warnings") or []
    notes = [RING_GROUND_NOTES[warning] for warning in warnings if warning in RING_GROUND_NOTES]
    return Outcome(
        section(OK, warnings=warnings, handle=handle,
                rings=[{key: ring.get(key) for key in RING_FIELDS} for ring in rings],
                resolution_m=payload.get("resolution_m"),
                speed_factor=payload.get("speed_factor"),
                # The feature's `ground`, renamed: a scenario's section names
                # the staged ground it stands on as `ground`.
                ground_facts=payload.get("ground"),
                detail_call=detail_call("travel_time_rings", arguments)),
        [ring_line(ring, index, of=len(rings), point_name=point_name, notes=notes)
         for index, ring in enumerate(rings)
         if (ring.get("outline") or {}).get("type") == "LineString"])


def ring_line(ring: dict, index: int, *, of: int = 1, point_name: str = "the point",
              notes: list[str] = ()) -> dict:
    """A ring's outline - the largest part's outer boundary, the loop the
    feature says to draw - as a CalTopo line Shape, its description the
    ring's facts, what the walk could not see (`notes`, from
    RING_GROUND_NOTES) and what the minutes are NOT. Ring `index` of `of`
    takes its colour spread evenly over RING_STROKES, shortest lightest."""
    lines = [f"Ground within {ring['minutes']:g} walking minutes of {point_name} at a "
             f"steady pace: {ring['area_acres']:,.1f} acres, as much as a circle of "
             f"{ring['equivalent_radius_m']:,.0f} m radius; it reaches farthest "
             f"{ring['max_reach_m']:,.0f} m at {ring['max_reach_bearing_deg']:.0f} deg."]
    if (ring.get("parts") or 1) > 1:
        lines.append(f"In {ring['parts']} parts; only the largest is outlined here.")
    if ring.get("reaches_edge"):
        lines.append("Cut at the edge of the staged ground: the ground it encloses is a "
                     "lower bound.")
    lines.extend(notes)
    lines.append("Walking time over a model, not time since last seen, and not a line the "
                 "subject cannot be beyond. Fences, private land and unmapped trails are "
                 "unseen.")
    return {"type": "Feature", "geometry": ring["outline"],
            "properties": {"class": "Shape", "title": f"{ring['minutes']:g} min walk",
                           "description": "\n".join(lines),
                           "stroke": RING_STROKES[_ring_shade(index, of)],
                           "stroke-width": 2, "stroke-opacity": 0.9}}


def _ring_shade(index: int, of: int) -> int:
    """Where ring `index` of `of` falls on RING_STROKES: the first lightest,
    the last darkest, the rest between - so three rings are not three near
    shades of one light cyan."""
    last = len(RING_STROKES) - 1
    if of < 2:
        return 0
    return min(last, max(0, round(index * last / (of - 1))))


def rings_facts(rings: dict) -> list[dict]:
    """From an answered `rings_section`: each ring's area and reach."""
    return [{key: ring.get(key) for key in
             ("minutes", "area_acres", "equivalent_radius_m", "max_reach_m", "reaches_edge")}
            for ring in rings["rings"]]


def hasty_pace_warnings(hasty: dict, rings: dict, ring_overrides: dict[str, Any]
                        ) -> list[str]:
    """A top-level warning when the walking rings were given a pace of their
    own (`ring_overrides`, `overrides['travel_rings']` as `check_overrides`
    returned them and `rings_ground` has already walked) beside a hasty
    section that answered. hasty_candidates takes no pace: its
    `travel_minutes_from_center` are always walked on the default model.
    Unsaid, a trail reads 6 minutes away at speed_factor 0.5 in the same
    answer as a 12-minute ring that stops short of it, and a planner banding
    the lines on the rings' pace sends teams wider than the subject could
    have walked.

    Nothing when the rings were left out (`unused_overrides` already says no
    call took their overrides), when hasty did not answer, or when every
    keyword given walks the default model anyway - a speed_factor of 1 given
    outright. With speed_factor the only change, every minute of the walk
    scales by it (the speeds and a creek's crossing time alike), so the
    warning says how to read a line's minutes at the rings' pace; any other
    keyword slows some ground and not the rest, and the order the lines
    come in can change with it, which the warning says instead.
    cliff_slope_deg is no pace at all: it moves the slope past which ground
    is closed (`planning.walking`), so a line the default walk never reaches
    - its minutes null - may be reached on the rings' walk, or the reverse.
    Left as "another order", a null would read as ground the subject could
    not reach at their own cliff limit, so the warning says that too."""
    if hasty.get("status") != OK or rings.get("status") == SKIPPED:
        return []
    default = rings_model()
    changed = {key: ring_overrides[key] for key in RINGS_MODEL_KEYWORDS
               if ring_overrides.get(key) is not None
               and rings_model(**{key: ring_overrides[key]}) != default}
    if not changed:
        return []
    given = ", ".join(
        f"{key} {json.dumps(value) if isinstance(value, dict) else format(value, 'g')}"
        for key, value in changed.items())
    text = ("travel_minutes_from_center on the hasty lines is walked on the default walking "
            f"model, not the one overrides['travel_rings'] gives the walking rings ({given}): "
            "hasty_candidates takes none of these, so a line's minutes cannot be read "
            "against a ring.")
    if set(changed) == {"speed_factor"}:
        text += (" At the rings' pace a line's minutes are its travel_minutes_from_center "
                 f"divided by {changed['speed_factor']:g}.")
    else:
        text += " On the rings' walk the lines may also come in another order."
    if "cliff_slope_deg" in changed:
        text += (" cliff_slope_deg also changes which ground can be walked at all: a line "
                 "whose minutes are null here may be reached on the rings' walk, and a line "
                 "with minutes may not be.")
    return [text]


# --- Urban blocks and assignments ----------------------------------------------


#: The populated_areas keywords that decide which blocks are drawn.
#: urban_assignments takes only `max_distance_m` of them: it draws its own
#: blocks, at every other rule's default, before grouping them.
BLOCK_RULES = ("pixel_m", "max_parcel_area_m2", "min_building_m2", "large_parcel_min_cover",
               "bridge_m", "min_parcels", "max_distance_m")
#: The block rules urban_assignments groups blocks drawn by: the defaults of
#: the domain function it calls (`planning.urban` passes none of them but
#: `max_distance_m`), read once at import, so a changed default there is
#: what the comparison in `urban_section` uses.
GROUPED_BLOCK_RULES: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(populated_areas_impl).parameters.items()
    if name in BLOCK_RULES}
#: The block rules a populated_areas call left at its own defaults draws by.
DRAWN_BLOCK_RULES: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(urban_feature.populated_areas).parameters.items()
    if name in BLOCK_RULES}
#: The house-marker colour each urban call draws in when not told, read
#: once from the real feature functions, as FEATURE_KEYWORDS is.
MARKER_COLOURS: dict[str, str] = {
    tool: inspect.signature(function).parameters["marker_color"].default
    for tool, (function, _fixed) in SECTION_PARTS["urban"].items()}
#: What an urban assignment keeps in a section: the facts a planner hands a
#: team by, never its polygon, its block names or its house markers -
#: `get_candidate` returns those. `markers` (a count) only when asked for.
URBAN_FACTS = ("streets", "blocks", "parcels", "buildings", "area_acres", "point",
               "distance_from_center_m", "bearing_from_center_deg", "parts", "water_links",
               "markers")
#: What the blocks keep in a section: populated_areas' counts. The blocks
#: themselves, with their ids and names, are its `detail_call`.
BLOCK_COUNTS = ("count", "total_before_limit", "truncated_to", "parcels_built_up",
                "parcels_in_blocks", "parcels_left_out", "large_parcels", "markers")


def check_target_acres(value) -> None:
    """`target_acres`: None (the corpus median for the resource type) or a
    size above zero - urban_assignments refuses zero and below, and a NaN
    or an infinity would group every block into one or none."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"target_acres must be a number of acres above zero, or null for "
                         f"the corpus median of the resource type; got {value!r}")


def check_resource_type(value) -> None:
    """`resource_type`: a CalTopo resource type such as GROUND or DOG. Any
    word is taken - for one the corpus has no median for, assignments are
    sized by the median of every assignment - but it must be a word: the
    features read it as text, and a number would raise once staged."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"resource_type must be a resource type such as 'GROUND'; "
                         f"got {value!r}")


def check_urban_overrides(overrides: dict[str, dict[str, Any]], *, max_distance_m: float,
                          building_markers: bool) -> None:
    """Refuse, before any work, an urban override value either feature would
    refuse once the ground is staged, in the domain's own words: a block
    rule out of range (`planning.populated.check_params`, which both calls
    run - urban_assignments draws its blocks through it with its own
    `max_distance_m`) and, with markers asked for, a marker colour CalTopo
    cannot read. `overrides` is what `check_overrides` returned for the
    section; `max_distance_m` and `building_markers` are what the helper
    passes when the overrides do not say."""
    drawn = {**DRAWN_BLOCK_RULES, "max_distance_m": max_distance_m, "limit": 0,
             **(overrides.get("populated_areas") or {})}
    grouped = {**GROUPED_BLOCK_RULES, "max_distance_m": max_distance_m, "limit": 0,
               **{key: value for key, value in (overrides.get("urban_assignments") or {}).items()
                  if key == "max_distance_m"}}
    for tool, rules in (("populated_areas", drawn), ("urban_assignments", grouped)):
        try:
            check_block_params(**{key: rules[key] for key in (*BLOCK_RULES, "limit")})
            # Whether it was given, not whether it is None: an explicit null
            # reaches the feature as None, which it refuses once staged.
            given = overrides.get(tool) or {}
            if building_markers and "marker_color" in given:
                marker_colour(given["marker_color"])
        except TypeError as error:
            # A string where a number goes, compared with one.
            raise ValueError(f"overrides['urban'][{tool!r}]: {', '.join(BLOCK_RULES)} and "
                             f"limit are numbers ({error})") from error
        except ValueError as error:
            raise ValueError(f"overrides['urban'][{tool!r}]: {error}") from error


def check_urban_radius(name: str, value: float, *, latitude: float, longitude: float) -> None:
    """A search radius whose circle populated_areas can read lot lines over,
    or a ValueError now: the panel is read at 1 m and coarsened to at most
    2 m to keep within 4,096 pixels a side, so a circle over about 8 km
    across (a radius over about 4,035 m) is refused - by the feature's own
    rule (`panel_pixel_m`) on the very circle `prepare_area` would stage,
    which reads no file and fetches nothing. A `pixel_m` override changes
    nothing here: it is at most 2 m (`check_urban_overrides`), and the panel
    is coarsened to 2 m before the limit bites. Call `check_radius` first."""
    area = build_area(center_latitude=latitude, center_longitude=longitude,
                      radius_m=float(value))
    try:
        panel_pixel_m(area.geometry)
    except ValueError as error:
        raise ValueError(f"{name} of {value:,} m is too wide to read lot lines over, so "
                         f"nothing was staged: populated_areas would refuse it - {error}; "
                         f"or leave urban out of include") from error


def urban_section(ctx: ServerContext, handle: str, *, radius_m: float,
                  target_acres: float | None, resource_type: str, building_markers: bool,
                  overrides: dict[str, dict[str, Any]]) -> Outcome:
    """`populated_areas`, then `urban_assignments`, on one staged ground,
    compact: the blocks' counts and set, and each assignment's facts without
    its polygon. `radius_m` is the staged search circle's radius: what both
    calls keep blocks within (`max_distance_m`) unless their overrides
    (`{tool: keyword arguments}`) say otherwise. Call
    `check_urban_overrides` first.

    Two calls because urban_assignments draws its blocks itself and keeps
    none: populated_areas saves them as a set, so a block's polygon is a
    `get_candidate` away. It also takes the block rules urban_assignments
    does not; when the two calls draw blocks by different rules the section
    says so, since the blocks counted are then not the ones grouped. The
    blocks come first: they fetch CalTopo's parcel and structures panels,
    which the grouping then reads back from the cache, and a refusal (no
    WMS token) is the blocks' own, in their own words. A grouping that
    raises after them fails the section with its own error and call, but
    the blocks stay under `blocks` - they are saved and still answer - so
    `overall_status` reads the section as partial, as it does an egress
    whose other mode answered."""
    blocks_arguments = {"handle": handle, "max_distance_m": radius_m,
                        "building_markers": building_markers,
                        **(overrides.get("populated_areas") or {})}
    blocks = call_feature(ctx, "populated_areas", urban_feature.populated_areas,
                          blocks_arguments)
    arguments = {"handle": handle, "target_acres": target_acres, "resource_type": resource_type,
                 "max_distance_m": radius_m, "building_markers": building_markers,
                 **(overrides.get("urban_assignments") or {})}
    # What the section says of the two calls themselves, whether or not the
    # grouping answers: each a caller's choice that reads as something else.
    notes = [note for note in (_beyond_circle(overrides, radius_m),
                               _block_rules_differ(blocks_arguments, arguments),
                               _colours_differ(blocks_arguments, arguments)) if note]
    block_summary = blocks.get("summary") or {}

    def blocks_entry(said: list[str]) -> dict[str, Any]:
        return {"set_id": blocks["set_id"], "created_at": blocks.get("created_at"),
                "reused": blocks.get("reused"),
                **{key: block_summary[key] for key in BLOCK_COUNTS if key in block_summary},
                # Said once: most of the blocks' warnings are the grouping's too.
                "warnings": [warning for warning in blocks.get("warnings") or []
                             if warning not in said],
                "detail_call": detail_call("populated_areas", blocks_arguments)}

    try:
        payload = call_feature(ctx, "urban_assignments", urban_feature.urban_assignments,
                               arguments)
    except FeatureFailed as failure:
        return Outcome(section(FAILED, error=error_text(ctx, failure.error), warnings=notes,
                               handle=handle, blocks=blocks_entry(notes),
                               detail_call=failure.call))
    summary = payload.get("summary") or {}
    grouping = (payload.get("assumptions") or {}).get("grouping") or {}
    warnings = [*(payload.get("warnings") or []), *notes]
    return Outcome(section(
        OK, warnings=warnings, handle=handle, set_id=payload["set_id"],
        created_at=payload.get("created_at"), reused=payload.get("reused"),
        assignments=[compact_urban(candidate) for candidate in payload["candidates"]],
        count=summary.get("count"), target_acres=grouping.get("target_acres"),
        target_source=grouping.get("target_source"), area_acres=summary.get("area_acres"),
        parcels=summary.get("parcels"), buildings=summary.get("buildings"),
        parcels_left_out=summary.get("parcels_left_out"),
        **({"markers": summary["markers"]} if "markers" in summary else {}),
        blocks=blocks_entry(warnings),
        detail_call=detail_call("urban_assignments", arguments)))


def _beyond_circle(overrides: dict[str, dict[str, Any]], radius_m: float) -> str | None:
    """A warning when an urban `max_distance_m` override reaches past the
    search circle (or is null, no limit), else None. It reads as a wider
    search but is not one: populated_areas reads lot lines over the staged
    circle only (plus a 60 m margin) and drops every block that does not
    touch it, so past the radius the keyword can add only blocks straddling
    the circle's edge, never one wholly outside. Not refused - keeping those
    straddling blocks is a fair thing to want - but said, with the way to
    search wider."""
    wide = {tool: given["max_distance_m"] for tool, given in overrides.items()
            if "max_distance_m" in given
            and (given["max_distance_m"] is None or given["max_distance_m"] > radius_m)}
    if not wide:
        return None
    said = "; ".join(f"{tool} {'null (no limit)' if value is None else f'{value:,g} m'}"
                     for tool, value in wide.items())
    return (f"max_distance_m reaches past the {radius_m:,.0f} m search radius ({said}), but "
            "lot lines are read over the staged search circle only: it can add only blocks "
            "straddling the circle's edge, never one wholly outside it - to search wider, "
            "pass a larger radius")


def _colours_differ(blocks_arguments: dict[str, Any], arguments: dict[str, Any]
                    ) -> str | None:
    """A warning when, with house markers on, a `marker_color` given to
    populated_areas is not the one the assignments are drawn in, else None.
    The draft file carries the assignments' markers, so a colour given to
    populated_areas alone never reaches CalTopo - which a caller asking for
    red houses would not guess. A colour given to urban_assignments alone is
    not said: it reaches the drafts, as asked, and only the blocks' own set
    keeps white. Compared as CalTopo writes a colour (upper case, one `#`)
    without refusing one: `check_urban_overrides` did that."""
    if not arguments.get("building_markers") or "marker_color" not in blocks_arguments:
        return None

    def written(tool: str, given: dict[str, Any]) -> str:
        colour = given.get("marker_color", MARKER_COLOURS[tool])
        return "#" + str(colour).strip().lstrip("#").upper()

    drawn = written("populated_areas", blocks_arguments)
    grouped = written("urban_assignments", arguments)
    if drawn == grouped:
        return None
    return (f"the blocks' house markers are {drawn} (populated_areas' marker_color) but the "
            f"assignments' - and so the draft file's - are {grouped} (urban_assignments' "
            "marker_color): give urban_assignments the colour the drafts should carry")


def _block_rules_differ(blocks_arguments: dict[str, Any], arguments: dict[str, Any]
                        ) -> str | None:
    """A warning when the blocks populated_areas drew are not the ones
    urban_assignments grouped - a block rule overridden for the blocks
    alone, or a different `max_distance_m` - else None. Not an error: the
    caller may want the blocks drawn another way to compare."""
    drawn = {key: blocks_arguments.get(key, DRAWN_BLOCK_RULES[key]) for key in BLOCK_RULES}
    grouped = {**GROUPED_BLOCK_RULES, "max_distance_m": arguments.get("max_distance_m")}
    differ = [key for key in BLOCK_RULES if drawn[key] != grouped[key]]
    if not differ:
        return None
    return ("the blocks under `blocks` were drawn with "
            + ", ".join(f"{key} {drawn[key]!r}" for key in differ)
            + "; the assignments were grouped from blocks drawn with "
            + ", ".join(f"{key} {grouped[key]!r}" for key in differ)
            + " (urban_assignments takes no block rule but max_distance_m), so the two "
              "do not count the same blocks")


def compact_urban(candidate: dict) -> dict:
    """One urban assignment without its polygon, block names or markers:
    what it is called, its streets and size, where its inside point lies
    from the centre, and its warnings (parts joined across water, an
    outline in pieces)."""
    facts = candidate.get("facts") or {}
    return {"id": candidate["id"], "name": candidate.get("name"),
            **{key: facts[key] for key in URBAN_FACTS if key in facts},
            "warnings": list(candidate.get("warnings") or [])}


def urban_facts(urban: dict) -> dict[str, Any]:
    """From an answered `urban_section`: how many assignments, the target
    they were grouped towards and the spread of their sizes round it (min,
    median, max and the total, null when there are none), and the lots and
    buildings in them."""
    return {key: urban.get(key) for key in
            ("count", "target_acres", "area_acres", "parcels", "buildings")}


# --- An assignment's ground: metrics, coverage, features -----------------------


#: What a drainage keeps in the metrics section: its measurements, without
#: the method sentence every one repeats word for word (`detail_call` has it).
DRAINAGE_FIELDS = ("id", "length_m", "full_length_m", "contributing_area_m2", "depth_m",
                   "sidewall_slope_deg", "transect_count", "any_transect_truncated")
#: The traverse facts kept: the rates, the sweep width and the time a pass
#: takes. The table of rates and widths by lifeform behind them is
#: `detail_call`'s - the same for every assignment, and a third of the block.
TRAVERSE_FIELDS = ("baseline_rate_kmh", "vegetation_adjusted_rate_kmh",
                   "effective_sweep_width_m", "sweep_time_hours_per_pass", "method")
#: What a coverage gap keeps: its size and the terrain shapes it leaves
#: unsearched, never its outline - `gap_point` puts a point inside it instead.
GAP_FIELDS = ("id", "area_m2", "length_m", "dominant_vegetation", "mean_slope_deg",
              "unsearched_shapes")
#: The coverage facts kept as coverage_for gives them.
COVERAGE_FIELDS = ("assignment_label", "assignment_kind", "teams", "effective_sweep_width_m",
                   "covered_fraction", "effort_coverage", "zone_area_m2", "line_length_m",
                   "time_entered", "time_exited", "time_searching_hours")


def check_track_names(track_names, *, needed: bool) -> list[str]:
    """The track names as a list, or a ValueError now. A string is refused -
    it would be read a letter at a time - and so is a name that is not text
    or is blank. An empty list is refused only when coverage is asked for:
    coverage_for would read it as no team at all and raise once the ground
    was staged, while with coverage left out it means nothing."""
    if isinstance(track_names, str):
        raise ValueError(f"track_names must be a list of track names, such as [{track_names!r}]")
    if not isinstance(track_names, (list, tuple)):
        raise ValueError(f"track_names must be a list of track names; got "
                         f"{type(track_names).__name__}")
    blank = [name for name in track_names if not isinstance(name, str) or not name.strip()]
    if blank:
        raise ValueError(f"track_names must be names as the map shows them; got {blank!r}")
    if not track_names and needed:
        raise ValueError("track_names is empty, but coverage is asked for: name the team's "
                         "tracks as the map shows them (a team number such as '101' finds the "
                         "tracks named for that team), or leave coverage out of include")
    return list(track_names)


def check_feature_kinds(kinds, *, needed: bool) -> list[str]:
    """The feature kinds, each once, in the order given, or a ValueError
    now in `identify_features`' own words (`terrain.features.validate_kinds`,
    which names the kinds there are). A direct caller is not held to the
    schema's enum, so a typo would otherwise fail the section after the
    staging. An empty list is refused only when the features are asked for."""
    from ..terrain.features import validate_kinds

    if isinstance(kinds, str):
        raise ValueError(f"kinds must be a list of feature kinds, such as [{kinds!r}]")
    if not isinstance(kinds, (list, tuple)):
        raise ValueError(f"kinds must be a list of feature kinds; got {type(kinds).__name__}")
    strangers = [kind for kind in kinds if not isinstance(kind, str)]
    if strangers:
        raise ValueError(f"kinds must be feature kind names; got {strangers!r}")
    if kinds or needed:
        validate_kinds(list(kinds))
    return list(dict.fromkeys(kinds))


def _staged_directory(ctx: ServerContext, handle: str) -> Path | None:
    """The directory staged under `handle`, or None when there is none.
    Read without revalidating - no map read - since only what is on disk is
    wanted, and the section that asks is about to revalidate anyway."""
    from ..handles import resolve

    try:
        return resolve(handle, ctx.reader, revalidate=False).directory
    except Exception:  # noqa: BLE001 - unknown, and the caller says so
        return None


def assignment_kind(ctx: ServerContext, handle: str) -> str | None:
    """"area" or "line": the shape of the assignment staged under `handle`,
    read from the staged subject, or None when it cannot be read."""
    from ..terrain.grid import read_subject

    directory = _staged_directory(ctx, handle)
    if directory is None:
        return None
    try:
        kind = (read_subject(directory).get("geometry") or {}).get("type")
    except Exception:  # noqa: BLE001 - unknown, and the caller says so
        return None
    return {"Polygon": "area", "MultiPolygon": "area",
            "LineString": "line", "MultiLineString": "line"}.get(kind)


def saved_metrics(ctx: ServerContext, handle: str) -> bool | None:
    """Whether `terrain_metrics` has saved its answer beside the ground
    staged under `handle` - the file `coverage_for` reads its sweep width
    from - or None when the staged ground cannot be found. A saved file
    always describes the ground as staged now: every re-staging discards
    it (`staging.discard_terrain_metrics`)."""
    from ..terrain.shapes import METRICS_FILENAME

    directory = _staged_directory(ctx, handle)
    return None if directory is None else (directory / METRICS_FILENAME).exists()


def metrics_section(ctx: ServerContext, handle: str, *, overrides: dict[str, Any]) -> Outcome:
    """`terrain_metrics` on a staged assignment, compact: what the ground
    is - elevation, slope, aspect, landform, drainages, vegetation, the
    walking rate and sweep width, routes and ways in - without one outline.
    Its warnings are the metrics' own data-quality warnings. It also saves
    the metrics beside the ground, which `coverage_for`, `slope_stats` and
    `elevation_profile` read: call it before `coverage_section`."""
    arguments = {"handle": handle, **overrides}
    payload = call_feature(ctx, "terrain_metrics", ground_component.terrain_metrics, arguments)
    quality = payload.get("data_quality") or {}
    return Outcome(section(OK, warnings=quality.get("warnings") or [], handle=handle,
                           **compact_metrics(payload),
                           detail_call=detail_call("terrain_metrics", arguments)),
                   staleness=payload.get("staleness"))


def compact_metrics(payload: dict) -> dict[str, Any]:
    """The terrain metrics without geometry: the measured blocks as they
    are, the drainages without the repeated method, the traverse without its
    lookup tables, the routes counted by type and named rather than drawn,
    and every identified shape counted by kind - 140 KB of outlines on a
    109-acre assignment, which say nothing here a count does not. The shape ids
    are the features section's; `detail_call` returns the shapes."""
    quality = payload.get("data_quality") or {}
    traverse = payload.get("traverse") or {}
    landforms = payload.get("landforms")
    hydrology = payload.get("hydrology")
    access = payload.get("access")
    shapes: dict[str, int] = {}
    for shape in payload.get("shapes") or []:
        shapes[shape["kind"]] = shapes.get(shape["kind"], 0) + 1
    drainages = (hydrology or {}).get("drainages") or []
    return {
        "assignment_label": payload.get("assignment_label"),
        "area_m2": quality.get("area_m2"),
        "dem_resolution_m": quality.get("dem_resolution_m"),
        "elevation": payload.get("elevation"),
        "slope": payload.get("slope"),
        "aspect": payload.get("aspect"),
        "landform": None if landforms is None else {
            "fractions": landforms.get("fractions"),
            "cliff_band_count": landforms.get("cliff_band_count")},
        "hydrology": None if hydrology is None else {
            "drainage_count": len(drainages),
            "upstream_truncated": hydrology.get("upstream_truncated"),
            "drainages": [{key: drainage.get(key) for key in DRAINAGE_FIELDS}
                          for drainage in drainages]},
        "vegetation": payload.get("vegetation"),
        "traverse": {key: traverse.get(key) for key in TRAVERSE_FIELDS},
        "access": None if access is None else compact_access(access),
        "shapes": shapes,
    }


def compact_access(access: dict) -> dict[str, Any]:
    """The metrics' `access` without a line: how many OpenStreetMap routes
    the staged ground holds, by type, the names among them, and how many
    points a walkable one crosses the boundary at."""
    by_type: dict[str, int] = {}
    names: list[str] = []
    routes = access.get("routes") or []
    for route in routes:
        properties = route.get("properties") or {}
        kind = properties.get("route_type") or "unknown"
        by_type[kind] = by_type.get(kind, 0) + 1
        name = properties.get("name")
        if name and name not in names:
            names.append(name)
    return {"routes": len(routes), "routes_by_type": by_type, "route_names": names,
            "ingress_count": access.get("ingress_count"), "source": access.get("source")}


def metrics_facts(metrics: dict) -> dict[str, Any]:
    """From an answered `metrics_section`: the size and relief of the
    ground, its slope, what grows on it, the sweep width and how long one
    pass takes, and how many drainages, cliff bands and ways in it has
    (ways in null when the ground has no OpenStreetMap)."""
    slope = metrics.get("slope") or {}
    vegetation = metrics.get("vegetation") or {}
    traverse = metrics.get("traverse") or {}
    access = metrics.get("access")
    return {"area_m2": metrics.get("area_m2"),
            "relief_m": (metrics.get("elevation") or {}).get("relief_m"),
            "slope_mean_deg": slope.get("mean_deg"), "slope_p90_deg": slope.get("p90_deg"),
            "dominant_lifeform": vegetation.get("dominant_lifeform"),
            "density_class": vegetation.get("density_class"),
            "effective_sweep_width_m": traverse.get("effective_sweep_width_m"),
            "sweep_time_hours_per_pass": traverse.get("sweep_time_hours_per_pass"),
            "drainages": (metrics.get("hydrology") or {}).get("drainage_count"),
            "cliff_bands": (metrics.get("landform") or {}).get("cliff_band_count"),
            "ingress_points": None if access is None else access.get("ingress_count")}


def coverage_section(ctx: ServerContext, handle: str, *, track_names: list[str],
                     overrides: dict[str, Any], metrics: dict | None = None) -> Outcome:
    """`coverage_for` on a staged assignment from the named tracks, compact:
    the coverage numbers, each track's summary, and each gap's size, the
    shapes it leaves unsearched and a point inside it - never its outline.

    `metrics` is the metrics section's dict when this scenario ran it, None
    when it was left out. Coverage of an AREA assignment takes its sweep
    width from the metrics `terrain_metrics` saves, so when that section
    failed this one does not run: DependencyFailed, naming the failure.
    Left out, coverage_for reads the metrics an earlier call saved on this
    ground (a re-staging discards them); with none saved an area's coverage
    does not run either: DependencyFailed, saying to add "metrics" to
    include - coverage_for's own refusal would name a local path and a
    command line, neither of which a scenario caller can use. A LINE
    assignment runs either way, on the width saved before or on
    coverage_for's default corridor, and whenever the metrics failed or
    none were saved a warning says which width and why."""
    notes = []
    if metrics is not None and metrics["status"] == FAILED:
        kind = assignment_kind(ctx, handle)
        if kind != "line":
            what = ("an area assignment" if kind == "area"
                    else "this assignment (whether it is an area or a line could not be read "
                         "from the staged ground)")
            raise DependencyFailed(
                f"coverage of {what} takes its sweep width from terrain_metrics, which the "
                f"metrics section runs first, and it failed, so this section did not run - "
                f"{metrics['error']}")
        notes.append(_line_width_note(left_out=False, saved=saved_metrics(ctx, handle)))
    elif metrics is None and saved_metrics(ctx, handle) is False:
        # Only a definite "none saved": ground that cannot be found at all
        # is coverage_for's to refuse, in its own words, when it resolves it.
        kind = assignment_kind(ctx, handle)
        if kind == "area":
            raise DependencyFailed(
                "coverage of an area assignment takes its sweep width from terrain_metrics, "
                "and none is saved on this ground (staging discards it whenever the ground "
                "changes), so this section did not run - add \"metrics\" to include")
        if kind == "line":
            notes.append(_line_width_note(left_out=True, saved=False))
    arguments = {"handle": handle, "track_names": list(track_names), **overrides}
    payload = call_feature(ctx, "coverage_for", coverage_feature.coverage_for, arguments)
    gaps = payload.get("gaps") or []
    closed = payload.get("closed_gaps") or []
    measure = {"area": "area_m2", "line": "length_m"}.get(payload.get("assignment_kind"))
    return Outcome(
        section(OK, warnings=[*(payload.get("warnings") or []), *notes], handle=handle,
                **{key: payload.get(key) for key in COVERAGE_FIELDS},
                tracks=list(payload.get("tracks") or []),
                gap_count=len(gaps), gap_area_m2=_total(gaps, "area_m2", measure),
                gap_length_m=_total(gaps, "length_m", measure),
                gaps=[{**{key: gap.get(key) for key in GAP_FIELDS},
                       "point": gap_point(gap.get("geometry"))} for gap in gaps],
                closed_gaps={"count": len(closed),
                             "area_m2": _total(closed, "area_m2", measure),
                             "length_m": _total(closed, "length_m", measure)},
                detail_call=detail_call("coverage_for", arguments)),
        staleness=payload.get("staleness"))


def _line_width_note(*, left_out: bool, saved: bool | None) -> str:
    """The warning a line assignment's coverage carries when this call's
    metrics did not give its corridor width: why they did not (`left_out`
    of include, or failed), which width coverage_for used instead, and -
    when metrics were left out - the scenario's own remedy."""
    why = "metrics was left out of include" if left_out else "the metrics section failed"
    if saved:
        return (f"{why}, so this line's corridor width is the one terrain_metrics saved on "
                "this ground at an earlier call")
    return (f"{why} and no metrics are saved on this ground, so this line's corridor is "
            "coverage_for's default width, as its own warning says"
            + (" - add \"metrics\" to include for the width this ground's vegetation gives"
               if left_out else ""))


def _total(items: list[dict], key: str, measure: str | None) -> float | None:
    """The sum of `key` over `items` when it is the assignment's own
    `measure` - area_m2 for an area, length_m for a line - and 0 when there
    are none; None for the measure it lacks, as coverage_for gives each gap,
    since a 0 there would read as "no gap" rather than "not measured"."""
    if key != measure:
        return None
    return float(sum(item.get(key) or 0.0 for item in items))


def gap_point(geometry: dict | None) -> dict | None:
    """A WGS84 Point inside a coverage gap - on it, for a line assignment's
    gap - or None when the gap has no outline: shapely's representative
    point, which lies inside the shape where a centroid of an L-shaped gap
    may not. coverage_for gives gap outlines in Web Mercator, the coverage
    grid's own coordinates, so the point is projected back here; rounded to
    six places, a tenth of a metre, since a gap is tens of metres across."""
    if not geometry:
        return None
    from shapely.geometry import shape

    from ..geo import to_wgs84_geojson

    inside = shape(geometry).representative_point()
    if inside.is_empty:
        return None
    point = to_wgs84_geojson(inside)
    return {"type": "Point", "coordinates": [round(value, 6) for value in point["coordinates"]]}


def coverage_facts(coverage: dict) -> dict[str, Any]:
    """From an answered `coverage_section`: how much of the assignment the
    tracks covered and the effort it took, how many tracks and gaps and the
    gaps' total, the largest gap - coverage_for lists them largest first -
    and how long the tracks were inside the assignment."""
    gaps = coverage["gaps"]
    largest = gaps[0] if gaps else None
    return {"assignment_kind": coverage.get("assignment_kind"),
            "covered_fraction": coverage.get("covered_fraction"),
            "effort_coverage": coverage.get("effort_coverage"),
            "tracks": len(coverage["tracks"]), "gaps": coverage["gap_count"],
            "gap_area_m2": coverage.get("gap_area_m2"),
            "gap_length_m": coverage.get("gap_length_m"),
            "largest_gap": None if largest is None else {
                key: largest.get(key) for key in ("id", "area_m2", "length_m", "point")},
            "time_searching_hours": coverage.get("time_searching_hours")}


def features_section(ctx: ServerContext, handle: str, *, kinds: list[str],
                     overrides: dict[str, Any]) -> Outcome:
    """`identify_features` on a staged assignment, compact: per kind, how
    many shapes (null when the kind's input was missing, `compact_kind`)
    and their ids - the ids coverage gaps cite and `slope_stats`
    takes - the summary's own facts, and the kind's warnings. Never a shape:
    `detail_call` returns them. A summary's per-shape lists (the drainages'
    measurements) are left out too; the metrics section carries those. The
    section's warnings are the kinds', each once, prefixed with the kinds
    that gave it unless every kind did. Call `check_feature_kinds` first."""
    arguments = {"handle": handle, "kinds": list(kinds), **overrides}
    payload = call_feature(ctx, "identify_features", ground_component.identify_features,
                           arguments)
    results = payload.get("results") or {}
    by_kind = {kind: list(result.get("warnings") or []) for kind, result in results.items()}
    return Outcome(
        section(OK, warnings=merged_warnings(by_kind, tuple(results)), handle=handle,
                kinds={kind: compact_kind(result) for kind, result in results.items()},
                detail_call=detail_call("identify_features", arguments)),
        staleness=payload.get("staleness"))


def compact_kind(result: dict) -> dict[str, Any]:
    """One kind's answer without its shapes: the count, the ids, the
    summary's single facts (a cliff pitch's tallest drop, whether drainages
    are truncated upstream) and the kind's warnings. The count is None when
    an input the kind reads was missing - a clearing with no vegetation
    panel, a route or way in with no osm.json - by the warning its detector
    declares for that (`terrain.features.input_missing`): the detector's 0
    there means nothing was looked at, and `features_facts` copies the
    count into a summary that keeps no warning beside it."""
    from ..terrain.features import input_missing

    shapes = result.get("shapes") or []
    summary = result.get("summary") or {}
    unknown = input_missing(result.get("kind"), result.get("warnings"))
    return {"count": None if unknown else summary.get("count", len(shapes)),
            "ids": [shape["id"] for shape in shapes],
            "summary": {key: value for key, value in summary.items()
                        if key != "count" and not isinstance(value, (list, tuple, dict))},
            "warnings": list(result.get("warnings") or [])}


def features_facts(features: dict) -> dict[str, int | None]:
    """From an answered `features_section`: how many of each kind - None
    for a kind whose input was missing (`compact_kind`), as the metrics'
    `ingress_points` is None on the same ground with no OpenStreetMap."""
    return {kind: entry["count"] for kind, entry in features["kinds"].items()}


# --- A candidate set as draft assignments -------------------------------------


def check_export_overrides(overrides: dict[str, Any]) -> None:
    """Refuse, before any work, an `overrides['export']` label prefix
    `caltopo_export` would refuse, or a blank resource type, which names no
    resource - it would draft "" as GROUND, and spaces as no type at all: a
    caller's mistake heard at once, not a warning after the grounds are
    staged. The prefix rule is the exporter's own
    (`planning.export.check_label_prefix`), which refuses a prefix that is
    not text itself; `check_overrides` has already refused a resource type
    that is not text."""
    try:
        if "label_prefix" in overrides:
            check_label_prefix(overrides["label_prefix"])
        if "resource_type" in overrides:
            check_resource_type(overrides["resource_type"])
    except ValueError as error:
        raise ValueError(f"overrides['export']: {error}") from error


def resource_type_warnings(ctx: ServerContext, resource_type) -> list[str]:
    """A warning when draft assignments carry a resource type the server's
    assignment corpus has no median for, else none - a typo such as DOGS
    for DOG, which CalTopo would show as written. Not refused: a real type
    nobody in the corpus used (OHV, say) is a fair one to draft in."""
    resource = (resource_type or "GROUND").strip().upper()
    if resource_median_ha(resource, ctx.norms) is not None:
        return []
    known = ", ".join((ctx.norms.get("area_ha_by_resource") or {}).get("medians") or {})
    return [f"the draft assignments carry resourceType {resource!r} as given, a type the "
            f"assignment corpus has no median for (it has one for {known or 'no type'}): "
            "check it is the type meant before importing them"]


def set_export_section(ctx: ServerContext, source_name: str, source: dict, *,
                       overrides: dict[str, Any],
                       settings: dict[str, Any] | None = None) -> Outcome:
    """`caltopo_export` of the candidate set another section generated -
    `source` is that section's dict, `source_name` its name - as DRAFT
    assignments with the search area (`include_area` True unless the
    overrides say otherwise), written by the component to
    `<staged dir>/exports/<set_id>.geojson`. Call `check_export_overrides`
    first. `settings` are keywords the scenario sets from its own arguments,
    such as plan_urban_search's `resource_type`. They are spread after the
    overrides, so they win as every keyword a helper sets does: held here,
    not left to each scenario, since an override that won would draft in
    the wrong resource type and say nothing. The scenario's `scenario_sets`
    should name each one too, so that `check_overrides` refuses it in the
    overrides with the argument to use, rather than leave it unused.

    An export failure is a warning, never a failed section - the sections'
    answers stand without the file - so every way this can draw nothing is
    an "ok" section with `path` null and a warning saying why: the source
    left out of `include`, the source failed (its own error says why), or
    `caltopo_export` raising (a disk that is full, a set gone from disk),
    which keeps the call as `detail_call` to make again. `overall_status`
    does not count this section either way (EXPORT_SECTION)."""
    if source["status"] == SKIPPED:
        return Outcome(section(OK, warnings=[
            f"{source_name} was left out of include, so there is no candidate set to draw "
            "as draft assignments"], path=None, set_id=None))
    if source["status"] != OK:
        return Outcome(section(OK, warnings=[
            f"the {source_name} section did not answer, so there is no candidate set to draw "
            "as draft assignments; its error says why"], path=None, set_id=None))
    arguments = {"handle": source["handle"], "set_id": source["set_id"], "include_area": True,
                 **overrides, **(settings or {})}
    try:
        payload = call_feature(ctx, "caltopo_export", candidates_component.caltopo_export,
                               arguments)
    except FeatureFailed as failure:
        return Outcome(section(OK, warnings=[
            f"the draft assignments could not be written ({error_text(ctx, failure.error)}); "
            "every section's answer stands, and detail_call makes the export again"],
            path=None, set_id=source["set_id"], detail_call=failure.call))
    # The drafts carry a type given them as given, so a typo is said here,
    # where the file is, as well as wherever it sized anything
    # (urban_assignments). Left to caltopo_export's own GROUND, nothing is.
    warnings = [*(payload.get("warnings") or []),
                *(resource_type_warnings(ctx, arguments["resource_type"])
                  if "resource_type" in arguments else [])]
    return Outcome(section(
        OK, warnings=warnings, path=payload["path"],
        set_id=payload.get("set_id"), assignments=payload.get("assignments"),
        features=payload.get("features"), labels=payload.get("labels"),
        area_included=payload.get("area_included"),
        detail_call=detail_call("caltopo_export", arguments)))


# --- The point itself, and the export file -----------------------------------


def point_marker(latitude: float, longitude: float, title: str, description: str) -> dict:
    """The scenario's own point as a CalTopo Marker, drawn as the medical
    tool draws its point: a blue dot."""
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {"class": "Marker", "title": title, "description": description,
                           "marker-symbol": ORIGIN_SYMBOL, "marker-color": ORIGIN_COLOR,
                           "marker-size": 1}}


def export_handle(staging: dict[str, dict[str, Any]],
                  preference: tuple[str, ...]) -> str | None:
    """The handle of the first ground in `preference` that was staged -
    the one a scenario's export file is written beside - or None."""
    return next((staging[name]["handle"] for name in preference
                 if name in staging and staging[name]["status"] == OK), None)


def write_export(ctx: ServerContext, staging: dict[str, dict[str, Any]],
                 preference: tuple[str, ...], filename: str,
                 features: list[dict], *, what: str = "the CalTopo export"
                 ) -> tuple[list[str], list[str]]:
    """(paths written, warnings). One GeoJSON FeatureCollection CalTopo
    imports, at `<staged dir>/exports/<filename>` beside the first ground in
    `preference` that was staged. Nothing is sent to CalTopo. A file that
    cannot be written is a warning, never a failed section: every section's
    answer stands without it. `what` names the file in that warning: a
    scenario that writes a second file says which one failed, since "the
    CalTopo export" beside the drafts' own warning reads as both."""
    handle = export_handle(staging, preference)
    if handle is None:
        return [], ["no ground was staged, so there is nowhere beside it to write the "
                    "CalTopo export; each section's detail_call returns what it found"]
    try:
        directory = ctx.staged(handle).directory
        path = directory / EXPORTS_DIRNAME / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        title = folder_title(Path(filename).stem.replace("_", " ").capitalize())
        atomic_write_text(path, json.dumps({"type": "FeatureCollection",
                                            "features": in_one_folder(features, title)},
                                           indent=1))
    except Exception as error:  # noqa: BLE001 - an export never fails a section
        return [], [f"{what} could not be written ({error_text(ctx, error)}); "
                    "every section's answer stands"]
    return [str(path)], []


def earlier_export_warnings(ctx: ServerContext, staging: dict[str, dict[str, Any]],
                            preference: tuple[str, ...], filename: str, *,
                            what: str) -> list[str]:
    """For a call that wrote no `filename`: a warning naming the one an
    earlier call left at `<staged dir>/exports/<filename>`, where this call
    would have written it, else none. The name is fixed, so a call that
    writes the file replaces it; one with nothing to put in it, or a disk
    that refused it, leaves the earlier call's as it was - beside this
    call's drafts, in the folder a planner may import whole - and what it
    holds, perhaps walked or asked for differently, would read as this
    answer's. It is named, not removed: the earlier answer gave that path,
    and it is still that answer's.

    Only a file whose write is this call's own last step, all or nothing
    (`write_export`), is named so: a null path for it is proof the call
    wrote none. Not the drafts `caltopo_export` leaves at `<set_id>.geojson`: it
    can raise after writing them, and this call's own file would then be
    called an earlier call's; and an earlier file under the same set id
    drew a set asked for with the same parameters on the same ground -
    most often the same lines, labelled or typed otherwise."""
    handle = export_handle(staging, preference)
    if handle is None:
        return []
    try:
        path = ctx.staged(handle).directory / EXPORTS_DIRNAME / filename
        if not path.is_file():
            return []
    except Exception:  # noqa: BLE001 - only a warning is at stake, never the answer
        return []
    return [f"{what} at {path} is an earlier call's, left as it was: this call wrote none, "
            "so it is not this answer's, and exports does not list it"]


def write_markers(ctx: ServerContext, staging: dict[str, dict[str, Any]],
                  outcomes: dict[str, Outcome], *, set_source: str,
                  sources: tuple[str, ...], export_overrides: dict[str, Any],
                  preference: tuple[str, ...], filename: str, point: dict
                  ) -> tuple[list[str], list[str]]:
    """The second file of a scenario whose `export` section drafts a
    candidate set (`set_export_section` of `set_source`'s set): what the
    `sources` sections found that the drafts do not hold - parking,
    landing zones, hospitals - after the scenario's own `point`, written
    with `write_export` and noted on the export section as `markers_path`
    and `markers` (a count by CalTopo class). Returns (every file the export
    wrote, the draft assignments first; the scenario's export warnings).

    The warnings are said apart because they are different news. Nothing
    found is the search's answer; a file not written is the disk's or the
    ground's, and the sections still hold everything found. Either file
    missing is said, naming which - one file in `exports` does not show
    the other was meant, and in plan_urban_search the drafts are the
    answer itself. An earlier call's markers file still where this call
    wrote none is named (`earlier_export_warnings`). And export overrides
    the set's absence left unused are said, as any section's are when it
    is left out."""
    export = outcomes[EXPORT_SECTION].section
    source = outcomes[set_source].section
    warnings = []
    if export_overrides and source["status"] != OK:
        why = (f"{set_source} was left out of include" if source["status"] == SKIPPED
               else f"the {set_source} section did not answer")
        warnings.append(f"overrides['export'] was not used: {why}, so there was no "
                        "candidate set for caltopo_export to draw")
    drew = [name for name in sources if outcomes[name].features]
    found = [feature for name in drew for feature in outcomes[name].features]
    export["markers_path"], export["markers"] = None, {}
    if found:
        paths, problems = write_export(ctx, staging, preference, filename, [point, *found],
                                       what=f"the markers file {filename}")
        export["warnings"].extend(problems)
        if paths:
            export["markers_path"] = paths[0]
            for feature in [point, *found]:
                kind = feature["properties"]["class"]
                export["markers"][kind] = export["markers"].get(kind, 0) + 1
    exports = [path for path in (export.get("path"), export["markers_path"]) if path]
    # An export section that raised says why in its error, not its warnings
    # (`guard`); a markers file refused beside it still says why in its
    # warnings. Pointing at an empty list would read as no reason at all.
    if export["status"] == OK:
        see = "sections.export.warnings says why"
    elif export["warnings"]:
        see = "sections.export.error and sections.export.warnings say why"
    else:
        see = "sections.export.error says why"
    see += ", and every section's answer stands"
    if not found and source["status"] != OK:
        warnings.append("nothing was found to draw, so no CalTopo export was written")
    elif not exports:
        warnings.append("no CalTopo export was written, although the sections found things "
                        f"to draw: {see}")
    elif source["status"] == OK and not export.get("path"):
        warnings.append(f"the draft assignments of the {set_source} set were not written - "
                        f"only the markers file {filename} was: {see}")
    elif found and export["markers_path"] is None:
        names = ", ".join(drew[:-1]) + (" and " if len(drew) > 1 else "") + drew[-1]
        warnings.append(f"the markers file {filename} was not written - only the draft "
                        f"assignments were - so what {names} found is in no CalTopo file: "
                        f"{see}")
    if export["markers_path"] is None:
        warnings.extend(earlier_export_warnings(ctx, staging, preference, filename,
                                                what="the markers file"))
    return exports, warnings
