"""Scenario: a patient at a point - where a helicopter can land, the way out
on the ground, and the nearest hospitals, in one call."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ._sections import (
    LZ_RESOLUTION_M,
    WALKING_RESOLUTION_M,
    Ground,
    check_egress_overrides,
    check_limit,
    check_overrides,
    check_per_category,
    check_point,
    check_radius,
    earlier_export_warnings,
    egress_facts,
    egress_section,
    ground_handle,
    guard,
    lz_facts,
    lz_section,
    medical_facts,
    medical_section,
    overall_status,
    point_marker,
    requested,
    skipped,
    stage_grounds,
    summarise,
    unused_overrides,
    write_export,
)

#: The sections, in the order they are answered and reported. A type, so the
#: three names reach the caller as an enum in the schema.
MedevacSection = Literal["lz", "egress", "medical"]
SECTIONS: tuple[str, ...] = ("lz", "egress", "medical")
EgressMode = Literal["walk_out", "carry_out"]
EGRESS_MODES: tuple[str, ...] = ("carry_out", "walk_out")

# The landing-zone ground is staged at LZ_RESOLUTION_M (2 m): 1,000 m round
# the patient plus the 1,000 m planning buffer is 4,000 m a side, 2,000 px.
# The exit ground at WALKING_RESOLUTION_M (10 m): 3,000 m round the patient
# plus the buffer is 8,000 m a side, 800 cells.

#: The default modes, carry-out first: a medevac's patient is usually
#: carried. One list both the function and the wrapper default to, never
#: changed in place.
DEFAULT_EGRESS_MODES: list = list(EGRESS_MODES)

#: What plan_medevac sets in each feature call from its own arguments, so an
#: override may not: {section: {keyword: the argument to use instead}}.
SCENARIO_SETS = {"lz": {"limit": "lz_limit"}, "egress": {"mode": "egress_modes"},
                 "medical": {"per_category": "medical_per_category"}}

#: What plan_medevac passes lz_candidates unless an override says otherwise:
#: the whole staged ground is graded, the planning buffer included, so a
#: landing zone just past `lz_radius_m` is not missed (a test run,
#: 2026-10-24: a marginal zone on the exit trail 1,290 m out was left out of
#: a 1,000 m circle that had been staged 2,000 m out).
LZ_DEFAULTS = {"whole_ground": True}

#: The landing-zone ground first: it is the one a planner opens to look at.
EXPORT_GROUNDS = ("lz", "egress")
EXPORT_FILENAME = "medevac.geojson"
#: The exit ground staged again, re-centred along a trail that ran off the
#: first one (`egress_section`'s `restage`), as `staging` names it.
RECENTRED_GROUND = "egress_recentred"
#: Which ground each section stands on, as the section and `staging` name it.
SECTION_GROUND = {"lz": "lz", "egress": "egress"}

NOTE = ("No verdict: this lays out what lz_candidates, egress_routes and "
        "nearest_medical_facilities found and chooses no landing zone, exit or hospital. "
        "Landing zones are graded from rasters - the pilot decides; exits are walked over "
        "a model - the team leader decides; hospitals are as HIFLD last validated them - "
        "confirm with the receiving facility and follow local EMS destination policy.")


def plan_medevac(ctx: ServerContext, latitude: float, longitude: float,
                 map_id: str = "", public: bool = True,
                 include: list[MedevacSection] | None = None,
                 lz_radius_m: float = 1000.0, egress_radius_m: float = 3000.0,
                 egress_modes: list[EgressMode] = DEFAULT_EGRESS_MODES,
                 lz_limit: int = 5, medical_per_category: int = 3,
                 overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The `plan_medevac` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    # Every refusal first, before anything is staged or fetched: a caller's
    # mistake is theirs to hear at once, not a failed section after minutes
    # of staging.
    check_point(latitude, longitude)
    asked = requested(include, SECTIONS)
    # Every argument is checked whether or not its section runs - a wrong
    # value is wrong either way - and each radius exactly as `prepare_area`
    # would check it round this point, at its ground's resolution.
    check_radius("lz_radius_m", lz_radius_m, latitude=latitude, longitude=longitude,
                 resolution_m=LZ_RESOLUTION_M)
    check_radius("egress_radius_m", egress_radius_m, latitude=latitude,
                 longitude=longitude, resolution_m=WALKING_RESOLUTION_M)
    modes = _modes(egress_modes, egress_asked="egress" in asked)
    check_limit("lz_limit", lz_limit, "zone")
    check_per_category(medical_per_category)
    keywords = check_overrides("plan_medevac", overrides, SECTIONS, SCENARIO_SETS)
    if "egress" in asked:
        check_egress_overrides(modes, keywords["egress"])

    grounds = {}
    if "lz" in asked:
        grounds["lz"] = Ground(lz_radius_m, LZ_RESOLUTION_M)
    if "egress" in asked:
        grounds["egress"] = Ground(egress_radius_m, WALKING_RESOLUTION_M)
    staging = stage_grounds(ctx, grounds, latitude=latitude, longitude=longitude,
                            map_id=map_id, public=public)

    def restage_egress(centre_latitude: float, centre_longitude: float) -> str:
        """The exit ground again, its size unchanged, centred along the trail
        that ran off it; recorded in `staging` beside the first."""
        staging.update(stage_grounds(ctx, {RECENTRED_GROUND: grounds["egress"]},
                                     latitude=centre_latitude, longitude=centre_longitude,
                                     map_id=map_id, public=public))
        return ground_handle(staging, RECENTRED_GROUND)

    outcomes = {
        "lz": guard(ctx, lambda: lz_section(
            ctx, ground_handle(staging, "lz"), latitude=latitude, longitude=longitude,
            limit=lz_limit, overrides={**LZ_DEFAULTS, **keywords["lz"]},
            point_name="the patient")) if "lz" in asked else skipped(),
        "egress": guard(ctx, lambda: egress_section(
            ctx, ground_handle(staging, "egress"), latitude=latitude, longitude=longitude,
            modes=modes, overrides=keywords["egress"],
            restage=restage_egress)) if "egress" in asked else skipped(),
        "medical": guard(ctx, lambda: medical_section(
            ctx, latitude=latitude, longitude=longitude,
            per_category=medical_per_category,
            overrides=keywords["medical"])) if "medical" in asked else skipped(),
    }
    sections = {name: outcome.section for name, outcome in outcomes.items()}
    for name, ground in SECTION_GROUND.items():
        if name in asked:
            sections[name]["ground"] = ground

    warnings = unused_overrides(keywords, asked)
    found = [feature for outcome in outcomes.values() for feature in outcome.features]
    exports: list[str] = []
    if found:
        # Only what is true of every line in the file: the medical section's
        # notes say how the hospitals were ranked - by straight-line
        # distance when no road time is known - and where the road times
        # start when that is some way off.
        patient = point_marker(
            latitude, longitude, "Patient",
            "\n".join(["Where the patient is. Landing-zone distances and bearings, and "
                       "every exit route, start here.",
                       *(note for outcome in outcomes.values()
                         for note in outcome.point_notes),
                       "From plan_medevac; nothing in this file is a decision."]))
        exports, export_warnings = write_export(ctx, staging, EXPORT_GROUNDS,
                                                EXPORT_FILENAME, [patient, *found])
        warnings.extend(export_warnings)
    else:
        warnings.append("nothing was found to draw, so no CalTopo export was written")
    if not exports:
        # The name is fixed, so an earlier call's file is still there when
        # this one wrote none, and would read as this answer's.
        warnings.extend(earlier_export_warnings(ctx, staging, EXPORT_GROUNDS,
                                                EXPORT_FILENAME, what="the CalTopo export"))

    return {
        "status": overall_status(sections),
        "patient": {"latitude": latitude, "longitude": longitude},
        # Egress is read in part too: a walk-out that answered while the
        # carry-out raised is still a way out to report.
        "summary": summarise(sections, asked, FACTS, in_part=("egress",)),
        "sections": sections,
        "staging": staging,
        "handles": {name: record["handle"] for name, record in staging.items()},
        "exports": exports,
        "warnings": warnings,
        "note": NOTE,
    }


def _modes(egress_modes, *, egress_asked: bool) -> tuple[str, ...]:
    """The modes asked for, each once, in the order given. A direct caller is
    not held to the schema's enum, so an unknown mode is refused here rather
    than failing the egress section after its ground is staged. An empty
    list is refused only when egress is asked for: with egress left out of
    `include`, "no modes" and "no egress" say the same thing."""
    if isinstance(egress_modes, str):
        raise ValueError(f"egress_modes must be a list, such as [{egress_modes!r}]")
    modes = tuple(dict.fromkeys(egress_modes or ()))
    unknown = sorted(set(modes) - set(EGRESS_MODES))
    if unknown:
        raise ValueError(f"egress_modes names {unknown}; the modes are {list(EGRESS_MODES)}")
    if not modes and egress_asked:
        raise ValueError("egress_modes is empty, but egress is asked for: name carry_out, "
                         "walk_out or both, or leave egress out of include")
    return modes


#: Each section's top-line facts, read off its answered section (`summarise`).
FACTS = {"lz": lz_facts, "egress": egress_facts, "medical": medical_facts}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="plan_medevac", tags={"scenario"})
    def _plan_medevac_tool(latitude: float, longitude: float,
                           map_id: str = "", public: bool = True,
                           include: list[MedevacSection] | None = None,
                           lz_radius_m: float = 1000.0, egress_radius_m: float = 3000.0,
                           egress_modes: list[EgressMode] = DEFAULT_EGRESS_MODES,
                           lz_limit: int = 5, medical_per_category: int = 3,
                           overrides: dict[str, dict[str, Any]] | None = None
                           ) -> dict[str, Any]:
        """A patient at a point - a subject found, or a searcher hurt - and the
        three things a medevac turns on, in one call: where a helicopter can
        put down near them, the way out on the ground, and the nearest
        hospitals. It runs `lz_candidates`, `egress_routes` and
        `nearest_medical_facilities` from the patient and returns their facts
        summarised, one section each. Nothing is decided.

        WHAT IT STAGES, AND WHAT THAT COSTS. Two grounds through
        `prepare_area`, both centred on the patient, with no credentials
        (elevation, vegetation and roads are public): for `lz`, `lz_radius_m`
        (default 1,000 m) at `resolution_m=2.0`, because lz_candidates will
        not grade coarser than 3 m a pixel - with the 1,000 m planning buffer
        4,000 m a side; for `egress`, `egress_radius_m` (default 3,000 m) at
        `resolution_m=10.0`, because egress_routes visits every cell and
        refuses ground over 1,800 x 1,800 cells - 8,000 m a side is 800 cells.
        Either radius must be under 4,500 m: `prepare_area` stages at most
        9,000 m across, measured on its own circle, which puts 4,500 m itself
        just over (and 4,499.9 m over near the poles); the radius is checked
        exactly as `prepare_area` checks it, before anything is staged.
        Measured on 2026-10-22, a first call on new ground took three and a
        half to five minutes: one to two minutes to stage each ground, up to
        one more for the landing zones' imagery, under one for the hospitals.
        The handle is a hash of the request, so ground already staged is
        reused for the cost of a file read, and a second call on the same
        patient took a minute or so, most of it the hospital lookup. A section
        left out of `include` stages nothing. `map_id` and `public` are
        optional map context, passed to staging as `prepare_area` takes them.
        Medical needs no ground; it asks HIFLD, OSRM and OpenStreetMap over the
        network. Landing zones also fetch CalTopo imagery panels and chips when
        the server has a WMS token, and are refused without one unless the
        ground carries a canopy panel.

        EVERY SECTION has `status` - "ok", "failed", or "skipped" when left out
        of `include` - `error` (null when ok, else "<ExceptionType>: <message>"),
        `warnings` (the feature's own, verbatim; egress merges its modes', see
        below) and, once it has answered, its compact facts. It also has a
        `detail_call` - `{tool, arguments}`, the exact feature call that
        returns the full answer, geometry included (for egress, one per mode).
        A section whose feature call raised keeps that call's `detail_call`, to
        retry it or make it directly; one whose ground could not be staged
        never made a call and has none - its error names the ground, and that
        ground's `staging` record carries the `prepare_area` call. A section
        standing on a ground names it in `ground` (`lz` or `egress`, as
        `staging` names them). Points come back; outlines and routes do not.

        `lz` - landing zones nearest the patient first, `lz_limit` of them
        (default 5; 0 for every one), on the ground in `handle`, graded over
        the WHOLE staged ground: `lz_radius_m` plus the 1,000 m planning
        buffer, so 2,000 m out by default (lz_candidates' `whole_ground`; an
        override of `{"lz": {"whole_ground": false}}` keeps to the circle).
        Each has `id`,
        `grade` (preferred, minimal or marginal, as lz_candidates grades them),
        the `touchdown` Point, `clear_diameter_m`, `slope_mean_deg` and
        `slope_max_deg`, `canopy_fraction`, `distance_m` and `bearing_deg` from
        the patient, `powerline`, `road` and `parking_lot` - each a fact, null,
        or the string "unknown", read exactly as lz_candidates says: null is
        "nothing mapped that close", never "nothing there" - `chips` (imagery
        squares; show each one's `attribution` beside it) and `chip_path`, and
        its own `warnings`. `grades` counts every grade BEFORE the limit;
        `total_before_limit` and `truncated_to` say what the limit dropped.
        `checks` says what was and was NOT checked - structures, trees, which
        water sources, which power-line kinds, road class. The patch outline is
        left out: `detail_call` returns it.

        `egress` - `egress_routes` from the patient once per mode in
        `egress_modes` (default carry_out then walk_out), under `modes`. Each
        mode has its own `status`, `error` and `detail_call`, and both exits,
        2WD first: a reachable one with `minutes`, `distance_m`, `gain_m`,
        `loss_m`, the `exit_point` Point, `road_name` (null when the exit lies
        on no named road, as an exit in a lot may), `road_access` (the road's
        OpenStreetMap access, such as "private"; null when untagged or
        unknown), `gate_on_road` (true when a gate stands on the exit's road
        itself; null - never false - on ground too old to say), `parking_lot`
        ({id, off} for a lot at the exit, `off` the road it opens onto; null,
        or "unknown" on ground too old to say), `gates_crossed` (a count; on
        ground that predates gates it is 0 and a warning says gates are
        unknown), `off_trail_m` and `on_trail_share` (how much of the route
        is off any mapped way - once on a trail a route stays on the trails,
        so this is mostly the way from the patient to the first one),
        `off_trail_after_joining_m` (the part of that after joining a trail),
        `steep_pitch_m` (for a carry, metres of trail steeper than the
        litter's limit, which the team belays rather than leave the trail),
        `litter_start` (null, or - when the patient's own spot is too steep
        to carry off at all - where the carry starts instead, its distance,
        bearing and drop from the patient; getting there needs a rope
        evacuation, which no minutes count) and its `warnings`; an
        unreachable one with
        `reachable: false` and the feature's `reason`. Each mode also has
        `trail_beyond` (egress_routes' own: where the trail ran off the ground
        when no exit here kept to one, else null) and `recentred`.

        A TRAIL THAT RUNS OFF THE EXIT GROUND IS FOLLOWED. When a mode's
        answer says its trail runs off the ground while no exit is reached
        without leaving a trail, the exit ground is staged once more at the
        same radius and resolution, centred where that answer says (along the
        trail, the patient kept 500 m inside), as `staging.egress_recentred`
        - a minute or two more on new ground - and each such mode is asked
        again there. The second answer is kept when its best exit leaves a
        trail for less (then its other exit); the mode's `recentred` then
        gives {handle, center, because}, its `detail_call` names the new
        handle, and a warning says so. Otherwise the first answer stands and
        a warning says the re-centred ground was asked and why it was not
        used. Once only: a trail that runs off the second ground too is said
        in its `trail_beyond`. The section's `handle` stays the first
        ground's. The section is "failed"
        when any mode failed, and the modes that answered are still there, so
        the overall status reads partial, never failed, while one did. A
        section warning not every mode asked for gave - a mode that failed gave
        none - is prefixed with the names of the modes that gave it. Route
        lines are left out.

        `medical` - `nearest_medical_facilities` at the patient. `categories`
        holds medical_center, trauma_center and pediatric_trauma_center, each
        up to `medical_per_category` hospitals (default 3, at most 10) ranked
        by road time - a hospital with no road time (the road router could not
        be reached, or found no road to it) after every routed one, by
        straight-line distance, as a warning says:
        `rank`, `name`, `trauma` (the level as a person says it),
        `helipad` (Y, N or unknown), `road_minutes`, `air_minutes` (only with a
        helipad), `road_distance_m`, `phone`, `latitude` and `longitude`. An
        empty category found nothing within the search radius, and a warning
        says so. `table` is the feature's Markdown table verbatim, to show as
        it is; `in_us` is false outside the US, where the list has no
        hospitals (null when that could not be checked); `road_start` is
        where the road time begins.

        OVERALL `status`: "ok" when every section asked for is ok, "failed"
        when every one failed with nothing to show, otherwise "partial" - an
        egress section with one mode answered counts as something to show.
        Skipped sections count neither way. A failed section never stops the
        others: a hospital lookup with no network still leaves the landing
        zones and the exits, and ground that could not be staged fails only
        the section standing on it, its error saying which ground and why.

        `summary` - the facts to read first, one block per section asked for,
        null when that section did not answer. `lz.nearest_any` and
        `lz.nearest_preferred` (id, grade, distance, bearing, clear diameter).
        lz_candidates orders by distance before grade, so on most ground the
        nearest few zones are minimal or marginal and the nearest preferred
        one lies past `lz_limit`: then `nearest_preferred` is null,
        `preferred_beyond_limit` counts the preferred zones the limit left
        out, and `preferred_detail_call` is the lz_candidates call that lists
        every zone, those included - it cuts an imagery chip for each, so it
        costs more than this call did - or call again with a larger
        `lz_limit`. Null `nearest_preferred` with `preferred_beyond_limit` 0
        is no preferred zone on the ground at all. `egress.<mode>`: that
        mode's quickest reachable exit - kind, minutes, distance, metres
        off trail (`off_trail_m`), `litter_start` when the carry starts away
        from the patient, and its
        vehicle end as the section gives it (`road_name`, `road_access`,
        `gate_on_road`, `parking_lot`), so a gated or private quickest exit
        says so here too where the ground can tell; a null reads as it does
        in the section, and a null `road_access` is never "public" - the
        string "none reachable" when neither could be
        reached, or null when that mode failed and another answered. Its
        gates crossed are left to the section, beside the warning that says
        when gates are unknown. `medical.nearest_trauma_center` (name,
        trauma level, helipad, road minutes, air minutes, and `ranked_by`:
        "road_time", or "straight_line" when no trauma center found has a
        road time - the road router could not be reached or found no road -
        so it is the nearest as the crow flies, which may not be the quickest
        by road), null when none was found.

        `staging` - one record per ground: `status`, `error`, `handle`,
        `radius_m` and `resolution_m` asked for, `dem_resolution_m`, `staged`
        (false when it was already on disk), `osm_available`, `warnings`, and
        the `prepare_area` call as `detail_call`. `handles` maps each ground to
        its handle (null when it could not be staged), for any tool that takes
        one: lz_candidates with other limits, site_facts on a touchdown, and
        on the exit ground (it is at 10 m) egress_routes with other factors
        or travel_time_rings from the patient.

        `exports` - the path of one GeoJSON file CalTopo imports (Import, choose
        the file): the patient as a blue dot, each landing zone as a Marker at
        its touchdown (green preferred, yellow minimal, orange marginal), each
        reachable exit's route as a line (purple carry-out, blue walk-out), and
        each hospital as the medical tool's red first-aid cross. Each line
        says how many metres of it are off any mapped way, and a carry that
        starts away from the patient says where and that it needs a rope
        evacuation to get there. Each marker's
        description carries its facts and what was NOT checked there. Each
        line's says where the vehicle meets the team - the exit road (an
        unnamed one by its kind), or the lot and the road it opens onto - the
        road's access as OpenStreetMap tags it, or that it is not tagged, and
        its gates: those crossed on the way; a gate on the stretch of the exit
        road the exit lies on (one OpenStreetMap way, within the staged
        ground), or none mapped there; what was not checked - the road in
        beyond that stretch, or the road into a lot; or unknown where the
        ground cannot say, never left out to read as none. An exit at the
        patient itself has no line: what its line would say goes on the
        patient's marker, which also says where the hospital road times
        start when that is over 200 m away. Written to
        `exports/medevac.geojson` beside the landing-zone ground, or beside
        the egress ground when that is the one staged; a later call on the
        same ground replaces it, and one that writes none leaves the earlier
        call's as it was, with a warning naming it as an earlier call's, not
        this answer's. Nothing is sent to CalTopo. A file that
        cannot be written is a warning, never a failed section; with no ground
        staged (medical alone) `exports` is empty and a warning says why.

        `overrides` - keyword arguments for the feature behind a section, by
        section: `{"lz": {"max_slope_deg": 6.0}, "medical": {"search_radius_km":
        200}}`, passed to every call that section makes. An egress override -
        `on_way_factor`, `off_way_factor`, `max_grade` - needs one mode in
        `egress_modes`: each has a different default per mode (a litter's
        40 % grade limit, none on foot), so one value for both is refused.
        Refused before anything is staged: a section this tool does not have;
        a keyword the feature does not take, the refusal listing what it does;
        and a keyword plan_medevac sets itself - the handle, the centre, and
        `limit`, `mode` and `per_category`, which are `lz_limit`,
        `egress_modes` and `medical_per_category` here. Bad coordinates, an
        unknown or empty `include`, a radius `prepare_area` would refuse, an
        unknown mode and the like are refused the same way, before any work.
        Each override value is read as the feature tool reads it: one the
        tool's schema converts is passed on converted - "6" as a number,
        "false" as false, never true - and one it refuses (text for a
        number, a null where a number goes) or its own checks would refuse
        (a `max_slope_deg` above `marginal_slope_deg`, a negative
        `max_grade`, a `search_radius_km` past 400) is refused the same way,
        naming the keyword. So is a number that is not finite ("nan",
        "inf"), though the tool would take it: no range check can be trusted
        to catch it.

        `patient` echoes the point the call was made for. `warnings` are the
        call's own, beside each section's: an override given for a section
        left out of `include` (accepted, but no call took it), and why no
        export was written. `note` says in a few sentences what NO VERDICT
        below says, to show beside any answer.

        NO VERDICT. This lays out what the three tools found. It chooses no
        landing zone, exit or hospital, and orders nothing beyond the tools'
        own orders - nearest first, quickest first. Landing zones are a filter
        on elevation and imagery rasters, not a site survey: the pilot decides.
        Exits are walked over a model that does not see fences or trails OSM
        lacks: the team leader decides. Hospitals are as HIFLD last validated
        them, with diversion unknown: confirm with the receiving facility and
        follow local EMS destination policy. Drill down with each section's
        `detail_call`.
        """
        return plan_medevac(ctx, latitude=latitude, longitude=longitude, map_id=map_id,
                            public=public, include=include, lz_radius_m=lz_radius_m,
                            egress_radius_m=egress_radius_m, egress_modes=egress_modes,
                            lz_limit=lz_limit, medical_per_category=medical_per_category,
                            overrides=overrides)
