"""Scenario: an urban or suburban search round a place last seen - the
populated blocks grouped into assignments of about one size, where to park
and stage, and the nearest hospitals, in one call, drafted for CalTopo."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..planning.area import RadiusUnit, radius_in_metres
from ._sections import (
    Ground,
    check_export_overrides,
    check_limit,
    check_overrides,
    check_per_category,
    check_point,
    check_radius,
    check_resource_type,
    check_target_acres,
    check_urban_overrides,
    check_urban_radius,
    ground_handle,
    guard,
    medical_facts,
    medical_section,
    overall_status,
    parking_facts,
    parking_section,
    point_marker,
    requested,
    set_export_section,
    skipped,
    small_radius_warnings,
    stage_grounds,
    summarise,
    unused_overrides,
    urban_facts,
    urban_section,
    write_markers,
)

#: The sections, in the order they are answered and reported. A type, so the
#: four names reach the caller as an enum in the schema.
UrbanSearchSection = Literal["urban", "parking", "medical", "export"]
SECTIONS: tuple[str, ...] = ("urban", "parking", "medical", "export")

#: The search area when no radius is given: half a mile, as urban_assignments
#: suggests - a few dozen blocks in a suburb, a handful of assignments. Its
#: circle, 1.6 km across, is read for lot lines at 1 m, well inside the
#: 4,096 pixels a side populated_areas reads before it coarsens; with the
#: 1,000 m planning buffer the staged ground is about 3.6 km a side.
#: Converted as `radius=0.5, radius_unit="mi"` is, so the two stage one ground.
DEFAULT_SEARCH_RADIUS_M = radius_in_metres(radius=0.5, radius_unit="mi")

#: What plan_urban_search sets in each feature call from its own arguments,
#: so an override may not: {section: {keyword: the argument}}. The urban
#: entry covers both tools the section calls; the export's two keep the
#: drafts' resource type and house markers those of the assignments.
SCENARIO_SETS = {"urban": {"target_acres": "target_acres", "resource_type": "resource_type",
                           "building_markers": "building_markers"},
                 "parking": {"limit": "parking_limit"},
                 "medical": {"per_category": "medical_per_category"},
                 "export": {"resource_type": "resource_type",
                            "include_markers": "building_markers"}}

#: The one ground: the draft assignments and the markers are written beside it.
EXPORT_GROUNDS = ("search",)
EXPORT_FILENAME = "urban_search.geojson"
#: Which ground each section stands on, as the section and `staging` name it.
SECTION_GROUND = {"urban": "search", "parking": "search"}

NOTE = ("No verdict: this lays out what populated_areas, urban_assignments, "
        "parking_candidates and nearest_medical_facilities found, and chooses no "
        "assignment, staging area or hospital. Assignments are ordered by distance from the "
        "centre, a fact and not a priority - which blocks are searched, by whom and in what "
        "order is the planner's call; they are a first cut from CalTopo's lot lines and "
        "building footprints and OpenStreetMap's streets, blind to an undrawn house, a gate "
        "or a yard; hospitals are as HIFLD last validated them - confirm with the receiving "
        "facility. The draft assignments are drafts: check each one before issuing it.")


def plan_urban_search(ctx: ServerContext, latitude: float, longitude: float,
                      radius: float | None = None, radius_unit: RadiusUnit = "m",
                      radius_m: float | None = None, target_acres: float | None = None,
                      resource_type: str = "GROUND", building_markers: bool = False,
                      map_id: str = "", public: bool = True,
                      include: list[UrbanSearchSection] | None = None,
                      parking_limit: int = 5, medical_per_category: int = 3,
                      overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The `plan_urban_search` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    # Every refusal first, before anything is staged or fetched: a caller's
    # mistake is theirs to hear at once, not a failed section after a minute
    # of staging. Every argument is checked whether or not its section runs.
    check_point(latitude, longitude)
    asked = requested(include, SECTIONS)
    search_radius = radius_in_metres(radius_m, radius, radius_unit)
    search_radius = DEFAULT_SEARCH_RADIUS_M if search_radius is None else search_radius
    check_radius("radius", search_radius, latitude=latitude, longitude=longitude)
    check_target_acres(target_acres)
    check_resource_type(resource_type)
    if not isinstance(building_markers, bool):
        # It names a different set and draws a marker on every roof: a
        # "false" string must not read as asking for them.
        raise ValueError(f"building_markers must be true or false; got {building_markers!r}")
    check_limit("parking_limit", parking_limit, "site")
    check_per_category(medical_per_category)
    keywords = check_overrides("plan_urban_search", overrides, SECTIONS, SCENARIO_SETS)
    check_urban_overrides(keywords["urban"], max_distance_m=search_radius,
                          building_markers=building_markers)
    check_export_overrides(keywords["export"])
    # Lot lines are read over the search circle itself, and populated_areas
    # refuses one over about 8 km across - a radius prepare_area still
    # stages. Checked only when the blocks are asked for: parking and the
    # hospitals stand on any radius prepare_area takes.
    if "urban" in asked:
        check_urban_radius("radius", search_radius, latitude=latitude, longitude=longitude)

    grounds = {}
    if "urban" in asked or "parking" in asked:
        grounds["search"] = Ground(search_radius)
    staging = stage_grounds(ctx, grounds, latitude=latitude, longitude=longitude,
                            map_id=map_id, public=public)

    outcomes = {
        "urban": guard(ctx, lambda: urban_section(
            ctx, ground_handle(staging, "search"), radius_m=search_radius,
            target_acres=target_acres, resource_type=resource_type,
            building_markers=building_markers,
            overrides=keywords["urban"])) if "urban" in asked else skipped(),
        "parking": guard(ctx, lambda: parking_section(
            ctx, ground_handle(staging, "search"), limit=parking_limit,
            overrides=keywords["parking"],
            point_name="the place last seen")) if "parking" in asked else skipped(),
        "medical": guard(ctx, lambda: medical_section(
            ctx, latitude=latitude, longitude=longitude, per_category=medical_per_category,
            overrides=keywords["medical"])) if "medical" in asked else skipped(),
    }
    # After urban, whose set it draws: an urban section that failed or was
    # left out leaves it nothing to draft but the markers, and it says so.
    outcomes["export"] = guard(ctx, lambda: set_export_section(
        ctx, "urban", outcomes["urban"].section, overrides=keywords["export"],
        settings={"resource_type": resource_type, "include_markers": building_markers})
    ) if "export" in asked else skipped()
    sections = {name: outcome.section for name, outcome in outcomes.items()}
    for name, ground in SECTION_GROUND.items():
        if name in asked:
            sections[name]["ground"] = ground

    # Under 50 m the circle holds a lot or two (`small_radius_warnings`).
    warnings = small_radius_warnings(
        search_radius, what="a lot or two across rather than a neighbourhood")
    warnings.extend(unused_overrides(keywords, asked))
    exports: list[str] = []
    if "export" in asked:
        # The place says only what is true of every line in its file: the
        # medical section's notes say how the hospitals were ranked - by
        # straight-line distance when no road time is known - and where the
        # road times start when that is some way off.
        place = point_marker(
            latitude, longitude, "Place last seen",
            "\n".join(["Where the subject was last seen, the centre of the search circle. "
                       "Parking distances and bearings, and the draft assignments' "
                       "distances, are measured from here.",
                       *(note for outcome in outcomes.values() for note in outcome.point_notes),
                       "From plan_urban_search; nothing in this file is a decision."]))
        exports, export_warnings = write_markers(
            ctx, staging, outcomes, set_source="urban", sources=("parking", "medical"),
            export_overrides=keywords["export"], preference=EXPORT_GROUNDS,
            filename=EXPORT_FILENAME, point=place)
        warnings.extend(export_warnings)

    return {
        "status": overall_status(sections),
        "place_last_seen": {"latitude": latitude, "longitude": longitude},
        "radius_m": search_radius,
        "summary": summarise(sections, asked, FACTS),
        "sections": sections,
        "staging": staging,
        "handles": {name: record["handle"] for name, record in staging.items()},
        "exports": exports,
        "warnings": warnings,
        "note": NOTE,
    }


#: Each section's top-line facts, read off its answered section (`summarise`).
#: The export has none: its paths are in `exports`. Not urban in part: an
#: urban section that failed after its blocks has no assignments to count.
FACTS = {"urban": urban_facts, "parking": parking_facts, "medical": medical_facts}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="plan_urban_search", tags={"scenario"})
    def _plan_urban_search_tool(latitude: float, longitude: float,
                                radius: float | None = None, radius_unit: RadiusUnit = "m",
                                radius_m: float | None = None,
                                target_acres: float | None = None,
                                resource_type: str = "GROUND",
                                building_markers: bool = False,
                                map_id: str = "", public: bool = True,
                                include: list[UrbanSearchSection] | None = None,
                                parking_limit: int = 5, medical_per_category: int = 3,
                                overrides: dict[str, dict[str, Any]] | None = None
                                ) -> dict[str, Any]:
        """An urban or suburban search round a place last seen, in one call:
        the populated blocks grouped into assignments of about one size,
        where to park and stage, and the nearest hospitals - with the
        assignments drafted as CalTopo assignments. It runs `populated_areas`
        and `urban_assignments` (the `urban` section), `parking_candidates`
        and `nearest_medical_facilities` from the place last seen, and
        `caltopo_export` on the assignments, and returns their facts
        summarised, one section each. Nothing is decided. For town only:
        rural ground has big lots and yields no blocks.

        THE SEARCH AREA is a circle round the place last seen: `radius` with
        `radius_unit` ("m", "km", "ft", "yd" or "mi", as `prepare_area` takes
        them, so half a mile is `radius=0.5, radius_unit="mi"`), or
        `radius_m`; give one form, not both. `radius` is read in metres
        unless `radius_unit` says otherwise, so `radius=0.5` alone is half a
        metre: a search radius under 50 m - a lot or two, not a
        neighbourhood - is answered with a top-level warning saying so.
        Default half a mile (804.672 m), as urban_assignments suggests: a
        few dozen blocks in a suburb, a handful of assignments. It must be
        under 4,500 m, the most
        `prepare_area` stages; and, with `urban` asked for, under about
        4,035 m: populated_areas reads lot lines at 1 m, coarsening to no
        worse than 2 m to stay within 4,096 pixels a side, so it refuses a
        circle over about 8 km across. Both limits are checked exactly as the
        tools check them, before anything is staged.

        WHAT IT STAGES, AND WHAT THAT COSTS. One ground through
        `prepare_area`, centred on the place last seen, at its default
        raster, with no credentials (elevation, vegetation and roads are
        public): the blocks need no finer, since lot lines and roofs are read
        from CalTopo's own panels at 1 m, and parking is read from the staged
        OpenStreetMap. Half a mile plus the 1,000 m planning buffer is about
        3.6 km a side. Measured on 2026-10-22 over San Carlos, CA: at a
        quarter mile staging took about 45 seconds, the blocks about 11 (most
        of it fetching CalTopo's parcel and structures panels) and the
        assignments about 1, reading those panels back from the cache; at
        the default half mile the blocks took about 19 seconds and the
        assignments 9; the hospital lookup took 35 to 45 seconds. Staging
        half a mile took a minute and a half while OpenStreetMap could not be
        reached - it retries before giving up, and without it streets,
        arterials and parking are unknown, which the warnings say. The
        handle is a hash of the request, so ground already staged is reused
        for the cost of a file read, and the same blocks and assignments on
        it are read back rather than drawn again (`reused`). A section left
        out of `include` stages nothing. `map_id` and `public` are passed to
        staging as `prepare_area` takes them. Medical needs no ground; it
        asks HIFLD, OSRM and OpenStreetMap over the network. THE URBAN
        SECTION NEEDS THE SERVER'S CALTOPO WMS TOKEN: lot lines and roofs are
        read through it, and without it the section fails with
        populated_areas' own reason while parking and the hospitals still
        answer.

        EVERY SECTION has `status` - "ok", "failed", or "skipped" when left out
        of `include` - `error` (null when ok, else "<ExceptionType>: <message>"),
        `warnings` (the feature's own, verbatim) and, once it has answered, its
        compact facts and `detail_call` - `{tool, arguments}`, the exact call
        that returns the full answer, geometry included. A section whose call
        raised keeps that call's `detail_call`, to retry it or make it
        directly; one whose ground could not be staged never made a call and
        has none - its error names the ground, and that ground's `staging`
        record carries the `prepare_area` call. A section standing on the
        ground names it in `ground`. Points come back; outlines do not.

        `urban` - the blocks, then the assignments, on the search ground.
        `populated_areas` draws the blocks: the built-up lots along each
        street, both sides and the street between them, one block per run of
        lots (a lot is built-up with 30 m2 of roof on it, and urban or
        suburban when no bigger than an acre; an apartment complex, school or
        shopping centre over that size is a block of its own).
        `urban_assignments` then merges blocks, smallest first, along the
        streets that join them (a cul-de-sac into the road it opens onto) -
        never two blocks that only back onto each other, unless one is left
        under 30 % of the target - into assignments of about `target_acres`
        (default: the corpus median for `resource_type`, 25 acres for GROUND;
        hand-drawn door-to-door assignments run 2.5 to 10 acres, so about 6 for
        teams of that size), never across an arterial - a motorway, trunk,
        primary or secondary road - and never past 140 % of the target;
        small groups still left pair across water, two at most. So every
        team has both sides of its streets, its edges back fences or
        streets. Both keep every block any part of which is within the
        search radius of the centre (`max_distance_m`, the radius unless
        overridden), whole: the ring is extended out to the back fences
        beyond the nearest street, never cut. Lot lines are read over the
        search circle and 200 m round it, then twice as far again while a
        kept block runs into the edge of what was read, up to 1,000 m past
        the circle, so a street the circle reaches is read to its end. A
        block the circle does not reach is never drawn, whatever
        `max_distance_m` says: to search wider, pass a larger radius. `set_id` names the
        assignment set, `created_at` dates it and `reused` says it was read
        back rather than drawn again. `assignments`, nearest the centre
        first, each with `id` (for `get_candidate`, with the section's
        `handle`, which returns its polygon, its block names and any house
        markers), `name` (the main street and "+N" others), `streets`,
        `blocks` (how many), `parcels`, `buildings`, `area_acres`, `point`
        (inside it), `distance_from_center_m` and `bearing_from_center_deg`,
        `parts` (over 1: the outline is in pieces, drafted as one assignment
        each), `water_links` (parts joined across water by a strip drawn for
        the map only - travel between them by road), `markers` (how many
        house markers, only with `building_markers`: `get_candidate` and the
        draft file carry the markers themselves) and its `warnings`. Then
        `count`, `target_acres` and `target_source` (given, or the corpus
        median for `resource_type`; for a type the corpus has no median
        for - a typo such as DOGS - the median of every assignment, 22.5 acres
        in the shipped corpus, and a warning names the type), `area_acres`
        (min, median, max and total; null with no assignment), `parcels`,
        `buildings`, `parcels_left_out` (built-up lots in runs too short to
        be a block: in no assignment) and, with markers, `markers`.
        `blocks` is populated_areas' answer as counts:
        its `set_id` (its `detail_call` lists every block with its id and
        name; `get_candidate` on a block id returns that block's polygon),
        `count`, `total_before_limit`, `truncated_to`, `parcels_built_up`,
        `parcels_in_blocks`, `parcels_left_out`, `large_parcels`, the
        `warnings` the assignments do not already give, and `detail_call`.
        Ground with no populated block in the circle - rural lots, or no lot
        lines drawn there - answers "ok" with `assignments` empty, `count` 0
        and `area_acres` null, and a warning says which: an empty list is
        nothing found, never a failure. Without OpenStreetMap on the staged
        ground, lots are grouped by adjacency alone, streets are unnamed and
        arterials unknown, and the warnings say so. When urban_assignments
        raises after the blocks were drawn, the section fails with its error
        and `detail_call` but keeps `blocks`: the blocks stand, saved, and
        the section counts as answered in part (see OVERALL `status`).

        BUILDING MARKERS ARE OFF UNLESS ASKED FOR. Pass
        `building_markers=true` only when the user asks for the houses to be
        marked: each assignment then holds one CalTopo house Marker per
        building - this answer gives only how many (`markers`); the draft
        file and `get_candidate` carry the markers themselves - white unless
        urban_assignments' `marker_color` override says otherwise. The
        drafts take the assignments' colour: one given to populated_areas
        alone colours only the blocks' markers, and a section warning says
        so. It names a different set, so it is drawn afresh.

        `parking` - `parking_candidates` on the search ground: the nearest
        public lots and hardstand, `parking_limit` of them (default 5; 0 for
        every one). Each has `id`, `kind` (parking - a lot OpenStreetMap maps;
        hardstand - a school, church or ballfield with room, never a promised
        surface), `name`, `point` (inside the lot), `area_m2`, `surface`,
        `capacity` (free text), `flags` (fee, access_private, access_permit,
        access_customers, covered_deck, parking_unverified and the like: an
        awkward lot is flagged, never hidden), `distance_from_center_m` and
        `bearing_from_center_deg` from the place last seen, `nearest_way`
        (the nearest named trail or road within 150 m) and `warnings`. Gates,
        opening hours and the road in are not checked. Outlines are left out.
        `total_before_limit` counts the sites before the limit: 0 when
        OpenStreetMap was read and maps none here, null when no lots could be
        read - the staged ground has no OpenStreetMap (an Overpass outage;
        the ground's `osm_available` is false), or OpenStreetMap fetched
        before parking was asked for. That is unknown, not none, and the
        `warnings` say which.

        `medical` - `nearest_medical_facilities` at the place last seen.
        `categories` holds medical_center, trauma_center and
        pediatric_trauma_center, each up to `medical_per_category` hospitals
        (default 3, at most 10) ranked by road time - a hospital with no road
        time (the road router could not be reached, or found no road to it)
        after every routed one, by straight-line distance, as a warning says:
        `rank`, `name`, `trauma`
        (the level as a person says it), `helipad` (Y, N or unknown),
        `road_minutes`, `air_minutes` (only with a helipad),
        `road_distance_m`, `phone`, `latitude` and `longitude`. An empty
        category found nothing within the search radius, and a warning says
        so. `table` is the feature's Markdown table verbatim; `in_us` is false
        outside the US, where the list has no hospitals (null when that could
        not be checked); `road_start` is where the road time begins.

        `export` - the CalTopo files. The assignments go through
        `caltopo_export` as DRAFT assignments of resource type
        `resource_type` as given (one the corpus has no median for is
        drafted as written, and a warning here says so), with the search
        area drawn round them (`include_area` true) and, with
        `building_markers`, the house markers (`include_markers`): `path`,
        `set_id`, `assignments` (how many drawn - an assignment in pieces is
        one per piece), `features`
        (a count by CalTopo class), `labels` (each label with the candidate
        it came from, such as `U03 Laurel Street +2`), `area_included`, the
        export's `warnings` and its `detail_call`. Everything else found goes
        into a second file, `markers_path` (`exports/urban_search.geojson`;
        `markers` counts it by CalTopo class): the place last seen as a blue
        dot, each parking site as a grey dot and each hospital as the
        medical tool's red first-aid cross - every description carrying its
        facts and what was NOT checked there. Both are written beside the
        search ground, and a later call on the same ground replaces them. A
        later call with `export` in `include` that writes no markers file
        leaves the earlier call's urban_search.geojson as it was, and a
        top-level warning names it as an earlier call's, not this answer's.
        When caltopo_export raised, an earlier call's drafts under the same
        `set_id` may still be there, and are not named. Import each in
        CalTopo (Import, choose the file); nothing is sent to CalTopo. An
        export failure is a warning, never a failed section: the section
        stays "ok" and a null `path` or `markers_path` with a warning says
        which file was not written and why - `urban` left out of `include`
        or failed (no set to draft; the markers file is still written), a
        disk that is full, or no ground staged to write beside.
        `markers_path` is also null, with no warning that it is missing,
        when there was nothing for it - parking and medical left out of
        `include` or finding nothing: the place last seen alone is not
        written. When caltopo_export itself raised, its `detail_call` makes
        it again. The top-level `warnings` then say either that nothing was
        found to draw or which file was not written although something found
        was for it - the draft assignments (the answer itself), the markers
        file, or both. Left out of `include`, nothing is written at all and
        no file already in the folder is named; `export` alone in `include`
        is refused, having nothing to draw.

        OVERALL `status`: "ok" when every section asked for is ok, "failed"
        when every one failed, otherwise "partial". Skipped sections count
        neither way, and nor does `export`: it finds nothing of its own, so a
        file it could not write never turns "ok" into "partial", nor an
        export with nothing to draw "failed" into "partial". An urban section
        that failed but kept its `blocks` counts as "partial", not "failed":
        it still holds the blocks of the town. A failed section
        never stops the others: no WMS token, or no network for the
        hospitals, still leaves the rest, and ground that could not be staged
        fails only the sections standing on it, each error saying which
        ground and why.

        `summary` - the facts to read first, one block per section asked for
        (the export's are its paths, in `exports`), null when that section did
        not answer. `urban`: `count`, `target_acres`, `area_acres` (min, median and
        max - the spread of sizes round the target - and the total ground,
        null with no assignment), `parcels` and `buildings`.
        `parking`: `count`, `total_before_limit` and the `nearest` site (id,
        kind, name, distance and bearing, flags). `nearest` is null when
        there is no site. `count` and `total_before_limit` are also null when
        no lots could be read, as in the section: unknown, not none.
        `medical`: `nearest_trauma_center` (name, trauma level, helipad, road
        minutes, air minutes, and `ranked_by`: "road_time", or
        "straight_line" when no trauma center found has a road time - the
        road router could not be reached or found no road - so it is the
        nearest as the crow flies, which may not be the quickest by road),
        null when none was found.

        `staging` - one record, `search`: `status`, `error`, `handle`,
        `radius_m` and `resolution_m` asked for, `dem_resolution_m`, `staged`
        (false when it was already on disk), `osm_available`, `warnings`, and
        the `prepare_area` call as `detail_call`. `handles` maps it to its
        handle (null when it could not be staged), for any tool that takes
        one: `get_candidate` on an assignment or block id, `describe_boundary`
        on an assignment, `caltopo_export` of chosen assignments.

        `overrides` - keyword arguments for the feature behind a section, by
        section, passed to every call that section makes: `{"parking":
        {"max_distance_m": 600}, "medical": {"search_radius_km": 150},
        "export": {"label_prefix": "T"}}`. The urban section calls two
        features, so its overrides are by tool - to keep only the blocks
        within 600 m of the centre, `{"urban": {"populated_areas":
        {"max_distance_m": 600}, "urban_assignments": {"max_distance_m":
        600}}}`. Both take `max_distance_m` (the search radius by default
        here) and `marker_color` (used with `building_markers`; the
        assignments and the drafts take urban_assignments'). A
        `max_distance_m` can only narrow the search: lot lines are read
        over the search circle alone, so one past the radius, or null, adds
        at most the blocks straddling its edge, and a section warning says
        so - to search wider, pass a larger `radius`. populated_areas also
        takes `limit` and the block rules - `pixel_m`,
        `max_parcel_area_m2`, `min_building_m2`, `large_parcel_min_cover`,
        `bridge_m` and `min_parcels`. urban_assignments takes none of the
        block rules: it groups blocks drawn by populated_areas' defaults
        within its own `max_distance_m`, so a block rule given to
        populated_areas never changes the assignments. A block rule given
        to populated_areas, or a `max_distance_m` given to only one of the
        two, makes the blocks counted differ from the blocks grouped -
        `blocks` follows populated_areas, the assignments follow
        urban_assignments - and a section warning says so. Refused before
        anything is staged: a section this tool does not have; urban
        keywords not under a tool; a tool the urban section does not call;
        a keyword the feature does not take, the refusal listing what it
        does; and a keyword plan_urban_search sets itself - the handle, the
        centre, `target_acres`, `resource_type` and `building_markers` (its own
        arguments, also for the export's `resource_type` and
        `include_markers`), the set the export draws (`set_id`,
        `candidate_ids`), and `limit` and `per_category`, which are
        `parking_limit` and `medical_per_category` here. Bad coordinates,
        an unknown or empty `include`, a radius `prepare_area` or
        populated_areas would refuse, a `target_acres` not above zero, a
        `building_markers` that is not true or false, a negative limit, a
        block rule out of range, a marker colour CalTopo cannot read (null
        included) and an export `label_prefix` caltopo_export would refuse
        are refused the same way, before any work. Each override value is
        read as the feature tool reads it: one the tool's schema converts is
        passed on converted - "600" as a number, "false" as false, never
        true - and one it refuses (text for a number, a null where a number
        goes) or its own checks would refuse (a `search_radius_km` past 400)
        is refused the same way, naming the keyword. So is a number that is
        not finite ("nan", "inf"), though the tool would take it: no range
        check can be trusted to catch it.

        `place_last_seen` echoes the point the call was made for, and
        `radius_m` the search radius in metres, converted once from `radius`
        and `radius_unit` (or the default). `warnings` are the call's own,
        beside each section's: a search radius under 50 m, an override given
        for a section left out of `include` (accepted, but no call took it),
        and the export's (see `export`). `note` says in a few sentences what
        NO VERDICT below says, to show beside any answer.

        NO VERDICT. This lays out what the tools found. It chooses no
        assignment, staging area or hospital, and orders nothing beyond the
        tools' own orders - nearest the centre first. An assignment's order
        is distance, a fact, not a priority: which blocks are searched, by
        whom and in what order is the planner's call, and sizes follow the
        streets, so an arterial-bound pocket stays small. The blocks are a
        first cut from three public layers - CalTopo's lot lines and
        building footprints, OpenStreetMap's streets: a house CalTopo has not
        drawn leaves its lot reading vacant, a lot line it has not drawn
        joins two lots, and gates, fences, dogs, yards and the inside of a
        building are not seen. Parking knows nothing of gates, hours or the
        road in. Hospitals are as HIFLD last validated them: confirm with the
        receiving facility. The draft assignments are drafts: check each on
        the map before issuing it. Drill down with each section's
        `detail_call`, and an assignment or block with `get_candidate`.
        """
        return plan_urban_search(
            ctx, latitude=latitude, longitude=longitude, radius=radius,
            radius_unit=radius_unit, radius_m=radius_m, target_acres=target_acres,
            resource_type=resource_type, building_markers=building_markers,
            map_id=map_id, public=public, include=include, parking_limit=parking_limit,
            medical_per_category=medical_per_category, overrides=overrides)
