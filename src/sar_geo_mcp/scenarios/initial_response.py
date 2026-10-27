"""Scenario: the first operational period of a wilderness search from a place
last seen - the hasty lines a team can run, where to park and stage, how far
a walker could have got, where a helicopter could land and the nearest
hospitals, in one call, drafted for CalTopo."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..planning.area import RadiusUnit, radius_in_metres
from ..planning.corridors import ALL_INCLUDE as HASTY_ALL_INCLUDE
from ._sections import (
    LZ_RESOLUTION_M,
    WALKING_RESOLUTION_M,
    Ground,
    check_export_overrides,
    check_hasty_overrides,
    check_limit,
    check_overrides,
    check_per_category,
    check_point,
    check_radius,
    check_ring_minutes,
    ground_handle,
    guard,
    hasty_facts,
    hasty_pace_warnings,
    hasty_section,
    lz_facts,
    lz_section,
    medical_facts,
    medical_section,
    overall_status,
    parking_facts,
    parking_section,
    point_marker,
    requested,
    rings_facts,
    rings_ground,
    rings_section,
    set_export_section,
    skipped,
    small_radius_warnings,
    stage_grounds,
    summarise,
    unused_overrides,
    write_markers,
)

#: The sections, in the order they are answered and reported. A type, so the
#: six names reach the caller as an enum in the schema.
InitialResponseSection = Literal["hasty", "parking", "travel_rings", "lz", "medical",
                                 "export"]
SECTIONS: tuple[str, ...] = ("hasty", "parking", "travel_rings", "lz", "medical", "export")

#: The search area when no radius is given: 1,500 m is the area the hasty
#: and parking tools were measured on (Windy Hill), and with the 1,000 m
#: planning buffer at `prepare_area`'s default 1,024-pixel raster it is
#: measured at about 4.9 m a pixel.
DEFAULT_SEARCH_RADIUS_M = 1500.0
#: The landing-zone ground, at LZ_RESOLUTION_M: plan_medevac's default, so a
#: medevac planned later from the same point stands on the ground already
#: staged here. 1,000 m plus the buffer is 4,000 m a side, 2,000 px.
LZ_RADIUS_M = 1000.0
#: The hasty axes a first operational period looks at when the caller names
#: none: the trails and roads a lost person most likely kept to and a hasty
#: team walks end to end. Waterway corridors come back with
#: `overrides={"hasty": {"include": None}}` (or the three names).
DEFAULT_HASTY_INCLUDE: tuple[str, ...] = ("trail", "road")
#: The walking rings drawn when none are asked for, in walking minutes - the
#: rings tool's own default. One list both the function and the wrapper
#: default to, never changed in place.
DEFAULT_RING_MINUTES: list = [15.0, 30.0, 60.0]

#: What plan_initial_response sets in each feature call from its own
#: arguments, so an override may not: {section: {keyword: the argument}}.
SCENARIO_SETS = {"hasty": {"limit": "hasty_limit"}, "parking": {"limit": "parking_limit"},
                 "travel_rings": {"minutes": "ring_minutes"}, "lz": {"limit": "lz_limit"},
                 "medical": {"per_category": "medical_per_category"}}

#: The search ground first: the draft assignments are written beside it, and
#: it is the one a planner opens to look at. The others only when it is not
#: staged.
EXPORT_GROUNDS = ("search", "rings", "lz")
EXPORT_FILENAME = "initial_response.geojson"
#: Which ground each section stands on, as the section and `staging` name it.
SECTION_GROUND = {"hasty": "search", "parking": "search", "travel_rings": "rings",
                  "lz": "lz"}

NOTE = ("No verdict: this lays out what hasty_candidates, parking_candidates, "
        "travel_time_rings, lz_candidates and nearest_medical_facilities found, and "
        "chooses no assignment, staging area, landing zone or hospital. Hasty lines are "
        "ordered by distance, a fact and not a ranking - which a team runs is the "
        "planner's call; the rings are walking minutes, not time since last seen; "
        "landing zones are graded from rasters - the pilot decides; hospitals are as "
        "HIFLD last validated them - confirm with the receiving facility. The draft "
        "assignments are drafts: check each one before issuing it.")


def plan_initial_response(ctx: ServerContext, latitude: float, longitude: float,
                          radius: float | None = None, radius_unit: RadiusUnit = "m",
                          radius_m: float | None = None,
                          map_id: str = "", public: bool = True,
                          include: list[InitialResponseSection] | None = None,
                          ring_minutes: list[float] = DEFAULT_RING_MINUTES,
                          hasty_limit: int = 20, parking_limit: int = 5,
                          lz_limit: int = 5, medical_per_category: int = 3,
                          overrides: dict[str, dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """The `plan_initial_response` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    # Every refusal first, before anything is staged or fetched: a caller's
    # mistake is theirs to hear at once, not a failed section after minutes
    # of staging. Every argument is checked whether or not its section runs.
    check_point(latitude, longitude)
    asked = requested(include, SECTIONS)
    search_radius = radius_in_metres(radius_m, radius, radius_unit)
    search_radius = DEFAULT_SEARCH_RADIUS_M if search_radius is None else search_radius
    check_radius("radius", search_radius, latitude=latitude, longitude=longitude)
    minutes = check_ring_minutes("ring_minutes", ring_minutes)
    check_limit("hasty_limit", hasty_limit, "candidate")
    check_limit("parking_limit", parking_limit, "site")
    check_limit("lz_limit", lz_limit, "zone")
    check_per_category(medical_per_category)
    keywords = check_overrides("plan_initial_response", overrides, SECTIONS, SCENARIO_SETS)
    keywords["hasty"] = check_hasty_overrides(keywords["hasty"])
    # The default goes to the call, never into `keywords`: those are the
    # caller's, and one they never gave must not read as an unused override.
    # An `include` they did give, null included, is theirs - null is every
    # axis, named in full: hasty_candidates reads null as its own default.
    hasty_keywords = {"include": list(DEFAULT_HASTY_INCLUDE), **keywords["hasty"]}
    if hasty_keywords["include"] is None:
        hasty_keywords["include"] = list(HASTY_ALL_INCLUDE)
    check_export_overrides(keywords["export"])
    # Sized from the longest ring at the speed the overrides give, and
    # checked as prepare_area would check it - but only when it is to be
    # staged: the caller never named this radius, so it must not refuse a
    # call that leaves the rings out.
    rings = rings_ground(minutes, keywords["travel_rings"])
    if "travel_rings" in asked:
        check_radius("the walking rings' ground radius", rings.radius_m, latitude=latitude,
                     longitude=longitude, resolution_m=rings.resolution_m)

    grounds = {}
    if "hasty" in asked or "parking" in asked:
        grounds["search"] = Ground(search_radius)
    if "travel_rings" in asked:
        grounds["rings"] = rings
    if "lz" in asked:
        grounds["lz"] = Ground(LZ_RADIUS_M, LZ_RESOLUTION_M)
    staging = stage_grounds(ctx, grounds, latitude=latitude, longitude=longitude,
                            map_id=map_id, public=public)

    outcomes = {
        "hasty": guard(ctx, lambda: hasty_section(
            ctx, ground_handle(staging, "search"), limit=hasty_limit,
            overrides=hasty_keywords)) if "hasty" in asked else skipped(),
        "parking": guard(ctx, lambda: parking_section(
            ctx, ground_handle(staging, "search"), limit=parking_limit,
            overrides=keywords["parking"],
            point_name="the place last seen")) if "parking" in asked else skipped(),
        "travel_rings": guard(ctx, lambda: rings_section(
            ctx, ground_handle(staging, "rings"), latitude=latitude, longitude=longitude,
            minutes=minutes, overrides=keywords["travel_rings"],
            point_name="the place last seen")) if "travel_rings" in asked else skipped(),
        "lz": guard(ctx, lambda: lz_section(
            ctx, ground_handle(staging, "lz"), latitude=latitude, longitude=longitude,
            limit=lz_limit, overrides=keywords["lz"],
            point_name="the place last seen")) if "lz" in asked else skipped(),
        "medical": guard(ctx, lambda: medical_section(
            ctx, latitude=latitude, longitude=longitude, per_category=medical_per_category,
            overrides=keywords["medical"])) if "medical" in asked else skipped(),
    }
    # After hasty, whose set it draws: a hasty section that failed or was
    # left out leaves it nothing to draw but the markers, and it says so.
    outcomes["export"] = guard(ctx, lambda: set_export_section(
        ctx, "hasty", outcomes["hasty"].section,
        overrides=keywords["export"])) if "export" in asked else skipped()
    sections = {name: outcome.section for name, outcome in outcomes.items()}
    for name, ground in SECTION_GROUND.items():
        if name in asked:
            sections[name]["ground"] = ground

    # Most likely half a mile asked without its unit, so said first, as
    # plan_urban_search says it. A hasty line is a candidate only where it
    # crosses the search circle, at any radius; a circle a metre across is
    # crossed by few if any, and an empty hasty section would read as "no
    # hasty lines here". Said of hasty only when hasty was asked for.
    what = f"a circle {2 * search_radius:g} m across rather than a search area"
    if "hasty" in asked:
        what += (", which few if any hasty lines cross - an empty hasty section says "
                 "nothing of the lines nearby")
    warnings = small_radius_warnings(search_radius, what=what)
    warnings.extend(unused_overrides(keywords, asked))
    # Widths for corridors the list leaves out reach the call and change
    # nothing; most likely the default trail-and-road list, not the caller's.
    axes = hasty_keywords["include"]
    if ("hasty" in asked and keywords["hasty"].get("corridor_widths_m")
            and axes is not None and "waterway" not in axes):
        warnings.append(
            "overrides['hasty']['corridor_widths_m'] was not used: the hasty list "
            f"includes {sorted(axes)} and no corridor axis (waterway) - "
            "name it in overrides['hasty']['include'], or give it null for every axis")
    # The rings can be walked at the subject's pace and the hasty minutes
    # cannot: hasty_candidates takes no pace. Said beside both, or a line 6
    # minutes away at the default pace reads as inside a 12-minute ring at
    # half pace that stops short of it.
    warnings.extend(hasty_pace_warnings(sections["hasty"], sections["travel_rings"],
                                        keywords["travel_rings"]))
    exports: list[str] = []
    if "export" in asked:
        # What the draft assignments do not hold - the place last seen,
        # parking, ring outlines, landing zones, hospitals - goes to a second
        # file. The place says only what is true of every line in it: the
        # medical section's notes say how the hospitals were ranked - by
        # straight-line distance when no road time is known - and where the
        # road times start when that is some way off.
        place = point_marker(
            latitude, longitude, "Place last seen",
            "\n".join(["Where the subject was last seen. Hasty-line, parking and "
                       "landing-zone distances and bearings, and every walking ring, are "
                       "measured from here.",
                       *(note for outcome in outcomes.values() for note in outcome.point_notes),
                       "From plan_initial_response; nothing in this file is a decision."]))
        exports, export_warnings = write_markers(
            ctx, staging, outcomes, set_source="hasty",
            sources=("parking", "travel_rings", "lz", "medical"),
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
#: The export has none: its paths are in `exports`.
FACTS = {"hasty": hasty_facts, "parking": parking_facts, "travel_rings": rings_facts,
         "lz": lz_facts, "medical": medical_facts}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="plan_initial_response", tags={"scenario"})
    def _plan_initial_response_tool(latitude: float, longitude: float,
                                    radius: float | None = None,
                                    radius_unit: RadiusUnit = "m",
                                    radius_m: float | None = None,
                                    map_id: str = "", public: bool = True,
                                    include: list[InitialResponseSection] | None = None,
                                    ring_minutes: list[float] = DEFAULT_RING_MINUTES,
                                    hasty_limit: int = 20, parking_limit: int = 5,
                                    lz_limit: int = 5, medical_per_category: int = 3,
                                    overrides: dict[str, dict[str, Any]] | None = None
                                    ) -> dict[str, Any]:
        """The first operational period of a wilderness search, from a place
        last seen, in one call: the hasty lines a team can run, where to park
        and stage, how far a walker could have got, where a helicopter could
        land and the nearest hospitals - with the hasty lines drafted as
        CalTopo assignments. It runs `hasty_candidates`, `parking_candidates`,
        `travel_time_rings`, `lz_candidates` and `nearest_medical_facilities`
        from the place last seen, and `caltopo_export` on the hasty set, and
        returns their facts summarised, one section each. Nothing is decided.

        THE SEARCH AREA is a circle round the place last seen: `radius` with
        `radius_unit` ("m", "km", "ft", "yd" or "mi", as `prepare_area` takes
        them, so half a mile is `radius=0.5, radius_unit="mi"`), or
        `radius_m`; give one form, not both. `radius` is read in metres
        unless `radius_unit` says otherwise, so `radius=0.5` alone is half a
        metre: a search radius under 50 m - a circle few if any hasty lines
        cross, so an empty hasty section would say nothing of the lines
        nearby - is answered with a top-level warning saying so. Default
        1,500 m. It must be
        under 4,500 m: `prepare_area` stages at most 9,000 m across, measured
        on its own circle, which puts 4,500 m itself just over; the radius is
        checked exactly as `prepare_area` checks it, before anything is
        staged.

        WHAT IT STAGES, AND WHAT THAT COSTS. Up to three grounds through
        `prepare_area`, each centred on the place last seen, with no
        credentials (elevation, vegetation and roads are public), because
        each feature needs its own: the SEARCH area at `prepare_area`'s
        default raster for `hasty` and `parking` (1,500 m plus the 1,000 m
        planning buffer is measured at about 4.9 m a pixel); the RINGS ground
        at `resolution_m=10.0` for `travel_rings`, which walks every cell and
        refuses ground over 1,800 x 1,800 cells; and 1,000 m at
        `resolution_m=2.0` for `lz`, because lz_candidates will not grade
        coarser than 3 m a pixel - the same ground plan_medevac stages by
        default, so a medevac planned later from this point reuses it. The
        rings ground is sized for the longest ring: the farthest anyone walks
        in that time at the model's top speed (6 km/h, on a trail), less the
        buffer, rounded up to 250 m - 2,000 m for 30 minutes - and at most
        4,499 m, the largest `prepare_area` stages, which with the buffer
        reaches about 5,500 m: 55 minutes on a good trail. So the default
        60-minute ring stages that largest ground (1,100 cells a side) and may
        still be cut at its edge, along a long trail or road; the ring then
        says `reaches_edge`. Fewer or shorter rings stage less; a
        `speed_factor` override scales the size, and an `off_way_factor` or
        `vegetation_factors` override that makes open ground quicker than a
        trail sizes it for that ground. Measured on 2026-10-22:
        staging the default rings ground took about a minute and a half, as
        did the landing-zone ground for plan_medevac; the hospital lookup took
        25 to 35 seconds, the landing zones a few more, the hasty lines about
        5. With the landing-zone ground already on disk the whole call took a
        little over two minutes, so a first call on new ground takes several.
        The handle is a hash of the request, so ground already staged is
        reused for the cost of a file read, and a second call on the same
        point mostly waits on the hospital lookup. A section left out of
        `include` stages nothing. `map_id` and `public` are passed to staging
        as `prepare_area` takes them; the hasty lines' overlap with existing
        assignments is measured against that map. Medical needs no ground; it
        asks HIFLD, OSRM and OpenStreetMap over the network. Landing zones
        also fetch CalTopo imagery panels and chips when the server has a WMS
        token, and are refused without one unless the ground carries a
        canopy panel.

        EVERY SECTION has `status` - "ok", "failed", or "skipped" when left out
        of `include` - `error` (null when ok, else "<ExceptionType>: <message>"),
        `warnings` (the feature's own, verbatim) and, once it has answered, its
        compact facts and `detail_call` - `{tool, arguments}`, the exact call
        that returns the full answer, geometry included. A section whose call
        raised keeps that call's `detail_call`, to retry it or make it
        directly; one whose ground could not be staged never made a call and
        has none - its error names the ground, and that ground's `staging`
        record carries the `prepare_area` call. A section standing on a ground
        names it in `ground`. Points come back; lines, corridors, rings and
        outlines do not.

        `hasty` - `hasty_candidates` on the search area: by default every
        trail and road, each a line end to end, nearest the place last seen
        first, `hasty_limit` of them (default 20; 0 for every one). Waterway
        corridors are left out unless asked for: `overrides={"hasty":
        {"include": null}}` brings back every axis (trail, road, waterway),
        and `{"include": ["trail", "waterway"]}` names the axes wanted.
        Drainages and ridges are not offered. The default `include` is in
        the section's `detail_call`. `set_id` names the set, `created_at` dates its facts and
        `reused` says it was read back rather than measured again. Each
        candidate has `id` (for `get_candidate`, with the section's `handle`,
        which returns the line or corridor itself), `kind` (line or
        corridor), `axis` (trail, road or waterway), `name`
        (null when unnamed), `length_m`; a line's `branches` (pieces; more
        than 1 is not one walk end to end) and `clipped_at_edge` (the way runs
        on past the staged ground); a corridor's `width_m` and `area_acres`;
        `distance_from_center_m` and `bearing_from_center_deg` (null when it
        passes through the place last seen); `travel_minutes_from_center`, the
        walking minutes to it on the rings' default walking model - null when
        the walk never reaches it. A `travel_rings` override of that model
        (`speed_factor`, `off_way_factor`, `creek_crossing_minutes`,
        `cliff_slope_deg`, `vegetation_factors`) walks the rings alone, never
        these minutes, and a top-level warning then says so;
        `overlap_existing_pct` and `overlap_labels`, how
        much of it an existing assignment on `map_id` already covers - null
        when no map was read: unknown, not zero; `flags`, where its length or
        width falls outside what real assignments on 65 searches look like -
        unusual, not wrong; and its `warnings`. `by_axis` and `by_kind` count
        the candidates kept - an axis absent kept none - except on ground
        staged with no OpenStreetMap (an Overpass outage; the ground's
        `osm_available` is false). There each axis read from OpenStreetMap
        that was asked for - trail, road, waterway - is null in `by_axis`:
        unknown, not none. In `by_kind`, `line` is null when trail or road
        was asked for, and `corridor` when waterway was the only corridor
        axis asked for. `total_before_limit`, `truncated_to` and
        `dropped_beyond_max_distance` say what the limit and any
        `max_distance_m` override dropped; `total_before_limit` is null when
        no axis asked for could be read. `existing_assignments` counts the
        map's assignments; `overlap_reason` says, once for every candidate,
        why overlap is null - null when the map was read.

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

        `travel_rings` - `travel_time_rings` from the place last seen, one per
        `ring_minutes` (default 15, 30 and 60; at most 8): the ground a walker
        reaches within that many WALKING minutes, stretched along trails,
        pinched by cliffs and unbridged rivers, slowed by brush and creek
        crossings. Each ring has `minutes`, `parts`, `area_acres`,
        `equivalent_radius_m` (the circle of the same area), `max_reach_m`
        and `max_reach_bearing_deg`, `farthest_point`, `reach_by_direction_m`
        (the farthest reachable ground in each of the eight directions N to
        NW, 0 where none) and `reaches_edge` (true when the ring is cut at the
        edge of the staged ground, so its area is a lower bound - a warning
        then says so). `ground_facts` (the feature's `ground`) gives the
        vegetation shares, mapped water by kind and the share of ground
        closed; `speed_factor` and `resolution_m` what was used. Ring
        polygons, outlines and the routes to the farthest ground are left
        out; `detail_call` returns them.

        `lz` - `lz_candidates` from the place last seen on the 1,000 m ground,
        nearest first, `lz_limit` of them (default 5; 0 for every one). Each
        has `id`, `grade` (preferred, minimal or marginal), the `touchdown`
        Point, `clear_diameter_m`, `slope_mean_deg` and `slope_max_deg`,
        `canopy_fraction`, `distance_m` and `bearing_deg`, `powerline`, `road`
        and `parking_lot` - each a fact, null, or the string "unknown": null
        is "nothing mapped that close", never "nothing there" - `chips`
        (imagery squares; show each one's `attribution`), `chip_path` and its
        `warnings`. `grades` counts every grade BEFORE the limit;
        `total_before_limit` and `truncated_to` say what the limit dropped;
        `checks` says what was and was NOT checked - structures, trees, which
        water sources, which power-line kinds, road class.

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

        `export` - the CalTopo files. The hasty set goes through
        `caltopo_export` as DRAFT assignments with the search area drawn
        round them (`include_area` true): `path`, `set_id`, `assignments`
        (how many drawn), `features` (a count by CalTopo class), `labels`
        (each label with the candidate it came from, such as `U03 Spring
        Ridge Trail`), `area_included`, the export's `warnings` and its
        `detail_call`. Everything else found goes into a second file,
        `markers_path` (`exports/initial_response.geojson`; `markers` counts
        it by CalTopo class): the place last seen as a blue dot, each parking
        site as a grey dot, each ring's outline as a cyan line (lightest for
        the shortest ring, darkest for the longest), each landing zone as a
        Marker at its touchdown (green preferred, yellow minimal, orange
        marginal), and each hospital as the medical tool's red first-aid
        cross - every description carrying its facts and what was NOT
        checked there. Both are written beside the search ground (or,
        without it, beside the rings or landing-zone ground) and a later call
        on the same ground replaces them. A later call with `export` in
        `include` that writes no markers file leaves the earlier call's
        initial_response.geojson as it was, and a top-level warning names it
        as an earlier call's, not this answer's. When caltopo_export raised,
        an earlier call's drafts under the same `set_id` may still be there,
        and are not named. Import each in CalTopo (Import, choose the file);
        nothing is sent to CalTopo. An export failure is a warning, never a
        failed section: the section stays "ok" and a null `path` or
        `markers_path` with a warning says which file was not written and
        why - `hasty` left out of `include` or failed (no set to draw; the
        markers file is still written), a disk that is full, or no ground
        staged to write beside. `markers_path` is also null, with no warning
        that it is missing, when there was nothing for it - parking, rings,
        landing zones and hospitals left out of `include` or finding
        nothing: the place last seen alone is not written. When
        caltopo_export itself raised, its `detail_call` makes it again. The
        top-level `warnings` then say either that nothing was found to draw
        or which file was not written although something found was for it -
        the draft assignments, the markers file, or both. Left out of
        `include`, nothing is written at all and no file already in the
        folder is named; `export` alone in `include` is refused, having
        nothing to draw.

        OVERALL `status`: "ok" when every section asked for is ok, "failed"
        when every one failed, otherwise "partial". Skipped sections count
        neither way, and nor does `export`: it finds nothing of its own, so a
        file it could not write never turns "ok" into "partial", nor an
        export with nothing to draw "failed" into "partial". A failed section
        never stops the others: a hospital lookup with no network still
        leaves the hasty lines and the rings, and ground that could not be
        staged fails only the sections standing on it, each error saying
        which ground and why.

        `summary` - the facts to read first, one block per section asked for
        (the export's are its paths, in `exports`), null when that section did
        not answer. A count that could not be made at all, its input
        missing, is null there - never 0, which is a real "none here".
        `hasty`: `count`, `by_kind` (lines and corridors kept), `by_axis`
        (trails, roads and waterways kept) and
        `total_before_limit`, with the section's nulls: with no
        OpenStreetMap, each of trail, road and waterway that was asked for
        is null in `by_axis`, and `line` or `corridor` in `by_kind` as the
        section says; `count` and `total_before_limit` are null when no axis
        asked for could be read, and otherwise count what could be.
        `parking`: `count`, `total_before_limit` and the `nearest` site (id,
        kind, name, distance and bearing, flags). `nearest` is null when
        there is no site. `count` and `total_before_limit` are also null when
        no lots could be read, as in the section: unknown, not none.
        `travel_rings`: per ring, `minutes`, `area_acres`,
        `equivalent_radius_m`, `max_reach_m` and `reaches_edge`. `lz`:
        `nearest_any` and `nearest_preferred` (id, grade, distance, bearing,
        clear diameter); lz_candidates orders by distance before grade, so
        when the limit left every preferred zone out `nearest_preferred` is
        null, `preferred_beyond_limit` counts them and
        `preferred_detail_call` is the call listing every zone (it cuts an
        imagery chip for each, so it costs more) - null with
        `preferred_beyond_limit` 0 is no preferred zone on the ground at all.
        `medical`: `nearest_trauma_center` (name, trauma level, helipad, road
        minutes, air minutes, and `ranked_by`: "road_time", or
        "straight_line" when no trauma center found has a road time - the
        road router could not be reached or found no road - so it is the
        nearest as the crow flies, which may not be the quickest by road),
        null when none was found.

        `staging` - one record per ground (`search`, `rings`, `lz`): `status`,
        `error`, `handle`, `radius_m` and `resolution_m` asked for,
        `dem_resolution_m`, `staged` (false when it was already on disk),
        `osm_available`, `warnings`, and the `prepare_area` call as
        `detail_call`. `handles` maps each ground to its handle (null when it
        could not be staged), for any tool that takes one: `get_candidate` on
        a hasty id, `describe_boundary` on a candidate, `site_facts` on a
        touchdown, `egress_routes` on the rings ground.

        `overrides` - keyword arguments for the feature behind a section, by
        section: `{"hasty": {"min_length_m": 593, "corridor_widths_m":
        {"waterway": 40}}, "travel_rings": {"speed_factor": 0.5}, "export":
        {"label_prefix": "H"}}`, passed to every call that section makes.
        Refused before anything is staged: a section this tool does not have;
        a keyword the feature does not take, the refusal listing what it does;
        and a keyword plan_initial_response sets itself - the handle, the
        centre, the set the export draws (`set_id`, `candidate_ids`), and
        `limit`, `minutes` and `per_category`, which are `hasty_limit`,
        `parking_limit`, `lz_limit`, `ring_minutes` and
        `medical_per_category` here. Bad coordinates, an unknown or empty
        `include`, a radius `prepare_area` would refuse, bad `ring_minutes`,
        a negative limit, a corridor width that is not one, a walking-model
        override the rings tool would refuse (a `speed_factor` or
        `off_way_factor` not above zero, an unknown vegetation lifeform) and
        an export `label_prefix` caltopo_export would refuse are refused the
        same way, before any work. Each override value is read as the
        feature tool reads it: one the tool's schema converts is passed on
        converted - "6" as a number, "false" as false, never true - and one
        it refuses (text for a number, a null where a number goes, a hasty
        `include` kind that is not one) or its own checks would refuse (a
        `min_length_m` of 0, a `search_radius_km` past 400, a `max_slope_deg`
        above `marginal_slope_deg`) is refused the same way, naming the
        keyword. So is a number that is not finite ("nan", "inf"), though
        the tool would take it: no range check can be trusted to catch it.

        `place_last_seen` echoes the point the call was made for, and
        `radius_m` the search radius in metres, converted once from `radius`
        and `radius_unit` (or the default). `warnings` are the call's own,
        beside each section's: a search radius under 50 m, an override given
        for a section left out of `include` (accepted, but no call took it),
        hasty minutes walked on another model than the rings (a `travel_rings`
        override of the walking model, beside a hasty section that answered),
        hasty `corridor_widths_m` given with no corridor axis in the hasty
        `include` (the default trail-and-road list draws no corridors), and the
        export's (see `export`). `note` says in a few sentences what
        NO VERDICT below says, to show beside any answer.

        NO VERDICT. This lays out what the tools found. It chooses no
        assignment, staging area, landing zone or hospital, and orders
        nothing beyond the tools' own orders - nearest first. A hasty line's
        order is distance, a fact, not a ranking: which lines are run is the
        planner's judgement. The rings are walking time at a steady pace, not
        time since last seen, and no lost-person behaviour or likelihood is
        modelled: turning elapsed time into walking time is the planner's
        call. Parking knows nothing of gates, hours or the road in. Landing
        zones are a filter on elevation and imagery rasters: the pilot
        decides. Hospitals are as HIFLD last validated them: confirm with the
        receiving facility. The draft assignments are drafts: check each on
        the map before issuing it. Drill down with each section's
        `detail_call`, and a candidate with `get_candidate`.
        """
        return plan_initial_response(
            ctx, latitude=latitude, longitude=longitude, radius=radius,
            radius_unit=radius_unit, radius_m=radius_m, map_id=map_id, public=public,
            include=include, ring_minutes=ring_minutes, hasty_limit=hasty_limit,
            parking_limit=parking_limit, lz_limit=lz_limit,
            medical_per_category=medical_per_category, overrides=overrides)
