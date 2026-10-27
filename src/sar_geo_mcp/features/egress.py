"""Feature: the way out, on foot or by litter."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..planning.egress import egress_routes as egress_routes_impl


def egress_routes(ctx: ServerContext, handle: str, center_latitude: float,
                  center_longitude: float,
                  mode: Literal["walk_out", "carry_out"] = "walk_out",
                  on_way_factor: float | None = None,
                  off_way_factor: float | None = None,
                  max_grade: float | None = None,
                  leave_way_factor: float | None = None) -> dict[str, Any]:
    """The `egress_routes` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    payload = egress_routes_impl(
        resolved.directory, center_latitude=center_latitude,
        center_longitude=center_longitude, mode=mode,
        on_way_factor=on_way_factor, off_way_factor=off_way_factor,
        max_grade=max_grade, leave_way_factor=leave_way_factor)
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    # No map_provenance, as for lz_candidates: an exit is found on
    # rasters and OpenStreetMap, and nothing here reads the CalTopo map.
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="egress_routes", tags={"feature"})
    def _egress_routes_tool(handle: str, center_latitude: float, center_longitude: float,
                            mode: Literal["walk_out", "carry_out"] = "walk_out",
                            on_way_factor: float | None = None,
                            off_way_factor: float | None = None,
                            max_grade: float | None = None,
                            leave_way_factor: float | None = None) -> dict[str, Any]:
        """The way out on the ground: the quickest exit a 2WD vehicle can
        reach and the quickest a 4x4 can, from a location, on foot
        (`walk_out`) or by litter (`carry_out`).

        Call `prepare_area` first - WITH `resolution_m=10.0`, centred on the
        location - and pass its handle. Walking needs no finer ground, and the
        search visits every cell: the default 3,000 m radius plus the 1,000 m
        planning buffer is 800 cells a side at 10 m. Ground of more than
        1,800 x 1,800 cells is refused as staged too fine for egress - a 2 m
        landing-zone handle would need gigabytes - naming the re-stage at
        `resolution_m=10.0`. A location more than a cell outside the staged
        ground is refused.

        QUICKEST MEANS LEAST WALKING TIME, AND THE CLIMB COUNTS. Each step to
        a neighbouring cell costs its ground length over Tobler's hiking speed
        on the step's SIGNED grade, 6 * exp(-3.5 * |grade + 0.05|) km/h -
        fastest on a gentle downhill, slower climbing, slower descending
        steeply - so a far exit along the contour can beat a near one over a
        ridge, and the same route costs differently up and down. The speed is
        scaled by a ground factor for the cell stepped into, on a mapped
        trail, track or road or off one: `walk_out` 1.0 on a way and 0.6 off
        it (Tobler's own off-path factor), `carry_out` 0.25 and 0.08 - about
        1.3 km/h on flat trail and 0.4 km/h off it, starting values from
        litter-carry rules of thumb. `on_way_factor` and `off_way_factor`
        override either; both must be above zero.

        ONCE ON A TRAIL, IT STAYS ON THE TRAILS. A route walks freely from
        the location to its first mapped way. After that, a step off every
        way counts 1 / `leave_way_factor` times its minutes (default 0.1:
        ten times) in choosing the route, so it leaves a trail only where the
        trail's way round is ten times longer than the ground cut across - a
        litter team does not leave a trail to save a minute. An exit's own
        cells (a lot beside the road) are never counted over. `minutes` are
        plain walking minutes, never counted over. `leave_way_factor` must be
        above zero and at most 1; 1 turns it off.

        THE LITTER GRADE LIMIT. `carry_out` takes no step steeper than
        `max_grade` (default 0.40, 40 %) either way, and does not enter ground
        whose own slope is over it, so a face cannot be climbed by zigzagging.
        A mapped way is exempt from the ground's slope - a benched trail reads
        steep on a 10 m model - and a step ALONG a way (both ends on it) from
        the per-step limit too: a pitch of trail steeper than the limit is
        belayed, never walked round over open ground. `steep_pitch_m` on the
        exit says how many metres of such pitch the carry takes, and a
        warning says to belay them; the minutes do not count it. `max_grade`
        may be set for `walk_out` too; by default walking has none, and Tobler
        makes steep ground slow. A gap in the elevation model is never
        crossed, and a motorway is never walked along or across, except where
        another way crosses it on a bridge or meets a ramp at its terminal.

        WHAT COUNTS AS AN EXIT. 2WD: a mapped ROAD - not a motorway, a
        motorway ramp or a pedestrian street - whose surface is not dirt
        (paved, gravel, compacted or untagged), or a mapped parking lot. 4x4:
        a TRACK, or a ROAD surfaced dirt, ground, earth, unpaved, sand, mud or
        grass; a surface stacked with ";" (`dirt;gravel`) is dirt when any part
        of it is. Busways, raceways, rest areas and other highways are walked
        faster but are never exits; unbuilt ways, via ferratas and guideways
        are plain ground. `assumptions` spells every rule out.

        BOTH KINDS ALWAYS COME BACK, 2WD first. One that cannot be reached is
        `{kind, reachable: false, reason}`, the reason saying why: none is
        mapped in the staged area; the litter grade limit, which walk-out gets
        past; for a `walk_out` given its own `max_grade`, that limit; the
        location itself too steep to carry off at all (a rope
        evacuation or a landing zone, when no ground near it lets a litter
        reach an exit either); the location on a gap in the elevation
        model; or the ground between cut off. A reachable one carries
        `exit_point` (the vehicle end) and `route` (location to exit, smoothed
        to about a cell), both WGS84; `minutes`; `distance_m` along the route;
        `gain_m` and `loss_m`, ignoring rises under a metre; `on_trail_share`
        and `off_trail_m`; `off_trail_after_joining_m`, the part of that after
        the route first joins a way - a trail left, not the walk to one;
        `steep_pitch_m`, metres of trail steeper than `max_grade` (null with
        no grade limit); `max_grade_pct`, the steepest step; `road`, the way
        the exit lies on, as {name, route_type, highway, surface, access,
        gate_on_road}, or null - an exit in a lot may have no road beside it;
        `parking_lot`, {id, off} for a lot touching the exit cell (the one
        nearest its centre, where two do), `off` naming the road it opens
        onto; `gates_crossed`, each {geometry, barrier, access} - a gate on a
        way the route walks, near its line; `litter_start`, null unless the
        carry could not start at the location (below); and `warnings` - a
        route more than half off trail says so.

        A TRAIL THAT RUNS OFF THE GROUND. The staged ground is a square round
        the location, and a trail's trailhead can lie past its edge. When no
        exit is reached without leaving a trail for more than 100 m after
        joining one (or none is reached at all) and the route can follow a
        trail to the edge of the ground, `trail_beyond` says where: `point`
        on the edge, `way_name`, the walking `minutes` and `distance_m` to it,
        `bearing_deg` from the location, `problems` (what is wrong with the
        exits here) and `recentre` {latitude, longitude} - a centre for ground
        of the same size moved along that bearing as far as it goes with the
        location still 500 m inside. A warning says the same. Stage there with
        `prepare_area` at the same radius and resolution and call again from
        the same location; `plan_medevac` does this itself. Null otherwise.

        A WALKER THE MAPS PUT IN WATER OR ON A CLIFF. The patient is where
        they are: when `walk_out` finds the location's own cell closed (open
        water on the 30 m land cover, a mapped shoreline or river, or ground
        over 45 degrees), the straight line to the nearest open ground is
        opened, and `closed_start` gives that ground's `point`,
        `distance_m`, `bearing_deg`, `drop_m` and `direction`, with a warning
        to check that first stretch. Null otherwise, and always for
        `carry_out`, which does not see water.

        A LITTER THAT CANNOT LEAVE THE LOCATION. When `carry_out` finds the
        location itself too steep to carry off at all, each exit's carry
        starts instead from the nearest ground - reached over the elevation
        model without the grade limit, a stand-in for the rope team's way -
        from which a litter can reach that exit. `litter_start` gives that
        point, `distance_m` and `bearing_deg` from the location, and `drop_m`
        (positive when lower). The route, minutes and distance start there;
        a warning says the patient must first be moved there by a technical
        or low-angle rope evacuation, which nothing here counts.

        GATES AND ACCESS ARE SAID, NEVER EXCLUDED. A route through a gate, or
        ending on a private road, is still the route: the planner can call for
        the combination or the ranger. `gate_on_road` is true when a gate
        stands on the exit's road itself - on its line, not up a side
        driveway - anywhere in the staged area. A gate is listed in
        `gates_crossed` when it stands on a way the route runs through for two
        or more cells. A trail the route crosses at a shallow angle (under
        about 45 degrees) can meet that rule, so a gate on such a trail near
        the crossing may be listed although the team does not pass through it.

        WHAT THE STAGED DATA CANNOT SAY. On ground staged at OpenStreetMap
        query version 4 or earlier, road class, access and gates are unknown:
        `highway` and `access` are null, `gate_on_road` is null - never false
        - and `gates_crossed` is empty, and a warning says so, and says a
        motorway cannot be told from any other road there. Gates are unknown
        in the same way on any file without a gate list, and its warning says
        to re-stage with `force=True`, since a plain re-stage keeps that file.
        On ground staged before parking was asked for (query version 1 or 2),
        lots are unknown: `parking_lot` is the string "unknown", a warning says
        so, and a 2WD exit that cannot be reached says lots are unknown there,
        never that none is mapped. Ground with no OpenStreetMap data at all is
        refused: with no roads there is nothing to exit to.

        WALK-OUT WALKS THE SHARED GROUND. `walk_out` walks the same ground
        `travel_time_rings` and hasty candidates' minutes do (one movement
        model, `assumptions.walking`): off a way, the 0.6 factor times the
        LANDFIRE vegetation factor (brush 0.4, timber 0.8, grass 0.9); 5
        minutes to get into a creek; rivers, canals, lake and coast shores,
        open water and ground over 45 degrees closed except within a cell of
        a mapped way across them - an exit unreachable for that says so. An
        exit's own cells are never closed. Warnings say when the ground has
        no LANDFIRE vegetation or predates waterway kinds. `carry_out` keeps
        the litter's factors and grade limit over the ways and motorways
        alone: it does not see vegetation or water.

        NOT A SURVEYED ROUTE. It walks the elevation model, OpenStreetMap and
        (walk-out) LANDFIRE; it does not see fences, private land or trails
        OSM lacks. Read the minutes as a rank between two exits, not a
        promise. The team leader decides.
        """
        return egress_routes(ctx, handle=handle, center_latitude=center_latitude,
                             center_longitude=center_longitude, mode=mode,
                             on_way_factor=on_way_factor, off_way_factor=off_way_factor,
                             max_grade=max_grade, leave_way_factor=leave_way_factor)
