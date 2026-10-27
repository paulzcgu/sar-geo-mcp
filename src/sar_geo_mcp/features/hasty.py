"""Feature: the lines a team can run on day one."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..components.candidates import CANDIDATE_FACTS_NOTE
from ..context import ServerContext, mcp_tool
from ..planning.boundary import ground_for
from ..planning.candidates import make_set_id, save_set, summaries
from ..planning.corridors import hasty_candidates as hasty_candidates_impl, hasty_params
from ..terrain.grid import read_subject
from ._sets import reusable_set, set_map_version

#: The kinds `hasty_candidates` can draw. A type, not prose in a docstring,
#: so the three allowed values reach the caller as an enum in the schema and
#: any other is refused before any ground is read.
HastyInclude = Literal["trail", "road", "waterway"]


class CorridorWidths(BaseModel):
    """Corridor widths in ground metres; a class left out keeps the corpus width."""

    waterway: float | None = Field(default=None, gt=0)

    def as_overrides(self) -> dict[str, float]:
        """Only the classes actually given: an absent class keeps the corpus
        width, and passing None down as a width would erase it."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


def hasty_candidates(ctx: ServerContext, handle: str,
                     include: list[HastyInclude] | None = None,
                     corridor_widths_m: CorridorWidths | None = None,
                     max_distance_m: float | None = None,
                     limit: int | None = None,
                     min_length_m: float | None = None) -> dict[str, Any]:
    """The `hasty_candidates` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    directory = resolved.directory
    widths = corridor_widths_m.as_overrides() if corridor_widths_m is not None else None
    # The id comes from the parameters, so it is known before the ground
    # is read; that is what makes reuse possible at all. The server's norms
    # are among them, as a digest: the flags are read against their bands,
    # so a set made under another `--norms` is measured again. The map version
    # is likewise read before the reuse decision, not after: a set is
    # only ever reused against the map state it will be compared to.
    params = hasty_params(include=include, corridor_widths_m=widths,
                          max_distance_m=max_distance_m, limit=limit,
                          min_length_m=min_length_m, norms=ctx.norms)
    map_version = set_map_version(resolved)
    cset = reusable_set(directory, make_set_id("hasty", params), map_version)
    reused = cset is not None
    if cset is None:
        subject = read_subject(directory)
        map_features = resolved.state.features if resolved.state is not None else None
        cset = hasty_candidates_impl(
            ground_for(str(directory.resolve())), subject, include=include,
            corridor_widths_m=widths, max_distance_m=max_distance_m, limit=limit,
            min_length_m=min_length_m, norms=ctx.norms, map_features=map_features,
            handle=handle, map_version=map_version)
        save_set(directory, cset)
    # `manifest_stamp` is bookkeeping for the staleness check on the saved
    # file; it means nothing to a caller and costs a line in every payload.
    payload = cset.model_dump(exclude={"candidates", "manifest_stamp"})
    payload["candidates"] = summaries(cset)
    payload["reused"] = reused
    payload["facts_note"] = CANDIDATE_FACTS_NOTE
    # An unreadable map is why overlap is unknown, and the reason lives in
    # the resolver's status; a caller reading `warnings` should not have to
    # go looking in a second place for the half of the story that explains
    # the first.
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    payload["map_provenance"] = (resolved.status.get("map_provenance")
                                 or resolved.status.get("provenance"))
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="hasty_candidates", tags={"feature"})
    def _hasty_candidates_tool(handle: str, include: list[HastyInclude] | None = None,
                               corridor_widths_m: CorridorWidths | None = None,
                               max_distance_m: float | None = None,
                               limit: int | None = None,
                               min_length_m: float | None = None) -> dict[str, Any]:
        """The lines a team can run on day one around a staged area: every
        trail and road as a line end to end, and - when asked for - every
        mapped waterway as a thin corridor, measured and ordered by distance
        from the centre. Drainages and ridges are not offered.

        Call `prepare_area` first and pass its handle. `include` names the
        kinds wanted: trail, road, waterway (default trail and road);
        `corridor_widths_m` overrides the waterway corridor's corpus width
        (`{"waterway": 40}`; 30 m by default); `max_distance_m` drops
        candidates farther than that from the centre; `limit` keeps only the
        first that many after the ordering - default 50 nearest; 0 for all -
        with `summary.truncated_to` and `summary.total_before_limit` saying
        so. On a large area, set `max_distance_m` and `limit` deliberately.

        `min_length_m` replaces the 30 m minimum for LINE candidates only, so
        a caller can ask for assignment-sized runs instead of 36 m trail
        stubs; corridor axes keep their own 100 m minimum. The corpus p10 for
        a line assignment is 593 m - the shortest tenth of the real
        first-period line assignments were at least that long - which is the
        figure to reach for when the question is "what could a team actually
        be assigned". It is one of the parameters the set id is hashed over.

        A named trail or road entering the area is ONE candidate, drawn to its
        ends: its ways are grouped by name into connected components, so a
        trail that loops or carries a spur is one assignment and `length_m` is
        its total length on the staged ground, which is what `min_length_m` is
        judged against. `facts.branches` counts the pieces - 1 for a simple
        line - and a candidate with more says in its warnings that it is not a
        single walk end to end. Its geometry is then a MultiLineString.
        Unnamed ways are merged only where one way's end meets another's.

        A LINE candidate runs to the edge of the STAGED GROUND - the area plus
        the planning buffer, the ground slope, climb, effort and travel time
        were measured on - and NOT to the edge of the search area: a team is
        handed a trail, not the arc where the circle happened to cut it. When
        the way carries on past that ground, `facts.clipped_at_edge` is true
        and a warning says the assignment stops where the terrain data stops,
        not where the way does. A waterway corridor is the other way about: it
        is clipped to the area.

        Returns summaries WITHOUT geometry - ids, kind, axis, name, facts
        (`length_m`, `branches` and `clipped_at_edge` for a line, `width_m`
        and `area_ha` for a
        corridor, distance and bearing from the centre, nearest point, slope,
        gain and loss, ingress junctions and area crossings, effort,
        `sources`, `axis_ids`,
        and overlap with the map's existing assignments) and norm flags -
        plus the set summary, assumptions, warnings, staleness and map
        provenance. `travel_minutes_from_center` is walking time from the
        centre on the same walk `travel_time_rings` draws (one shared movement
        model, `assumptions.travel.movement`): Tobler on each step's signed
        grade, faster on mapped trails and roads, slowed off them by the
        LANDFIRE vegetation type, creeks costing minutes, and rivers, open
        water and ground too steep to walk closed. It is the fact to band
        candidates on, because a creek 200 m away across a cliff band is
        farther than a trail 800 m away along a contour. It is a rank, not a
        promise: no fences, private land or unmapped trails, and null when
        the walk never reaches the candidate (or, with a warning, on ground
        staged too fine to walk - more than 1,800 x 1,800 cells). `effort`
        reads the same model: a trail or road is walked at the on-trail
        speed, which vegetation does not slow; a corridor is swept at the
        off-trail speed times its vegetation factor. Canopy density changes
        sweep width, never speed. `assumptions.travel` states the cost model. Numbers are rounded to what the measurement supports,
        and the effort assumptions that are the same for every candidate sit
        once in `assumptions.effort`. `overlap_existing_pct` is null with an
        `overlap_reason` when no map was readable: unknown, not zero. Fetch
        geometry with `get_candidate` for the shortlist only. The order is a
        fact (distance), never a ranking: choosing among candidates is the
        caller's judgement. The same parameters under the same assignment
        norms always return the same set id and the same candidates (a server
        started with other norms measures the set afresh under a new id): a
        set already generated on this handle is read back rather than
        measured again, and `reused` says which
        happened - the facts in a reused set date from its `created_at`. A
        set is regenerated, even with the same parameters, once the map
        changes underneath it - `map_version` identifies the map state its
        overlap facts were measured against, and is null when there is no map.
        """
        return hasty_candidates(ctx, handle=handle, include=include,
                                corridor_widths_m=corridor_widths_m,
                                max_distance_m=max_distance_m, limit=limit,
                                min_length_m=min_length_m)
