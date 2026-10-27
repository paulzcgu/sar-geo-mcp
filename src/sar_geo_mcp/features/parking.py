"""Feature: where a search can park."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.parking import parking_candidates as parking_candidates_impl


def parking_candidates(ctx: ServerContext, handle: str, limit: int = 0,
                       max_distance_m: float | None = None) -> dict[str, Any]:
    """The `parking_candidates` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    payload = parking_candidates_impl(resolved.directory, limit=limit,
                                      max_distance_m=max_distance_m)
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    payload["map_provenance"] = (resolved.status.get("map_provenance")
                                 or resolved.status.get("provenance"))
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="parking_candidates", tags={"feature"})
    def _parking_candidates_tool(handle: str, limit: int = 0,
                                 max_distance_m: float | None = None) -> dict[str, Any]:
        """Where a search can park near a staged area: public parking lots and
        public hardstand, nearest first.

        Call `prepare_area` first and pass its handle. Two kinds come back.
        `parking` is a lot OpenStreetMap has mapped as such; `hardstand` is a
        public site whose lot it has NOT mapped - a school, a church, a
        ballfield, a picnic site - offered only where no mapped lot falls
        inside it and always carrying the `parking_unverified` flag: it is a
        place with room, not a promise of a surface.

        NOTHING IS HIDDEN FOR BEING AWKWARD. A lot that is private, permit-only,
        customers-only or charges a fee is offered with a flag saying so, because
        a locked commercial gate is a phone call a search manager can make and a
        lot they never see is a lot they cannot make it about. Residential
        parking is the one exception and never appears: a resident's parking
        space is somebody's home. Flags are `fee`, `permissive`, `access_private`,
        `access_no`, `access_permit`, `access_customers`, `access_unknown`,
        `covered_deck` (a parking structure - no aircraft, poor radio), plus
        `parking_unverified` and `node_only`.

        Each candidate carries a `point` GUARANTEED to lie inside its own
        polygon (the pole of inaccessibility, not the centroid, which on an
        L-shaped lot lands in the trees), `area_m2`, `surface` and `capacity`
        where OSM has them - `capacity` is free text, not a number - geodesic
        `distance_from_center_m` and `bearing_from_center_deg`, and the polygon
        itself in WGS84 for a caller that wants to draw it. `limit` 0 returns
        every one; `summary.total_before_limit` and `summary.truncated_to` say
        what a limit dropped, and `max_distance_m` drops sites farther out than
        that (the limit applies after it).

        The order is a fact (distance), never a ranking: which lot a search
        stages in is the planner's judgement, and this tool knows nothing about
        gates, opening hours or the road in.

        Answered from the OSM data staged with the handle, so it costs no
        Overpass traffic. Ground staged before parking was asked for carries
        a `query_version` below 3, and the warnings then say so and name the
        re-stage rather than reporting an empty list as "no lots here". Ground
        at version 3 or later holds parking and is read as it is: version 4
        added distribution power lines and version 5 road class, access,
        tracktype and gates, none of which this tool reads, so parking is
        unaffected by them.
        """
        return parking_candidates(ctx, handle=handle, limit=limit,
                                  max_distance_m=max_distance_m)
