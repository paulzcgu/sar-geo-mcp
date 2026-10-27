"""Feature: rings of equal walking time round a place last seen."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.travel_rings import travel_time_rings as travel_time_rings_impl


def travel_time_rings(ctx: ServerContext, handle: str, center_latitude: float,
                      center_longitude: float,
                      minutes: list[float] | None = None,
                      speed_factor: float = 1.0,
                      off_way_factor: float | None = None,
                      creek_crossing_minutes: float | None = None,
                      cliff_slope_deg: float | None = None,
                      vegetation_factors: dict[str, float] | None = None
                      ) -> dict[str, Any]:
    """The `travel_time_rings` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    payload = travel_time_rings_impl(
        resolved.directory, center_latitude=center_latitude,
        center_longitude=center_longitude, minutes=minutes,
        speed_factor=speed_factor, off_way_factor=off_way_factor,
        creek_crossing_minutes=creek_crossing_minutes,
        cliff_slope_deg=cliff_slope_deg, vegetation_factors=vegetation_factors)
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    # No map_provenance: the rings are drawn on rasters and
    # OpenStreetMap, and nothing here reads the CalTopo map.
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="travel_time_rings", tags={"feature"})
    def _travel_time_rings_tool(handle: str, center_latitude: float, center_longitude: float,
                                minutes: list[float] | None = None,
                                speed_factor: float = 1.0,
                                off_way_factor: float | None = None,
                                creek_crossing_minutes: float | None = None,
                                cliff_slope_deg: float | None = None,
                                vegetation_factors: dict[str, float] | None = None
                                ) -> dict[str, Any]:
        """Rings of equal walking time round a place last seen: the travel-time
        counterpart of a radius circle. Each ring is the ground a walker can
        reach from the location within so many WALKING minutes (default 15,
        30 and 60; at most 8 rings), and every point on its outline is, to
        this model, the same walking time away. It stretches along trails,
        pulls in at cliffs and unbridged rivers, and shrinks in brush.

        Call `prepare_area` first - WITH `resolution_m=10.0`, centred on the
        location - and pass its handle. Ground of more than 1,800 x 1,800
        cells is refused as staged too fine. A location more than a cell
        outside the staged ground, or on a gap in the elevation model, is
        refused. The largest area prepare_area stages (radius 4,500 m) reaches
        about 5,500 m from its centre - roughly an hour on a good trail - so
        a longer ring may be cut there: `reaches_edge` is then true, and a
        warning says so and names the radius that would fit, or that none
        does.

        THE MODEL. Each step to a neighbouring cell costs its ground length
        over Tobler's hiking speed on the step's SIGNED grade, 6 * exp(-3.5 *
        |grade + 0.05|) km/h, so climbing is slower than descending. On a
        mapped trail, track or road the factor is 1.0; off one it is 0.6
        (`off_way_factor`) times a VEGETATION factor for the LANDFIRE lifeform
        on the cell: Tree 0.8, Shrub 0.4, Herb 0.9, Sparse 0.9, Barren 0.7,
        Agriculture 0.8, Developed 1.0, Snow-Ice 0.5 (`vegetation_factors`
        overrides any of them by lifeform name). Getting into a CREEK (OSM
        waterway stream, brook, ditch, drain) costs `creek_crossing_minutes`
        (default 5). A RIVER, canal or tidal channel, a lake, reservoir or
        coast shoreline, and LANDFIRE open water are never entered, except
        within a cell of a mapped way across them - a bridge or ford. Ground
        off a way steeper than `cliff_slope_deg` (default 45) is never
        entered. Motorways are closed as in `egress_routes`. `speed_factor`
        scales every speed (0.5 for a subject who walks at half pace) and
        divides the crossing time.

        EACH RING carries `minutes`; `geometry`, a WGS84 Polygon or
        MultiPolygon with holes where ground inside it cannot be reached in
        time (a cliff band, a lake); `outline`, the largest part's outer
        boundary as a LineString - the loop to draw; `parts`; `area_acres`;
        `equivalent_radius_m`, the radius of a circle of the same area;
        `max_reach_m`, `max_reach_bearing_deg` and `farthest_point`;
        `reach_by_direction_m`, the farthest reachable ground in each of the
        eight 45-degree sectors N to NW - set beside a circle's radius it
        shows where the ground stretches and where it is pinched;
        `routes_by_direction`, the same eight sectors, each the quickest
        route to that farthest ground (null where nothing is in reach):
        `reach_m`, `bearing_deg`, `minutes`, `trail_minutes` and
        `off_trail_minutes` (a step counts as on a trail when it enters a
        mapped way's cell; creek crossings are off-trail time),
        `leaves_trail_at` (the last trail point before the route goes off
        for good; null when it never uses a trail or ends on one), `trail`
        ({name, route_type, highway} of the way it last walked, or null),
        `point` and `route` (a LineString, smoothed to about a cell) - so a
        planner sees, for instance, "22 minutes on the Ridge Trail, then 8
        down the slope"; and `reaches_edge`. `ground` gives the lifeform shares, the mapped water
        counted by kind, and the share of measured ground closed.
        `assumptions` spells every rule and factor out.

        WHAT THE STAGED DATA CANNOT SAY. With no OpenStreetMap data staged,
        no trail, river or creek is seen and a warning says so; with no
        LANDFIRE vegetation, brush slows no one and the rings are too large
        in thick vegetation, and a warning says so. On ground staged before
        0.21.0, every waterway is read as a creek - a river does not stop
        anyone - and a warning says to re-stage with `force=True`.

        NOT A PREDICTION. The minutes are walking time at a steady pace, not
        time since the subject was last seen: a lost person stops, rests and
        backtracks, so turning elapsed time into walking time is the
        planner's call. No lost-person behaviour or likelihood is modelled,
        and fences, private land and unmapped trails are unseen. Read the
        rings as the shape of the reachable ground, not a line the subject
        cannot be beyond.
        """
        return travel_time_rings(ctx, handle=handle, center_latitude=center_latitude,
                                 center_longitude=center_longitude, minutes=minutes,
                                 speed_factor=speed_factor, off_way_factor=off_way_factor,
                                 creek_crossing_minutes=creek_crossing_minutes,
                                 cliff_slope_deg=cliff_slope_deg,
                                 vegetation_factors=vegetation_factors)
