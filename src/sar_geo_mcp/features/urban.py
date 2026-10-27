"""Features: populated blocks in town, and urban assignments built from them."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.candidates import make_set_id, save_set, summaries
from ..planning.populated import populated_params, populated_set
from ..planning.urban import urban_params, urban_set
from ._sets import reusable_set


def populated_areas(ctx: ServerContext, handle: str, pixel_m: float = 1.0,
                    max_parcel_area_m2: float = 4047.0,
                    min_building_m2: float = 30.0,
                    large_parcel_min_cover: float = 0.10,
                    bridge_m: float = 12.0, min_parcels: int = 3,
                    max_distance_m: float | None = None,
                    limit: int = 0, building_markers: bool = False,
                    marker_color: str = "#FFFFFF") -> dict[str, Any]:
    """The `populated_areas` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    directory = resolved.directory
    params = populated_params(pixel_m=pixel_m, max_parcel_area_m2=max_parcel_area_m2,
                              min_building_m2=min_building_m2,
                              large_parcel_min_cover=large_parcel_min_cover,
                              bridge_m=bridge_m, min_parcels=min_parcels,
                              max_distance_m=max_distance_m, limit=limit,
                              building_markers=building_markers,
                              marker_color=marker_color)
    # No map is read: blocks come from CalTopo's layers and OSM, so the
    # set has no map version to go stale against - only the staging.
    cset = reusable_set(directory, make_set_id("populated", params), None)
    reused = cset is not None
    if cset is None:
        cset = populated_set(directory, params, handle=handle,
                             wms_base_url=ctx.config.wms_base_url(),
                             cache_dir=ctx.config.cache_dir)
        save_set(directory, cset)
    payload = cset.model_dump(exclude={"candidates", "manifest_stamp", "map_version"})
    payload["candidates"] = summaries(cset)
    payload["reused"] = reused
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    return payload


def urban_assignments(ctx: ServerContext, handle: str, target_ha: float | None = None,
                      resource_type: str = "GROUND",
                      max_distance_m: float | None = None,
                      building_markers: bool = False,
                      marker_color: str = "#FFFFFF") -> dict[str, Any]:
    """The `urban_assignments` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    directory = resolved.directory
    # The server's norms go into the id when they choose the target, so a
    # set grouped under another `--norms` is grouped again, not read back.
    params = urban_params(target_ha=target_ha, resource_type=resource_type,
                          max_distance_m=max_distance_m,
                          building_markers=building_markers, marker_color=marker_color,
                          norms=ctx.norms)
    cset = reusable_set(directory, make_set_id("urban", params), None)
    reused = cset is not None
    if cset is None:
        cset = urban_set(directory, params, handle=handle, norms=ctx.norms,
                         wms_base_url=ctx.config.wms_base_url(),
                         cache_dir=ctx.config.cache_dir)
        save_set(directory, cset)
    payload = cset.model_dump(exclude={"candidates", "manifest_stamp", "map_version"})
    payload["candidates"] = summaries(cset)
    payload["reused"] = reused
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="populated_areas", tags={"feature"})
    def _populated_areas_tool(handle: str, pixel_m: float = 1.0,
                              max_parcel_area_m2: float = 4047.0,
                              min_building_m2: float = 30.0,
                              large_parcel_min_cover: float = 0.10,
                              bridge_m: float = 12.0, min_parcels: int = 3,
                              max_distance_m: float | None = None,
                              limit: int = 0, building_markers: bool = False,
                              marker_color: str = "#FFFFFF") -> dict[str, Any]:
        """Populated blocks in town (urban and suburban ground only): the
        built-up lots along each street, both sides and the street between
        them, outlined as one block per street - the block a planner draws
        by hand on CalTopo's parcel layer for an urban search.

        Call `prepare_area` first over the town and pass its handle; the
        area must be under about 8 km across. Lots are read from CalTopo's
        parcel layer (its drawn lot lines), buildings from its structures
        layer, both through the WMS token - with no token, or when either
        layer cannot be had, the call is refused. Street names come from
        the staged OpenStreetMap; without it lots are grouped by adjacency
        and `street` is null.

        A lot is BUILT-UP when at least `min_building_m2` of roof stands on
        it, and URBAN OR SUBURBAN when it is no bigger than
        `max_parcel_area_m2` (default one acre). Rural ground has big lots
        and yields no blocks, with a warning saying it reads as rural. A
        built-up lot over the size limit whose roofs cover at least
        `large_parcel_min_cover` of it - an apartment complex, a school, a
        shopping centre - is its own block, `axis` "large_parcel"; a
        farmhouse on a big lot never is. Parcels drawn round single condo
        units are folded into the complex around them.

        Each lot belongs to the street whose right-of-way it touches most
        on the parcel map (a lot touching none: the named street it has the
        most edge along within 20 m, or the nearest within 60 m). A corner
        lot can therefore land on the street its address is not on. A street's lots are joined across
        the street and between neighbours by a closing of `bridge_m`
        (12 m spans a 24 m right-of-way), and every lot of another block is
        cut back out, so no house is in two blocks. A run of fewer than
        `min_parcels` houses joins the block whose lots it touches (its
        warnings say so); only one touching no block is left out, counted
        in `summary.parcels_left_out` with a warning. A vacant lot touching
        its street's houses is searched with them (`vacant_lots`).

        Returns summaries WITHOUT geometry, nearest the area's centre
        first: id, `name` (the street, numbered "(1)", "(2)" when it has
        several runs), `axis` ("street" or "large_parcel") and facts -
        `street`, `parcels`, `buildings`, `area_ha`, `median_parcel_m2`,
        `roof_cover_pct`, a `point` guaranteed inside the block, distance
        and bearing from the centre, `clipped_at_edge` (a lot runs past the
        ground read) and `holes`. Fetch a block's polygon with
        `get_candidate`. The same parameters on the same handle return the
        same set, read back rather than fetched again (`reused`).

        BUILDING MARKERS ARE OFF UNLESS ASKED FOR. Pass
        `building_markers=true` only when the user asks for the houses to be
        marked: each block then carries one CalTopo Marker per building on
        it - CalTopo's house symbol (`hut`), `marker_color` (default white,
        `#FFFFFF`; any `#RRGGBB`), standing on the roof - in place of any
        tracing of the buildings. The markers come back with the block from
        `get_candidate` (`markers`, a list of GeoJSON Point features carrying
        CalTopo's `marker-symbol` and `marker-color`); summaries carry only
        `facts.markers`, the count, and `summary.markers` the total. A
        building is marked once, in the block holding most of its roof.
        Markers or not, the block outline is the same.

        A FIRST CUT, NOT A CENSUS: a house CalTopo has not drawn leaves its
        lot reading as vacant, and a lot line it has not drawn joins two
        lots. The order is distance, never priority; which blocks to search
        and in what order is the planner's call.
        """
        return populated_areas(ctx, handle=handle, pixel_m=pixel_m,
                               max_parcel_area_m2=max_parcel_area_m2,
                               min_building_m2=min_building_m2,
                               large_parcel_min_cover=large_parcel_min_cover,
                               bridge_m=bridge_m, min_parcels=min_parcels,
                               max_distance_m=max_distance_m, limit=limit,
                               building_markers=building_markers, marker_color=marker_color)

    @mcp_tool(mcp, name="urban_assignments", tags={"feature"})
    def _urban_assignments_tool(handle: str, target_ha: float | None = None,
                                resource_type: str = "GROUND",
                                max_distance_m: float | None = None,
                                building_markers: bool = False,
                                marker_color: str = "#FFFFFF") -> dict[str, Any]:
        """Urban search assignments: the populated blocks of a staged town
        (see `populated_areas`) merged into assignments of about one size,
        the way a planner groups streets for a team.

        Call `prepare_area` over the town first (`radius=0.5,
        radius_unit="mi"` for half a mile) and pass its handle.
        `target_ha` is the size to aim for; left out, it is the corpus
        median for `resource_type` (10 ha for GROUND, from
        `assignment_norms`); hand-drawn door-to-door assignments run 1 to 4
        ha, so pass `target_ha` (about 2.5) for teams of that size.
        `max_distance_m` keeps every block any part of which is within that
        distance of the centre, whole - pass the radius to keep the blocks
        of the search circle: the ring is extended out to the back fences
        beyond the nearest street, never cut through a street's lots.

        THE RULES. Each block is one street, both sides, back fence to back
        fence, and an assignment is whole blocks: a team always has both
        sides of its streets, and its edges are back fences or streets.
        Blocks merge smallest first ALONG THE STREETS THAT JOIN THEM - a
        cul-de-sac into the road it opens onto, two runs of one street -
        into the most compact shape, until a group reaches 60 % of the
        target, never growing one past 140 %. Blocks that only back onto
        each other (a fence between them) merge only when a group is left
        under 30 % of the target with no street to grow along. A group
        NEVER crosses an arterial - an OpenStreetMap motorway, trunk,
        primary or secondary road - so the boulevards are the edges. Small groups still left
        then pair across water (two isles a team drives between), two at
        most, drawn as one polygon by a thin strip for the drawing only. No
        two assignments share ground; holes are filled unless another
        assignment is inside. On ground staged before road classes were
        kept, arterials are unknown and a warning says so.

        Returns summaries WITHOUT geometry, nearest the centre first: id,
        `name` (the main street and "+N" others), facts {`streets`,
        `blocks`, `block_names`, `parcels`, `buildings`, `area_ha`,
        `point`, distance and bearing, `water_links`, `parts`, `markers`
        when asked}, plus `summary` (count, sizes, totals) and
        `assumptions.grouping`. `building_markers` (off unless asked for)
        carries each block's white house markers into its assignment. Fetch
        a polygon with `get_candidate`; write the set for CalTopo with
        `caltopo_export`. Sizes follow the streets, so they spread round
        the target: an arterial-bound pocket stays small. The planner
        adjusts and decides; nothing here is issued.
        """
        return urban_assignments(ctx, handle=handle, target_ha=target_ha,
                                 resource_type=resource_type, max_distance_m=max_distance_m,
                                 building_markers=building_markers,
                                 marker_color=marker_color)
