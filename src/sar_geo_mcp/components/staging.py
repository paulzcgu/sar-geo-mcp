"""Components: staging ground and handing back a handle for it."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..handles import prepare_terrain as prepare_terrain_impl
from ..planning.area import RadiusUnit
from ..planning.area import prepare_area as prepare_area_impl


def prepare_terrain(ctx: ServerContext, map_id: str, label: str, public: bool = True,
                    force: bool = False) -> dict[str, Any]:
    """The `prepare_terrain` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    return prepare_terrain_impl(map_id, label, ctx.reader, public=public,
                                force=force)


def prepare_area(ctx: ServerContext, center_latitude: float | None = None,
                 center_longitude: float | None = None,
                 radius_m: float | None = None,
                 geometry: dict | None = None,
                 map_id: str = "", public: bool = True,
                 force: bool = False,
                 resolution_m: float | None = None,
                 radius: float | None = None,
                 radius_unit: RadiusUnit = "m") -> dict[str, Any]:
    """The `prepare_area` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    return prepare_area_impl(ctx.reader, center_latitude=center_latitude,
                             center_longitude=center_longitude, radius_m=radius_m,
                             geometry=geometry, map_id=map_id, public=public,
                             force=force, resolution_m=resolution_m,
                             radius=radius, radius_unit=radius_unit)


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="prepare_terrain", tags={"component"})
    def _prepare_terrain_tool(map_id: str, label: str, public: bool = True,
                              force: bool = False) -> dict[str, Any]:
        """Stage the ground for one assignment and return a handle for it.

        Call this before any terrain question. Staging downloads elevation,
        vegetation and roads for the assignment's extent - no credentials
        needed, all public services - and hands back an opaque handle that
        every later question names.

        Re-staging is skipped when the ground is already staged AND the
        assignment boundary has not moved since, which is the common case.
        `force` re-stages regardless.

        The tiles underneath survive a boundary change, so re-staging after
        an edit costs compute rather than bandwidth.
        """
        return prepare_terrain(ctx, map_id=map_id, label=label, public=public, force=force)

    @mcp_tool(mcp, name="prepare_area", tags={"component"})
    def _prepare_area_tool(center_latitude: float | None = None,
                           center_longitude: float | None = None,
                           radius_m: float | None = None,
                           geometry: dict | None = None,
                           map_id: str = "", public: bool = True,
                           force: bool = False,
                           resolution_m: float | None = None,
                           radius: float | None = None,
                           radius_unit: RadiusUnit = "m") -> dict[str, Any]:
        """Stage the ground around a planner-chosen area and return its handle.

        Give exactly one of a centre plus radius (a place last seen and how
        far to look) or a GeoJSON polygon in WGS84. The radius is either
        `radius_m`, or `radius` with `radius_unit` - "m", "km", "ft", "yd"
        or "mi" - so "half a mile" is `radius=0.5, radius_unit="mi"` and is
        never converted by hand; give one form, not both. The result's
        `area` is the area itself - kind, centre, `radius_m` and its WGS84
        outline (the range circle, not the buffered ground) - to draw the
        range on a map or pass to `caltopo_export`. `map_id` is optional
        context: the assignments and tracks on that map are what a boundary
        is compared against. The area may be at most 9,000 m across before
        the buffer; a wider request is refused with the limit named.

        The ground staged is the area plus a 1,000 m planning buffer, which
        is what a line candidate may run to: a trail that leaves the search
        circle is still offered whole. The wider box costs a first staging
        about 1.6x the ground to fetch, and at a fixed 1,024-pixel raster a
        1,500 m area is measured at 4.9 m per pixel instead of 3.9.

        `resolution_m` asks for ground metres per pixel instead of that fixed
        raster, for a question the default cannot answer. At the default, a
        1,500 m area is measured at 4.9 m per pixel and a 25 m rock face a
        couple of metres wide is simply not there: sweeps of El Corte de
        Madera and Purisima both found NO cliffs at 6-7 m and real ones at 1 m.
        A request finer than 3DEP can serve (8,000 px per axis) is refused
        with the finest achievable value named, never quietly downgraded.
        Staging at 1 m costs proportionally more bandwidth and time, so ask
        for it when the question is a small steep feature, not by default.

        The handle is `aoi-` plus a hash of the request INCLUDING that buffer
        and the resolution, so the same area on the same map is staged once and
        a change of either stages afresh rather than serving a different
        raster under the old name.
        An area never goes stale - its input
        is immutable - so every later question answers without a network
        check; only the map context is re-read, with provenance. Staging
        needs no credentials: elevation, vegetation and roads are public.
        """
        return prepare_area(ctx, center_latitude=center_latitude,
                            center_longitude=center_longitude, radius_m=radius_m,
                            geometry=geometry, map_id=map_id, public=public, force=force,
                            resolution_m=resolution_m, radius=radius,
                            radius_unit=radius_unit)
