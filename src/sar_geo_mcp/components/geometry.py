"""Components: stateless GeoJSON geometry questions. No staged ground needed."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..geo import (
    bbox_center_wgs84,
    boundary_clearance as boundary_clearance_impl,
    buffer_geometry_wgs84,
    geometry_bounds_wgs84,
    to_web_mercator,
)


def point_in_geometry(ctx: ServerContext, latitude: float, longitude: float,
                      geometry: dict) -> dict[str, Any]:
    """The `point_in_geometry` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    from shapely.geometry import Point, shape

    from ..geo import distance_to_boundary_m

    polygon = shape(geometry)
    # Measured before `contains` is asked, because the measurement refuses
    # a point that is not on the globe (latitude 95, or not a number),
    # which `contains` would answer with a plain False. It measures to the
    # nearest ring, a hole's or another part's included, in ground metres
    # whichever way the edge lies (see `geo.distance_to_boundary_m`).
    distance = distance_to_boundary_m(latitude, longitude, geometry)
    return {
        "inside": bool(polygon.contains(Point(longitude, latitude))),
        "distance_to_boundary_m": distance,
    }


def geometry_bounds(ctx: ServerContext, geometry: dict) -> dict[str, Any]:
    """The `geometry_bounds` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    west, south, east, north = geometry_bounds_wgs84(geometry)
    latitude, longitude = bbox_center_wgs84(
        to_web_mercator((west, south, east, north)))
    return {
        "west": west, "south": south, "east": east, "north": north,
        "center": {"latitude": latitude, "longitude": longitude},
    }


def boundary_clearance(ctx: ServerContext, geometry: dict, boundary: dict,
                       access_latitude: float | None = None,
                       access_longitude: float | None = None,
                       ) -> dict[str, Any]:
    """The `boundary_clearance` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    access = None
    if access_latitude is not None and access_longitude is not None:
        access = (access_latitude, access_longitude)
    elif access_latitude is not None or access_longitude is not None:
        raise ValueError(
            "an access point needs both access_latitude and access_longitude")
    return boundary_clearance_impl(geometry, boundary, access=access)


def buffer_geometry(ctx: ServerContext, geometry: dict, distance_m: float,
                    single_ring: bool = False) -> dict[str, Any]:
    """The `buffer_geometry` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    return buffer_geometry_wgs84(geometry, distance_m, single_ring=single_ring)


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="point_in_geometry", tags={"component"})
    def _point_in_geometry_tool(latitude: float, longitude: float,
                                geometry: dict) -> dict[str, Any]:
        """Whether a point falls inside a GeoJSON geometry.

        Takes the geometry straight from `get_assignment`, so "is this clue
        inside the assignment?" is one call. Distance to the boundary comes
        back too, in true ground metres, because "just outside" and "far
        outside" are different operational facts.
        """
        return point_in_geometry(ctx, latitude=latitude, longitude=longitude,
                                 geometry=geometry)

    @mcp_tool(mcp, name="geometry_bounds", tags={"component"})
    def _geometry_bounds_tool(geometry: dict) -> dict[str, Any]:
        """The WGS84 bounding box and centre of a GeoJSON geometry."""
        return geometry_bounds(ctx, geometry=geometry)

    @mcp_tool(mcp, name="boundary_clearance", tags={"component"})
    def _boundary_clearance_tool(geometry: dict, boundary: dict,
                                 access_latitude: float | None = None,
                                 access_longitude: float | None = None,
                                 ) -> dict[str, Any]:
        """How much of a boundary's inside surrounds this, and does the way in
        stay inside it. Stateless - no staged ground needed.

        Land tenure disqualifies sites that pass every terrain check. A
        candidate in El Corte de Madera Creek OSP had 28 m of drop at a 62
        degree mean, eight minutes from a lot, clean vegetation - and sat on a
        narrow finger of preserve 47 m from the edge, reached by a road that was
        entirely outside the boundary, from a lot that was also outside it,
        along a line only half on preserve ground. That is what this measures.

        `geometry` and `boundary` are GeoJSON Polygons or MultiPolygons in
        WGS84. Supply both `access_latitude` and `access_longitude` - a gate,
        trailhead or lot - to add the `approach` block.

        `radial_clearance_m` gives the distance to the boundary on all eight
        compass points, because one number hides a narrow finger: the candidate
        above had 60 m to the south-east and over 1.5 km to the east, and no
        single figure describes that site honestly. A null on a bearing means
        the boundary was not reached within 2 km.

        The approach is a STRAIGHT LINE. It flags a private-land risk; it does
        not resolve a parcel line, and it knows nothing about trails, fences or
        where a vehicle can stop. A boundary is also only as current as its
        source.
        """
        return boundary_clearance(ctx, geometry=geometry, boundary=boundary,
                                  access_latitude=access_latitude,
                                  access_longitude=access_longitude)

    @mcp_tool(mcp, name="buffer_geometry", tags={"component"})
    def _buffer_geometry_tool(geometry: dict, distance_m: float,
                              single_ring: bool = False) -> dict[str, Any]:
        """Widen a line into the polygon CalTopo's Buffer tool would draw:
        `distance_m` ground metres each side of it, round ends. Takes a
        GeoJSON LineString, MultiLineString, Polygon or MultiPolygon in
        WGS84 and returns the polygon with `width_m` (twice the distance)
        and `area_ha`. A line search on a CalTopo map is such a ribbon,
        15 m each side by default. Distances above 1,000 m are refused.

        Set `single_ring` for a result bound for CalTopo. A way that closes
        into a loop buffers into a donut, and CalTopo carries one ring per
        polygon: it drops the hole and fills the middle of the loop. With
        `single_ring` the hole is cut open along a zero-width bridge
        instead, so the ribbon arrives as a ribbon. The area is unchanged.
        """
        return buffer_geometry(ctx, geometry=geometry, distance_m=distance_m,
                               single_ring=single_ring)
