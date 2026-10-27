"""Components: reading, writing and measuring between coordinates."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..coords import format_ddm, format_dms, format_mgrs, format_utm, is_near, parse_location


def parse_coordinate(ctx: ServerContext, text: str, near_latitude: float | None = None,
                     near_longitude: float | None = None) -> dict[str, Any]:
    """The `parse_coordinate` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    near = None
    if near_latitude is not None and near_longitude is not None:
        near = (near_latitude, near_longitude)

    point = parse_location(text, near=near)
    if point is None:
        return {
            "parsed": False,
            "stated": text,
            "reason": ("not a recognisable coordinate, or a grid reference "
                       "that decoded too far from the reference point to "
                       "be trusted"),
        }
    latitude, longitude = point
    result: dict[str, Any] = {
        "parsed": True,
        "stated": text,
        "latitude": latitude,
        "longitude": longitude,
    }
    if near is not None:
        result["near_reference"] = is_near(latitude, longitude, near)
    return result


def convert_coordinate(ctx: ServerContext, latitude: float, longitude: float) -> dict[str, Any]:
    """The `convert_coordinate` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    return {
        "decimal_degrees": f"{latitude:.5f}, {longitude:.5f}",
        "degrees_decimal_minutes": format_ddm(latitude, longitude),
        "degrees_minutes_seconds": format_dms(latitude, longitude),
        "utm": format_utm(latitude, longitude),
        "mgrs": format_mgrs(latitude, longitude),
    }


def distance_bearing(ctx: ServerContext, from_latitude: float, from_longitude: float,
                     to_latitude: float, to_longitude: float) -> dict[str, Any]:
    """The `distance_bearing` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    forward, _back, metres = geod.inv(from_longitude, from_latitude,
                                      to_longitude, to_latitude)
    return {
        "metres": metres,
        "kilometres": metres / 1000.0,
        "bearing_degrees": forward % 360.0,
    }


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="parse_coordinate", tags={"component"})
    def _parse_coordinate_tool(text: str, near_latitude: float | None = None,
                               near_longitude: float | None = None) -> dict[str, Any]:
        """Read a coordinate a person wrote or said, in any common format.

        Handles decimal degrees, degrees-decimal-minutes, DMS, UTM, and
        MGRS/USNG; a hemisphere letter may come after the number or before
        it, as a GPS displays it (`N37°30.000' W122°15.000'`). Give `near_latitude`/`near_longitude` (the assignment's
        own position) whenever you have them: they are what let a transposed
        pair or a dropped western minus be resolved, and what lets a grid
        reference that decodes to the wrong 100km square be rejected.

        `parsed` is false when the text is not a coordinate - a description
        like "the second creek crossing", or a grid reference that decoded
        implausibly. That is a complete answer, not an error: record the
        words. NEVER convert a coordinate yourself and pass the result here;
        pass what the person actually said.
        """
        return parse_coordinate(ctx, text=text, near_latitude=near_latitude,
                                near_longitude=near_longitude)

    @mcp_tool(mcp, name="convert_coordinate", tags={"component"})
    def _convert_coordinate_tool(latitude: float, longitude: float) -> dict[str, Any]:
        """One point in every format a team might need it in.

        Decimal degrees, degrees-decimal-minutes (what most handhelds
        display), degrees-minutes-seconds (a paper map margin, a dispatch
        log, a phone's compass app: `37°30'00.0"N 122°15'00.0"W`, to a
        tenth of a second), UTM, and MGRS/USNG (what CalTopo calls USNG).
        Every form parses back with `parse_coordinate`. UTM and MGRS come
        back null above 84N or below 80S, where the grids do not apply.
        """
        return convert_coordinate(ctx, latitude=latitude, longitude=longitude)

    @mcp_tool(mcp, name="distance_bearing", tags={"component"})
    def _distance_bearing_tool(from_latitude: float, from_longitude: float,
                               to_latitude: float, to_longitude: float) -> dict[str, Any]:
        """Ground distance and initial bearing between two points.

        Geodesic on the WGS84 ellipsoid, so it is true ground distance -
        not a planar measurement on a projection, which over-reads by
        1/cos(latitude) and has been a recurring source of error in this
        project's history.
        """
        return distance_bearing(ctx, from_latitude=from_latitude,
                                from_longitude=from_longitude, to_latitude=to_latitude,
                                to_longitude=to_longitude)
