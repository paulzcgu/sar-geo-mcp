"""Feature: the nearest hospitals and trauma centers, by road and by air."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.medical import nearest_medical_facilities as nearest_medical_facilities_impl


def nearest_medical_facilities(ctx: ServerContext, latitude: float, longitude: float,
                               per_category: int = 3,
                               search_radius_km: float = 100.0,
                               helicopter_speed_kmh: float = 240.0,
                               air_overhead_minutes: float = 5.0,
                               check_osm_helipads: bool = True,
                               ) -> dict[str, Any]:
    """The `nearest_medical_facilities` tool as a plain function: `ctx`,
    then the tool's arguments; the tool's payload back. What each argument
    and field means is in the tool's description, in `register` below."""
    return nearest_medical_facilities_impl(
        latitude, longitude, per_category=per_category,
        search_radius_km=search_radius_km,
        helicopter_speed_kmh=helicopter_speed_kmh,
        air_overhead_minutes=air_overhead_minutes,
        check_osm_helipads=check_osm_helipads)


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="nearest_medical_facilities", tags={"feature"})
    def _nearest_medical_facilities_tool(latitude: float, longitude: float,
                                         per_category: int = 3,
                                         search_radius_km: float = 100.0,
                                         helicopter_speed_kmh: float = 240.0,
                                         air_overhead_minutes: float = 5.0,
                                         check_osm_helipads: bool = True,
                                         ) -> dict[str, Any]:
        """The closest MEDICAL CENTER (acute-care hospital), TRAUMA CENTER and
        PEDIATRIC (child) TRAUMA CENTER to a point, ranked by ROAD time, with
        a ready-made table. Stateless: no staged ground or map needed.

        `table` is a Markdown table to show as-is, one row per hospital per
        category (`per_category`, default 3, nearest by road first): trauma
        level (Level I-V adult and/or pediatric, or a state's own code),
        helipad Y / N / unknown, road distance (km and miles), road time, air
        time (only where there is a helipad), phone and address. `rows` holds
        the same as data - plus `faster` ("air" or "road"), straight-line
        distance, coordinates, `validated` (the date HIFLD last checked the
        record) and `notes` - and `markers` is a GeoJSON FeatureCollection
        that imports into CalTopo as Markers: a red first-aid cross per
        hospital whose description holds its table row, and a blue dot for the
        point. Nothing is written to CalTopo.

        SOURCES. Hospitals, trauma levels, helipads, phones and addresses:
        the HIFLD Hospitals list (US states, DC and territories only), open
        hospitals; psychiatric, rehabilitation and long-term care are left
        out. Road: OSRM over OpenStreetMap roads, free-flow car time from
        where the point joins the road network - `road_start` says where, and
        a warning says when that is more than 200 m from the point (the walk
        or carry to it is NOT counted). Helipad: HIFLD, cross-checked against
        OpenStreetMap helipads within 300 m (`check_osm_helipads`); either
        saying yes is a Y. A category with nothing within `search_radius_km`
        (default 100, at most 400) is searched again out to 400 km.

        US ONLY. `point.in_us` says whether the point is inside a US state,
        DC or territory (US Census boundaries). When it is not - another
        country, or offshore - the first warning says so: that country's
        hospitals are not in the list and a closer one may exist.

        AIR TIME is straight-line distance at `helicopter_speed_kmh` (default
        240, about 130 knots) plus `air_overhead_minutes` (default 5) for
        lift-off and landing - the transport leg alone, not dispatch, the
        flight to the scene, loading, weather, terrain or night.

        NOT A DESTINATION DECISION. Designations and phone numbers are as
        HIFLD last validated them, and emergency-department status and
        diversion are unknown here. A children's hospital is never offered as
        an adult trauma center, even when its record claims one. Confirm with
        the receiving facility and follow local EMS destination policy. Road
        or helipad lookups that fail degrade to warnings; a failed hospital
        lookup is an error.
        """
        return nearest_medical_facilities(ctx, latitude=latitude, longitude=longitude,
                                          per_category=per_category,
                                          search_radius_km=search_radius_km,
                                          helicopter_speed_kmh=helicopter_speed_kmh,
                                          air_overhead_minutes=air_overhead_minutes,
                                          check_osm_helipads=check_osm_helipads)
