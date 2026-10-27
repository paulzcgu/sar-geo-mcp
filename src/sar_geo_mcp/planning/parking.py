"""Where a search puts its vehicles: parking lots and public hardstand on
staged ground, nearest first.

NOT `sar_geo_mcp/staging.py`, which stages the raster. The planning agent
that consumes this calls these staging areas; here they are parking and
hardstand, because in this repo "staging" is already the ground being
prepared and one word cannot mean both.

The facts are the hub's, as everywhere: the app that shows them holds no
geometry maths. Distance and bearing are geodesic from the area centre to
the point the marker will go on - one measurement of one place, rather than
a distance to the nearest kerb and a bearing to somewhere else.

OSM rarely names a car park, so each candidate also carries `nearest_way`:
the named trail, track or road nearest the LOT'S OWN shape (not its point),
in ground metres, within NEAREST_WAY_CEILING_M - or `None` when nothing
named is that close. See `_nearest_named_way` for the reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..geo import geometry_centroid, ground_meters_per_mercator_unit, to_wgs84_geojson
from ..terrain.grid import read_subject
from ..sources.osm import OSM_QUERY_VERSION_PARKING
from ..terrain.linear import SITE_HARDSTAND

#: Ground-metre ceiling on "nearest named way". A lot's own access road runs
#: along its edge, and a big lot's centre can be 80 m from the road it opens
#: onto - both comfortably inside this. Past 150 m the nearest named thing
#: is no longer the way a team drives in on; a lot with nothing named that
#: close gets `None`, which is the honest answer rather than a guess.
NEAREST_WAY_CEILING_M = 150.0

#: Route classes eligible to name a lot. TRAIL and TRACK are in here on equal
#: footing with ROAD/ROAD_OTHER/OTHER - see `_nearest_named_way`'s docstring
#: for why nearest wins regardless of class. BARRIER, WATERWAY and
#: WATER_EDGE are deliberately absent: a lot named after a fence, a creek or
#: a lakeshore is a worse answer than an unnamed lot.
NEAREST_WAY_ELIGIBLE_CLASSES = frozenset({"TRAIL", "TRACK", "ROAD", "ROAD_OTHER", "OTHER"})

MISSING_OSM_WARNING = (
    "osm.json is missing from the staged ground, so parking here is unknown "
    "rather than absent")

#: Only below `OSM_QUERY_VERSION_PARKING` (3): a version-3 file already holds
#: parking, and lacks only distribution power lines, which no lot needs.
STALE_QUERY_WARNING = (
    "the staged ground was fetched by OSM query version {version}, before parking "
    "was asked for, so no lots can be read from it - re-stage this area with "
    "force=True to look for them")

NO_SUBJECT_WARNING = (
    "the staged ground has no readable subject, so distance and bearing from the "
    "centre could not be measured")

__all__ = ["MISSING_OSM_WARNING", "NEAREST_WAY_CEILING_M",
           "NEAREST_WAY_ELIGIBLE_CLASSES", "NO_SUBJECT_WARNING",
           "STALE_QUERY_WARNING", "parking_candidates"]


def _point_inside(geometry):
    """A point the marker can stand on, guaranteed inside the shape.

    Pole of inaccessibility - the point farthest from any edge - and not the
    centroid: an L-shaped lot's centroid lands in the notch between its arms,
    which on the ground is somebody's hedge. `polylabel` is documented as
    taking a single `Polygon`, but in shapely 2.x it is a thin wrapper over
    GEOS's `maximum_inscribed_circle`, which also accepts a `MultiPolygon`
    (a repaired bowtie lot, for instance) and correctly returns a point
    inside whichever lobe is largest - verified directly rather than assumed.
    What it does raise on is an EMPTY geometry or a `GeometryCollection`
    (`GEOSException`, not a `ValueError`); `representative_point` is the
    fallback for those and for anything else unforeseen, and a Point is
    simply itself.
    """
    from shapely.errors import GEOSException
    from shapely.ops import polylabel

    if geometry.geom_type == "Point":
        return geometry
    try:
        return polylabel(geometry, tolerance=1.0)
    except (GEOSException, ValueError, TypeError, AttributeError):
        return geometry.representative_point()


def _named_ways(routes: list[dict]) -> list[dict]:
    """The routes worth naming a lot after: a real name, and a class that
    is not a fence, a creek or a lakeshore (NEAREST_WAY_ELIGIBLE_CLASSES).

    An unnamed service road tells a planner nothing, so it is filtered here
    rather than merely scored low - `name` missing, `None`, or blank after
    stripping are all treated the same as absent.
    """
    named = []
    for route in routes:
        properties = route.get("properties") or {}
        name = properties.get("name")
        if not name or not str(name).strip():
            continue
        if properties.get("route_type") not in NEAREST_WAY_ELIGIBLE_CLASSES:
            continue
        named.append(route)
    return named


def _nearest_way_index(named_routes: list[dict]):
    """A shapely `STRtree` over `named_routes`' geometries (already Web
    Mercator, the same CRS the lot shapes below are measured in before their
    own WGS84 reprojection), or `None` when there is nothing to index.

    A nested loop over lots x routes is fine at Windy Hill's scale (a few
    dozen lots, a few hundred routes) but wasteful, and the tree costs
    nothing extra: `class_distances` in `planning/boundary.py` already builds
    one for the same nearest-line-to-many-points shape of problem, so this
    is the established idiom rather than a new one.
    """
    from shapely import STRtree
    from shapely.geometry import shape as shapely_shape

    if not named_routes:
        return None
    return STRtree([shapely_shape(route["geometry"]) for route in named_routes])


def _nearest_named_way(geometry, named_routes: list[dict], tree, scale: float) -> dict | None:
    """The named way nearest `geometry` (ground metres, true polygon-to-line
    distance) within NEAREST_WAY_CEILING_M, or `None` when nothing named is
    that close.

    Nearest wins whatever its class - a trailhead lot beside "Anniversary
    Trail" is genuinely best described by that trail, which IS what the lot
    is for. Preferring a farther road over a nearer trail because a road is
    "more drivable" is the obvious-looking change to make here, and it would
    be wrong: `route_type` is reported so a consumer can word the row, but it
    never overrides distance.
    """
    if tree is None:
        return None
    ceiling_units = NEAREST_WAY_CEILING_M / scale
    indices, mercator_distances = tree.query_nearest(
        geometry, max_distance=ceiling_units, return_distance=True, all_matches=True)
    if len(indices) == 0:
        return None
    # `query_nearest` can return several exactly-tied ways (two mapped copies
    # of the same road, or two ways crossing at this lot's closest point) in
    # visitation order, which is not a stable order between runs. Sorted the
    # same way the candidate list itself is: distance, then name, then class.
    candidates = [
        (float(mercator_distances[i]) * scale,
         named_routes[idx]["properties"]["name"],
         named_routes[idx]["properties"]["route_type"])
        for i, idx in enumerate(indices)
    ]
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1], candidate[2]))
    distance_m, name, route_type = candidates[0]
    return {"name": name, "route_type": route_type, "distance_m": distance_m}


def parking_candidates(directory, *, limit: int = 0,
                       max_distance_m: float | None = None) -> dict:
    """The lots and hardstand staged under `directory`, nearest first.

    `limit` 0 returns every one; the summary says what a limit dropped.
    `max_distance_m` is applied FIRST: `summary.total_before_limit` counts
    what survived the distance filter, not everything staged, so "5 lots
    were near enough" and "the top 2 of those 5" stay two separate facts
    rather than one number that quietly means both. A hardstand that
    CONTAINS a mapped lot is not offered: the lot is the better answer to
    the same question, and two rows for one place is how a planner ends up
    sending two teams to the same asphalt.

    Distance and bearing use one ground-scale factor
    (`ground_meters_per_mercator_unit`) evaluated once at the staged area's
    centre rather than per lot. Over an area a few kilometres across at this
    latitude that leaves `area_m2` off by well under a tenth of a percent -
    smaller than the rounding already in an OSM `capacity` tag - so one
    shared factor is measurement, not a shortcut.

    A site row with no usable `id` is a bug in what staged this directory,
    not a data-quality question this function is positioned to answer -
    `osm.json` is written by this hub's own `stage_terrain`, never by hand,
    so it is let raise (`KeyError`) rather than silently skipped, the same
    way `stage_terrain` lets a DEM failure raise instead of guessing.
    """
    from pyproj import Geod, Transformer
    from shapely.geometry import shape as shapely_shape

    directory = Path(directory)
    warnings: list[str] = []

    path = directory / "osm.json"
    payload: dict = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        warnings.append(MISSING_OSM_WARNING)

    # The first query wrote no version: version 1, as the other tools say.
    version = int(payload.get("query_version") or 1)
    # Below the version that fetched parking, not below the current one: a
    # later bump (4 added distribution power lines) leaves the lots intact.
    if payload and version < OSM_QUERY_VERSION_PARKING:
        warnings.append(STALE_QUERY_WARNING.format(version=version))

    centre_lonlat = None
    try:
        centre_lonlat = geometry_centroid(read_subject(directory).get("geometry") or {})
    except (OSError, ValueError):
        centre_lonlat = None
    if centre_lonlat is None:
        warnings.append(NO_SUBJECT_WARNING)
        return {"candidates": [], "summary": {"count": 0, "total_before_limit": 0,
                                              "truncated_to": None},
                "query_version": version, "center": None, "warnings": warnings}

    centre_lon, centre_lat = centre_lonlat
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    scale = ground_meters_per_mercator_unit(centre_lat)
    geod = Geod(ellps="WGS84")

    shapes = [(site, shapely_shape(site["geometry"])) for site in payload.get("parking") or []]
    # Tested against a point INSIDE the lot rather than the lot's own
    # geometry: a lot on a school's boundary line is inside the school for
    # this purpose, and `contains` on two touching polygons can go False on
    # floating-point noise alone (a lot reprojected to 3857 can land a
    # fraction of a millimetre outside a boundary it is meant to share) -
    # confirmed directly: `school.contains(lot)` flipped to False on such a
    # lot while `school.contains(point_inside(lot))` still read True.
    lots = [geometry for site, geometry in shapes if site.get("kind") != SITE_HARDSTAND]

    named_routes = _named_ways(payload.get("routes") or [])
    ways_tree = _nearest_way_index(named_routes)

    entries: list[dict] = []
    for site, geometry in shapes:
        if geometry.is_empty:
            continue
        if site.get("kind") == SITE_HARDSTAND and any(
                geometry.contains(_point_inside(lot)) for lot in lots):
            continue
        inside = _point_inside(geometry)
        longitude, latitude = to_wgs84.transform(inside.x, inside.y)
        bearing, _back, distance_m = geod.inv(centre_lon, centre_lat, longitude, latitude)
        if max_distance_m is not None and distance_m > float(max_distance_m):
            continue
        properties = site.get("properties") or {}
        entries.append({
            "id": site["id"],
            "kind": site.get("kind"),
            "name": properties.get("name"),
            "point": {"latitude": latitude, "longitude": longitude},
            "area_m2": (geometry.area * scale * scale
                        if geometry.geom_type != "Point" else None),
            "surface": properties.get("surface"),
            "capacity": properties.get("capacity"),
            "flags": list(properties.get("flags") or []),
            "distance_from_center_m": distance_m,
            "bearing_from_center_deg": bearing % 360.0,
            "nearest_way": _nearest_named_way(geometry, named_routes, ways_tree, scale),
            "geometry": to_wgs84_geojson(geometry),
            "warnings": [],
        })

    # Distance, then name, then id: two lots the same distance out keep one
    # order between runs, which is what makes a set of candidates something a
    # second planner can be told about by row number.
    entries.sort(key=lambda entry: (entry["distance_from_center_m"],
                                    entry["name"] or "", entry["id"]))
    total = len(entries)
    truncated = None
    if limit and int(limit) > 0 and total > int(limit):
        entries = entries[:int(limit)]
        truncated = int(limit)

    return {
        "candidates": entries,
        "summary": {"count": len(entries), "total_before_limit": total,
                    "truncated_to": truncated},
        "query_version": version,
        "center": {"latitude": centre_lat, "longitude": centre_lon},
        "warnings": warnings,
    }
