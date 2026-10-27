"""What OpenStreetMap says of the ways around an egress search.

Reading the staged osm.json and what it cannot say; sorting its routes into
the ways a search walks faster, the motorways it may not, and the roads each
kind of exit lies on; and naming what an exit ends on - its road, the lot it
is in, the gates its route passes through. `egress` burns these onto the grid
and walks them; nothing here reads the elevation model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geo import to_wgs84_geojson
from ..terrain.linear import (OSM_QUERY_VERSION_PARKING, OSM_QUERY_VERSION_ROUTE_TAGS,
                              SITE_PARKING, WALKABLE_CLASSES)
from .parking import _nearest_named_way

#: A ROAD with one of these surfaces is a 4x4 exit, not a 2WD one (spec 3.2).
#: Anything else - paved, gravel, compacted, or no surface tag at all - is 2WD:
#: a residential road with no surface tag is almost always paved. OSM stacks
#: values with ";" (`gravel;dirt`), and a road is dirt when ANY part is: the
#: cautious reading, since a 4x4 may be needed (`_surface_parts`).
DIRT_SURFACES = frozenset({"dirt", "ground", "earth", "unpaved", "sand", "mud", "grass"})
#: Nobody walks a freeway out, and no exit is on one (spec 3.1).
MOTORWAYS = frozenset({"motorway", "motorway_link"})
#: A ROAD a vehicle cannot be sent to: walked, never an exit (ruled 2026-09-23).
NOT_AN_EXIT_ROADS = frozenset({"pedestrian"})
#: `highway` values for a way that is not on the ground - not yet, or not any
#: more. Such a way is neither walked faster nor an exit: a route ending on a
#: proposed road ends nowhere.
NOT_BUILT = frozenset({"proposed", "construction", "planned", "abandoned", "disused",
                       "razed"})
#: `highway` values nobody walks a casualty along: a via ferrata is a cabled
#: climb, a guideway a concrete bus track. They are plain ground here.
NOT_WALKED = frozenset({"via_ferrata", "bus_guideway"})
#: A gate this many cells from a route line may be on it: the line is
#: simplified to one cell, so the walked path can lie a cell from it, plus
#: half a cell. It must also bar a way the route walks (GATE_ON_WAY_M).
GATE_ON_ROUTE_CELLS = 1.5
#: An OSM gate is a node ON the way it bars: within this much ground of a
#: way's line, it bars that way; farther, it bars something else - a
#: driveway, a path - that happens to be near.
GATE_ON_WAY_M = 0.5

NO_OSM_ERROR = ("no OpenStreetMap data is staged for this ground, so there is no road "
                "to exit to; stage it again once Overpass answers")
ROAD_FACTS_UNKNOWN_WARNING = (
    "this ground was staged at OpenStreetMap query version {version}, before road class, "
    "access and gates were asked for: road class, access and gates are unknown here, never "
    "\"none\", and a motorway cannot be told from any other road, so a route may cross one "
    "or end on one; stage the area again to read them")
GATES_UNKNOWN_WARNING = (
    "the staged OpenStreetMap data carries no gate list, so gates are unknown on this "
    "ground, never \"none\"; stage the area again with prepare_area and force=True to read "
    "them - a plain re-stage keeps this file as it is, being up to date in all else")
PARKING_UNKNOWN_WARNING = (
    "this ground was staged at OpenStreetMap query version {version}, before parking lots "
    "were asked for: lots are unknown here, never \"none\", and an exit may lie on one this "
    "cannot see; stage the area again to read them")


def _osm(directory: Path) -> dict:
    """The staged osm.json, with `query_version` (absent means 1) and the
    three lists always present. No file is refused: with no roads there is
    nothing to exit to.

    `gates_known` is whether a gate list was asked for: query version 5 or
    later AND a `gates` list in the file. A file without the key - ground
    staged before gates were written, whatever its version - has gates
    UNKNOWN, never "no gates". `parking_known` is whether parking was asked
    for at all: query version 3 or later. Below it lots are UNKNOWN, never
    "no lots" - the same reading `lz_candidates` gives the same ground."""
    path = Path(directory) / "osm.json"
    if not path.is_file():
        raise ValueError(NO_OSM_ERROR)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["query_version"] = int(payload.get("query_version", 1))
    payload["gates_known"] = (payload["query_version"] >= OSM_QUERY_VERSION_ROUTE_TAGS
                              and isinstance(payload.get("gates"), list))
    payload["parking_known"] = payload["query_version"] >= OSM_QUERY_VERSION_PARKING
    for key in ("routes", "parking", "gates"):
        payload[key] = payload.get(key) or []
    return payload


def _osm_warnings(osm: dict) -> list[str]:
    """What the staged OpenStreetMap data cannot say on this ground."""
    warnings = []
    if osm["query_version"] < OSM_QUERY_VERSION_ROUTE_TAGS:
        warnings.append(ROAD_FACTS_UNKNOWN_WARNING.format(version=osm["query_version"]))
    elif not osm["gates_known"]:
        warnings.append(GATES_UNKNOWN_WARNING)
    if not osm["parking_known"]:
        warnings.append(PARKING_UNKNOWN_WARNING.format(version=osm["query_version"]))
    return warnings


def _surface_parts(value) -> list[str]:
    """A `surface` split on ";", each part stripped and lowercased, as
    `linear._access_parts` reads `access`."""
    return [stripped for part in str(value or "").lower().split(";")
            if (stripped := part.strip())]


def _classify(routes: list[dict]) -> dict[str, list[tuple[dict, object]]]:
    """The staged routes as (record, EPSG:3857 geometry) pairs, sorted into
    `ways` (walked faster), `motorways` (closed; ramps included), `links`
    (the ramps alone), `two_wd` and `four_wd` (the ways each kind of exit is
    on). The rules are `_way_masks`' docstring's."""
    from shapely.geometry import shape as shapely_shape

    classified: dict[str, list] = {"ways": [], "motorways": [], "links": [],
                                   "two_wd": [], "four_wd": []}
    for route in routes:
        properties = route.get("properties") or {}
        route_type = properties.get("route_type")
        highway = str(properties.get("highway") or "").strip().lower()
        if route_type not in WALKABLE_CLASSES or highway in NOT_BUILT | NOT_WALKED:
            continue
        pair = (route, shapely_shape(route["geometry"]))
        if highway in MOTORWAYS:
            classified["motorways"].append(pair)
            if highway == "motorway_link":
                classified["links"].append(pair)
            continue
        classified["ways"].append(pair)
        dirt = any(part in DIRT_SURFACES for part in _surface_parts(properties.get("surface")))
        road = route_type == "ROAD" and highway not in NOT_AN_EXIT_ROADS
        if road and not dirt:
            classified["two_wd"].append(pair)
        if route_type == "TRACK" or (road and dirt):
            classified["four_wd"].append(pair)
    return classified


def _lots(parking: list[dict]) -> list[tuple[dict, object]]:
    """The mapped parking lots (kind `parking`, polygon or node) as (record,
    EPSG:3857 geometry) pairs; a hardstand is not a lot."""
    from shapely.geometry import shape as shapely_shape

    return [(row, shapely_shape(row["geometry"])) for row in parking
            if row.get("kind") == SITE_PARKING and row.get("geometry")]


def _exit_road(point, ways: list[tuple[dict, object]], tree, *, reach: float,
               gate_points: list | None, gate_reach: float) -> dict | None:
    """The nearest way of the exit's own kind within `reach` (a cell's
    diagonal, map units) of the exit point, or None - an exit on a parking
    lot can have no road beside it. Ties go to the name, so two runs agree.

    `gate_on_road` is whether a gate lies ON that way - within `gate_reach`
    (GATE_ON_WAY_M of ground) of its line - ANYWHERE in the staged area: the
    4x4 may need the combination to get to it. A gate up a side driveway is
    not on the road. None, never False, when gates are unknown."""
    if tree is None:
        return None
    indices, distances = tree.query_nearest(point, max_distance=reach,
                                            return_distance=True, all_matches=True)
    if len(indices) == 0:
        return None
    _, index = min(zip(distances, indices), key=lambda pair: (
        float(pair[0]), str((ways[int(pair[1])][0].get("properties") or {}).get("name") or "")))
    route, geometry = ways[int(index)]
    properties = route.get("properties") or {}
    return {"name": properties.get("name"),
            "route_type": properties.get("route_type"),
            "highway": properties.get("highway"),
            "surface": properties.get("surface"),
            "access": properties.get("access"),
            "gate_on_road": (None if gate_points is None else
                             any(geometry.distance(gate) <= gate_reach
                                 for _, gate in gate_points))}


def _gates_crossed(line, path_area: list, gate_points: list, ways: list, ways_tree, *,
                   reach: float, on_way: float) -> list[dict]:
    """The gates the route passes through, as {geometry (WGS84 Point),
    barrier, access}: within `reach` (GATE_ON_ROUTE_CELLS) of the route's
    line, AND within `on_way` (GATE_ON_WAY_M of ground) of a way the route
    FOLLOWS - one running through at least two of the route's cells
    (`path_area`, the cells' footprints). A way that only touches the route
    at one cell - a driveway leaving it, a trail it steps across - is passed,
    not followed, and a gate on it is not crossed, however near it stands.

    `path_area` is a list of footprints rather than one MultiPolygon: squares
    sharing edges make an invalid MultiPolygon, which predicates do not
    promise to handle."""
    near = [(record, point) for record, point in gate_points
            if line.distance(point) <= reach]
    if not near or ways_tree is None:
        return []
    # (footprint, way) pairs that meet; each pair once, so a way's count is
    # the number of distinct route cells it runs through.
    _, way_indices = ways_tree.query(np.asarray(path_area, dtype=object),
                                     predicate="intersects")
    cells_per_way = np.bincount(np.asarray(way_indices, dtype=np.int64),
                                minlength=len(ways))
    walked = [ways[int(index)][1] for index in np.flatnonzero(cells_per_way >= 2)]
    return [{"geometry": to_wgs84_geojson(point),
             "barrier": (record.get("properties") or {}).get("barrier"),
             "access": (record.get("properties") or {}).get("access")}
            for record, point in near
            if any(way.distance(point) <= on_way for way in walked)]


def _parking_at(cell, lots: list[tuple[dict, object]], named: list[dict], named_tree,
                scale: float) -> dict | None:
    """The lot the exit cell lies in, as {id, off}, or None. `off` is the
    road the lot opens onto, named exactly as the staging-area tab names it
    (`parking._nearest_named_way`). Where lots overlap the cell, the one
    nearest its centre wins."""
    touching = [(geometry.distance(cell.centroid), str(row.get("id")), row, geometry)
                for row, geometry in lots if geometry.intersects(cell)]
    if not touching:
        return None
    _, _, row, geometry = min(touching, key=lambda entry: (entry[0], entry[1]))
    nearest = _nearest_named_way(geometry, named, named_tree, scale)
    return {"id": row.get("id"), "off": nearest["name"] if nearest else None}
