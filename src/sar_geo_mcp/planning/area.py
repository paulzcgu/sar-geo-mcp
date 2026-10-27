"""A planner-chosen area, staged behind an immutable handle.

WHY AN AREA MAY BE STAGED WHEN ARBITRARY GEOMETRY COULD NOT
-----------------------------------------------------------
The README refused to stage geometry that is on no map, because a staged
assignment can move on the map and a free-floating polygon has nothing to
revalidate against. A supplied point and radius cannot move: the input IS
the fingerprint, so revalidation is a constant. Map context an area is
compared against (other assignments, tracks) is still fetched fresh and
reported with provenance.

THE KEY
-------
`aoi-` plus sixteen hex characters of a SHA-256 over the canonical request:
kind, coordinates rounded to seven decimals (about a centimetre), radius
rounded to a tenth of a metre, the map id or null, and the planning buffer.
Two agents staging the same area on the same map share one directory. The
prefix cannot collide with an assignment handle, whose first character is a
map id's alphanumeric.

The buffer is in the key because it is not a detail of how the ground was
fetched, it is how much ground there IS. A line candidate runs to the edge
of the staged raster, so a directory staged with a smaller buffer answers a
different question under the same name. Changing the buffer therefore
changes every handle: a directory staged with the old one is never asked for
again and is left for the cache to evict, rather than quietly serving a
raster its consumer no longer assumes.

THE EXTENT LIMIT
----------------
9,000 m across before the buffer. With the 1,000 m planning buffer that is
11,000 m over 1,024 pixels, or 10.7 m per pixel - a little past the 10 m the
terrain maths assumes, and only at the very widest area anyone can ask for.
The 1,500 m radius a planner actually stages is 5,000 m over 1,024 pixels,
4.9 m. A wider request is refused rather than staged coarser still, because
everything downstream is a measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..geo import geometry_bounds_wgs84, ground_meters_per_mercator_unit, to_web_mercator

#: Longest side of the area itself, in ground metres, before the buffer.
MAX_EXTENT_M = 9000.0

#: Ground metres of margin staged around a PLANNING area - its own constant,
#: not the margin an assignment is staged with (`staging.DEFAULT_BUFFER_M`,
#: 200 m), because the two are sized for different questions. An assignment's
#: buffer is the ground beside a boundary searchers stray onto. A planning
#: area's buffer is how far a candidate may RUN: a line candidate is offered
#: to the edge of the staged raster, so the buffer decides whether a team is
#: handed a trail or the arc where the circle happened to cut it.
#:
#: WHY A KILOMETRE, MEASURED
#: Windy Hill, a 1,500 m area at 37.35687, -122.24093. Four trails enter the
#: circle; between them the circle cut off 3,073 m of trail (Spring Ridge
#: 2,788 m of its 4,584; Razorback Ridge 2,094 of 4,402; Hamms Gulch 979 of
#: 4,470), and the farthest END of any of them is 2,242 m from the centre -
#: 742 m outside the circle. A 1,000 m buffer covers every one with room to
#: spare, and the four trails come back whole.
#:
#: THE COST, HONESTLY
#: The staged raster around that area grows from a 2,000 m radius to a
#: 2,500 m one: 1.56x the ground, and at the fixed 1,024-pixel raster
#: 4.9 ground metres per pixel where it was 3.9. Elevation, vegetation and
#: Overpass are all fetched over that wider box, so a first staging is
#: correspondingly slower. Roads are why this is not larger: 6 roads enter
#: the same circle and 25,441 m of them is cut, and Skyline Boulevard runs
#: for tens of kilometres - there is no buffer that draws a road to its end.
PLANNING_AREA_BUFFER_M = 1000.0

AREA_PREFIX = "aoi-"

#: Vertices on the circle polygon; a 1.5 km radius at 64 vertices is within
#: a metre of the true circle.
CIRCLE_SEGMENTS = 64

#: Ground metres in one of each unit a radius may be given in. A planner
#: says "half a mile" or "800 feet"; the conversion is done here, once, with
#: the international definitions, so no agent converts by hand and the
#: handle is the same whichever unit named the same ground.
RADIUS_UNITS = {"m": 1.0, "km": 1000.0, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
RadiusUnit = Literal["m", "km", "ft", "yd", "mi"]


def radius_in_metres(radius_m: float | None = None, radius: float | None = None,
                     radius_unit: str = "m") -> float | None:
    """The radius in ground metres from either `radius_m` or `radius` in
    `radius_unit`, never both; None when neither is given.

    The unit is checked whatever is given, and a unit other than metres
    beside `radius_m` is refused: `radius_m` is metres by its name, so
    `radius_m=2, radius_unit="km"` can only be two kilometres meant and two
    metres read - a search circle a thousand times too small, staged and
    answered as though nothing were wrong."""
    unit = (radius_unit or "").strip().lower()
    if unit not in RADIUS_UNITS:
        raise ValueError(f"radius_unit {radius_unit!r} is not one of "
                         f"{', '.join(RADIUS_UNITS)}")
    if radius is None:
        if radius_m is not None and unit != "m":
            raise ValueError(f"radius_unit {radius_unit!r} applies to radius, not to radius_m, "
                             f"which is always metres: for {radius_m!r} {unit} give "
                             f"radius={radius_m!r}, radius_unit={unit!r}; for {radius_m!r} m "
                             "leave radius_unit out")
        return radius_m
    if radius_m is not None:
        raise ValueError("give the radius once: radius_m, or radius with radius_unit")
    return float(radius) * RADIUS_UNITS[unit]


class AreaOfInterest(BaseModel):
    """The staged subject for an area handle.

    `label` mirrors `Assignment.label` so `stage_terrain` treats both alike.
    """

    kind: Literal["circle", "polygon"]
    center: tuple[float, float] | None = None   # (latitude, longitude)
    radius_m: float | None = None
    geometry: dict                               # WGS84 Polygon
    map_id: str | None = None
    key: str
    extent_m: float
    #: Ground metres per pixel the caller asked for, or None for the default
    #: 1,024-pixel raster. Recorded so the staged directory says what it is.
    resolution_m: float | None = None

    @property
    def label(self) -> str:
        return self.key


def area_key(kind: str, coordinates, radius_m: float | None, map_id: str | None,
             buffer_m: float = PLANNING_AREA_BUFFER_M,
             resolution_m: float | None = None) -> str:
    """The handle for a request, stable under float noise.

    `buffer_m` is part of the request, not a detail of it: the staged raster
    is the area plus the buffer, and a line candidate runs to the edge of
    that raster. Two stagings differing only by the buffer are two different
    grounds and get two different directories.

    `resolution_m` joins it for the same reason. A 6 m raster and a 1 m raster
    over identical ground answer different questions - at 6 m neither El Corte
    de Madera nor Purisima contains a single cliff, and at 1 m both do - so
    serving one under the other's handle would be the same silent substitution
    the buffer rule exists to prevent.

    It is omitted from the payload entirely when None, so every handle staged
    before this parameter existed still resolves to its own directory.
    """

    def rounded(value):
        if isinstance(value, float):
            return round(value, 7)
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        return value

    payload = {
        "kind": kind,
        "coordinates": rounded(coordinates),
        "radius_m": None if radius_m is None else round(float(radius_m), 1),
        "map_id": map_id or None,
        "buffer_m": round(float(buffer_m), 1),
    }
    if resolution_m is not None:
        payload["resolution_m"] = round(float(resolution_m), 3)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return AREA_PREFIX + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _circle_polygon(latitude: float, longitude: float, radius_m: float) -> dict:
    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    centre = shapely_transform(to_3857.transform, Point(longitude, latitude))
    radius_units = radius_m / ground_meters_per_mercator_unit(latitude)
    circle = centre.buffer(radius_units, quad_segs=CIRCLE_SEGMENTS // 4)
    geometry = mapping(shapely_transform(to_wgs84.transform, circle))
    # shapely's coordinates are tuples; round-trip through JSON so the
    # in-memory geometry matches what `AreaOfInterest.model_validate_json`
    # hands back (JSON arrays deserialize as lists, and a tuple != a list).
    return json.loads(json.dumps(geometry))


def _extent_m(geometry: dict) -> float:
    west, south, east, north = geometry_bounds_wgs84(geometry)
    minx, miny, maxx, maxy = to_web_mercator((west, south, east, north))
    scale = ground_meters_per_mercator_unit((south + north) / 2.0)
    return max(maxx - minx, maxy - miny) * scale


def build_area(*, center_latitude: float | None = None,
               center_longitude: float | None = None,
               radius_m: float | None = None,
               geometry: dict | None = None,
               map_id: str | None = None,
               resolution_m: float | None = None,
               radius: float | None = None,
               radius_unit: str = "m") -> AreaOfInterest:
    """Exactly one of a centre plus radius or a WGS84 polygon.

    The radius is `radius_m`, or `radius` in `radius_unit` (m, km, ft, yd,
    mi) - converted to metres before the handle is made, so half a mile and
    804.672 m are one handle. `resolution_m` asks for ground metres per pixel
    instead of the default 1,024-pixel raster, and travels into the handle -
    see `area_key`.
    """
    radius_m = radius_in_metres(radius_m, radius, radius_unit)
    if resolution_m is not None and resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive; got {resolution_m}")
    has_centre = center_latitude is not None or center_longitude is not None or radius_m is not None
    if has_centre == (geometry is not None):
        raise ValueError(
            "give exactly one of (center_latitude, center_longitude, radius_m) "
            "or a GeoJSON polygon `geometry`")
    if has_centre:
        if center_latitude is None or center_longitude is None or radius_m is None:
            raise ValueError("a circle needs center_latitude, center_longitude and radius_m")
        if radius_m <= 0:
            raise ValueError("radius_m must be positive")
        polygon = _circle_polygon(center_latitude, center_longitude, radius_m)
        kind, centre = "circle", (float(center_latitude), float(center_longitude))
        key = area_key(kind, [centre[0], centre[1]], radius_m, map_id,
                       resolution_m=resolution_m)
    else:
        if geometry.get("type") != "Polygon":
            raise ValueError("geometry must be a GeoJSON Polygon in WGS84")
        polygon, kind, centre = geometry, "polygon", None
        key = area_key(kind, geometry["coordinates"], None, map_id,
                       resolution_m=resolution_m)

    extent = _extent_m(polygon)
    if extent > MAX_EXTENT_M:
        raise ValueError(
            f"the area is {extent:.0f} m across; the limit is {MAX_EXTENT_M:.0f} m "
            f"so the staged raster stays at 10 m. Use a smaller radius or polygon.")

    return AreaOfInterest(kind=kind, center=centre, radius_m=radius_m,
                          geometry=polygon, map_id=map_id or None, key=key,
                          extent_m=extent, resolution_m=resolution_m)


def area_directory(cache_dir: Path, key: str) -> Path:
    from ..handles import STAGED_DIRNAME

    return Path(cache_dir) / STAGED_DIRNAME / key


def prepare_area(reader, *, center_latitude: float | None = None,
                 center_longitude: float | None = None,
                 radius_m: float | None = None,
                 geometry: dict | None = None,
                 map_id: str = "", public: bool = True,
                 force: bool = False,
                 resolution_m: float | None = None,
                 radius: float | None = None,
                 radius_unit: str = "m") -> dict:
    """Stage the ground for an area and return its handle.

    `reader` is the server's `MapReader`; it is read only when `map_id` is
    given, for the provenance of the map context, and an unreadable map is
    a warning rather than a failure: the ground can be staged offline.
    `public` says which of its paths the map is read by, here and on every
    later read of the map for this handle: the path of the call that
    staged the ground, or of a later call that found it already staged and
    read the map by it. A later call whose own read by its path failed
    leaves the recorded path as it was.

    The radius may be `radius_m`, or `radius` in `radius_unit` (m, km, ft,
    yd, mi). The result carries `area`: the kind, centre, radius in metres
    and the WGS84 outline of the area itself (the circle, not the buffered
    raster), to draw the range on a map.
    """
    from ..handles import record_access
    from ..staging import (
        STAGE_PIXELS,
        manifest_warnings,
        pixels_for_resolution,
        refresh_osm,
        stage_terrain,
    )

    if reader.cache_dir is None:
        raise ValueError("staging needs a cache directory; none is configured")

    area = build_area(center_latitude=center_latitude, center_longitude=center_longitude,
                      radius_m=radius_m, geometry=geometry, map_id=map_id or None,
                      resolution_m=resolution_m, radius=radius, radius_unit=radius_unit)
    drawn = {"kind": area.kind,
             "center": ({"latitude": area.center[0], "longitude": area.center[1]}
                        if area.center else None),
             "radius_m": area.radius_m, "geometry": area.geometry}
    # Resolved before any network call so an unservable request fails at once,
    # naming the finest this extent can be staged at.
    stage_pixels = STAGE_PIXELS
    if resolution_m is not None:
        stage_pixels = pixels_for_resolution(
            area.extent_m + 2 * PLANNING_AREA_BUFFER_M, resolution_m)
    directory = area_directory(reader.cache_dir, area.key)
    warnings: list[str] = []

    map_provenance = None
    if map_id:
        try:
            state = reader.read(map_id, public=public, max_age_seconds=0)
            map_provenance = state.provenance()
        except Exception as error:
            warnings.append(f"could not read map {map_id} ({type(error).__name__}); "
                            "the ground is staged, map context is unavailable")

    if (directory / "manifest.json").exists() and not force:
        # A directory this old may hold an `osm.json` behind the current
        # query (0.11.0 added parking; any later bump repeats this) - top
        # it up here rather than leaving `parking_candidates` to explain,
        # every time, why nothing can be read from it. See
        # `staging.refresh_osm` for what this costs when there is nothing
        # to do: one file read, never a network call.
        warnings.extend(refresh_osm(directory, reader.cache_dir))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        # From the file on disk, not from the warnings just collected: a
        # directory staged during an outage and topped up by `refresh_osm`
        # two minutes ago carries warnings ABOUT that outage and a complete
        # osm.json. The question a caller is asking is "can trails, roads
        # and parking be read from this ground right now", and only the
        # file answers that. On 2026-09-21 the app had no field to ask, so
        # it inferred the answer by string-matching a hub warning constant
        # - and the hub has three differently-worded constants all named
        # `MISSING_OSM_WARNING` (boundary.py, parking.py, routes.py), so the
        # app's match only ever caught one of them. (Until 0.17.1 that
        # outage warning was also returned for ever after the top-up; it is
        # now retired once osm.json exists - see `manifest_warnings` - but
        # the field still comes from the file, not from which warnings are
        # left.)
        osm_available = (directory / "osm.json").is_file()
        # The key hashes the map id but not `public`: how the map is read
        # does not change the ground. So a call corrected from public=True
        # to public=False on a private map lands here, its own read having
        # worked - and every later read of the map, each section's among
        # them, goes by the path recorded for this ground. The path this
        # call has just read the map by is made that path; nothing is
        # staged again (PR #4 review B49). Only a path that worked: a later
        # call leaving `public` at its default true on a private map would
        # otherwise move a handle that reads the map off the one path that
        # does, every section after it losing the map. That call's own
        # warning says its read failed; the path recorded is left as it was.
        if map_id and map_provenance is not None:
            record_access(directory, public)
        return {"handle": area.key, "staged": False, "reason": "already staged",
                "extent_m": area.extent_m,
                "dem_resolution_m": manifest.get("dem_resolution_m"),
                # Not the raw list: an outage recorded at staging is retired
                # once any osm.json exists (`staging.manifest_warnings`).
                "warnings": warnings + manifest_warnings(directory, manifest),
                "map_provenance": map_provenance,
                "osm_available": osm_available, "area": drawn}

    result = stage_terrain(
        area, map_id, directory, cache_dir=reader.cache_dir,
        buffer_m=PLANNING_AREA_BUFFER_M, fingerprint=area.key.removeprefix(AREA_PREFIX),
        access="public" if public else "authenticated", subject_kind="area",
        stage_pixels=stage_pixels)
    # Same reasoning as the already-staged branch above: ask the file, not
    # the warnings `stage_terrain` just returned.
    osm_available = (directory / "osm.json").is_file()
    return {"handle": area.key, "staged": True,
            "reason": "forced" if force else "not staged before",
            "extent_m": area.extent_m,
            "dem_resolution_m": result.get("dem_resolution_m"),
            "warnings": warnings + list(result["warnings"]),
            "map_provenance": map_provenance,
            "osm_available": osm_available, "area": drawn}
