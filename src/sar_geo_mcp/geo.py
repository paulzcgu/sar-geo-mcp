"""Pure coordinate math. No network, no I/O."""

import json
import math
import numbers
from dataclasses import dataclass

from pyproj import Transformer

Bbox = tuple[float, float, float, float]

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _coordinate_positions(coordinates) -> list[tuple[float, float]]:
    """Flatten arbitrarily nested GeoJSON coordinate arrays down to positions."""
    if not coordinates:
        return []
    if isinstance(coordinates[0], (int, float)):
        return [(float(coordinates[0]), float(coordinates[1]))]
    positions: list[tuple[float, float]] = []
    for item in coordinates:
        positions.extend(_coordinate_positions(item))
    return positions


_BOUNDABLE_GEOMETRIES = frozenset(
    {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
)


def geometry_bounds_wgs84(geometry: dict) -> Bbox:
    """Bounds for any GeoJSON geometry CalTopo attaches to an assignment.

    Area assignments are Polygons; line assignments (roads, drainages) are
    LineStrings. Both must work.
    """
    geometry_type = geometry.get("type")
    if geometry_type not in _BOUNDABLE_GEOMETRIES:
        raise ValueError(
            f"unsupported geometry type {geometry_type!r}; "
            f"expected one of {sorted(_BOUNDABLE_GEOMETRIES)}"
        )

    positions = _coordinate_positions(geometry.get("coordinates") or [])
    if not positions:
        raise ValueError(f"{geometry_type} geometry has no coordinates")

    longitudes = [position[0] for position in positions]
    latitudes = [position[1] for position in positions]
    return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))


def polygon_bounds_wgs84(geometry: dict) -> Bbox:
    """Strict Polygon-only bounds. Use geometry_bounds_wgs84 to accept lines too."""
    geometry_type = geometry.get("type")
    if geometry_type != "Polygon":
        raise ValueError(f"expected Polygon geometry, got {geometry_type!r}")
    return geometry_bounds_wgs84(geometry)


def bbox_center_latitude(bbox_wgs84: Bbox) -> float:
    return (bbox_wgs84[1] + bbox_wgs84[3]) / 2.0


def to_web_mercator(bbox_wgs84: Bbox) -> Bbox:
    """Project a WGS84 bbox to EPSG:3857, the only CRS CalTopo's WMS accepts."""
    min_x, min_y = _TO_3857.transform(bbox_wgs84[0], bbox_wgs84[1])
    max_x, max_y = _TO_3857.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (min_x, min_y, max_x, max_y)


def to_wgs84_geojson(geometry_3857) -> dict:
    """A shapely EPSG:3857 geometry as plain GeoJSON in WGS84.

    THE RULE THIS ENFORCES
    ----------------------
    GeoJSON without a stated CRS is WGS84 by definition (RFC 7946), and every
    tool in this package emits its coordinate fields suffixed `_wgs84` beside
    a `crs` that describes the RASTER rather than the geometry. So a
    `geometry` handed to a caller is reprojected HERE, at the emission
    boundary, and never carries Mercator units out of the package.

    Measuring in Web Mercator and emitting in WGS84 is the pattern the
    planning layer already follows - `corridors.py` (which this function was
    lifted out of), `parking.py` and `planning/area.py` all did their own
    copy of it. One copy, so the next emitter gets it by importing rather
    than by remembering.

    Why it matters, measured: a lineament emitted in 3857 reads
    `(-13618745.5, 4486605.0)`; run through `site_facts`'s documented
    WGS84 -> 3857 reprojection - which `slope_shading`'s docstring advertises
    as the next step for a patch - that becomes `(inf, inf)`. The advertised
    workflow cannot work on un-reprojected output, and nothing in either
    envelope says why.

    The `json` round trip converts shapely's tuples to lists so the result is
    JSON-identical to what a caller receives on the wire, rather than
    something that merely serialises the same.
    """
    from shapely.geometry import mapping
    from shapely.ops import transform as shapely_transform

    return json.loads(json.dumps(
        mapping(shapely_transform(_TO_4326.transform, geometry_3857))))


def bbox_center_wgs84(bbox_3857: Bbox) -> tuple[float, float]:
    """The (latitude, longitude) centre of a Web Mercator bbox.

    The one place a stored `bbox_3857` gets turned back into somewhere on
    the globe. Used to give the coordinate parser a reference point - it can
    only resolve a transposed or unsigned pair against somewhere it knows
    the assignment actually is (see debrief/coords.py).
    """
    min_x, min_y, max_x, max_y = bbox_3857
    longitude, latitude = _TO_4326.transform((min_x + max_x) / 2,
                                             (min_y + max_y) / 2)
    return latitude, longitude


#: Square metres in one international acre. Areas are given in acres, the
#: unit a search planner works in; the corpus and the grouping keep their
#: own figures and are converted where an answer is written.
M2_PER_ACRE = 4046.8564224
#: Hectares in one acre, for converting the corpus's hectare tables.
HA_PER_ACRE = M2_PER_ACRE / 10_000.0


def acres(area_m2: float) -> float:
    """`area_m2` square metres in acres."""
    return area_m2 / M2_PER_ACRE


def ground_meters_per_mercator_unit(latitude_deg: float) -> float:
    """Web Mercator units are inflated by 1/cos(latitude); this is the inverse.

    Multiply a Web Mercator distance by this to get true ground metres.
    """
    return math.cos(math.radians(latitude_deg))


def buffer_bbox(bbox_3857: Bbox, ground_meters: float, latitude_deg: float) -> Bbox:
    """Expand a Web Mercator bbox by a true ground distance.

    Web Mercator units are inflated by 1 / cos(latitude), so buffering by raw
    metres would under-buffer everywhere off the equator.
    """
    scale = 1.0 / math.cos(math.radians(latitude_deg))
    delta = ground_meters * scale
    return (
        bbox_3857[0] - delta,
        bbox_3857[1] - delta,
        bbox_3857[2] + delta,
        bbox_3857[3] + delta,
    )


@dataclass(frozen=True)
class Tile:
    """One WMS request within a larger mosaic."""

    col: int
    row: int
    bbox: Bbox
    width_px: int
    height_px: int
    x_off: int
    y_off: int


def _spans(total_px: int, max_px: int) -> list[tuple[int, int]]:
    """Split a pixel length into (offset, size) chunks of at most max_px."""
    spans = []
    offset = 0
    while offset < total_px:
        spans.append((offset, min(max_px, total_px - offset)))
        offset += max_px
    return spans


def plan_tiles(
    bbox_3857: Bbox,
    width_px: int,
    height_px: int,
    max_px: int = 1024,
) -> list[Tile]:
    """Split a request into tiles no larger than CalTopo's 1024 px GetMap ceiling.

    Image row 0 is the top of the image, which is maximum northing in Web
    Mercator, so Y offsets count down from the bbox's max_y.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"width and height must be positive, got {width_px}x{height_px}")

    min_x, min_y, max_x, max_y = bbox_3857
    x_per_px = (max_x - min_x) / width_px
    y_per_px = (max_y - min_y) / height_px

    tiles = []
    for row, (y_off, tile_height) in enumerate(_spans(height_px, max_px)):
        top = max_y - y_off * y_per_px
        bottom = top - tile_height * y_per_px
        for col, (x_off, tile_width) in enumerate(_spans(width_px, max_px)):
            left = min_x + x_off * x_per_px
            right = left + tile_width * x_per_px
            tiles.append(
                Tile(
                    col=col,
                    row=row,
                    bbox=(left, bottom, right, top),
                    width_px=tile_width,
                    height_px=tile_height,
                    x_off=x_off,
                    y_off=y_off,
                )
            )
    return tiles


# ---------------------------------------------------------------------------
# Centroids.
#
# Moved here from the debrief app's interview planner, which is where they
# were first needed. They are geometry computation, so every consumer of
# this package wants them, and leaving them there meant the next caller
# would write them again.
#
# Both return None rather than raising, and neither ever guesses. A
# geometry that will not parse, or that parses to nothing, has no centre -
# and inventing one would put a marker somewhere nobody chose.
# ---------------------------------------------------------------------------


def parse_geometry(geometry: dict):
    """A GeoJSON dict as a shapely geometry, or None if it is not usable.

    Empty geometry is a real case, not a malformed one: a placeholder like
    `{"type": "Polygon", "coordinates": []}` turns up in fixtures and in
    partially-drawn CalTopo features.
    """
    if not geometry:
        return None
    try:
        from shapely.geometry import shape

        parsed = shape(geometry)
    except Exception:
        return None
    return None if parsed.is_empty else parsed


def geometry_centroid(geometry: dict) -> tuple[float, float] | None:
    """One geometry's centroid as a plain (x, y), in its own coordinates."""
    parsed = parse_geometry(geometry)
    if parsed is None:
        return None
    centroid = parsed.centroid
    return None if centroid.is_empty else (centroid.x, centroid.y)


def union_centroid(geometries: list[dict]) -> tuple[float, float] | None:
    """The centroid of everything taken together.

    Used where no single boundary is available and the best "where is the
    middle of this" reference is the union of whatever shapes are present.
    """
    parsed = [g for g in (parse_geometry(item) for item in geometries)
              if g is not None]
    if not parsed:
        return None
    from shapely.ops import unary_union

    unioned = unary_union(parsed)
    if unioned.is_empty:
        return None
    return (unioned.centroid.x, unioned.centroid.y)


# ---------------------------------------------------------------------------
# Distance to an edge: how far inside or outside a geometry a point lies.
# ---------------------------------------------------------------------------

#: How short a piece of a boundary is measured to, in degrees: about 110 m of
#: latitude. GeoJSON draws an edge straight in longitude and latitude, which
#: is the edge `contains` answers against, and a projection bends a long one:
#: measured between its projected corners alone, a point lying on a
#: 0.5-degree diagonal edge read 76 m from it, and one 10 m inside a
#: 0.5-degree east-west edge read 39 m. Cut into pieces this short, each one
#: projects true to well under a millimetre.
BOUNDARY_PIECE_DEGREES = 0.001
#: The most pieces one measurement cuts a boundary into. A boundary longer
#: than 200 degrees is cut into longer pieces rather than millions of them.
BOUNDARY_MAX_PIECES = 200_000


def _finite_number(value) -> bool:
    """A real number that is not NaN or infinite. A bool is not one, though
    Python counts it as an int: True passed as a latitude is a mistake."""
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value))


def require_point(latitude: float, longitude: float) -> None:
    """Refuse a point that is not on the globe, naming the coordinate.

    A latitude past 90 degrees is not somewhere a measurement can start
    from: before this check, latitude 95 read -558,651 m from a boundary
    and a latitude that was not a number read NaN, which is not JSON. A
    string is refused by name too, not left to fail a comparison with a
    TypeError that names nothing the caller passed. Worded as the
    scenarios' `check_point` words the same refusal, so a caller meets one
    phrasing for it.
    """
    for name, value, limit in (("latitude", latitude, 90.0),
                               ("longitude", longitude, 180.0)):
        if _finite_number(value) and abs(value) <= limit:
            continue
        # The longitude in the latitude's place is the usual way to get a
        # latitude past 90, so the refusal asks - but only when the pair
        # swapped would be on the globe. Asked of a pair that cannot be
        # swapped either, as (95, -122.31), the question would send the
        # caller to the swapped pair, refused with the same question.
        swappable = (name == "latitude"
                     and _finite_number(latitude) and abs(latitude) <= 180.0
                     and _finite_number(longitude) and abs(longitude) <= 90.0)
        raise ValueError(
            f"{name} must be a number within -{limit:g}..{limit:g}; got {value!r}"
            + (". Was the longitude passed as the latitude?" if swappable else ""))


def _edges(parsed) -> list:
    """The parts a point's distance to `parsed` is measured to, each a single
    point or line: every ring of a polygon - holes and every part included -
    and a line or point itself, however a collection nests them. An empty
    part is left out, having nothing in it to measure to."""
    import shapely

    if parsed.is_empty:
        return []
    if parsed.geom_type in ("Point", "LineString", "LinearRing"):
        return [parsed]
    if parsed.geom_type in ("Polygon", "MultiPolygon"):
        return _edges(parsed.boundary)
    return [edge for part in shapely.get_parts(parsed) for edge in _edges(part)]


def distance_to_boundary_m(latitude: float, longitude: float,
                           geometry: dict) -> float:
    """Ground metres from a point to the nearest edge of `geometry`, the
    point inside it or out.

    The nearest edge is any ring: a hole's edge, and the edge of every part
    of a MultiPolygon, not only the outer ring of the first. So a point deep
    inside one part of a multi-part assignment reads how far inside it is,
    not 0 m - which would put it on the line - and a point in an inholding
    reads how far it is from the inholding's edge. A line or a point is
    measured to itself, and a line whose positions are all one place - a
    track part of a single fix - to that place.

    Measured in an azimuthal equidistant projection centred on the point,
    where the distance from the centre to anything is its true ground
    distance on the WGS84 ellipsoid, in every direction. One
    degrees-to-metres scale cannot do that: 111,320 m x cos(latitude) is
    right east-west and reads north-south short by that same cos(latitude),
    79 % of the ground at 37.4 N and half of it at 60 N. The boundary is
    cut into short pieces first (`BOUNDARY_PIECE_DEGREES`), so the edge
    measured to is the one GeoJSON draws. Near the far side of the Earth
    from the point, where that projection fails, the answer is held to
    within half a piece (about 56 m) of the truth by the pieces' ends,
    measured on the ellipsoid one by one.

    Refuses, with a ValueError saying what is wrong, rather than answer a
    negative or NaN figure: a point not on the globe (`require_point`), a
    geometry with no coordinates (`"coordinates": []`), a geometry with a
    coordinate that is not a number, and one with a latitude past a pole.
    """
    import numpy as np
    import shapely
    from shapely.geometry import GeometryCollection, Point, shape

    require_point(latitude, longitude)
    parts = _edges(shape(geometry))
    if not parts:
        raise ValueError(
            "the geometry has no coordinates, so it has no boundary to measure "
            "a distance to")
    # Checked rather than left to the measurement: a NaN vertex came back
    # from it as 0.0001 m, a point on the line, and a latitude past the
    # pole is most often a [latitude, longitude] pair the wrong way round
    # for GeoJSON. Refused in two messages, as `buffer_geometry_wgs84`
    # refuses them, so the caller knows which it was and, past a pole,
    # which figure.
    coordinates = shapely.get_coordinates(parts)
    if not np.isfinite(coordinates).all():
        raise ValueError(
            "a coordinate in the geometry is not a number, so no distance can "
            "be measured to its boundary")
    latitudes = coordinates[:, 1]
    if (np.abs(latitudes) > 90.0).any():
        offending = float(latitudes[np.argmax(np.abs(latitudes))])
        raise ValueError(
            f"the geometry has a latitude of {offending}, past a pole, so no "
            f"distance can be measured to its boundary; GeoJSON coordinates "
            f"are [longitude, latitude]")
    # A line whose positions are all one place - valid GeoJSON, which asks
    # only for two positions, and what a track part of a single fix is -
    # has no length to cut, and GEOS refuses to cut it with an error that
    # names nothing the caller passed. Measured to, it is that place.
    edges = GeometryCollection([
        Point(part.coords[0]) if part.geom_type != "Point" and part.length == 0
        else part
        for part in parts])
    piece = max(BOUNDARY_PIECE_DEGREES, edges.length / BOUNDARY_MAX_PIECES)
    pieces = shapely.segmentize(edges, piece)

    # The end of every piece, measured on the ellipsoid itself. The nearest
    # edge is no farther than the nearest end, and no nearer than half a
    # piece short of it: a piece is at most `piece` degrees long, and a
    # degree is at most 111,700 m of ground.
    from pyproj import Geod

    ends = shapely.get_coordinates(pieces)
    to_ends = Geod(ellps="WGS84").inv(
        np.full(len(ends), float(longitude)), np.full(len(ends), float(latitude)),
        ends[:, 0], ends[:, 1])[2]
    nearest_end = float(to_ends.min())
    half_piece_m = piece * 111_700.0 / 2.0

    local = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={float(latitude)!r} +lon_0={float(longitude)!r} "
        f"+datum=WGS84 +units=m",
        always_xy=True)
    measured = float(shapely.transform(pieces, local.transform, interleaved=False)
                     .distance(Point(0.0, 0.0)))
    # The projection is true to well under a metre out to 19,000 km from
    # the point. Within about 100 km of the point's antipode, where the
    # projection spreads one place round a circle, it is not - it read
    # 11,417 km for ground 19,995 km away - so the measurement is held
    # between the bounds the piece ends set, which a sound one never
    # leaves. A figure that is not a number is never answered.
    if not math.isfinite(measured):
        return nearest_end
    return min(max(measured, nearest_end - half_piece_m), nearest_end)


# ---------------------------------------------------------------------------
# Buffering: the polygon CalTopo's Buffer tool draws around a line.
# ---------------------------------------------------------------------------

#: The widest buffer `buffer_geometry_wgs84` draws, in ground metres each
#: side. A line search is a 30 m ribbon (15 m each side, CalTopo's default);
#: 1,000 m each side is a 2 km swath, which is an area assignment drawn the
#: wrong way round, and a distance past it is almost certainly a width typed
#: where a half-width was meant.
MAX_BUFFER_M = 1000.0

_BUFFERABLE = frozenset({"LineString", "MultiLineString", "Polygon", "MultiPolygon"})


def _cut_holes_open(polygon):
    """`polygon` drawn as ONE ring, each hole reached along a zero-width
    bridge - the way a shapefile has always had to carry a donut.

    CalTopo keeps one ring per polygon. Handed a ribbon round a closed way
    it drops the hole and fills the middle, so a 1,698 m loop trail
    imported as a 21-acre area of hillside instead of a 12-acre corridor.
    Cut open, the same ribbon arrives as the ribbon.

    The bridge adds no area, and the hole is walked the opposite way round
    from the outer ring, so the middle stays empty under the nonzero fill
    rule as well as the even-odd one - we do not know which CalTopo uses.
    The result is deliberately self-touching, which is to say invalid as
    OGC counts validity; that is the price of the one-ring model and the
    reason this is asked for rather than always done.
    """
    from shapely.geometry import Polygon

    if not polygon.interiors:
        return polygon
    ring = list(polygon.exterior.coords)[:-1]
    outward = polygon.exterior.is_ccw
    for interior in polygon.interiors:
        hole = list(interior.coords)[:-1]
        if interior.is_ccw == outward:
            hole.reverse()
        at, into = _bridge_ends(ring, hole)
        # Out along the bridge at `at`, right round the hole from `into`
        # and back to it, then home to `at` and on round the outer ring.
        ring = ring[:at + 1] + hole[into:] + hole[:into + 1] + ring[at:]
    cut = Polygon(ring)
    # Shapely will not measure a self-touching ring, so the check is on the
    # ground the ring encloses; a bridge that crossed something would show
    # up here as area gained or lost.
    if not _same_ground(cut, polygon):
        raise ValueError(
            "the ribbon round this way could not be cut open without changing the ground "
            "it covers, so it was not cut; buffer it without single_ring for the ribbon "
            "whole, holes and all, and draw the assignment by hand")
    return cut


def _bridge_ends(ring: list, hole: list) -> tuple[int, int]:
    """Where to bridge: the vertex of `ring` and the vertex of `hole` that
    lie closest together. Sought from shapely's nearest points rather than
    by comparing every pair, so a ribbon of a few thousand vertices costs
    one pass and not a few million."""
    from shapely.geometry import LineString
    from shapely.ops import nearest_points

    on_ring, on_hole = nearest_points(LineString(ring + ring[:1]),
                                      LineString(hole + hole[:1]))

    def nearest(points: list, to) -> int:
        return min(range(len(points)),
                   key=lambda i: (points[i][0] - to.x) ** 2 + (points[i][1] - to.y) ** 2)

    return nearest(ring, on_ring), nearest(hole, on_hole)


def _same_ground(cut, polygon) -> bool:
    """Whether the cut ring paints exactly the polygon it came from."""
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    made = make_valid(cut)
    filled = unary_union([part for part in getattr(made, "geoms", [made])
                          if part.geom_type in ("Polygon", "MultiPolygon")])
    return filled.symmetric_difference(polygon).area <= polygon.area * 1e-9


def _single_ring(geometry):
    """Every polygon of `geometry` cut open; anything else unchanged."""
    if geometry.geom_type == "Polygon":
        return _cut_holes_open(geometry)
    if geometry.geom_type == "MultiPolygon":
        from shapely.geometry import MultiPolygon

        return MultiPolygon([_cut_holes_open(part) for part in geometry.geoms])
    return geometry


def buffer_geometry_wgs84(geometry: dict, distance_m: float, *,
                          single_ring: bool = False) -> dict:
    """`geometry` widened by `distance_m` ground metres each side, as GeoJSON.

    Round ends and joins, which is what CalTopo's own Buffer draws and what
    makes a buffered trail read as a ribbon rather than a chain of boxes.
    Computed in Web Mercator inflated by 1 / cos(latitude) at the
    geometry's centroid, the scaling `planning.corridors` uses for its
    corridor axes; the ends here are round, as CalTopo draws them.
    Mercator units are not metres off the equator, and a raw 15 m buffer
    at 37 N would be 12 m of ground.

    A `LineString` or `MultiLineString` becomes a `Polygon`, or a
    `MultiPolygon` when branches lie further apart than the width; a
    polygon is widened. A `Point` is refused - a buffered point is a range
    ring, which CalTopo already has - and so is an empty geometry.

    `single_ring` draws a result that encloses ground as ONE ring per
    polygon, each hole cut open along a zero-width bridge, because CalTopo
    carries one ring and otherwise fills the middle of a loop. The ground
    covered and `area_acres` are the same either way; only the drawing
    changes. See `_cut_holes_open`.

    Returns `geometry`, `distance_m`, `width_m` (twice the distance) and
    `area_acres` of the result in ground acres.
    """
    kind = geometry.get("type") if isinstance(geometry, dict) else type(geometry).__name__
    if kind not in _BUFFERABLE:
        raise ValueError(
            f"buffer_geometry takes a LineString, MultiLineString, Polygon or "
            f"MultiPolygon; got {kind!r}")
    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        raise ValueError(
            f"distance_m must be a number of metres between 0 and {MAX_BUFFER_M:.0f}; "
            f"got {distance_m!r}") from None
    # `not (0 < d <= MAX)` rather than `d <= 0 or d > MAX` so NaN is refused:
    # every comparison with NaN is False, so the parenthesised test is
    # False and the refusal fires.
    if not (0.0 < distance <= MAX_BUFFER_M):
        raise ValueError(
            f"distance_m must be a number of metres between 0 and {MAX_BUFFER_M:.0f}; "
            f"got {distance_m!r}")
    parsed = parse_geometry(geometry)
    if parsed is None:
        raise ValueError("the geometry is empty or will not parse, so there is nothing to buffer")

    from shapely import get_coordinates
    from shapely.geometry import mapping
    from shapely.ops import transform

    # GEOS leaves NaN out of `bounds`, so the coordinates themselves are
    # checked; a NaN that got through would crash in the centroid.
    if not all(math.isfinite(v) for xy in get_coordinates(parsed) for v in xy):
        raise ValueError("a coordinate in the geometry is not a number, so it cannot be buffered")
    _minx, miny, _maxx, maxy = parsed.bounds
    # Past 85 degrees Mercator projects to nothing and the buffer comes
    # back empty. The realistic way to get there is a lat/lon-swapped pair.
    if max(abs(miny), abs(maxy)) > 85.0:
        offending = miny if abs(miny) > abs(maxy) else maxy
        raise ValueError(
            f"the geometry has a latitude of {offending}, which is off the map; "
            f"GeoJSON coordinates are [longitude, latitude]")

    latitude = parsed.centroid.y
    scale = 1.0 / math.cos(math.radians(latitude))
    buffered = transform(_TO_3857.transform, parsed).buffer(distance * scale)
    area_m2 = buffered.area / (scale * scale)
    back = transform(_TO_4326.transform, buffered)
    # After the area, and after the return to WGS84: the cut is a drawing
    # decision, so it must not touch what the ribbon measures, and the
    # bridge is shortest in the coordinates the file is written in.
    if single_ring:
        back = _single_ring(back)
    return {
        # Through json so the coordinates are lists, as GeoJSON is, rather
        # than shapely's tuples.
        "geometry": json.loads(json.dumps(mapping(back))),
        "distance_m": distance,
        "width_m": 2.0 * distance,
        "area_acres": round(acres(area_m2), 2),
    }


#: Compass points the radial clearance probe walks, and the ceiling it gives up
#: at. One clearance number hides a narrow finger of ground: a candidate in El
#: Corte de Madera had 60 m of preserve to the south-east and over 1.5 km to
#: the east, and no single figure describes that site honestly.
CLEARANCE_BEARINGS = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                      "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
CLEARANCE_LIMIT_M = 2000.0
CLEARANCE_STEP_M = 10.0


def boundary_clearance(geometry: dict, boundary: dict, *,
                       access: tuple[float, float] | None = None) -> dict:
    """How much of the boundary's inside surrounds `geometry`, and whether the
    way in stays inside it.

    WHY THIS IS A TOOL
    ------------------
    A candidate site in El Corte de Madera Creek OSP survived every terrain and
    vegetation check and was disqualified by land tenure alone. It sat wholly
    inside the preserve but 47 m from the edge, on a narrow western finger; the
    road 76 m away was entirely outside the preserve; the parking lot was
    outside too, and the straight line from that lot to the site was only half
    on preserve ground. None of that was askable here, so it was hand-rolled.

    `geometry` and `boundary` are GeoJSON in WGS84 (Polygon or MultiPolygon).
    `access` is an optional (latitude, longitude) - a trailhead, gate or lot -
    and adds the `approach` block.

    WHAT IT DOES NOT DO
    -------------------
    The approach is a STRAIGHT LINE, not a route: it flags a private-land risk,
    it does not resolve a parcel line, and it knows nothing about trails,
    fences or where a vehicle can actually stop. A boundary is only as current
    as its source - MROSD's published GIS was 2018 when this was written.
    """
    import math

    from shapely.geometry import LineString, Point, shape as shapely_shape
    from shapely.ops import transform as shapely_transform

    from pyproj import Transformer

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    subject = shapely_transform(to_3857, shapely_shape(geometry)).buffer(0)
    limit = shapely_transform(to_3857, shapely_shape(boundary)).buffer(0)
    if subject.is_empty or limit.is_empty:
        raise ValueError("geometry and boundary must both be non-empty polygons")

    latitude = bbox_center_latitude(geometry_bounds_wgs84(geometry))
    scale = ground_meters_per_mercator_unit(latitude)

    inside = subject.intersection(limit)
    fraction = (inside.area / subject.area) if subject.area else 0.0
    within = subject.within(limit)
    # Distance from the shape to the boundary LINE, which is the number that
    # matters whether the shape is inside or out. `clearance_m` is that
    # distance for a shape that is inside, and zero for one that crosses.
    to_edge = subject.distance(limit.boundary) * scale

    centre = inside.centroid if not inside.is_empty else subject.centroid
    radial: dict[str, float | None] = {}
    for name, bearing in CLEARANCE_BEARINGS.items():
        radians = math.radians(bearing)
        reached = None
        step = CLEARANCE_STEP_M
        while step <= CLEARANCE_LIMIT_M:
            probe = Point(centre.x + math.sin(radians) * step / scale,
                          centre.y + math.cos(radians) * step / scale)
            if not limit.contains(probe):
                reached = step
                break
            step += CLEARANCE_STEP_M
        radial[name] = reached

    approach = None
    if access is not None:
        access_point = Point(to_3857(access[1], access[0]))
        corridor = LineString([access_point, subject.centroid])
        length = corridor.length
        approach = {
            "access_latitude": access[0],
            "access_longitude": access[1],
            "access_inside_boundary": bool(limit.contains(access_point)),
            "distance_m": round(length * scale),
            "fraction_inside": (round(corridor.intersection(limit).length / length, 3)
                                if length else 1.0),
        }

    return {
        "within": bool(within),
        "fraction_inside": round(fraction, 3),
        "clearance_m": round(to_edge) if within else 0.0,
        "distance_to_boundary_m": round(to_edge),
        "radial_clearance_m": radial,
        "approach": approach,
        "assumptions": {
            "radial_limit_m": CLEARANCE_LIMIT_M,
            "radial_step_m": CLEARANCE_STEP_M,
            "radial_null_means": (f"the boundary was not reached within "
                                  f"{CLEARANCE_LIMIT_M:.0f} m on that bearing"),
            "approach": ("a straight line from the access point to the "
                         "geometry's centroid - it flags a private-land risk "
                         "and is not a route: it knows nothing about trails, "
                         "fences, or where a vehicle can stop"),
            "currency": ("a boundary is only as current as its source; MROSD's "
                         "published GIS was version 2018.09.20 when this was "
                         "written"),
        },
    }
