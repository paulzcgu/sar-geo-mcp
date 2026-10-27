"""Trails, roads, waterways, and staging sites (public parking and hardstand)
from OpenStreetMap: what the cached Overpass tiles mean for a search.

CalTopo's rendered road layers are pixels; measuring where a trail crosses an
assignment boundary needs vector geometry, which is what Overpass provides.
Fetching and caching that geometry is `sources.osm`; this module reads the
cached tiles and decides what each element is - a walkable way, a road and
its class, a lot or public hardstand to park on, a gate.
"""

from collections.abc import Callable
from pathlib import Path

from ..geo import Bbox
from ..sources.osm import (
    BARRIER_TAGS,
    GATE_BARRIERS,
    OVERPASS_ASSUMPTIONS,
    SITE_TAGS,
    WATER_EDGE_NATURAL,
    fetch_osm_tiles,
    tiles_covering_osm,
)


#: What every routes result reports of how its OpenStreetMap was fetched.
#: The same dict as `sources.osm.OVERPASS_ASSUMPTIONS`, which the fetch
#: keeps current (the mirrors a run was allowed), not a copy.
LINEAR_ASSUMPTIONS = OVERPASS_ASSUMPTIONS

_TRAIL = {"path", "footway", "bridleway", "steps", "cycleway"}
_TRACK = {"track"}
_ROAD = {
    "service", "residential", "unclassified", "tertiary", "secondary",
    "primary", "trunk", "motorway", "road", "living_street",
    "tertiary_link", "secondary_link", "primary_link", "trunk_link",
    "motorway_link", "pedestrian",
}

# ---------------------------------------------------------------------------
# Site rules: what ground a search can put vehicles on.
#
# From here to `site_flags` decides whether an OSM element is staging ground
# at all, and what a planner needs to know before sending rigs there.
# `parking_for`, near the bottom of the file (it needs the tile-fetch
# machinery the routes section below defines first), is the one reader that
# turns these rules into rows. `routes_for`, also below, deliberately calls
# `is_site_shaped` rather than `classify_site` for its own exclusion guard -
# see that function's docstring for why the two are not interchangeable.
# ---------------------------------------------------------------------------

#: The two kinds of ground a search can put vehicles on. `parking` is a lot
#: OSM has actually mapped; `hardstand` is a public site whose lot it has
#: not, which is a weaker claim and is flagged as one.
SITE_PARKING = "parking"
SITE_HARDSTAND = "hardstand"

#: An `access` value that means "not you" - or not without asking first.
#: 2026-09-09 operator ruling: these are FLAGGED, not refused. Whether a lot
#: is private, permit-only, or residents/customers-only is, in the
#: operator's words, "something that just let user know" - a note on the
#: row, not a reason the planner never sees the row at all. The planner is a
#: trained search manager looking at a map at 06:00: a locked gate is a fact
#: they can verify with one phone call to the owner, faster than this module
#: can guess, and a lot this module never shows them is a lot they cannot
#: verify at all. `site_flags` names which of these values landed here so
#: the planner knows which call to make.
ACCESS_FLAGGED = frozenset({"private", "no", "permit", "customers"})

#: An `access` value that means this ground belongs to whoever lives there.
#: 2026-09-09 operator ruling: EXCLUDED from `is_site_shaped`, not flagged
#: like ACCESS_FLAGGED above - `residents` used to live in that set,
#: producing an `access_residents` flag, and it has moved out rather than
#: simply being deleted so the split stays visible to the next reader.
#:
#: The distinction is not degree of restriction, it is kind. A locked
#: commercial gate (`private`, `permit`, `customers`, `no`) is a phone call
#: the planner - a trained search manager - is qualified to make. A
#: resident's parking space is somebody's home: staging a SAR team there at
#: 06:00 is not a decision this module should hand the planner as a row on
#: a map, because getting it wrong makes trouble for the team and for the
#: agency, not just a wasted drive. The operator's ruling: avoid apartment
#: and residential parking; trailhead parking is fine. Do not fold this back into ACCESS_FLAGGED - that
#: would make every `access_residents` row visible again, right after the
#: whole point of this ruling was to make sure it never is.
ACCESS_EXCLUDED = frozenset({"residents"})

#: A `parking` sub-tag for covered or enclosed parking - no comms, no
#: helicopter, and a rig with a light bar does not fit under the entrance.
#: 2026-09-09 operator ruling: flagged (`covered_deck`), not refused, for
#: the same reason as ACCESS_FLAGGED above - deciding whether that still
#: rules a site out is the planner's call, not this module's guess. A
#: parking structure like this is a commercial building, not a home, so it
#: gets the phone-call treatment, not the exclusion below.
PARKING_FLAGGED = frozenset({"underground", "multi-storey", "rooftop"})

#: A `parking` sub-tag naming ground this module excludes outright rather
#: than flags - the same "somebody's home, not a phone call" reasoning as
#: ACCESS_EXCLUDED above, plus one more case that is not about homes at
#: all:
#: - `garage_boxes`, `carports`, `sheds` are private residential garages.
#:   `carports` and `sheds` used to live in PARKING_FLAGGED, producing
#:   `covered_deck`; they moved out on the same 2026-09-09 ruling as
#:   ACCESS_EXCLUDED, and for the same reason - do not fold them back in.
#: - `lane`, `street_side` are on-street parking spaces along a road, not a
#:   lot a team can stage in at all. There is no ground here to offer,
#:   flagged or otherwise, so these are excluded rather than flagged for a
#:   different reason than the residential values above: not "whose home is
#:   this", but "this was never a lot".
PARKING_EXCLUDED = frozenset({
    "garage_boxes", "carports", "sheds", "lane", "street_side",
})


#: Public sites whose parking OSM frequently has not mapped as its own
#: polygon: a school, a church, a ballfield. Offered only when no mapped lot
#: falls inside them (`planning.parking`), and always flagged
#: `parking_unverified` - this is "a place with room", not "a surface".
HARDSTAND_TAGS = SITE_TAGS


def _tag(tags: dict, key: str) -> str:
    """A tag's value, stripped and lowercased, so `Parking`, `PRIVATE`, and a
    trailing space all compare the same as the canonical spelling every set
    in this module is written in. Every tag comparison below goes through
    this - `amenity`, `leisure`, `access`, `fee` included - so which tags
    happen to arrive normalised from Overpass is not something this module
    has to track."""
    return str((tags or {}).get(key) or "").strip().lower()


def _access_parts(tags: dict) -> list[str]:
    """`access` split on `;`, each part normalised. OSM stacks values there:
    `private;customers` means "private, and customers may also come" - not
    "pick whichever reads better" - so a lot tagged that way is exactly as
    closed as one tagged `private` alone, and every part has to be checked
    against ACCESS_FLAGGED, not just the one a naive read would notice
    first."""
    return [
        stripped
        for part in _tag(tags, "access").split(";")
        if (stripped := part.strip())
    ]


def is_site_shaped(tags: dict) -> bool:
    """Does this element carry one of the tags query version 3 added for
    staging ground - the structural question, independent of whether a
    search may actually use it?

    This is the ONE place that answers "is this element site-shaped";
    `classify_site` below builds on it rather than re-testing the tags
    itself. `routes_for`'s guard calls this function and not
    `classify_site`, and that choice is deliberate rather than
    interchangeable: a PRIVATE school is as much "not a route" as a public
    one, and before this function existed, `routes_for` filtered on
    `classify_site` instead - which let every refused site (a gated park,
    an underground deck, a customers-only lot) leak through as a route of
    class OTHER, which was most of the parking tagged in a suburb.
    `classify_site`'s access/deck refusals are gone as of the 2026-09-09
    ruling below, so the two functions happen to agree on every input
    today - but a future refusal added back to `classify_site` (or to this
    function) must not make `routes_for` start asking the offer question
    again, which is why the guard names `is_site_shaped` and not
    `classify_site`.

    2026-09-09 operator ruling (residential parking): an element is not
    site-shaped at all - not flagged, EXCLUDED - when `access` names
    ACCESS_EXCLUDED (`residents`) or `parking` names PARKING_EXCLUDED
    (`garage_boxes`, `carports`, `sheds`, `lane`, `street_side`). See those
    constants for why a residential value is excluded here while a private
    or customers-only one is only flagged by `site_flags` below: a locked
    commercial gate is a phone call, a resident's parking space is
    somebody's home. Because this exclusion lives in `is_site_shaped`
    itself, it reaches `classify_site`, `parking_for`, AND `routes_for`'s
    guard in one place - an excluded lot is not offered, gets no row, and
    (unlike a merely-flagged private lot) is no longer treated as
    site-shaped by `routes_for` either, so it falls through to an ordinary
    OTHER-class route instead of being dropped. That is examined and
    correct, not a leak: see `routes_for`'s own comment on this.
    """
    tags = tags or {}
    if _tag(tags, "amenity") == "parking":
        shaped = True
    else:
        shaped = any(_tag(tags, key) in values for key, values in HARDSTAND_TAGS)
    if not shaped:
        return False
    if any(part in ACCESS_EXCLUDED for part in _access_parts(tags)):
        return False
    if _tag(tags, "parking") in PARKING_EXCLUDED:
        return False
    return True


def classify_site(tags: dict) -> str:
    """`"parking"`, `"hardstand"`, or `""` for an element that is not
    site-shaped at all (`is_site_shaped(tags)` is `False`).

    Parking wins over hardstand: a lot inside a school is a lot, and the
    school is only offered when no lot was mapped inside it.

    2026-09-09 operator ruling: nothing is refused for access or for being a
    deck any more - see ACCESS_FLAGGED and PARKING_FLAGGED for why. `""`
    therefore does not mean "a search may not stage here" for one of those
    reasons - a planner reads `site_flags` for that. It DOES still mean
    "not a parking or hardstand element", and, as of the same date's
    residential-parking ruling, `""` is also what a residential lot gets:
    `is_site_shaped` excludes it outright (see ACCESS_EXCLUDED and
    PARKING_EXCLUDED), so it never reaches the parking/hardstand branches
    below at all.
    """
    tags = tags or {}
    if not is_site_shaped(tags):
        return ""
    if _tag(tags, "amenity") == "parking":
        return SITE_PARKING
    return SITE_HARDSTAND


def site_flags(tags: dict) -> list[str]:
    """What a planner must know before sending rigs there, in the order a
    row reads them: `fee`; then `permissive`, if the `access` tag carries
    it; then one `access_<value>` flag per `;`-stacked `access` part that is
    in ACCESS_FLAGGED (`access_private`, `access_no`, `access_permit`,
    `access_customers`), in the order the parts are written; then
    `access_unknown` for any part not recognised as
    `yes`/`public`/`permissive`/an ACCESS_FLAGGED value; then `covered_deck`
    if the `parking` sub-tag is in PARKING_FLAGGED.

    This function is never reached for a residential lot: `is_site_shaped`
    excludes ACCESS_EXCLUDED/PARKING_EXCLUDED tags (2026-09-09 operator
    ruling) before `parking_for` calls this, so there is no
    `access_residents` flag any more, and no flag at all for
    `garage_boxes`/`carports`/`sheds`/`lane`/`street_side` - excluded rows
    carry no flags because they carry no row.

    Structural flags (`parking_unverified`, `node_only`) are not tags and
    are added where the shape is built, not here.
    """
    tags = tags or {}
    flags: list[str] = []
    fee = _tag(tags, "fee")
    charge = _tag(tags, "charge")
    free_values = {"no", "false", "0", "none"}
    if (fee and fee not in free_values) or (charge and charge not in free_values):
        # `fee=interval`, `fee=donation`, a literal price, and a bare
        # `charge=*` with no `fee` tag at all are exactly as much "bring
        # cash" as `fee=yes` - the county-park and trailhead lots this is
        # for use all four. Only an explicit "no" / "0" / "none" (or
        # nothing tagged at all) means free.
        flags.append("fee")
    parts = _access_parts(tags)
    if "permissive" in parts:
        flags.append("permissive")
    for part in parts:
        if part in ACCESS_FLAGGED:
            # Named per value, not a shared bucket: `access_private` and
            # `access_customers` are different phone calls for the planner.
            flags.append(f"access_{part}")
    recognised = {"yes", "public", "permissive"} | ACCESS_FLAGGED
    if not parts or not all(part in recognised for part in parts):
        # A value nobody here parsed - `employees`, `destination`, the
        # `fee` in `permissive;fee` - is MORE unknown than an absent tag,
        # not less: an absent tag at least never claimed anything, where an
        # unparsed one might be a real restriction this module doesn't
        # understand. That holds even alongside a `permissive` sibling:
        # `permissive;fee` is two claims stacked, and understanding one of
        # them does not vouch for the other. A value already named by an
        # `access_*` flag above is not unknown, though - this module DID
        # understand it, which is exactly why it has its own flag.
        flags.append("access_unknown")
    if _tag(tags, "parking") in PARKING_FLAGGED:
        flags.append("covered_deck")
    return flags


# ---------------------------------------------------------------------------
# Routes: the lines a team can walk or drive, off the same OSM tiles.
#
# Classifying a way and `routes_for` itself live here - everything
# downstream of "which OSM ways are worth reading," as opposed to the site
# rules above, which answer "which OSM elements are ground to park on." The
# Overpass query, its retries and the tile cache are `sources.osm`'s;
# `routes_for`, `parking_for` and `gates_for` all read the same cached tiles
# through it (`tiles_covering_osm`, `fetch_osm_tiles`).
# ---------------------------------------------------------------------------

#: The classes a person can walk along - what `ingress_candidates` counts.
WALKABLE_CLASSES = frozenset({"TRAIL", "TRACK", "ROAD", "ROAD_OTHER"})


def classify_way(tags: dict) -> str:
    """Map OSM tags onto the line classes the anatomy report matched
    boundaries to: TRAIL, TRACK, ROAD, ROAD_OTHER (any other highway, treated
    as road), WATERWAY, BARRIER (power line, railway, fence, wall, pipeline),
    WATER_EDGE (lake, reservoir, coastline; a closed way contributes its
    ring), else OTHER.

    Every tag is read through `_tag` (stripped, lowercased), the way the
    stored `highway`, `access`, `tracktype` and `power` are, so a way stored
    as highway "track" is always a TRACK and a blank value is no value.
    Classification happens when a tile is read, never in the tile, so this
    needs no query-version bump."""
    highway = _tag(tags, "highway")
    if highway in _TRAIL:
        return "TRAIL"
    if highway in _TRACK:
        return "TRACK"
    if highway in _ROAD:
        return "ROAD"
    if highway:
        return "ROAD_OTHER"
    if _tag(tags, "waterway"):
        return "WATERWAY"
    if (_tag(tags, "power") in ("line", "minor_line") or _tag(tags, "railway")
            or _tag(tags, "barrier") in BARRIER_TAGS
            or _tag(tags, "man_made") == "pipeline"):
        return "BARRIER"
    if (_tag(tags, "natural") in WATER_EDGE_NATURAL
            or _tag(tags, "landuse") == "reservoir"):
        return "WATER_EDGE"
    return "OTHER"


def routes_for(bbox_3857: Bbox, cache_dir: Path,
               get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Routes clipped to the AOI, as metric shapes with stable ids."""
    from pyproj import Transformer
    from shapely.geometry import LineString, box, mapping
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)

    elements: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        elements.extend(payload.get("elements", []))

    results: list[dict] = []
    counter = 0
    seen_ids: set[int] = set()
    for element in elements:
        if element.get("type") != "way" or element.get("id") in seen_ids:
            continue
        seen_ids.add(element.get("id"))

        tags = element.get("tags", {})
        # Query version 3 fetches parking and hardstand out of the same tile.
        # A school is not a route: it arrives classified OTHER, and it is
        # dropped here whenever it is site-shaped - whether or not a search
        # may actually stage there - so nothing the version 2 query used to
        # return has changed class.
        #
        # 2026-09-09 operator ruling: a residential lot (access=residents,
        # or a residential/on-street `parking` sub-tag) is excluded from
        # `is_site_shaped` itself, so it no longer trips this guard either.
        # That is intentional, not a gap this ruling reopened: an excluded
        # lot is not "a refused site slipping through as a route" the way
        # the version-2 gap above was - it is just an ordinary,
        # untagged-for-our-purposes closed way again, exactly as it would
        # have been read before query version 3 ever taught this module
        # about parking. It comes back as class OTHER, same as any other
        # way this module does not recognise.
        route_class = classify_way(tags)
        if route_class == "OTHER" and is_site_shaped(tags):
            continue

        coordinates = [(pt["lon"], pt["lat"]) for pt in element.get("geometry", [])]
        if len(coordinates) < 2:
            continue

        line = shapely_transform(to_3857.transform, LineString(coordinates))
        clipped = line.intersection(clip)
        if clipped.is_empty:
            continue

        results.append(
            {
                "id": f"route_{counter}",
                "kind": "route",
                "geometry": mapping(clipped),
                "properties": {
                    "route_type": route_class,
                    "name": tags.get("name"),
                    "osm_id": element.get("id"),
                    "surface": tags.get("surface"),
                    # Why a wire needs its own field: `classify_way` folds
                    # power lines, railways, fences, walls and pipelines
                    # into one BARRIER class, which is right for a boundary
                    # and useless for a helicopter. The key is ALWAYS
                    # written, null on any way that carries no `power` tag,
                    # so a record without the key is one staged before this
                    # field existed - and `planning/lz.py` reads that
                    # absence as "unknown", never as "no wires".
                    # Normalised as `classify_way` reads it, so a "Line"
                    # it classed a barrier is a "line" to `planning/lz.py`.
                    "power": _tag(tags, "power") or None,
                    # Query version 5, same rule: ALWAYS written, null where
                    # the way has no such tag, so a record without the key
                    # was staged before version 5 and its road class, access
                    # and tracktype are unknown, never "none". Normalised
                    # through `_tag` like every other tag comparison here,
                    # since egress compares them against lowercase sets.
                    "highway": _tag(tags, "highway") or None,
                    "access": _tag(tags, "access") or None,
                    "tracktype": _tag(tags, "tracktype") or None,
                    # Same rule again, for the travel-time rings (0.21.0):
                    # `waterway` tells a river (closed but at a bridge) from
                    # a creek (crossed, slowly), and `tunnel` a culvert (no
                    # crossing at all) from open water. No query change:
                    # `out geom` already returns every tag, so these come
                    # from the cached tile; a record without the keys was
                    # staged before they were kept.
                    "waterway": _tag(tags, "waterway") or None,
                    "tunnel": _tag(tags, "tunnel") or None,
                },
            }
        )
        counter += 1

    return results


def _repaired_polygon(shape):
    """`shape` made valid by geometry rather than by winding rule.

    `buffer(0)` repairs an invalid ring by winding rule, and a bowtie whose
    two lobes wind OPPOSITE ways loses one of them - a lot silently reported
    at half its real size, which is a wrong answer nobody notices rather
    than a crash somebody would. `make_valid` repairs the same ring by
    geometry and keeps both lobes. Same repair, same filtering, as
    `geo._same_ground`: a degenerate ring can make `make_valid` return a
    `GeometryCollection` mixing polygons with stray `LineString`s (a
    collinear ring, an out-and-back spike), and a line is not a lot, so only
    the polygonal parts survive. An input with no polygonal part left comes
    back empty, exactly as an unrepairable `buffer(0)` result did.
    """
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    made = make_valid(shape)
    parts = [part for part in getattr(made, "geoms", [made])
             if part.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts)


def parking_for(bbox_3857: Bbox, cache_dir: Path,
                get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Parking lots and public hardstand from the tiles the routes come from.

    A site is kept WHOLE when it touches the box, where a route is clipped to
    it. The difference is what the two are for: a route is measured, so the
    part outside the staged ground is not ours to report; a site is
    identified, and half a lot reported as the lot would understate the room
    a planner is judging.

    An OSM area is a CLOSED way. An open one is a kerb, a barrier, or a lot
    somebody half-drew, and a polygon forced out of it would be a shape
    nobody mapped.

    A consumer of these rows has to know three things this function does not
    say by name:

    - `geometry` is metric, but not in the sense `routes_for` means it -
      it is a Web Mercator (EPSG:3857) shape, not ground metres. Area is
      `geometry.area * ground_meters_per_mercator_unit(lat) ** 2`, not
      `geometry.area` alone; anything that expects lon/lat (CalTopo, a
      GeoJSON export) needs the same reprojection to WGS84 this function's
      own inputs went through first. Guessing wrong on either gives a
      plausible-looking number off by about 1.6x at 37 N - not a crash, a
      wrong answer a planner has no way to catch.
    - `capacity` is OSM's free text, verbatim: `"40"`, `"40 (10 disabled)"`,
      `"unknown"`, `"~15"`, or absent. It is passed through unparsed - never
      call `int()` on it. It looks like a number often enough that this is
      the mistake a consumer makes once, on real data rather than on a
      fixture.
    - No row carries an area, for a Point or a Polygon alike. Computing one
      for every row, used or not, is work this function has no way to know
      is wanted; the caller derives it from `geometry` using the scale
      factor above, only when it actually needs to.
    """
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon, box, mapping
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)

    elements: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        elements.extend(payload.get("elements", []))

    results: list[dict] = []
    seen: set[tuple[str, object]] = set()
    for element in elements:
        # `node` and `way` are the only types `build_overpass_query` ever
        # asks for, so a `relation` never actually arrives here - this guard
        # is cheap insurance against that changing, not relation support
        # half-built. That distinction matters: a large park, or a lot with
        # a hole in it, is commonly mapped in OSM as a multipolygon
        # RELATION, so "no hardstand here" can quietly mean "the park is a
        # relation this module has never been asked to read."
        osm_type, osm_id = element.get("type"), element.get("id")
        if osm_type not in ("node", "way") or (osm_type, osm_id) in seen:
            continue
        tags = element.get("tags") or {}
        kind = classify_site(tags)
        if not kind:
            continue

        flags = site_flags(tags)
        if osm_type == "node":
            if element.get("lat") is None or element.get("lon") is None:
                continue
            shape = Point(element["lon"], element["lat"])
            # A dot has no area, so nothing downstream can say how many rigs
            # fit; the planner has to look at the map.
            flags.append("node_only")
        else:
            coordinates = [(pt["lon"], pt["lat"]) for pt in element.get("geometry") or []]
            if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
                continue
            shape = Polygon(coordinates)
            if not shape.is_valid:
                shape = _repaired_polygon(shape)
            if shape.is_empty:
                continue
        if kind == SITE_HARDSTAND:
            flags.append("parking_unverified")

        metric = shapely_transform(to_3857.transform, shape)
        if not metric.intersects(clip):
            continue

        # Marked seen only now, once the element has actually earned a row:
        # an element that clears `classify_site` but fails the shape checks
        # above (an open way, a node with no coordinates) must not block a
        # later tile's copy of the SAME element from being tried - which a
        # `seen.add` any earlier than this would risk, if a way's geometry
        # ever arrived incomplete from one tile and complete from another.
        seen.add((osm_type, osm_id))
        results.append({
            "id": f"{kind}-{osm_type[0]}{osm_id}",
            "kind": kind,
            "geometry": mapping(metric),
            "properties": {
                "name": tags.get("name"),
                "osm_id": osm_id,
                "osm_type": osm_type,
                "surface": tags.get("surface"),
                "capacity": tags.get("capacity"),
                "flags": flags,
            },
        })
    return results


def gates_for(bbox_3857: Bbox, cache_dir: Path,
              get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Gate nodes (`barrier` in GATE_BARRIERS) inside the box, as EPSG:3857
    points: {id: "gate-n<osm_id>", kind: "gate", geometry,
    properties: {osm_id, barrier, access}}. Read from the same cached tiles
    as routes and parking - no extra Overpass traffic.

    `access` is null where the gate carries no such tag: an unlocked-looking
    gate is not an open one, and nothing here claims it is. A gate is
    reported, never used to exclude a way."""
    from pyproj import Transformer
    from shapely.geometry import Point, box, mapping

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)
    seen: set[int] = set()
    gates: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        for element in payload.get("elements", []):
            if element.get("type") != "node" or element.get("id") in seen:
                continue
            tags = element.get("tags") or {}
            barrier = _tag(tags, "barrier")
            if barrier not in GATE_BARRIERS:
                continue
            if element.get("lat") is None or element.get("lon") is None:
                continue
            point = Point(*to_3857.transform(element["lon"], element["lat"]))
            if not clip.intersects(point):
                continue
            # Seen only once it has earned a row, as in `parking_for`.
            seen.add(element["id"])
            gates.append({"id": f"gate-n{element['id']}", "kind": "gate",
                          "geometry": mapping(point),
                          "properties": {"osm_id": element["id"], "barrier": barrier,
                                         "access": _tag(tags, "access") or None}})
    return gates


def ingress_candidates(routes: list[dict], assignment_3857) -> list[dict]:
    """Points where a route crosses the assignment boundary.

    These are where a team can actually walk in, which is usually a more useful
    briefing item than the assignment's centroid.

    Only walkable classes count: a creek or a fence crossing the boundary is
    not a way in, and before this filter a waterway crossing was reported as
    one.
    """
    from shapely.geometry import mapping
    from shapely.geometry import shape as shapely_shape

    boundary = assignment_3857.boundary
    results: list[dict] = []
    counter = 0

    for route in routes:
        if route["properties"].get("route_type") not in WALKABLE_CLASSES:
            continue
        line = shapely_shape(route["geometry"])
        crossing = line.intersection(boundary)
        if crossing.is_empty:
            continue

        points = [crossing] if crossing.geom_type == "Point" else list(
            getattr(crossing, "geoms", [])
        )
        for point in points:
            if point.geom_type != "Point":
                continue
            results.append(
                {
                    "id": f"ingress_{counter}",
                    "kind": "ingress",
                    "geometry": mapping(point),
                    "properties": {
                        "route_id": route["id"],
                        "route_type": route["properties"]["route_type"],
                        "route_name": route["properties"].get("name"),
                    },
                }
            )
            counter += 1

    return results
