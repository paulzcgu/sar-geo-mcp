"""Populated areas in town: built-up parcels grouped by the street they
front, each group outlined as one search block.

WHAT THIS DRAWS
---------------
The urban-search block a planner draws by hand on CalTopo's parcel layer:
every built-up lot along one street, both sides, with the street between
them, as one polygon - "Berwick Drive", not forty separate lots and not the
whole suburb as one blob. A lot is BUILT-UP when CalTopo's structures layer
puts a building inside it; it is URBAN OR SUBURBAN when it is no bigger than
`max_parcel_area_m2` (an acre by default). Rural ground - ranches, forest,
open space - has big lots, so it yields nothing here, and that is the
intended answer: this tool is for town.

One exception to the size rule. A big lot that is mostly roof - an apartment
complex, a school, a shopping centre, a mobile-home park - is built-up
ground in its own right, but it fronts its street the way one house does,
and a planner searches it as its own block. Any lot over the size limit
whose buildings cover at least `large_parcel_min_cover` of it comes back as
its own `large_parcel` block. A farmhouse on forty acres covers a fraction
of a percent and never qualifies.

HOW PARCELS ARE READ
--------------------
CalTopo serves its parcel layer as rendered outlines, not as a parcel
database, so the lots are recovered from the picture: the outline pixels
are the walls, and every 4-connected run of ground between them is one
region. A 1 px diagonal wall is watertight to 4-connected ground, so no
dilation is needed and no thin lot is eaten. The outline panel is resampled
with `wms.MAX_ALPHA`, as the structures panel is, so a 1 px line survives
the shrink onto the pixel grid.

A region is one of three things:
  - a lot;
  - road right-of-way, which a parcel map leaves as the gap between lots.
    It is told apart by OpenStreetMap: a region more than
    `ROW_ROAD_SHARE` of whose ground lies within `ROW_ROAD_REACH_M` of a
    mapped road centreline, and that holds almost no roof, is street;
  - ground off the parcel map altogether (beyond the layer's coverage),
    which reads as one huge region and is dropped by the size rule.

WHICH STREET A LOT IS ON
------------------------
Frontage, measured on the parcel map. The street right-of-way the lot's
edge touches - the gap between lots, across the lot line - votes cell by
cell for the named road nearest that right-of-way cell (within
`STREET_CEILING_M`), and the most votes wins. A corner lot goes to the
street it has more edge along, which is usually the one its address is on.
Voting from the right-of-way, not from the lot, matters where
OpenStreetMap's centreline sits 10-15 m off the parcel map: measured from
the lot, the back row of the next street can be nearer to the line than a
lot's own front, and a street's far side would be filed under the street
behind it. A lot touching no right-of-way falls back to its own edge:
every edge cell within `FRONTAGE_REACH_M` of a named road votes for the
nearest one; failing that the nearest named road within `STREET_CEILING_M`,
and failing that no street: those lots are still grouped, by adjacency,
with `street` null.

HOW A BLOCK IS OUTLINED
-----------------------
The lots of one street, a morphological closing of radius `bridge_m` over
them (which spans the street between the two sides and the outline pixels
between neighbours), holes filled, and then every lot belonging to ANOTHER
block cut back out so two blocks never claim the same house. A street whose
lots fall into separate runs - a park between them, a cross street with no
houses on it - gives one block per run. A run with fewer than `min_parcels`
houses is no block of its own: it is searched with the block whose lots it
touches (or, failing that, whose ground comes within the closing's reach),
and that block's warnings say so; so is a piece of a run the cut left with
too few. Only a run touching no block is left out, counted in the summary,
so nothing is dropped silently. A lot with no building on it (a vacant
lot, a yard parcel) that touches its street's houses is searched with
them - one layer, so an empty stretch past the last house is not carried
along - and counted as `vacant_lots`, not in `parcels`.

HOUSE MARKERS, ONLY WHEN ASKED
------------------------------
Off by default. With `building_markers`, every building in a block gets one
CalTopo Marker on its roof - the house symbol, white unless `marker_color`
says otherwise - in place of tracing the building. A building is marked in
the one block holding most of its roof, and the block's `buildings` count
uses the same rule, so markers and count agree. The outline does not change.

WHAT IT IS NOT
--------------
Not an address list and not a census: a house CalTopo's structures layer has
not drawn makes its lot read as vacant, and a lot line CalTopo does not draw
makes two lots read as one. It is a first cut of the blocks a planner would
draw, measured from three public layers, for the planner to adjust.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..geo import geometry_centroid, ground_meters_per_mercator_unit, to_wgs84_geojson
from ..net_errors import safe_reason
from ..terrain.grid import read_subject

PARCEL_LAYER = "parcel"
STRUCTURES_LAYER = "structures"

#: CalTopo's house symbol. Its name is `hut`, but it is the plain house
#: glyph, and it takes the marker colour - black by default, white with
#: #FFFFFF (checked against caltopo.com/icon.png). The other house-shaped
#: symbols all MEAN something: the `evac-` set records a door knocked, an
#: evacuation, no answer, and a house marked with one of those says
#: something about it no one has found out.
MARKER_SYMBOL = "hut"
DEFAULT_MARKER_COLOR = "#FFFFFF"

#: Least alpha a parcel-panel pixel needs to be an outline. CalTopo draws
#: the lines opaque; anything this faint is antialiasing off a line that the
#: MAX_ALPHA shrink already carried onto its own cell.
PARCEL_ALPHA_MIN = 64

#: Least alpha a structures-panel pixel needs to be a building: the same
#: floor `lz.STRUCTURE_ALPHA_MIN` reads that layer with (footprints are drawn
#: semi-transparent, 64 to 127).
STRUCTURE_ALPHA_MIN = 32

#: Ground metres per pixel never coarser than this: an 18 m lot is nine
#: cells at 2 m, and a parcel outline drawn at 1 px stops separating lots
#: well before a lot is three cells wide.
MAX_PIXEL_M = 2.0

#: Pixels per side of the parcel panel. Past this the pixel is coarsened
#: (up to MAX_PIXEL_M) rather than the panel grown: 4,096 at 1 m is 4 km of
#: town, already a big urban search.
MAX_SIDE_PX = 4096

#: Ground metres of panel kept beyond the staged area on every side, so a
#: street block the area boundary crosses is read whole - both sides of the
#: street out to the back fences - rather than cut where the ring falls. A
#: run of lots reaching further than this past the ring is still cut, and
#: says so (`clipped_at_edge`).
MARGIN_M = 200.0

#: The block method a set id is hashed over (`populated_params`).
METHOD = 3

#: A lot with no building on it is searched with its street when it touches
#: that street's built-up lots: a vacant lot, a yard parcel, a garden - a
#: child hides there as readily as behind a house. Under this it is no lot
#: at all (a sliver between drawn lines).
VACANT_MIN_M2 = 100.0

#: A region is street right-of-way when at least this share of it lies
#: within ROW_ROAD_REACH_M of a mapped road centreline (and it is not
#: roofed). An 18 m right-of-way has 6 m of its width inside a 3 m reach,
#: a third; a lot that merely abuts the road has a strip of a few metres.
ROW_ROAD_SHARE = 0.20
ROW_ROAD_REACH_M = 3.0
#: ...and less than this share of it is roof. A lot with a house on it is a
#: lot, however close OSM's centreline has been drawn to it.
ROW_MAX_ROOF_SHARE = 0.05

#: A region under this is a unit parcel, never a lot of its own: no house
#: stands on 120 m2 with a yard, but a condominium unit's parcel does.
UNIT_LOT_MAX_M2 = 120.0
#: ...and so is a region at least this share roof and under
#: FOOTPRINT_LOT_MAX_M2: a parcel drawn round a building's footprint. A
#: townhouse lot with a yard is about half roof.
FOOTPRINT_ROOF_SHARE = 0.75
FOOTPRINT_LOT_MAX_M2 = 400.0
#: Ground metres a unit parcel looks out, step by step, for the lot to fold
#: into. Measured on a Carmel Valley condominium complex: the lines round
#: each unit ran four to six pixels thick at 1 m, so a 3 px look saw lines
#: and nothing else.
UNIT_FOLD_REACH_M = (3.0, 6.0, 10.0)

#: A unit parcel is never folded into a region bigger than this: that is
#: ground off the parcel map, or the whole street network, not a complex's
#: common area.
FOLD_PARENT_MAX_M2 = 100_000.0

#: An edge cell votes for the nearest named road within this many ground
#: metres. Half a wide residential right-of-way plus a little for OSM's
#: centreline not lying in the middle of it.
FRONTAGE_REACH_M = 20.0

#: A lot with no frontage vote takes the nearest named road within this.
STREET_CEILING_M = 60.0

#: OpenStreetMap classes a lot can front. Not TRAIL: a house is not
#: addressed off a footpath.
STREET_CLASSES = frozenset({"ROAD", "ROAD_OTHER", "TRACK"})

NO_TOKEN = ("no parcel panel: CalTopo's parcel and structures layers are read "
            "through the WMS token, and the server has none")
NO_OSM_WARNING = ("osm.json is missing from the staged ground, so streets are "
                  "unknown: lots are grouped by adjacency alone, every block's "
                  "street is null, and road right-of-way cannot be told from a "
                  "vacant lot")
NO_PARCELS_WARNING = ("CalTopo's parcel layer draws no lot lines on this ground: "
                      "either it is not mapped or it is not town")
NOT_TOWN_WARNING = ("no built-up lot on this ground is under {limit:g} m2: it reads "
                    "as rural, and this tool outlines urban and suburban blocks only")

__all__ = ["check_params", "panel_pixel_m", "populated_areas", "populated_params",
           "populated_set"]


# --------------------------------------------------------------------------
# panels


def _panel_grid(subject_geometry: dict, pixel_m: float) -> tuple[Any, int, int, float, float]:
    """(affine transform, width_px, height_px, pixel_m actually used, ground
    metres per Mercator unit) for a panel over the subject plus MARGIN_M,
    north-up in EPSG:3857.

    Raises ValueError when even MAX_PIXEL_M would need more than
    MAX_SIDE_PX a side."""
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    area = shapely_transform(to_3857.transform, shapely_shape(subject_geometry))
    _lon, latitude = geometry_centroid(subject_geometry)
    scale = ground_meters_per_mercator_unit(latitude)
    margin = MARGIN_M / scale
    west, south, east, north = area.bounds
    west, south, east, north = west - margin, south - margin, east + margin, north + margin
    span_m = max(east - west, north - south) * scale
    used_m = max(float(pixel_m), span_m / MAX_SIDE_PX)
    if used_m > MAX_PIXEL_M + 1e-9:
        raise ValueError(
            f"the staged area is {span_m / 1000:.1f} km across; reading lot lines "
            f"needs pixels no coarser than {MAX_PIXEL_M:g} m, which would take more "
            f"than {MAX_SIDE_PX} px a side. Stage a smaller area (under "
            f"{MAX_SIDE_PX * MAX_PIXEL_M / 1000:.0f} km across) over the town")
    unit = used_m / scale
    width = max(1, int(math.ceil((east - west) / unit)))
    height = max(1, int(math.ceil((north - south) / unit)))
    return Affine(unit, 0.0, west, 0.0, -unit, north), width, height, used_m, scale


def panel_pixel_m(subject_geometry: dict, pixel_m: float = 1.0) -> float:
    """The ground metres per pixel lot lines over `subject_geometry` would
    be read at, or the ValueError `populated_areas` refuses an area too wide
    with. It reads no file and fetches nothing, so a scenario can refuse an
    area before it stages one it could never read lot lines over."""
    return _panel_grid(subject_geometry, pixel_m)[3]


def _alpha(path: Path, width: int, height: int) -> np.ndarray:
    """The panel's alpha channel as a (height, width) uint8 array. A panel
    with no transparency would read as opaque everywhere - every pixel an
    outline or a roof - so it is refused."""
    from PIL import Image

    with Image.open(path) as image:
        if not (image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info):
            raise ValueError("the panel has no transparency")
        alpha = image.convert("RGBA").getchannel("A")
        if alpha.size != (width, height):
            alpha = alpha.resize((width, height), Image.NEAREST)
        return np.asarray(alpha)


def _fetch_alpha(fetch, layer: str, *, wms_base_url: str, transform, width: int,
                 height: int, cache_dir: Path) -> np.ndarray:
    """One layer's alpha over the panel grid. A failed fetch is refused in
    the hub's own words: the fetcher's message carries the GetMap URL, token
    and all, so it is never passed on (see `net_errors.safe_reason`)."""
    bbox = (transform.c, transform.f - height * abs(transform.e),
            transform.c + width * transform.a, transform.f)
    reason = None
    try:
        path = fetch(wms_base_url, [layer], bbox, width, height, Path(cache_dir))
        return _alpha(Path(path), width, height)
    except Exception as error:  # noqa: BLE001 - both panels are this tool's one refusal
        reason = safe_reason(error)
    # Raised outside the except block so the fetch error is not chained on.
    raise ValueError(f"no {layer} panel: the CalTopo {layer} layer could not be read "
                     f"({reason}); no lot can be classified without it")


# --------------------------------------------------------------------------
# streets


def _streets(payload: dict | None) -> dict[str, list]:
    """Named roads a lot can front, as {name: [shapely line, ...]} in
    EPSG:3857 (the staged routes are already Web Mercator)."""
    from shapely.geometry import shape as shapely_shape

    streets: dict[str, list] = {}
    for route in (payload or {}).get("routes") or []:
        properties = route.get("properties") or {}
        name = str(properties.get("name") or "").strip()
        if not name or properties.get("route_type") not in STREET_CLASSES:
            continue
        streets.setdefault(name, []).append(shapely_shape(route["geometry"]))
    return streets


def _roads(payload: dict | None) -> list:
    """Every mapped road centreline, named or not, for telling right-of-way
    from a lot."""
    from shapely.geometry import shape as shapely_shape

    return [shapely_shape(route["geometry"]) for route in (payload or {}).get("routes") or []
            if (route.get("properties") or {}).get("route_type") in STREET_CLASSES]


def _burn(lines_by_value: list[tuple[Any, int]], transform, shape) -> np.ndarray:
    """Lines burned onto the panel grid (every cell they touch), each with
    its value; 0 elsewhere."""
    from rasterio.features import rasterize

    if not lines_by_value:
        return np.zeros(shape, dtype=np.int32)
    return rasterize(lines_by_value, out_shape=shape, transform=transform,
                     fill=0, all_touched=True, dtype="int32")


# --------------------------------------------------------------------------
# geometry helpers


def _polygon_3857(window: np.ndarray, transform, row0: int, col0: int, tolerance: float):
    """The True cells of `window` as one shapely polygon in EPSG:3857,
    simplified to `tolerance` Mercator units (about one pixel)."""
    from affine import Affine
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    local = transform @ Affine.translation(col0, row0)
    pieces = [shapely_shape(geom) for geom, value in
              rasterio_shapes(window.astype("uint8"), mask=window, transform=local)
              if value == 1]
    polygon = unary_union(pieces)
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    return simplified if not simplified.is_empty else polygon


def _point_inside(geometry):
    """A point guaranteed inside the polygon (pole of inaccessibility), as
    `parking._point_inside`."""
    from .parking import _point_inside as point_inside

    return point_inside(geometry)


def _close(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Morphological closing with a disc of `radius_px`, by distance
    transforms (linear in the window, whatever the radius)."""
    from scipy import ndimage

    if radius_px <= 0:
        return mask
    dilated = ndimage.distance_transform_edt(~mask) <= radius_px
    # The erosion must see ground outside the window as open, or a block
    # touching the window edge would erode back from it.
    padded = np.pad(dilated, 1, constant_values=False)
    return (ndimage.distance_transform_edt(padded) > radius_px)[1:-1, 1:-1]


def _fold_unit_parcels(regions: np.ndarray, outline: np.ndarray, area_m2: np.ndarray,
                       roof_share: np.ndarray, pixel_m: float, no_parent=None) -> int:
    """Fold every unit parcel into the lot around it, in place, and return
    how many were folded.

    A condominium or townhouse complex is mapped as one common-area lot with
    a parcel drawn tight round each unit - a footprint parcel, mostly roof,
    or a sliver under UNIT_LOT_MAX_M2. Read as lots they make a dozen
    forty-square-metre blocks out of one apartment complex. Each is
    relabelled as the neighbouring region it shares the most edge with (a
    region that is not itself a unit parcel), together with the outline
    cells between them, so the complex reads as the one lot a team searches.
    A region `no_parent` marks (street right-of-way, ground off the parcel
    map) never takes one in: a unit with no other neighbour stays a lot.
    """
    from scipy import ndimage

    unit = (area_m2 < UNIT_LOT_MAX_M2) | ((roof_share >= FOOTPRINT_ROOF_SHARE)
                                          & (area_m2 < FOOTPRINT_LOT_MAX_M2))
    unit[0] = False
    unit_ids = np.flatnonzero(unit)
    if unit_ids.size == 0:
        return 0
    # Unit parcels are drawn close together, so the lines between a unit and
    # the common area can run several pixels thick: look farther out, step by
    # step, until a lot that is not itself a unit parcel is reached.
    reaches = sorted({max(2, int(math.ceil(metres / pixel_m)))
                      for metres in UNIT_FOLD_REACH_M})
    pad = reaches[-1] + 1
    slices = ndimage.find_objects(regions)
    height, width = regions.shape
    folded = 0
    for region_id in unit_ids:
        window = slices[region_id - 1]
        if window is None:
            continue
        rows = slice(max(window[0].start - pad, 0), min(window[0].stop + pad, height))
        cols = slice(max(window[1].start - pad, 0), min(window[1].stop + pad, width))
        local = regions[rows, cols]
        mine = local == region_id
        parent = None
        for reach in reaches:
            ring = ndimage.binary_dilation(mine, iterations=reach) & ~mine
            around = local[ring]
            around = around[(around > 0) & ~unit[around]]
            if no_parent is not None:
                around = around[~no_parent[around]]
            if around.size:
                parent = int(np.bincount(around).argmax())
                break
        if parent is None:
            continue
        walls = ndimage.binary_dilation(mine, iterations=reach) & outline[rows, cols]
        local[mine | walls] = parent
        folded += 1
    return folded


# --------------------------------------------------------------------------
# the tool


def _largest_part(geometry):
    """The largest polygon of `geometry`, or an empty polygon. A cut can
    leave a sliver beside the main piece; the sliver is the cut's noise."""
    from shapely.geometry import Polygon

    parts = [geometry] if geometry.geom_type == "Polygon" else [
        part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]
    parts = [part for part in parts if not part.is_empty]
    return max(parts, key=lambda part: part.area) if parts else Polygon()


def _one_piece_each(blocks_raw: list[dict], *, regions: np.ndarray, lot_mask: np.ndarray,
                    block_owner: np.ndarray, transform, centre: tuple[float, float]
                    ) -> list[dict]:
    """The blocks as single pieces of ground that no other block shares.

    Made final one at a time, nearest the centre first. Each block loses
    every other block's lots - no house is in two blocks - and every cell an
    earlier block already took: the street and lot lines two closings both
    reach, which until 0.20.0 were drawn into both. What is left can fall
    apart, where another block's lots cut a run in two: each 4-connected
    piece holding a lot of its own becomes a block, its lots the ones with
    most of their ground in it, and `split` says so. A piece holding no lot
    (a strip of street, a lot line) is dropped.

    `block_owner` maps each lot to its raw block's number (one-based, in
    `blocks_raw` order), for the cut.
    """
    from scipy import ndimage

    for number, block in enumerate(blocks_raw, start=1):
        block_owner[block["members"]] = number
    inverse = ~transform
    centre_col, centre_row = inverse * centre

    def distance(block) -> float:
        r0, _r1, c0, _c1 = block["window"]
        rows, cols = np.nonzero(block["mask"])
        if rows.size == 0:
            return math.inf
        return math.hypot(rows.mean() + r0 - centre_row, cols.mean() + c0 - centre_col)

    order = sorted(range(len(blocks_raw)), key=lambda i: (distance(blocks_raw[i]), i))
    taken = np.zeros(regions.shape, dtype=bool)
    pieces: list[dict] = []
    for index in order:
        block = blocks_raw[index]
        number = index + 1
        r0, r1, c0, c1 = block["window"]
        local = regions[r0:r1, c0:c1]
        others = lot_mask[local] & (block_owner[local] != number) & (block_owner[local] > 0)
        mask = block["mask"] & ~others & ~taken[r0:r1, c0:c1]
        parts, count = ndimage.label(mask)
        if count == 0:
            continue
        home = {}
        for member in block["members"]:
            votes = np.bincount(parts[local == member], minlength=count + 1)[1:]
            if votes.any():
                home.setdefault(int(votes.argmax()) + 1, []).append(member)
        for part, members in sorted(home.items()):
            piece_mask = parts == part
            taken[r0:r1, c0:c1] |= piece_mask
            pieces.append({**block, "members": members, "mask": piece_mask,
                           "split": len(home) > 1})
    return pieces


def marker_colour(colour: str) -> str:
    """`colour` as CalTopo writes a marker colour, `#RRGGBB` upper case; a
    leading `#` may be left off. Anything else is refused rather than passed
    on, because CalTopo draws a colour it cannot read as black."""
    text = str(colour or "").strip()
    if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", text):
        raise ValueError(f"marker_color ({colour!r}) must be a hex colour such as "
                         f"{DEFAULT_MARKER_COLOR}")
    return "#" + text.lstrip("#").upper()


def populated_params(*, pixel_m: float, max_parcel_area_m2: float, min_building_m2: float,
                     large_parcel_min_cover: float, bridge_m: float, min_parcels: int,
                     max_distance_m: float | None, limit: int,
                     building_markers: bool = False,
                     marker_color: str = DEFAULT_MARKER_COLOR) -> dict:
    """The parameters a set id is hashed over, normalised so that the same
    call always names the same set.

    The two marker parameters join the hash only when markers are asked
    for, so a call without them names the same set it always did, and the
    colour of markers that were never drawn cannot split one set into two."""
    params = {
        # How lots are given to streets and kept at the ring: a set drawn by
        # an earlier method is drawn again rather than read back (0.35.0,
        # frontage from the right-of-way, whole blocks at the ring).
        "method": METHOD,
        "pixel_m": float(pixel_m),
        "max_parcel_area_m2": float(max_parcel_area_m2),
        "min_building_m2": float(min_building_m2),
        "large_parcel_min_cover": float(large_parcel_min_cover),
        "bridge_m": float(bridge_m),
        "min_parcels": int(min_parcels),
        "max_distance_m": None if max_distance_m is None else float(max_distance_m),
        "limit": int(limit),
    }
    if building_markers:
        params["building_markers"] = True
        params["marker_color"] = marker_colour(marker_color)
    return params


def _building_blocks(roof: np.ndarray, buildings: np.ndarray, regions: np.ndarray,
                     block_owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(building labels, block numbers): every building that is in a block,
    with the block it is in.

    A building belongs to the ONE block whose lots hold most of its roof, so
    a footprint a lot line runs across - a garage on the boundary, a roof
    drawn a metre over the line - counts once, and only once. Both
    `facts.buildings` and the markers are counted by this rule, so the two
    always agree."""
    labels = buildings[roof]
    owners = regions[roof]
    keep = (owners > 0) & (labels > 0)
    labels, owners = labels[keep], owners[keep]
    if labels.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    # The region holding most of each building's roof: count every
    # (building, region) pair, then keep each building's largest.
    pairs, counts = np.unique(np.stack([labels, owners]), axis=1, return_counts=True)
    order = np.lexsort((-counts, pairs[0]))
    pairs = pairs[:, order]
    first = np.concatenate([[True], pairs[0, 1:] != pairs[0, :-1]])
    building_ids, region_ids = pairs[0, first], pairs[1, first]
    blocks = block_owner[region_ids]
    in_block = blocks > 0
    return building_ids[in_block], blocks[in_block]


def _building_markers(roof: np.ndarray, buildings: np.ndarray, building_ids: np.ndarray,
                      blocks: np.ndarray, transform,
                      to_wgs84) -> dict[int, list[tuple[float, float]]]:
    """{block number: [(longitude, latitude), ...]}, one point per building
    `_building_blocks` put in a block. The point is the roof cell farthest
    from the roof's own edge - on the roof, never in the yard beside an
    L-shaped house, the rule `_point_inside` keeps for a block's point - at
    that cell's centre."""
    from scipy import ndimage

    if building_ids.size == 0:
        return {}
    depth = ndimage.distance_transform_edt(roof)
    positions = ndimage.maximum_position(depth, buildings, index=building_ids)
    markers: dict[int, list[tuple[float, float]]] = {}
    for block, (row, col) in zip(blocks.tolist(), positions):
        x, y = transform * (col + 0.5, row + 0.5)
        markers.setdefault(int(block), []).append(to_wgs84.transform(x, y))
    return markers


def _marker_features(points: list[tuple[float, float]], colour: str) -> list[dict]:
    """CalTopo Marker features, the house symbol in `colour`, in a fixed
    order (west to east, then south to north) so a set reads back the same."""
    return [{"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
             "properties": {"class": "Marker", "title": "", "description": "",
                            "marker-symbol": MARKER_SYMBOL, "marker-color": colour,
                            "marker-size": 1}}
            for longitude, latitude in sorted(points)]


def check_params(*, pixel_m, max_parcel_area_m2, min_building_m2, large_parcel_min_cover,
                 bridge_m, min_parcels, max_distance_m, limit) -> None:
    """The parameter checks `populated_areas` makes before it reads
    anything: a ValueError naming the first one out of range. Public so a
    scenario refuses a bad override in these words before it stages."""
    if not 0 < pixel_m <= MAX_PIXEL_M:
        raise ValueError(f"pixel_m ({pixel_m}) must be above 0 and at most "
                         f"{MAX_PIXEL_M:g}; lot lines are not readable coarser")
    if max_parcel_area_m2 <= 0:
        raise ValueError(f"max_parcel_area_m2 ({max_parcel_area_m2}) must be above zero")
    if min_building_m2 <= 0:
        raise ValueError(f"min_building_m2 ({min_building_m2}) must be above zero; "
                         "a lot with no roof on it is not built-up")
    if not 0 < large_parcel_min_cover <= 1:
        raise ValueError(f"large_parcel_min_cover ({large_parcel_min_cover}) is a "
                         "share of the lot, above 0 and at most 1")
    if bridge_m < 0:
        raise ValueError(f"bridge_m ({bridge_m}) is below zero")
    if min_parcels < 1:
        raise ValueError(f"min_parcels ({min_parcels}) must be at least 1")
    if max_distance_m is not None and max_distance_m <= 0:
        raise ValueError(f"max_distance_m ({max_distance_m}) must be above zero")
    if limit < 0:
        raise ValueError(f"limit ({limit}) is negative; 0 means every block")


def populated_areas(directory, *, pixel_m: float = 1.0,
                    max_parcel_area_m2: float = 4047.0,
                    min_building_m2: float = 30.0,
                    large_parcel_min_cover: float = 0.10,
                    bridge_m: float = 12.0, min_parcels: int = 3,
                    max_distance_m: float | None = None, limit: int = 0,
                    building_markers: bool = False,
                    marker_color: str = DEFAULT_MARKER_COLOR,
                    wms_base_url: str | None = None, cache_dir: Path | None = None,
                    fetch=None) -> dict[str, Any]:
    """The populated blocks on the ground staged under `directory`, nearest
    the area's centre first. See the module docstring for the method.

    Returns `{"blocks": [...], "summary", "assumptions", "center",
    "warnings"}`; each block carries its polygon (`geometry`, WGS84). The
    MCP tool saves them as a candidate set and returns summaries.

    `building_markers` - off unless asked for - adds `markers` to every
    block: one CalTopo Marker per building in it, CalTopo's house symbol
    (`hut`) in `marker_color`, white by default, standing on the roof. The
    block's outline is the same either way; markers are what replaces
    tracing each building, not an addition to the outline.

    `fetch` is the panel fetcher, injected so tests need no network; it
    defaults to `caltopo.wms.fetch_panel`. Raises ValueError on a parameter
    out of range, an area too big to read lot lines over, no WMS token, or
    a parcel or structures panel that cannot be had.
    """
    import json

    from pyproj import Geod, Transformer
    from scipy import ndimage
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform

    check_params(pixel_m=pixel_m, max_parcel_area_m2=max_parcel_area_m2,
                 min_building_m2=min_building_m2,
                 large_parcel_min_cover=large_parcel_min_cover, bridge_m=bridge_m,
                 min_parcels=min_parcels, max_distance_m=max_distance_m, limit=limit)
    colour = marker_colour(marker_color) if building_markers else None
    directory = Path(directory)
    if not wms_base_url:
        raise ValueError(NO_TOKEN)
    if cache_dir is None:
        raise ValueError("no parcel panel: the server has no cache directory to "
                         "fetch it into")
    if fetch is None:
        from ..caltopo.wms import fetch_panel as fetch

    subject = read_subject(directory)
    subject_geometry = subject.get("geometry") or {}
    centre = geometry_centroid(subject_geometry)
    if centre is None:
        raise ValueError("the staged ground has no readable subject area")
    centre_lon, centre_lat = centre
    transform, width, height, used_m, scale = _panel_grid(subject_geometry, pixel_m)
    shape = (height, width)
    cell_m2 = used_m * used_m
    warnings: list[str] = []

    panels = {}
    for layer in (PARCEL_LAYER, STRUCTURES_LAYER):
        panels[layer] = _fetch_alpha(fetch, layer, wms_base_url=wms_base_url,
                                     transform=transform, width=width, height=height,
                                     cache_dir=cache_dir)
    outline = panels[PARCEL_LAYER] >= PARCEL_ALPHA_MIN
    roof = panels[STRUCTURES_LAYER] >= STRUCTURE_ALPHA_MIN
    roof = ndimage.binary_fill_holes(roof)
    if outline.mean() > 0.5:
        raise ValueError("the parcel panel is mostly opaque, so it is not a parcel "
                         "layer (an error image or a basemap); nothing is classified")

    osm_path = directory / "osm.json"
    payload = json.loads(osm_path.read_text(encoding="utf-8")) if osm_path.is_file() else None
    if payload is None:
        warnings.append(NO_OSM_WARNING)
    streets = _streets(payload)
    street_names = sorted(streets)

    assumptions = {
        "sources": {"parcels": "CalTopo parcel layer (rendered lot lines)",
                    "buildings": "CalTopo structures layer",
                    "streets": ("OpenStreetMap, as staged" if payload is not None
                                else "unknown (no osm.json)")},
        "pixel_m": round(used_m, 3),
        "max_parcel_area_m2": max_parcel_area_m2,
        "min_building_m2": min_building_m2,
        "large_parcel_min_cover": large_parcel_min_cover,
        "bridge_m": bridge_m,
        "min_parcels": min_parcels,
        "frontage_reach_m": FRONTAGE_REACH_M,
        "street_ceiling_m": STREET_CEILING_M,
        "note": ("lots are read from CalTopo's drawn lot lines and buildings from its "
                 "structures layer: an undrawn house reads as a vacant lot and an "
                 "undrawn lot line as one lot. A first cut for the planner to adjust."),
    }
    empty_summary = {"count": 0, "total_before_limit": 0, "truncated_to": None,
                     "parcels_built_up": 0, "parcels_in_blocks": 0,
                     "parcels_left_out": 0, "large_parcels": 0}
    centre_out = {"latitude": centre_lat, "longitude": centre_lon}
    if not outline.any():
        warnings.append(NO_PARCELS_WARNING)
        return {"blocks": [], "summary": empty_summary, "assumptions": assumptions,
                "center": centre_out, "warnings": warnings}

    # ---- regions between the lot lines
    regions, count = ndimage.label(~outline)
    labels = np.arange(count + 1)
    area_px = np.bincount(regions.ravel(), minlength=count + 1)
    roof_px = np.bincount(regions.ravel(), weights=roof.ravel(), minlength=count + 1)
    road_lines = _roads(payload)
    near_road = None
    if road_lines:
        road_cells = _burn([(line, 1) for line in road_lines], transform, shape) > 0
        near_road = ndimage.distance_transform_edt(~road_cells) * used_m <= ROW_ROAD_REACH_M

    def road_px_of():
        if near_road is None:
            return np.zeros(count + 1)
        return np.bincount(regions.ravel(), weights=near_road.ravel(), minlength=count + 1)

    # A unit parcel is folded into the lot round it, never into the street:
    # a row of footprint lots on a downtown street shares most of its edge
    # with the right-of-way, and folded there it was searched by no one
    # (and its roofs stopped the street reading as street). Nor into ground
    # off the parcel map.
    with np.errstate(divide="ignore", invalid="ignore"):
        roof_share_raw = roof_px / np.maximum(area_px, 1)
        road_share_raw = road_px_of() / np.maximum(area_px, 1)
    no_parent = (((road_share_raw >= ROW_ROAD_SHARE) & (roof_share_raw < ROW_MAX_ROOF_SHARE))
                 | (area_px * cell_m2 > FOLD_PARENT_MAX_M2))
    folded = _fold_unit_parcels(regions, outline, area_px * cell_m2, roof_share_raw, used_m,
                                no_parent=no_parent)
    if folded:
        area_px = np.bincount(regions.ravel(), minlength=count + 1)
        roof_px = np.bincount(regions.ravel(), weights=roof.ravel(), minlength=count + 1)
    border = np.zeros(count + 1, dtype=bool)
    border[np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0],
                                     regions[:, -1]]))] = True
    border[0] = False
    road_px = road_px_of()

    with np.errstate(divide="ignore", invalid="ignore"):
        roof_share = np.where(area_px > 0, roof_px / np.maximum(area_px, 1), 0.0)
        road_share = np.where(area_px > 0, road_px / np.maximum(area_px, 1), 0.0)
    area_m2 = area_px * cell_m2
    roof_m2 = roof_px * cell_m2
    right_of_way = (road_share >= ROW_ROAD_SHARE) & (roof_share < ROW_MAX_ROOF_SHARE)
    built = (roof_m2 >= min_building_m2) & ~right_of_way
    small_lot = built & (area_m2 <= max_parcel_area_m2)
    large_lot = built & (area_m2 > max_parcel_area_m2) & (roof_share >= large_parcel_min_cover)
    small_lot[0] = large_lot[0] = False
    small_ids = labels[small_lot]
    large_ids = labels[large_lot]
    vacant = (~built & ~right_of_way & (area_m2 >= VACANT_MIN_M2)
              & (area_m2 <= max_parcel_area_m2))
    vacant[0] = False
    vacant_ids = labels[vacant]

    # One label per building; each is counted in one block only, see
    # `_building_blocks`.
    buildings, n_buildings = ndimage.label(roof, structure=np.ones((3, 3)))

    if small_ids.size == 0 and large_ids.size == 0:
        warnings.append(NOT_TOWN_WARNING.format(limit=max_parcel_area_m2))
        return {"blocks": [], "summary": empty_summary, "assumptions": assumptions,
                "center": centre_out, "warnings": warnings}

    # ---- which street each lot fronts
    slices = ndimage.find_objects(regions)
    street_of = np.zeros(count + 1, dtype=np.int32)       # 0: no street
    if street_names:
        burned = _burn([(line, index + 1) for index, name in enumerate(street_names)
                        for line in streets[name]], transform, shape)
        distance_px, (near_rows, near_cols) = ndimage.distance_transform_edt(
            burned == 0, return_indices=True)
        nearest_street = burned[near_rows, near_cols]
        reach_px = FRONTAGE_REACH_M / used_m
        ceiling_px = STREET_CEILING_M / used_m
        # Right-of-way cells near enough a named road to say which it is.
        row_cells = right_of_way[regions] & (distance_px <= ceiling_px)
        for region_id in np.concatenate([small_ids, large_ids, vacant_ids]):
            window = slices[region_id - 1]
            rows = slice(max(window[0].start - 3, 0), min(window[0].stop + 3, height))
            cols = slice(max(window[1].start - 3, 0), min(window[1].stop + 3, width))
            mine = regions[rows, cols] == region_id
            # Three cells reach across a lot line (1 px, or 2 where the
            # panel's shrink thickened it) into the ground beyond.
            front = ndimage.binary_dilation(mine, iterations=3) & ~mine & row_cells[rows, cols]
            votes = nearest_street[rows, cols][front]
            if votes.size:
                street_of[region_id] = int(np.bincount(votes).argmax())
                continue
            edge = ndimage.binary_dilation(mine, iterations=2) & ~mine
            close = edge & (distance_px[rows, cols] <= reach_px)
            votes = nearest_street[rows, cols][close]
            if votes.size:
                street_of[region_id] = int(np.bincount(votes).argmax())
                continue
            inside = distance_px[rows, cols][mine]
            if inside.size and inside.min() <= ceiling_px:
                where = np.argmin(np.where(mine, distance_px[rows, cols], np.inf))
                street_of[region_id] = int(nearest_street[rows, cols].ravel()[where])

    # ---- blocks
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    area_3857 = shapely_transform(to_3857.transform, shapely_shape(subject_geometry))
    geod = Geod(ellps="WGS84")
    tolerance = (used_m / scale) * 1.0
    bridge_px = bridge_m / used_m
    pad = int(math.ceil(bridge_px)) + 3

    block_owner = np.zeros(count + 1, dtype=np.int32)      # region -> block number
    lot_mask = small_lot | large_lot
    blocks_raw: list[dict] = []
    left_out = 0

    def bounds(ids):
        r0 = min(slices[i - 1][0].start for i in ids)
        r1 = max(slices[i - 1][0].stop for i in ids)
        c0 = min(slices[i - 1][1].start for i in ids)
        c1 = max(slices[i - 1][1].stop for i in ids)
        return (max(r0 - pad, 0), min(r1 + pad, height), max(c0 - pad, 0), min(c1 + pad, width))

    groups: dict[int, list[int]] = {}
    for region_id in small_ids:
        groups.setdefault(int(street_of[region_id]), []).append(int(region_id))
    vacant_by_street: dict[int, list[int]] = {}
    for region_id in vacant_ids:
        vacant_by_street.setdefault(int(street_of[region_id]), []).append(int(region_id))
    orphans: list[dict] = []

    for street_index, built_ids in sorted(groups.items()):
        # The street's vacant lots that touch its built-up lots (across a
        # lot line or the street) join it: one layer, so an empty stretch
        # of street past the last house is not carried along.
        ids = list(built_ids)
        spare = vacant_by_street.get(street_index, [])
        if spare:
            r0, r1, c0, c1 = bounds(built_ids + spare)
            local = regions[r0:r1, c0:c1]
            near = ndimage.distance_transform_edt(~np.isin(local, built_ids)) <= bridge_px * 2
            touching = set(np.unique(local[near]).tolist())
            ids += [v for v in spare if v in touching]
            lot_mask[[v for v in spare if v in touching]] = True
        r0, r1, c0, c1 = bounds(ids)
        local = regions[r0:r1, c0:c1]
        mine = np.isin(local, ids)
        # Each lot's own outline is part of it: dilate one cell into walls only.
        mine = mine | (ndimage.binary_dilation(mine) & outline[r0:r1, c0:c1])
        closed = ndimage.binary_fill_holes(_close(mine, bridge_px))
        runs, n_runs = ndimage.label(closed)
        for run in range(1, n_runs + 1):
            run_mask = runs == run
            members = [i for i in ids if np.any(run_mask & (local == i))]
            if sum(1 for i in members if built[i]) < min_parcels:
                # Too few houses for a block of their own: searched with
                # the block they touch, below, rather than left out.
                orphans.append({"street_index": street_index, "members": members,
                                "window": (r0, r1, c0, c1), "mask": run_mask})
                continue
            blocks_raw.append({"kind": "street", "street_index": street_index,
                               "members": members, "window": (r0, r1, c0, c1),
                               "mask": run_mask})
    for region_id in large_ids:
        r0, r1, c0, c1 = bounds([int(region_id)])
        local = regions[r0:r1, c0:c1]
        mine = local == region_id
        mine = mine | (ndimage.binary_dilation(mine) & outline[r0:r1, c0:c1])
        blocks_raw.append({"kind": "large_parcel",
                           "street_index": int(street_of[region_id]),
                           "members": [int(region_id)], "window": (r0, r1, c0, c1),
                           "mask": ndimage.binary_fill_holes(mine)})

    # Short runs join the block whose ground comes nearest (within the
    # closing that joins a street's lots); one touching none is left out.
    # A block whose LOTS it touches (across a lot line) first, by the length
    # of the touch - not a block's street ground, which the nearer block
    # takes first and would leave the orphan a piece of its own; failing
    # that the most ground within the closing's reach.
    def contact(orphan, pool, iterations, lots_only):
        o0, o1, p0, p1 = orphan["window"]
        grown = ndimage.binary_dilation(orphan["mask"], iterations=iterations)
        best, most = None, 0
        for block in pool:
            if block is orphan:
                continue
            b0, b1, q0, q1 = block["window"]
            r0, r1, c0, c1 = max(o0, b0), min(o1, b1), max(p0, q0), min(p1, q1)
            if r0 >= r1 or c0 >= c1:
                continue
            theirs = block["mask"][r0 - b0:r1 - b0, c0 - q0:c1 - q0]
            if lots_only:
                theirs = theirs & np.isin(regions[r0:r1, c0:c1], block["members"])
            touch = int((grown[r0 - o0:r1 - o0, c0 - p0:c1 - p0] & theirs).sum())
            if touch > most:
                best, most = block, touch
        return best

    def join(best, orphan):
        o0, o1, p0, p1 = orphan["window"]
        b0, b1, q0, q1 = best["window"]
        r0, r1, c0, c1 = min(o0, b0), max(o1, b1), min(p0, q0), max(p1, q1)
        mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        mask[b0 - r0:b1 - r0, q0 - c0:q1 - c0] |= best["mask"]
        mask[o0 - r0:o1 - r0, p0 - c0:p1 - c0] |= orphan["mask"]
        closed = ndimage.binary_fill_holes(_close(mask, bridge_px))
        # Keep the closing only where it joins the two: the piece holding them.
        parts, _n = ndimage.label(closed)
        keep = np.unique(parts[mask])
        best["mask"] = np.isin(parts, keep[keep > 0])
        best["window"] = (r0, r1, c0, c1)
        best["members"] = best["members"] + orphan["members"]
        best["joined"] = best.get("joined", []) + [orphan["street_index"]]

    # In passes until none joins: a short run touching only another short
    # run joins once that one has joined a block.
    reach = int(math.ceil(bridge_px * 2))
    waiting = list(orphans)
    while waiting:
        still = []
        for orphan in waiting:
            best = (contact(orphan, blocks_raw, 3, True)
                    or contact(orphan, blocks_raw, reach, False))
            if best is None:
                still.append(orphan)
            else:
                join(best, orphan)
        if len(still) == len(waiting):
            break
        waiting = still
    left_out += sum(1 for orphan in waiting for i in orphan["members"] if built[i])

    blocks = _one_piece_each(blocks_raw, regions=regions, lot_mask=lot_mask,
                             block_owner=block_owner, transform=transform,
                             centre=to_3857.transform(centre_lon, centre_lat))
    # A piece the cut left with too few houses to stand alone - a lot or two
    # beyond the nearer block's ground - joins the block whose lots it
    # touches, as a short run does.
    for piece in [b for b in blocks if b["split"]
                  and sum(1 for i in b["members"] if built[i]) < min_parcels]:
        best = contact(piece, blocks, 3, True)
        if best is not None:
            join(best, piece)
            blocks.remove(piece)
    block_owner[:] = 0
    for number, block in enumerate(blocks, start=1):
        block_owner[block["members"]] = number
    building_ids, building_block = (_building_blocks(roof, buildings, regions, block_owner)
                                    if n_buildings else (np.zeros(0, int), np.zeros(0, int)))
    buildings_per_block = np.bincount(building_block, minlength=len(blocks) + 1)
    markers_by_block = (_building_markers(roof, buildings, building_ids, building_block,
                                          transform, to_wgs84)
                        if building_markers else {})

    from shapely.geometry import Point

    centre_3857 = Point(*to_3857.transform(centre_lon, centre_lat))
    entries: list[dict] = []
    claimed = None          # the outlines already drawn, nearest first
    for number, block in enumerate(blocks, start=1):
        r0, r1, c0, c1 = block["window"]
        polygon = _polygon_3857(block["mask"], transform, r0, c0, tolerance)
        # Two neighbours' outlines are simplified apart, so their shared edge
        # can cross by a fraction of a cell: the nearer block keeps it.
        if claimed is not None and polygon.intersects(claimed):
            polygon = _largest_part(polygon.difference(claimed))
        if polygon.is_empty:
            continue
        claimed = polygon if claimed is None else claimed.union(polygon)
        if not polygon.intersects(area_3857):
            continue
        # A block any part of which is within `max_distance_m` is kept whole:
        # the ring is extended out to the back fences beyond the nearest
        # street, never cut through a street's lots.
        if (max_distance_m is not None
                and polygon.distance(centre_3857) * scale > float(max_distance_m)):
            continue
        inside = _point_inside(polygon)
        longitude, latitude = to_wgs84.transform(inside.x, inside.y)
        bearing, _back, distance_m = geod.inv(centre_lon, centre_lat, longitude, latitude)
        members = block["members"]
        street = street_names[block["street_index"] - 1] if block["street_index"] else None
        lot_areas = area_m2[members]
        edge_cut = bool(border[members].any())
        facts = {
            "street": street,
            "parcels": int(built[members].sum()),
            "vacant_lots": int((~built[members]).sum()),
            "buildings": int(buildings_per_block[number]),
            "area_ha": round(polygon.area * scale * scale / 10_000, 3),
            "median_parcel_m2": round(float(np.median(lot_areas)), 0),
            "roof_cover_pct": round(100 * float(roof_m2[members].sum() / lot_areas.sum()), 1),
            "point": {"latitude": latitude, "longitude": longitude},
            "distance_from_center_m": round(distance_m, 0),
            "bearing_from_center_deg": round(bearing % 360.0, 0),
            "clipped_at_edge": edge_cut,
            "holes": sum(len(p.interiors) for p in getattr(polygon, "geoms", [polygon])),
        }
        block_warnings = []
        if edge_cut:
            block_warnings.append("a lot in this block runs past the ground read; the "
                                  "block may carry on beyond it")
        if facts["holes"]:
            block_warnings.append("the outline has a hole (another block's lots inside "
                                  "it); CalTopo keeps one ring per polygon, so cut it "
                                  "open with buffer_geometry(single_ring=True) or "
                                  "draw it as two")
        if block.get("joined"):
            joined = sorted({street_names[i - 1] if i else "an unnamed street"
                             for i in block["joined"]})
            block_warnings.append(
                f"lots on {', '.join(joined)}, too few for a block of their own, are "
                "searched with this one")
        if block["split"]:
            block_warnings.append("this is one piece of a run of lots that another "
                                  "block's lots cut in two; each piece is a block of "
                                  "its own")
        entry = {"kind": block["kind"], "name": street, "facts": facts,
                 "geometry": to_wgs84_geojson(polygon), "warnings": block_warnings,
                 "_order": (distance_m, street or "", block["kind"])}
        if building_markers:
            entry["markers"] = _marker_features(markers_by_block.get(number, []), colour)
            facts["markers"] = len(entry["markers"])
        entries.append(entry)

    entries.sort(key=lambda entry: entry.pop("_order"))
    # Two runs of one street read as one name twice; number them in order,
    # street blocks and large parcels separately ("Elm Street" and "Elm
    # Street - large parcel" are already told apart).
    seen: dict[tuple, int] = {}
    totals: dict[tuple, int] = {}
    for entry in entries:
        if entry["name"]:
            key = (entry["name"], entry["kind"])
            totals[key] = totals.get(key, 0) + 1
    for entry in entries:
        name = entry["name"]
        key = (name, entry["kind"])
        if name and totals[key] > 1:
            seen[key] = seen.get(key, 0) + 1
            entry["name"] = f"{name} ({seen[key]})"
        if entry["kind"] == "large_parcel" and entry["name"]:
            entry["name"] = f"{entry['name']} - large parcel"

    total = len(entries)
    truncated = None
    if limit and total > limit:
        entries = entries[:limit]
        truncated = limit
    summary = {
        "count": len(entries), "total_before_limit": total, "truncated_to": truncated,
        "parcels_built_up": int(small_ids.size + large_ids.size),
        "parcels_in_blocks": int(sum(e["facts"]["parcels"] for e in entries)),
        "parcels_left_out": int(left_out),
        "large_parcels": int(sum(1 for e in entries if e["kind"] == "large_parcel")),
    }
    if building_markers:
        summary["markers"] = int(sum(len(e["markers"]) for e in entries))
        assumptions["markers"] = {"symbol": MARKER_SYMBOL, "color": colour,
                                  "rule": ("one per building on CalTopo's structures "
                                           "layer, in the block holding most of its "
                                           "roof, placed on the roof")}
    if left_out:
        warnings.append(f"{left_out} built-up lot(s) were in runs of fewer than "
                        f"{min_parcels} along their street and are in no block")
    return {"blocks": entries, "summary": summary, "assumptions": assumptions,
            "center": centre_out, "warnings": warnings}


def populated_set(directory, params: dict, *, handle: str, wms_base_url: str | None,
                  cache_dir: Path | None, fetch=None):
    """`populated_areas` over `params` (from `populated_params`) as a
    `CandidateSet` ready to save: one `block` candidate per block, ids
    `populated-xxxxxx.N` in the order the blocks came back, the polygon
    kept for `get_candidate`."""
    from datetime import datetime, timezone

    from .candidates import Candidate, CandidateSet, candidate_id, make_set_id

    result = populated_areas(directory, wms_base_url=wms_base_url, cache_dir=cache_dir,
                             fetch=fetch, **{k: v for k, v in params.items() if k != "method"})
    set_id = make_set_id("populated", params)
    records = [Candidate(id=candidate_id(set_id, index), kind="block", axis=block["kind"],
                         axis_ids=[], sources=["parcel", "structures", "osm"],
                         name=block["name"], geometry=block["geometry"],
                         facts=block["facts"], norm_flags=[], warnings=block["warnings"],
                         markers=block.get("markers"))
               for index, block in enumerate(result["blocks"], start=1)]
    return CandidateSet(
        set_id=set_id, tool="populated_areas", params=params, handle=handle,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        candidates=records,
        summary={**result["summary"], "center": result["center"]},
        assumptions=result["assumptions"], warnings=result["warnings"])
