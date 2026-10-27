"""A fixed Web Mercator tile grid, shared by every static layer.

WHY
---
Measured on a real cache: 179 MB from a handful of overlapping assignments
on one training map. 17 DEMs (62 MB) and 102 WMS panels (115 MB) keyed by
the hash of an exact bbox and pixel size, sharing nothing between
assignments 100 m apart - against 10 EVT and OSM tiles (2.7 MB) covering
the same ground, because those two were keyed on a fixed grid instead.

`terrain/evt.py` and `terrain/linear.py` already had the answer. This
module generalises it so DEM and WMS can use it too, and so the next
fetcher somebody writes gets it for free.

THE LEVELS
----------
Resolutions follow the standard slippy-map pyramid - level `z` has a pixel
size of `WORLD_M / (256 * 2**z)` - so a cached tile means the same thing
here as it does in any other mapping tool, and the numbers are familiar to
anyone who has debugged a basemap.

Tiles are 1024 px rather than the customary 256. That is the exact ceiling
CalTopo's WMS GetMap accepts, so one tile is one server request: a panel
covers four to nine tiles instead of sixty-odd, and no tile is ever
stitched from sub-requests. Fewer round trips on a cold fetch, and the same
sharing on a warm one.

CHOOSING A LEVEL
----------------
`level_for` takes the level nearest the requested resolution, and refuses
to go more than `COARSER_TOLERANCE` softer than what was asked for. A
request almost never lands exactly on a level, so some resampling is
unavoidable; what matters is that it is a downsample from real data rather
than an upsample inventing detail that was never fetched.

That resampling is not free. Measured against 3DEP on a real assignment,
elevations moved by a mean of 0.13 m (max 1.22 m, well inside the source's
own vertical accuracy) and mean slope by 0.07 degrees - but individual
cells on sharp breaks moved by as much as 15 degrees. Aggregate metrics
are unaffected; a single steepest-cell reading may not be.

Two assignments of similar size over the same ground land on the same
level and share every tile. Two of very different sizes land on different
levels and share none, which is inherent: they genuinely need different
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geo import Bbox

#: The Web Mercator world extent, edge to edge, in metres.
WORLD_M = 40_075_016.685_578_488

#: Tile edge in pixels. Matches CalTopo's GetMap ceiling exactly - see the
#: module docstring.
TILE_PX = 1024

#: Deepest level worth asking for. Level 20 is ~0.15 m/px, finer than any
#: source this project reads, so anything beyond it would be interpolation
#: dressed up as data.
MAX_LEVEL = 20
MIN_LEVEL = 0

#: How much coarser than the request a level may be before the next finer
#: one is taken instead. Only helps when a request lands just below a level
#: boundary; a request at a fractional level of, say, .35 still takes the
#: finer level and downsamples, because the alternative would be handing
#: back a grid 28% coarser than asked for.
#:
#: Kept deliberately tight. Widening it would remove more resampling, but
#: it would do so by returning less resolution than the caller requested,
#: and everything downstream of a DEM in this project is a measurement.
COARSER_TOLERANCE = 1.06


def pixel_size_m(level: int) -> float:
    """Mercator units per pixel at a level. Standard slippy-map resolutions.

    These are MERCATOR units, not ground metres - EPSG:3857 is inflated by
    1/cos(latitude). Convert with `geo.ground_meters_per_mercator_unit`
    before reporting anything to a human.
    """
    return WORLD_M / (256 * 2 ** level)


def tile_size_m(level: int) -> float:
    return pixel_size_m(level) * TILE_PX


def level_for(bbox_3857: Bbox, width_px: int, height_px: int) -> int:
    """The level closest to the requested resolution without going soft.

    Taking the next FINER level unconditionally sounds safer and is not: it
    means every request fetches at up to twice the resolution it wants and
    then halves it, and that resample is what perturbs the numbers a metric
    is computed from. Taking the nearest level instead - while refusing to
    go more than `COARSER_TOLERANCE` coarser than asked - usually lands on
    a level that needs almost no resampling at all.

    Clamped to the pyramid. A request outside it gets the nearest level
    that exists rather than an error: the caller asked for a picture of
    some ground, and refusing over a resolution technicality helps nobody.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"width and height must be positive, "
                         f"got {width_px}x{height_px}")
    min_x, min_y, max_x, max_y = bbox_3857
    requested = min((max_x - min_x) / width_px, (max_y - min_y) / height_px)
    if requested <= 0:
        raise ValueError(f"degenerate bbox: {bbox_3857}")

    exact = math.log2(WORLD_M / (256 * requested))
    level = round(exact)
    if pixel_size_m(level) > requested * COARSER_TOLERANCE:
        level += 1  # too soft - take the finer one and downsample
    return max(MIN_LEVEL, min(MAX_LEVEL, level))


@dataclass(frozen=True)
class Tile:
    level: int
    col: int
    row: int

    @property
    def bbox(self) -> Bbox:
        """This tile's extent in EPSG:3857.

        The grid origin is the Mercator origin (0, 0), not the world corner,
        so `col` and `row` go negative in the western and southern
        hemispheres. That keeps the arithmetic trivial and the keys stable;
        it is the same convention `evt.py` already uses.
        """
        size = tile_size_m(self.level)
        return (self.col * size, self.row * size,
                (self.col + 1) * size, (self.row + 1) * size)

    @classmethod
    def from_slippy(cls, level: int, x: int, y: int) -> Tile:
        """The tile a slippy-map client addresses as z/x/y, when that client
        uses `TILE_PX` tiles (Leaflet's `tileSize: 1024`).

        Same pixel size per level, same edge; only the origin differs. A
        slippy grid counts from the world's north-west corner, this grid
        from the Mercator origin, so the shift is half the world in tiles:
        `2 ** (level - 3)` (the world is `2 ** (level - 2)` tiles across at
        1024 px). Levels below 3 have fewer than one tile per hemisphere
        and no integer shift, so they are refused rather than approximated.
        The pixel size matches because Leaflet's CRS scale is `256 * 2**z`
        regardless of `tileSize`, so a 1024 px client simply has
        `2**(z-2)` tiles across; a WMTS 1024 px tile matrix set numbers its
        levels differently and must not be fed here.
        """
        if not (3 <= level <= MAX_LEVEL):
            raise ValueError(f"slippy addressing needs 3 <= level <= {MAX_LEVEL} with "
                             f"{TILE_PX} px tiles; got level {level}")
        half = 2 ** (level - 3)
        if not (0 <= x < 2 * half and 0 <= y < 2 * half):
            raise ValueError(f"tile x={x}, y={y} is off the grid at level {level} "
                             f"(0 to {2 * half - 1})")
        return cls(level, x - half, half - 1 - y)

    def key(self, layer: str, extension: str) -> str:
        """Cache-relative path. Layer first so one layer can be dropped."""
        return f"tiles/{layer}/{self.level}/{self.col}/{self.row}.{extension}"


def tiles_covering(bbox_3857: Bbox, level: int) -> list[Tile]:
    """Every tile at `level` that intersects the bbox, row-major from the top.

    Ordered north-to-south then west-to-east, so a caller pasting into an
    image canvas can walk the list without sorting: image row 0 is maximum
    northing.
    """
    size = tile_size_m(level)
    min_x, min_y, max_x, max_y = bbox_3857
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"degenerate bbox: {bbox_3857}")

    min_col = math.floor(min_x / size)
    max_col = math.ceil(max_x / size) - 1
    min_row = math.floor(min_y / size)
    max_row = math.ceil(max_y / size) - 1

    return [
        Tile(level, col, row)
        for row in range(max_row, min_row - 1, -1)
        for col in range(min_col, max_col + 1)
    ]


def mosaic_bbox(tiles: list[Tile]) -> Bbox:
    """The combined extent of a tile set - the mosaic's own georeference."""
    if not tiles:
        raise ValueError("no tiles")
    boxes = [tile.bbox for tile in tiles]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def pixel_window(outer: Bbox, inner: Bbox, level: int) -> tuple[int, int, int, int]:
    """Where `inner` sits inside `outer`, as (left, top, right, bottom) pixels.

    Top-left origin, because that is what both PIL and raster row order use.
    Rounded rather than truncated so a window never loses a pixel column to
    floating-point drift at a tile boundary.
    """
    size = pixel_size_m(level)
    left = round((inner[0] - outer[0]) / size)
    right = round((inner[2] - outer[0]) / size)
    # Rows count down from the top, which is maximum northing.
    top = round((outer[3] - inner[3]) / size)
    bottom = round((outer[3] - inner[1]) / size)
    return left, top, max(right, left + 1), max(bottom, top + 1)
