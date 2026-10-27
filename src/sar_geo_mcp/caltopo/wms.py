"""CalTopo WMS client.

Verified server behaviour this module encodes:
  - EPSG:3857 only, image/png only, WMS 1.1.1 (SRS, not CRS)
  - GetMap is capped at 1024x1024; larger returns HTTP 200 with an HTML error page
  - Comma-separated LAYERS composite server-side
"""

import hashlib
import io
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import httpx
from PIL import Image

from ..geo import Bbox
from ..cache import atomic_output, atomic_write_bytes, note_hit, note_write
from ..cache.layout import source_folder
from ..tiles import (
    TILE_PX,
    Tile,
    level_for,
    mosaic_bbox,
    pixel_window,
    tiles_covering,
)

MAX_PIXELS = 1024

#: This source's folder under the cache (`sources.caltopo_wms`), and where it
#: wrote before 0.43.0: its grid tiles, and its panels named
#: "<layers>_<32 hex>.png" at the root.
FOLDER = "caltopo_wms"
LEGACY_CACHE = ("tiles/wms-*", "*_" + "[0-9a-f]" * 32 + ".png")

LAYERS: dict[str, str] = {
    "mbt": "MapBuilder Topo",
    "mbh": "MapBuilder Hybrid",
    "t": "Scanned Topos",
    "imagery": "Imagery",
    "n3": "NAIP 2019 to 2021",
    "ir": "NAIP False Color IR",
    "r": "Relief Normal",
    "r2": "Relief Enhanced",
    "r3": "Relief Terrain",
    "sf": "Slope Angle Shading",
    "sg": "Gradient Slope Shading",
    "c": "Contours",
    "cf10": "10 Foot Contours",
    "cm5": "5 Meter Contours",
    "mba": "MapBuilder Overlay",
    "mbr": "Roads",
    "mvum": "Motor Vehicle MVUM",
    "om": "OpenStreetMap",
    "structures": "Structures",
    "parcel": "Parcels",
    "sma": "Public Lands",
    "geology": "Geology",
}


#: Layers whose pixel COLOUR is a CATEGORY rather than a shade, and which
#: therefore must never be resampled by an interpolating filter.
#:
#: WHY THIS IS A PROPERTY OF THE LAYER AND NOT OF THE CALL
#: -------------------------------------------------------
#: "Slope Angle Shading paints each band one flat colour" is a fact about
#: what CalTopo renders. It does not change between call sites, and a call
#: site that got it wrong would be wrong silently - the panel comes back the
#: right size, the right extent and the right shape, just with a fringe of
#: invented colours along every band boundary. So the default is derived from
#: the layers themselves (`resample_filter_for`) and the per-call `resample`
#: parameter exists only to override a deliberate exception.
#:
#: MEASURED, on a synthetic `sf` render with large smooth band patches and a
#: PERFECTLY CORRECT legend. A staged 2.0 Mercator-unit pixel sends
#: `level_for` to level 17 (1.1943 u/px), a 1.675x LANCZOS downsample:
#:
#:   band-boundary density   LANCZOS unmatched   NEAREST unmatched
#:   sparse (6x6 field)                  0.033               0.000
#:   moderate (12x12)                    0.074               0.000
#:   broken (16x16)                      0.094               0.000
#:   very broken (32x32)                 0.192               0.000
#:
#: Every blended pixel is exactly the colour the legend refuses, so it
#: decodes to UNKNOWN. `UNMATCHED_WARNING_FRACTION` is 0.10, which broken
#: ground crosses on the fetch alone - and the warning then blames the
#: legend, which is the wrong suspect. `tiles.py` notes that a request
#: "almost never lands exactly on a level", so this was the normal path.
#:
#: The second harm is erosion, not just noise. UNKNOWN pixels are excluded
#: from `comparable`, so the fringe eats the EDGE of every disagreement
#: patch. Measured on a 3-native-pixel-wide steep face: 600 native band
#: pixels came back as 361 surviving plus a 250-pixel UNKNOWN halo under
#: LANCZOS, against 119 surviving and no halo at all under NEAREST. With
#: `MIN_PATCH_PIXELS` at 4, a genuinely narrow vertical face can erode below
#: the floor - the narrow-face blindness this whole design was built to
#: avoid, re-entering through the fetch path.
#:
#: ADDING A LAYER HERE. The test is whether anything reads the rendered
#: colour back as a VALUE. `sg` (Gradient Slope Shading) is a continuous
#: ramp by design and nothing decodes it; contours are line art, where an
#: interpolating filter is the correct antialiasing. `sf` is the one layer
#: this package decodes against a legend, so it is the one layer listed.
CATEGORICAL_LAYERS = frozenset({"sf"})

#: Not a Pillow filter: a resample that gives every output pixel the
#: MAXIMUM alpha over the source pixels its footprint touches, and colour by
#: a box average. For line art read back as a MASK through an alpha floor,
#: where an averaging filter is the wrong answer.
#:
#: MEASURED, on CalTopo's `structures` layer, which `planning.lz` reads as
#: "alpha at least 32 is a building": footprints are drawn at alpha 64 to
#: 127, and a building can be a 1 px outline. The weakest of a set of 1 px
#: outlines at alpha 64, at every phase against the output grid, after the
#: 1.676x downsample a staged 2 m cell gets (tests/test_wms_fetch.py):
#:
#:   LANCZOS 24   BILINEAR 19   BOX 32   MAX_ALPHA 64
#:
#: LANCZOS - what this layer got before - is under the floor, so the
#: building is not there and the ground beside it is offered as a
#: touchdown. BOX lands exactly ON the floor at this ratio, with no margin
#: at all, and falls under it at any coarser one (21 at 2.5x, 16 at 3.3x).
#: The maximum keeps 64 at every ratio and phase. It can widen a line by
#: one output pixel, never narrow it: for a closed-ground mask that is the
#: cautious side.
MAX_ALPHA = -1

#: Layers whose ALPHA is read back as a mask with a floor, and so must be
#: resampled with `MAX_ALPHA`. `structures` and `parcel`: a building can be a
#: 1 px outline, and every lot line is one, and `planning.populated` reads the
#: lot lines as walls - a line faded under the floor is a gap two lots leak
#: through into one. Contours are line art too, but only ever looked at,
#: where antialiasing is right.
MASK_LAYERS = frozenset({"structures", "parcel"})


def resample_filter_for(layers: list[str]) -> int:
    """The Pillow filter a panel of these layers must be resized with, or
    `MAX_ALPHA` for a layer read back as a mask.

    Any categorical layer in a composite decides it for the whole request:
    a composite is one rendered image, and there is no way to interpolate
    half of it. A mask layer decides it next, for the same reason.
    """
    if any(layer in CATEGORICAL_LAYERS for layer in layers):
        return Image.NEAREST
    if any(layer in MASK_LAYERS for layer in layers):
        return MAX_ALPHA
    return Image.LANCZOS


def max_pool(values, width: int, height: int):
    """`values` (a 2-D numpy array) resized to `height` rows by `width`
    columns, each output cell the maximum over every source cell its
    footprint touches - so nothing present in the source falls between
    output cells, at any ratio or phase. Works for upsampling too, where
    each output cell touches one or two source cells."""
    import numpy as np

    def along_rows(array, size):
        count = array.shape[0]
        edges = np.arange(size + 1) * count / size
        starts = np.floor(edges[:-1]).astype(int)
        ends = np.minimum(np.maximum(np.ceil(edges[1:] - 1e-9).astype(int), starts + 1),
                          count)
        return np.stack([array[start:end].max(axis=0)
                         for start, end in zip(starts, ends)])

    return along_rows(along_rows(np.asarray(values), height).T, width).T


def _resize(image: Image.Image, size: tuple[int, int], resample: int) -> Image.Image:
    """`image` resized to `size` with `resample`, a Pillow filter or
    `MAX_ALPHA`."""
    if resample != MAX_ALPHA:
        return image.resize(size, resample)
    import numpy as np

    rgba = image.convert("RGBA")
    colour = rgba.resize(size, Image.BOX)
    alpha = max_pool(np.asarray(rgba.getchannel("A")), size[0], size[1])
    colour.putalpha(Image.fromarray(alpha.astype("uint8"), "L"))
    return colour


def _filter_name(resample: int) -> str:
    """A Pillow filter as the name that goes into the cache key."""
    if resample == MAX_ALPHA:
        return "MAX_ALPHA"
    try:
        return Image.Resampling(resample).name
    except ValueError:
        return str(resample)


def validate_layers(layers: list[str]) -> None:
    unknown = [layer for layer in layers if layer not in LAYERS]
    if unknown:
        raise ValueError(
            f"unknown CalTopo layer(s): {unknown}. Known layers: {sorted(LAYERS)}"
        )


def build_getmap_url(
    base_url: str,
    layers: list[str],
    bbox_3857: Bbox,
    width_px: int,
    height_px: int,
    transparent: bool = True,
) -> str:
    """Build one GetMap URL. Callers above 1024 px must tile first."""
    if not layers:
        raise ValueError("at least one layer is required")
    validate_layers(layers)

    if width_px > MAX_PIXELS or height_px > MAX_PIXELS:
        raise ValueError(
            f"CalTopo WMS caps GetMap at {MAX_PIXELS}px per axis; "
            f"got {width_px}x{height_px}. Use geo.plan_tiles() and mosaic."
        )

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": ",".join(layers),
        "SRS": "EPSG:3857",
        "BBOX": ",".join(f"{value:.6f}" for value in bbox_3857),
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
        "FORMAT": "image/png",
        "TRANSPARENT": "true" if transparent else "false",
    }
    return f"{base_url}?{urlencode(params)}"


def cache_key(
    layers: list[str],
    bbox_3857: Bbox,
    width_px: int,
    height_px: int,
    resample: int = Image.LANCZOS,
) -> str:
    """Content hash for a panel request. Deliberately excludes the token.

    `resample` is part of the hash because it is part of the CONTENT: the
    same layers over the same ground at the same pixel size come back as
    genuinely different images under NEAREST and under LANCZOS, and a cache
    keyed without it would serve a blended `sf` panel to a caller that asked
    for an unblended one, forever. Only the assembled panel is rekeyed -
    the underlying grid tiles are fetched at their native size and never
    resized, so nothing has to be re-downloaded.
    """
    canonical = "|".join(
        [
            ",".join(layers),
            ",".join(f"{value:.3f}" for value in bbox_3857),
            str(width_px),
            str(height_px),
            _filter_name(resample),
        ]
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def ensure_png(content: bytes, url: str) -> None:
    """CalTopo serves HTML error pages with HTTP 200, so sniff the content."""
    if not content.startswith(PNG_MAGIC):
        preview = content[:120].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"CalTopo did not return a PNG for {url}\nFirst bytes: {preview!r}"
        )


def _http_get(url: str) -> bytes:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def wms_folder(cache_dir: Path | str) -> Path:
    """This source's folder under `cache_dir`, older files moved in."""
    return source_folder(cache_dir, FOLDER, LEGACY_CACHE)


def panel_tile_path(cache_dir: Path | str, layers: list[str], tile: Tile) -> Path:
    return wms_folder(cache_dir) / tile.key(f"wms-{'-'.join(layers)}", "png")


def fetch_panel(
    base_url: str,
    layers: list[str],
    bbox_3857: Bbox,
    width_px: int,
    height_px: int,
    cache_dir: Path,
    get_bytes: Callable[[str], bytes] = _http_get,
    resample: int | None = None,
) -> Path:
    """Fetch one panel, assembled from tiles on the shared grid.

    Returns the path to a cached PNG at exactly the requested bbox and
    pixel size. Underneath, the ground is fetched as fixed grid tiles that
    any other panel over the same ground reuses - before this, two
    assignments 100 m apart shared nothing, and a real cache held 102
    panels and 115 MB for one training map.

    The assembled panel is cached too, keyed by a content hash that
    excludes the token, so credentials never reach the disk.

    RESAMPLING

    A request almost never lands exactly on a pyramid level (`tiles.py` says
    so in its own docstring), so the crop out of the tile mosaic is usually
    NOT the requested pixel size and has to be resized. That resize is the
    normal path, not an edge case: a staged 2.0 Mercator-unit pixel takes
    level 17 at 1.1943 u/px and is downsampled 1.675x.

    `resample` chooses the filter. None - the default - lets the LAYER
    decide, via `resample_filter_for`: nearest-neighbour for a categorical
    layer whose colours are decoded back to values (`CATEGORICAL_LAYERS`,
    where the measured cost of interpolating is recorded), `MAX_ALPHA` for
    a layer whose alpha is read back as a mask (`MASK_LAYERS`), LANCZOS for
    everything else, which is what a human-viewed basemap wants and what
    every non-categorical caller got before this parameter existed. Pass a
    Pillow filter explicitly only to override that.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if resample is None:
        resample = resample_filter_for(layers)
    level = level_for(bbox_3857, width_px, height_px)
    key = cache_key(layers, bbox_3857, width_px, height_px, resample)
    destination = wms_folder(cache_dir) / f"{'-'.join(layers)}_{key}.png"
    if destination.exists():
        note_hit(cache_dir, destination)
        return destination

    grid = tiles_covering(bbox_3857, level)
    columns = len({tile.col for tile in grid})
    rows = len({tile.row for tile in grid})
    min_col = min(tile.col for tile in grid)
    max_row = max(tile.row for tile in grid)

    canvas = Image.new("RGBA", (columns * TILE_PX, rows * TILE_PX))
    for tile in grid:
        path = fetch_panel_tile(base_url, layers, tile, cache_dir, get_bytes)
        with Image.open(path) as image:
            canvas.paste(image.convert("RGBA"),
                         ((tile.col - min_col) * TILE_PX,
                          (max_row - tile.row) * TILE_PX))

    left, top, right, bottom = pixel_window(mosaic_bbox(grid), bbox_3857, level)
    window = canvas.crop((left, top, right, bottom))
    if window.size != (width_px, height_px):
        window = _resize(window, (width_px, height_px), resample)

    # Pillow opens the file itself, so it is handed a unique temp PATH.
    with atomic_output(destination) as temporary:
        window.save(temporary, format="PNG")
    note_write(cache_dir, destination, layer=f"wms-panel:{'-'.join(layers)}")
    return destination


def fetch_panel_tile(base_url: str, layers: list[str], tile: Tile,
                     cache_dir: Path,
                     get_bytes: Callable[[str], bytes] = _http_get) -> Path:
    """One cached basemap tile, shared by every panel that touches it.

    A tile is exactly `TILE_PX` square, which is exactly the GetMap
    ceiling, so it is always a single server render. That matters for
    labelled layers: a tile stitched from sub-requests would place labels
    independently in each piece and show duplicates at the joins.

    Seams can still fall on tile boundaries, as they already did whenever a
    panel exceeded the ceiling and had to be mosaicked. What changes is
    that they now fall on fixed lines rather than wherever a request
    happened to be split - and the pieces are reusable.

    Takes no `resample`, and needs none: a tile is requested from the server
    at exactly `TILE_PX` and written to disk as the bytes came back. Nothing
    here resizes anything, so the categorical-layer question
    (`CATEGORICAL_LAYERS`) does not arise at this level - which is why
    `prefetch.py`, the only caller of this function, is untouched by it.
    """
    cache_dir = Path(cache_dir)
    destination = panel_tile_path(cache_dir, layers, tile)
    if destination.exists():
        note_hit(cache_dir, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = build_getmap_url(base_url, layers, tile.bbox, TILE_PX, TILE_PX)
    content = get_bytes(url)
    ensure_png(content, url)
    atomic_write_bytes(destination, content)
    note_write(cache_dir, destination, layer=f"wms:{'-'.join(layers)}")
    return destination
