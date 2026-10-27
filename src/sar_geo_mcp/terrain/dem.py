"""USGS 3DEP elevation access.

The rendered CalTopo layers cannot answer 'how deep' or 'how steep' in real
units. This module supplies the float32 elevation grid that all measurement
depends on.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from ..geo import Bbox
from ..cache import atomic_output, atomic_write_bytes, note_hit, note_write
from ..tiles import (
    TILE_PX,
    Tile,
    level_for,
    mosaic_bbox,
    pixel_window,
    tiles_covering,
)

EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
MAX_DEM_PIXELS = 8000

TIFF_MAGIC_LE = b"II\x2a\x00"
TIFF_MAGIC_BE = b"MM\x00\x2a"


def build_export_url(bbox_3857: Bbox, width_px: int, height_px: int) -> str:
    if width_px > MAX_DEM_PIXELS or height_px > MAX_DEM_PIXELS:
        raise ValueError(
            f"3DEP exportImage caps at {MAX_DEM_PIXELS}px per axis; "
            f"got {width_px}x{height_px}"
        )

    params = {
        "bbox": ",".join(f"{value:.3f}" for value in bbox_3857),
        "bboxSR": "3857",
        "size": f"{width_px},{height_px}",
        "imageSR": "3857",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    return f"{EXPORT_URL}?{urlencode(params)}"


def ensure_tiff(content: bytes, url: str) -> None:
    """The service returns a JSON error body with HTTP 200 on failure."""
    if not (content.startswith(TIFF_MAGIC_LE) or content.startswith(TIFF_MAGIC_BE)):
        preview = content[:160].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"3DEP did not return a GeoTIFF for {url}\nFirst bytes: {preview!r}"
        )


def _http_get(url: str) -> bytes:
    """One tile body, with the shared retry policy in front of it.

    A staged area is up to sixteen of these, so a single transient 5xx used to
    discard the whole `fetch_dem` call - and 3DEP does serve them. See
    `sar_geo_mcp.net` for what is and is not retried.
    """
    from ..net import bytes_with_retry

    return bytes_with_retry(url)


def fetch_dem_tile(tile: Tile, cache_dir: Path,
                   get_bytes: Callable[[str], bytes] = _http_get) -> Path:
    """One cached elevation tile, shared by every request that touches it."""
    cache_dir = Path(cache_dir)
    destination = cache_dir / tile.key("dem", "tif")
    if destination.exists():
        note_hit(cache_dir, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = build_export_url(tile.bbox, TILE_PX, TILE_PX)
    content = get_bytes(url)
    ensure_tiff(content, url)
    # Write via a temp name so an interrupted download cannot poison the
    # cache with a truncated tile that later looks like a hit - and a name
    # unique to this writer, because two sessions staging overlapping
    # ground fetch this tile at the same instant. See `cache.atomic`.
    atomic_write_bytes(destination, content)
    note_write(cache_dir, destination, layer="dem", source_version="3dep")
    return destination


def fetch_dem(
    bbox_3857: Bbox,
    width_px: int,
    height_px: int,
    cache_dir: Path,
    get_bytes: Callable[[str], bytes] = _http_get,
) -> Path:
    """A float32 elevation GeoTIFF for exactly this bbox and pixel size.

    Assembled from tiles on the shared grid rather than requested whole, so
    two assignments over the same ground share their elevation data instead
    of downloading the same mountain twice. The returned file is still
    exactly what the caller asked for - the tiling is an implementation
    detail of how it was obtained.

    The tile level is chosen no coarser than the requested resolution, so
    the mosaic is downsampled to the requested grid and never upsampled.
    Inventing detail that was never fetched would be a lie about resolution,
    and everything downstream of here is a measurement.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    level = level_for(bbox_3857, width_px, height_px)
    tiles = tiles_covering(bbox_3857, level)

    # The assembled result is cached too. Mosaicking is cheap next to a
    # download, but the analysis step reads this file repeatedly and there
    # is no reason to redo the work on every read.
    canonical = f"{','.join(f'{v:.3f}' for v in bbox_3857)}|{width_px}|{height_px}|z{level}"
    key = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    destination = cache_dir / f"dem_{key}.tif"
    if destination.exists():
        note_hit(cache_dir, destination)
        return destination

    paths = [fetch_dem_tile(tile, cache_dir, get_bytes) for tile in tiles]

    outer = mosaic_bbox(tiles)
    columns = len({tile.col for tile in tiles})
    rows = len({tile.row for tile in tiles})
    canvas = np.full((rows * TILE_PX, columns * TILE_PX), np.nan, dtype=np.float32)

    min_col = min(tile.col for tile in tiles)
    max_row = max(tile.row for tile in tiles)
    nodata_seen = None
    for tile, path in zip(tiles, paths):
        with rasterio.open(path) as source:
            block = source.read(1)
            nodata_seen = source.nodata if source.nodata is not None else nodata_seen
        # Grid rows count down from the top, which is maximum northing.
        y_off = (max_row - tile.row) * TILE_PX
        x_off = (tile.col - min_col) * TILE_PX
        canvas[y_off:y_off + block.shape[0], x_off:x_off + block.shape[1]] = block

    if nodata_seen is not None:
        # 3DEP marks voids with a large negative sentinel. Carrying it into
        # a resample would smear the sentinel across real elevations, so it
        # becomes NaN before any interpolation happens.
        canvas[canvas == nodata_seen] = np.nan

    left, top, right, bottom = pixel_window(outer, bbox_3857, level)
    window = canvas[top:bottom, left:right]

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": height_px, "width": width_px, "crs": "EPSG:3857",
        "transform": from_bounds(*bbox_3857, width_px, height_px),
        "nodata": float("nan"),
    }
    # rasterio opens the file itself, so it is handed a unique temp PATH
    # rather than bytes; `atomic_output` renames it on a clean exit.
    with atomic_output(destination) as temporary:
        with rasterio.open(temporary, "w", **profile) as target:
            target.write(
                _resample(window, height_px, width_px, Resampling.bilinear), 1)
    note_write(cache_dir, destination, layer="dem-mosaic", source_version="3dep")
    return destination


def _resample(block: np.ndarray, height_px: int, width_px: int,
              resampling) -> np.ndarray:
    """Resize an elevation block to the requested grid.

    Bilinear, matching the interpolation 3DEP itself was being asked for
    before this module tiled - so the numbers a caller gets stay comparable
    with the ones it produced before.
    """
    if block.shape == (height_px, width_px):
        return block.astype(np.float32)

    from PIL import Image

    image = Image.fromarray(block.astype(np.float32), mode="F")
    resized = image.resize((width_px, height_px), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def read_dem(path: Path) -> tuple[np.ndarray, float]:
    """Read elevations and the raster's pixel size in WEB MERCATOR UNITS.

    The second value is NOT ground metres. EPSG:3857 units are inflated by
    1 / cos(latitude); convert with geo.ground_meters_per_mercator_unit().
    """
    import rasterio

    with rasterio.open(path) as dataset:
        elevations = dataset.read(1).astype("float32")
        resolution_mercator_units = abs(dataset.transform.a)

    return elevations, resolution_mercator_units


def read_dem_grid(path: Path) -> tuple[float, Bbox]:
    """Read pixel size (Web Mercator units) and bounds without loading the array.

    The 3DEP service may return a wider extent than requested, so the DEM's own
    bounds are authoritative and must not be assumed equal to the request bbox.
    """
    import rasterio

    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        return abs(dataset.transform.a), (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
