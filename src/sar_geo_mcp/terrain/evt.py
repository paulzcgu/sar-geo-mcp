"""LANDFIRE Existing Vegetation Type: what KIND of vegetation covers the zone.

The CIR proxy says how DENSE the vegetation is; EVT says whether it is timber,
chaparral, or grass — which changes effective sweep width more than density
does. 30 m pixels are coarse against the DEM, but sweep width is a class-level
number, not a per-pixel one.

Cache design: the key is a fixed ground TILE anchored at the Web Mercator
origin, not the assignment bbox. One CalTopo map carries ~119 assignments in a
few square kilometres, so a per-bbox key would refetch the same raster ~119
times. Each assignment still gets its own clipped evt.tif.

Verified live 2026-08-14: S16 codes, JSON errors served with HTTP 200 (same
sniffing defence as 3DEP), and an empty server-side rasterAttributeTable —
hence the vendored code table.
"""

import csv
import math
from collections.abc import Callable
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from ..geo import Bbox
from .dem import ensure_tiff
from ..cache import atomic_write_bytes, note_hit, note_write

EVT_EXPORT_URL = (
    "https://edcintl.cr.usgs.gov/arcgis/rest/services/"
    "Landfire_LF2024/LF2024_EVT_CONUS/ImageServer/exportImage"
)
EVT_PIXEL_M = 30.0  # LANDFIRE native resolution

EVT_TILE_PX = 256
EVT_TILE_M = EVT_PIXEL_M * EVT_TILE_PX  # 7 680 m


@lru_cache(maxsize=1)
def load_lifeform_table() -> dict[int, tuple[str, str, str]]:
    """VALUE -> (lifeform, physiognomy, name) from the vendored LANDFIRE CSV.

    The CSV is a vendored constant, so it is parsed once per process: a hasty
    set calls `composition` once per corridor candidate and would otherwise
    re-read several thousand rows for each. The returned dict is shared;
    callers read it and never mutate it.
    """
    table: dict[int, tuple[str, str, str]] = {}
    data = resources.files("sar_geo_mcp.terrain").joinpath("data/lf2024_evt.csv")
    with data.open("r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            table[int(row["VALUE"])] = (row["EVT_LF"], row["EVT_PHYS"], row["EVT_NAME"])
    return table


def tile_bounds(col: int, row: int) -> Bbox:
    """Web Mercator bounds of one cache tile."""
    return (
        col * EVT_TILE_M,
        row * EVT_TILE_M,
        (col + 1) * EVT_TILE_M,
        (row + 1) * EVT_TILE_M,
    )


def tiles_covering(bbox_3857: Bbox) -> list[tuple[int, int]]:
    """Every tile the bbox touches, as (col, row) on the fixed grid."""
    min_col = math.floor(bbox_3857[0] / EVT_TILE_M)
    max_col = math.ceil(bbox_3857[2] / EVT_TILE_M) - 1
    min_row = math.floor(bbox_3857[1] / EVT_TILE_M)
    max_row = math.ceil(bbox_3857[3] / EVT_TILE_M) - 1
    return [
        (col, row)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    ]


def snap_to_pixel_grid(bbox_3857: Bbox) -> Bbox:
    """Expand a bbox outward to the shared 30 m pixel grid.

    rasterio.merge rounds each source tile's destination window against the
    requested bounds; unaligned bounds drop a row or column at the seam
    between adjacent tiles, leaving fill pixels. Tiles are anchored at the
    Web Mercator origin, so snapping to EVT_PIXEL_M multiples aligns exactly
    with every tile's internal grid and no rounding occurs.
    """
    return (
        math.floor(bbox_3857[0] / EVT_PIXEL_M) * EVT_PIXEL_M,
        math.floor(bbox_3857[1] / EVT_PIXEL_M) * EVT_PIXEL_M,
        math.ceil(bbox_3857[2] / EVT_PIXEL_M) * EVT_PIXEL_M,
        math.ceil(bbox_3857[3] / EVT_PIXEL_M) * EVT_PIXEL_M,
    )


def build_tile_url(col: int, row: int) -> str:
    params = {
        "bbox": ",".join(f"{value:.3f}" for value in tile_bounds(col, row)),
        "bboxSR": "3857",
        "size": f"{EVT_TILE_PX},{EVT_TILE_PX}",
        "imageSR": "3857",
        "format": "tiff",
        "pixelType": "S16",
        "interpolation": "RSP_NearestNeighbor",  # categorical: never blend codes
        "f": "image",
    }
    return f"{EVT_EXPORT_URL}?{urlencode(params)}"


def _http_get(url: str) -> bytes:
    """One EVT tile body, behind the same retry policy as elevation.

    LANDFIRE is the optional layer - `stage_terrain` downgrades its absence to
    a warning - but a transient 5xx that costs a staging its vegetation is
    still worth one more ask. See `sar_geo_mcp.net`.
    """
    from ..net import bytes_with_retry

    return bytes_with_retry(url)


def fetch_tile(
    col: int, row: int, cache_dir: Path, get_bytes: Callable[[str], bytes] = _http_get
) -> Path:
    """One cached EVT tile. Shared by every assignment that touches it."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    destination = cache_dir / f"evt_tile_c{col}_r{row}.tif"
    if destination.exists():
        note_hit(cache_dir, destination)
        return destination

    url = build_tile_url(col, row)
    content = get_bytes(url)
    ensure_tiff(content, url)
    # Write via a temp name so an interrupted download cannot poison the
    # cache, and one unique to this writer, because two sessions staging
    # overlapping ground share this tile. See `cache.atomic`.
    atomic_write_bytes(destination, content)
    note_write(cache_dir, destination, layer="evt")
    return destination


def write_evt_clip(
    bbox_3857: Bbox,
    out_path: Path,
    cache_dir: Path,
    get_bytes: Callable[[str], bytes] = _http_get,
) -> Path:
    """Assemble the assignment's evt.tif from cached tiles.

    The output directory stays self-contained (the terrain step never reaches
    the network) while the tile cache is shared across every assignment.
    """
    import rasterio
    from rasterio.merge import merge

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapped = snap_to_pixel_grid(bbox_3857)

    tile_paths = [
        fetch_tile(col, row, cache_dir, get_bytes) for col, row in tiles_covering(snapped)
    ]

    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets, bounds=snapped)
        profile = datasets[0].profile
    finally:
        for dataset in datasets:
            dataset.close()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        dtype="int16",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0].astype("int16"), 1)

    return out_path


@lru_cache(maxsize=2)
def _read_evt_keyed(path_str: str, mtime_ns: int, size: int):
    """The decoded raster for one (path, mtime, size). See `_read_evt`."""
    import rasterio

    with rasterio.open(path_str) as dataset:
        codes = dataset.read(1)
        transform = dataset.transform
    codes.setflags(write=False)   # the array is shared between callers
    return codes, transform


def _read_evt(path: Path):
    """`(codes, transform)` for a staged evt.tif, read once per file version.

    A hasty set asks for the composition of every corridor candidate against
    the SAME staged raster, and decoding it per candidate is the whole cost of
    the second one onward. The key carries the file's mtime and size, so a
    re-staged directory is never served the raster it replaced. Two entries is
    enough for the one directory a call works in, plus the one before it.
    """
    stat = Path(path).stat()
    return _read_evt_keyed(str(path), stat.st_mtime_ns, stat.st_size)


_read_evt.cache_clear = _read_evt_keyed.cache_clear


def codes_at_cells(evt_path: Path, grid_transform, rows, cols) -> np.ndarray:
    """LANDFIRE codes at the centres of grid cells (rows, cols) of another
    raster - the elevation model's - or -1 for a cell off the EVT raster.

    The one place a DEM cell is matched to its 30 m EVT pixel. The two
    rasters differ in transform and resolution (the DEM can be 1 m), so each
    cell centre is projected into EVT row/col space rather than either raster
    being resampled: exact, and cheaper than warping. `site_facts` reads it
    for one footprint's cells, the walking ground for every cell.
    """
    codes, evt_transform = _read_evt(evt_path)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    xs = grid_transform.c + (cols + 0.5) * grid_transform.a
    ys = grid_transform.f + (rows + 0.5) * grid_transform.e
    evt_cols = np.floor((xs - evt_transform.c) / evt_transform.a).astype(np.int64)
    evt_rows = np.floor((ys - evt_transform.f) / evt_transform.e).astype(np.int64)
    inside = ((evt_rows >= 0) & (evt_rows < codes.shape[0])
              & (evt_cols >= 0) & (evt_cols < codes.shape[1]))
    out = np.full(rows.shape, -1, dtype=np.int64)
    out[inside] = codes[evt_rows[inside], evt_cols[inside]]
    return out


def lifeform_of(code: int) -> str:
    """The LANDFIRE lifeform of a code; "Unknown" for -1 or a code not in
    the vendored table."""
    if code < 0:
        return "Unknown"
    return load_lifeform_table().get(int(code), ("Unknown",))[0]


def composition(evt_path: Path, geometry_3857) -> dict:
    """Lifeform fractions and top vegetation types inside the assignment."""
    from rasterio.features import rasterize

    table = load_lifeform_table()

    codes, transform = _read_evt(evt_path)
    mask = rasterize(
        [(geometry_3857, 1)],
        out_shape=codes.shape,
        transform=transform,
        fill=0,
        all_touched=True,
    ).astype(bool)

    selected = codes[mask] if mask.any() else codes.ravel()

    lifeform_counts: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    for code, count in zip(*np.unique(selected, return_counts=True)):
        lifeform, _, name = table.get(int(code), ("Unknown", "Unknown", f"code {code}"))
        lifeform_counts[lifeform] = lifeform_counts.get(lifeform, 0) + int(count)
        name_counts[name] = name_counts.get(name, 0) + int(count)

    total = sum(lifeform_counts.values())
    unknown_fraction = lifeform_counts.get("Unknown", 0) / max(total, 1)
    fractions = {lf: count / total for lf, count in lifeform_counts.items()}
    top_types = sorted(
        ((name, count / total) for name, count in name_counts.items()),
        key=lambda item: -item[1],
    )[:5]

    return {
        "fractions": fractions,
        "dominant_lifeform": max(fractions, key=fractions.get),
        "top_types": top_types,
        "unknown_fraction": unknown_fraction,
        "source": "LANDFIRE LF2024 EVT, 30 m",
    }
