"""LANDFIRE LF2024 Existing Vegetation Type: the source of what KIND of
vegetation covers the ground.

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
import hashlib
import math
from collections.abc import Callable
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlencode

from ..geo import Bbox
from .base import Area, RasterResult, RasterSource, SourceInfo
from .dem import ensure_tiff
from ..cache import atomic_write_bytes, note_hit, note_write
from ..cache.layout import source_folder

EVT_EXPORT_URL = (
    "https://edcintl.cr.usgs.gov/arcgis/rest/services/"
    "Landfire_LF2024/LF2024_EVT_CONUS/ImageServer/exportImage"
)
EVT_PIXEL_M = 30.0  # LANDFIRE native resolution

EVT_TILE_PX = 256
EVT_TILE_M = EVT_PIXEL_M * EVT_TILE_PX  # 7 680 m

#: This source's folder under the cache, and where it wrote before 0.43.0.
FOLDER = "landfire_evt"
LEGACY_CACHE = ("evt_tile_*.tif",)


@lru_cache(maxsize=1)
def load_lifeform_table() -> dict[int, tuple[str, str, str]]:
    """VALUE -> (lifeform, physiognomy, name) from the vendored LANDFIRE CSV.

    The CSV is a vendored constant, so it is parsed once per process: a hasty
    set calls `composition` once per corridor candidate and would otherwise
    re-read several thousand rows for each. The returned dict is shared;
    callers read it and never mutate it.
    """
    table: dict[int, tuple[str, str, str]] = {}
    data = resources.files("sar_geo_mcp.sources").joinpath("data/lf2024_evt.csv")
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

    destination = source_folder(cache_dir, FOLDER, LEGACY_CACHE) / f"evt_tile_c{col}_r{row}.tif"
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


def lifeform_of(code: int) -> str:
    """The LANDFIRE lifeform of a code; "Unknown" for -1 or a code not in
    the vendored table."""
    if code < 0:
        return "Unknown"
    return load_lifeform_table().get(int(code), ("Unknown",))[0]


class LandfireEvtSource(RasterSource):
    """LANDFIRE vegetation type as a source: an int16 GeoTIFF of LANDFIRE
    codes over an area, snapped outward to the 30 m grid. The codes are the
    service's own; `load_lifeform_table` decodes them."""

    info = SourceInfo(
        name="landfire_evt",
        title="LANDFIRE LF2024 Existing Vegetation Type",
        kind="raster",
        provider="LANDFIRE (USDA Forest Service, U.S. Department of the Interior)",
        default_url=EVT_EXPORT_URL,
        attribution="LANDFIRE LF2024 EVT",
        coverage="Lower 48 states (CONUS)",
        native_resolution_m=EVT_PIXEL_M,
        notes=(
            "Vegetation type, not density: nothing here measures understory.",
            "Categorical codes, never resampled by blending.",
        ),
    )

    def fetch(self, area: Area, width_px: int | None = None,
              height_px: int | None = None, *, out_path: Path | str | None = None,
              **options) -> RasterResult:
        """The codes over `area` at the native 30 m. LANDFIRE is categorical,
        so there is no other pixel size: `width_px` and `height_px` are
        ignored, and the result's are the clip's own."""
        if options:
            raise ValueError(f"landfire_evt takes no options; got {sorted(options)}")
        import rasterio

        cache_dir = self.require_cache()
        snapped = snap_to_pixel_grid(area.bbox_3857)
        if out_path is None:
            key = ",".join(f"{value:.0f}" for value in snapped)
            digest = hashlib.sha256(key.encode()).hexdigest()[:32]
            out_path = source_folder(cache_dir, FOLDER, LEGACY_CACHE) / f"evt_clip_{digest}.tif"
        get_bytes = self.transport.get_bytes if self.transport else _http_get
        path = write_evt_clip(area.bbox_3857, Path(out_path), cache_dir, get_bytes=get_bytes)
        with rasterio.open(path) as dataset:
            width, height = dataset.width, dataset.height
        return RasterResult(path=path, bbox_3857=snapped, width_px=width,
                            height_px=height, attribution=self.info.attribution)
