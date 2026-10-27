"""USGS NAIP aerial imagery: a source of red, green, blue and near-infrared
tiles over a staged box.

THE SERVICE
-----------
USGS National Map NAIP, two public ImageServers with the same four bands
(red, green, blue, near-infrared), no credential:
  - `USGSNAIPImagery` - tried first. Four 1,000 px tiles in parallel took
    11.4-25.2 s (2026-10-21).
  - `USGSNAIPPlus` - the fallback. The same tile took 26-55 s, and one
    3,800 px request timed out at 90 s with a 504.
A request costs about the same whatever its size (250 px and 1,000 px both
took about 27 s on NAIPPlus), so the box is fetched in few, large tiles,
in parallel, each cached in the staged directory on its own.

What the bands say about the ground - the bare-corridor test - is
`planning.imagery`'s.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..net_errors import safe_reason
from .base import USER_AGENT, Area, RasterResult, RasterSource, SourceInfo

NAIP_URLS = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer",
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer",
)
NAIP_URLS_ENV = "ZA_NAIP_URLS"


IMAGERY_DIRNAME = "imagery"
IMAGERY_VERSION = 1

#: Ground metres per imagery pixel. Dirt roads here are 4-6 m wide; 1 m
#: puts several pixels across one, where the DEM's 2 m puts two.
IMAGERY_PIXEL_M = 1.0

#: Pixels per tile side, and the most tiles a box is fetched in. Past
#: MAX_TILES the pixel is coarsened instead - a 16 km staging at 1 m would
#: be 256 requests.
TILE_PX = 1000
MAX_TILES = 36

WORKERS = 4
ATTEMPTS_PER_URL = 2
TIMEOUT_S = 120.0


def naip_urls() -> tuple[str, ...]:
    configured = os.environ.get(NAIP_URLS_ENV, "").strip()
    if configured:
        return tuple(url.strip().rstrip("/") for url in configured.split(",") if url.strip())
    return NAIP_URLS


def _http_get(url: str, params: dict, timeout: float):
    import httpx

    return httpx.get(url, params=params, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": USER_AGENT})


def tile_grid(bbox_3857, latitude_deg: float) -> tuple[list[tuple], float]:
    """[(row, col, (w, s, e, n)), ...] covering the box, and the ground metres
    per pixel used - IMAGERY_PIXEL_M, coarsened only to stay under MAX_TILES."""
    west, south, east, north = bbox_3857
    scale = math.cos(math.radians(latitude_deg))            # ground m per unit
    side_m = max(east - west, north - south) * scale
    pixel_m = max(IMAGERY_PIXEL_M, side_m / (math.isqrt(MAX_TILES) * TILE_PX))
    tile_units = TILE_PX * pixel_m / scale
    tiles = []
    for row in range(max(1, math.ceil((north - south) / tile_units))):
        for col in range(max(1, math.ceil((east - west) / tile_units))):
            w = west + col * tile_units
            n = north - row * tile_units
            tiles.append((row, col, (w, n - tile_units, w + tile_units, n)))
    return tiles, pixel_m


def _fetch_tile(bounds, *, get) -> bytes:
    """One tile as GeoTIFF bytes, trying each service in turn."""
    w, s, e, n = bounds
    params = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 3857, "imageSR": 3857,
              "size": f"{TILE_PX},{TILE_PX}", "format": "tiff", "pixelType": "U8",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    last: Exception | None = None
    for url in naip_urls():
        for attempt in range(ATTEMPTS_PER_URL):
            try:
                response = get(f"{url}/exportImage", params, TIMEOUT_S)
                if response.status_code == 200 and response.content[:2] in (b"II", b"MM"):
                    return response.content
                last = RuntimeError(f"NAIP answered {response.status_code}")
            except Exception as error:  # noqa: BLE001 - try the next one
                last = error
            time.sleep(1.0 + attempt)
    raise last or RuntimeError("no NAIP service configured")


def load_naip(directory: Path, bbox_3857, latitude_deg: float, *, refresh: bool = False,
              get=None) -> dict:
    """Fetch (or reuse) the tiles over the box. Returns `status` ("ok",
    "partial" or "failed"), `tiles`, `missing`, `pixel_m`, `reason` and
    `paths` - the cached tile files, in tile order."""
    get = get or _http_get
    folder = Path(directory) / IMAGERY_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    tiles, pixel_m = tile_grid(bbox_3857, latitude_deg)
    stamp = {"version": IMAGERY_VERSION, "pixel_m": round(pixel_m, 4),
             "urls": list(naip_urls())}
    meta = folder / "naip.json"
    if refresh or not meta.exists() or json.loads(meta.read_text(encoding="utf-8")) != stamp:
        for old in folder.glob("naip_r*_c*.tif"):
            old.unlink()
        meta.write_text(json.dumps(stamp), encoding="utf-8")

    def path_of(row, col):
        return folder / f"naip_r{row}_c{col}.tif"

    needed = [(row, col, bounds) for row, col, bounds in tiles if not path_of(row, col).exists()]
    reasons: list[str] = []

    def fetch(job):
        row, col, bounds = job
        try:
            data = _fetch_tile(bounds, get=get)
        except Exception as error:  # noqa: BLE001 - a missing tile, reported
            return job, None, safe_reason(error) if not isinstance(error, RuntimeError) else str(error)
        return job, data, None

    from ..cache import atomic_write_bytes

    with ThreadPoolExecutor(WORKERS) as pool:
        for (row, col, _), data, reason in pool.map(fetch, needed):
            if data is None:
                reasons.append(reason)
            else:
                atomic_write_bytes(path_of(row, col), data)

    present = [(row, col) for row, col, _ in tiles if path_of(row, col).exists()]
    missing = [[row, col] for row, col, _ in tiles if not path_of(row, col).exists()]
    status = "ok" if not missing else ("failed" if not present else "partial")
    return {"status": status, "tiles": len(tiles), "missing": missing,
            "pixel_m": round(pixel_m, 3),
            "reason": "; ".join(sorted(set(reasons))) or None,
            "paths": [str(path_of(row, col)) for row, col in present]}


class _BytesResponse:
    """What `_fetch_tile` reads off a response, from a transport's bytes."""

    def __init__(self, content: bytes):
        self.status_code = 200
        self.content = content


class UsgsNaipSource(RasterSource):
    """USGS NAIP as a source: a four-band (red, green, blue, near-infrared)
    GeoTIFF of an area at about 1 m, built from the same cached tiles
    `unmarked_paths` reads. The pixel size is the service's tiling, so
    `width_px` and `height_px` are ignored; the result's are the mosaic's.
    A tile that cannot be fetched is a warning; none fetched is
    `SourceUnavailable`."""

    info = SourceInfo(
        name="usgs_naip",
        title="USGS NAIP imagery",
        kind="raster",
        provider="USDA National Agriculture Imagery Program, served by the USGS National Map",
        default_url=NAIP_URLS[0],
        attribution="USDA NAIP, via USGS National Map",
        coverage="Lower 48 states",
        url_env=NAIP_URLS_ENV,
        native_resolution_m=IMAGERY_PIXEL_M,
        notes=("Seasonal: as of the flight, and late-summer grass reads bare everywhere.",
               "Slow: a box takes 15-60 s on its first fetch."),
    )

    def endpoint(self) -> str:
        configured = [part.strip() for part in
                      (self.env.get(NAIP_URLS_ENV) or "").split(",") if part.strip()]
        return configured[0].rstrip("/") if configured else self.info.default_url

    def fetch(self, area: Area, width_px: int | None = None,
              height_px: int | None = None, **options) -> RasterResult:
        import hashlib

        import rasterio
        from rasterio.merge import merge

        from .base import SourceUnavailable

        if options:
            raise ValueError(f"usgs_naip takes no options; got {sorted(options)}")
        key = hashlib.sha256(",".join(f"{v:.1f}" for v in area.bbox_3857).encode()).hexdigest()[:32]
        directory = self.require_cache() / key
        get = None
        if self.transport is not None:
            transport = self.transport
            get = lambda url, params, timeout: _BytesResponse(transport.get_bytes(url))  # noqa: E731
        answer = load_naip(directory, area.bbox_3857, area.latitude, get=get)
        if answer["status"] == "failed":
            raise SourceUnavailable(self.info.name, answer.get("reason") or "no tile fetched")
        warnings = ([f"{len(answer['missing'])} of {answer['tiles']} tiles could not be "
                     f"fetched ({answer['reason']}); their ground is empty"]
                    if answer["missing"] else [])
        mosaic_path = directory / "naip_mosaic.tif"
        datasets = [rasterio.open(path) for path in answer["paths"]]
        try:
            mosaic, transform = merge(datasets)
            profile = datasets[0].profile
        finally:
            for dataset in datasets:
                dataset.close()
        profile.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                       transform=transform, count=mosaic.shape[0])
        with rasterio.open(mosaic_path, "w", **profile) as target:
            target.write(mosaic)
        return RasterResult(path=mosaic_path, bbox_3857=area.bbox_3857,
                            width_px=mosaic.shape[2], height_px=mosaic.shape[1],
                            attribution=self.info.attribution, warnings=warnings)
