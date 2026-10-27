"""NAIP aerial imagery over a staged box, and the bare-corridor test.

WHY IMAGERY
-----------
Every other line source here is either an agency's record or the
elevation. Imagery is a third, independent witness: a dirt road or a
landing is bare soil, and in the near-infrared band bare soil reads low
NDVI against the vegetation either side. Measured on real ground (Mendocino
County oak woodland, 900 m round a test point, 2026-10-21): of 10
DEM lines in a 440 x 260 m window, 8 ran along a corridor at least 0.10
NDVI barer than the ground 8 m to either side (typically about 0.0 along
the line against 0.3 beside it); of 200 random 60 m lines in the same
window, 19 did. So a line that passes is roughly ten times likelier to be
a cleared corridor than chance - not proof, but independent agreement.

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

FAILURE IS NOT LOOKED FOR
-------------------------
A tile that cannot be fetched leaves its ground NaN. A line over NaN
ground gets `bare: None` - not looked at - never False, the same rule the
agency sources follow.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..net_errors import safe_reason

NAIP_URLS = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer",
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer",
)
NAIP_URLS_ENV = "ZA_NAIP_URLS"

USER_AGENT = "sar-geo-mcp (search and rescue planning)"

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

#: The bare-corridor test: NDVI along a line must sit at least this far
#: below the median of the ground FLANK_M to either side.
BARE_CONTRAST = 0.10
FLANK_M = 8.0
SAMPLES = 30

#: Share of a line's samples that must land on fetched imagery for the
#: test to say anything; below it the answer is None.
MIN_COVERAGE = 0.6

IMAGERY_ASSUMPTIONS = {
    "source": "USGS National Map NAIP (USGSNAIPImagery, then USGSNAIPPlus)",
    "test": (f"median NDVI along the line at least {BARE_CONTRAST} below the median "
             f"{FLANK_M:.0f} m either side, over {SAMPLES} samples"),
    "measured": ("Mendocino oak woodland, 2026-10-21: 8 of 10 DEM lines passed, 19 "
                 "of 200 random 60 m lines did"),
    "limits": ("a road under closed canopy can fail it, and so can one in "
               "late-summer grassland where everything reads bare; a firebreak, a "
               "fence clearing or a dry creek bed can pass it. Agreement, not proof"),
    "none_means": "not looked at - no imagery fetched there",
}


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


class NdviSampler:
    """NDVI at points in EPSG:3857, from cached GeoTIFF tiles. NaN where no
    tile was fetched. Bands are read as red, green, blue, near-infrared."""

    def __init__(self, paths):
        import rasterio

        self._tiles = []
        for path in paths:
            with rasterio.open(path) as source:
                red = source.read(1).astype("float32")
                nir = source.read(4).astype("float32")
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndvi = (nir - red) / (nir + red)
                ndvi[(nir + red) == 0] = np.nan
                self._tiles.append((source.bounds, ~source.transform, ndvi))

    @classmethod
    def from_bytes(cls, blobs):
        """For tests: tiles straight from GeoTIFF bytes."""
        import tempfile

        folder = Path(tempfile.mkdtemp())
        paths = []
        for index, blob in enumerate(blobs):
            path = folder / f"t{index}.tif"
            path.write_bytes(blob)
            paths.append(path)
        return cls(paths)

    def at(self, x: float, y: float) -> float:
        for bounds, inverse, ndvi in self._tiles:
            if bounds.left <= x < bounds.right and bounds.bottom < y <= bounds.top:
                col, row = inverse * (x, y)
                row, col = int(row), int(col)
                if 0 <= row < ndvi.shape[0] and 0 <= col < ndvi.shape[1]:
                    return float(ndvi[row, col])
        return float("nan")


def bare_corridor(line_3857, sampler: NdviSampler, latitude_deg: float) -> dict:
    """Is this line barer than the ground either side of it?

    Returns `bare` (True, False, or None when under MIN_COVERAGE of the
    samples landed on imagery), `along_ndvi`, `flank_ndvi` and `coverage`.
    """
    scale = math.cos(math.radians(latitude_deg))
    offset = FLANK_M / scale
    along, flanks, valid = [], [], 0
    length = line_3857.length
    for t in np.linspace(0.05, 0.95, SAMPLES):
        here = line_3857.interpolate(t, normalized=True)
        ahead = line_3857.interpolate(min(t + 0.02, 1.0), normalized=True)
        behind = line_3857.interpolate(max(t - 0.02, 0.0), normalized=True)
        dx, dy = ahead.x - behind.x, ahead.y - behind.y
        norm = math.hypot(dx, dy) or (length or 1.0)
        nx, ny = -dy / norm, dx / norm
        value = sampler.at(here.x, here.y)
        left = sampler.at(here.x + nx * offset, here.y + ny * offset)
        right = sampler.at(here.x - nx * offset, here.y - ny * offset)
        if math.isfinite(value) and (math.isfinite(left) or math.isfinite(right)):
            valid += 1
            along.append(value)
            flanks.extend(v for v in (left, right) if math.isfinite(v))
    coverage = valid / SAMPLES
    if coverage < MIN_COVERAGE:
        return {"bare": None, "along_ndvi": None, "flank_ndvi": None,
                "coverage": round(coverage, 2)}
    along_median = float(np.median(along))
    flank_median = float(np.median(flanks))
    return {"bare": bool(flank_median - along_median >= BARE_CONTRAST),
            "along_ndvi": round(along_median, 3), "flank_ndvi": round(flank_median, 3),
            "coverage": round(coverage, 2)}
