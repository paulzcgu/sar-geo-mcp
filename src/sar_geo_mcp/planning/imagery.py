"""The bare-corridor test, over NAIP imagery staged by `sources.naip`.

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

The imagery itself - which service, its tiles and their cache - is
`sources.naip`.

FAILURE IS NOT LOOKED FOR
-------------------------
A tile that cannot be fetched leaves its ground NaN. A line over NaN
ground gets `bare: None` - not looked at - never False, the same rule the
agency sources follow.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

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
