"""Slope and aspect from the elevation grid. Pure math, no I/O.

Conventions (pinned by tests):
  - slope_deg: 0 = flat, 90 = vertical
  - aspect_deg: compass bearing of steepest DESCENT; 0 = north, 90 = east
  - image row 0 is north (max y), so the axis-0 gradient points south
"""

from dataclasses import dataclass

import numpy as np

# Spec section 5 bands; chosen to line up with CalTopo's slope shading reading.
SLOPE_BANDS = ("0_15", "15_27", "27_35", "35_45", "45_plus")
_BAND_EDGES = (0.0, 15.0, 27.0, 35.0, 45.0, 90.0)

# Aspect is meaningless on near-flat ground; exclude it from the distribution.
MIN_SLOPE_FOR_ASPECT_DEG = 5.0

_SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class SlopeAspect:
    slope_deg: np.ndarray
    aspect_deg: np.ndarray


def _gradients(elevation: np.ndarray, pixel_size_m: float):
    """(dz/drow, dz/dcol) by np.gradient, in rise over run. An axis one cell
    long has no gradient and contributes zero rather than raising."""
    elevation = np.asarray(elevation, dtype="float64")
    height, width = elevation.shape
    zero = np.zeros_like(elevation)
    dz_drow = np.gradient(elevation, pixel_size_m, axis=0) if height > 1 else zero
    dz_dcol = np.gradient(elevation, pixel_size_m, axis=1) if width > 1 else zero
    return dz_drow, dz_dcol


def slope_gradient(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Each cell's steepest gradient as rise over run - tan(slope_deg) of
    `compute_slope_aspect`, from the same gradients. NaN beside a void. The
    walking graphs compare it with a grade limit, so they read it here
    rather than converting degrees back."""
    dz_drow, dz_dcol = _gradients(elevation, pixel_size_m)
    return np.hypot(dz_drow, dz_dcol)


def compute_slope_aspect(elevation: np.ndarray, pixel_size_m: float) -> SlopeAspect:
    dz_drow, dz_dcol = _gradients(elevation, pixel_size_m)
    dz_dx = dz_dcol  # east
    dz_dy = -dz_drow  # north (row index increases southward)

    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    # Descent vector is -(dz_dx, dz_dy); bearing = atan2(east, north).
    aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0

    return SlopeAspect(slope_deg=slope_deg.astype("float32"),
                       aspect_deg=aspect_deg.astype("float32"))


def slope_statistics(slope_deg: np.ndarray, mask: np.ndarray) -> dict:
    values = slope_deg[mask]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean_deg": None, "p90_deg": None, "max_deg": None,
                "distribution": {band: 0.0 for band in SLOPE_BANDS}}

    histogram, _ = np.histogram(values, bins=_BAND_EDGES)
    distribution = {
        band: float(count) / values.size
        for band, count in zip(SLOPE_BANDS, histogram)
    }
    return {
        "mean_deg": float(values.mean()),
        "p90_deg": float(np.percentile(values, 90)),
        "max_deg": float(values.max()),
        "distribution": distribution,
    }


def aspect_statistics(aspect_deg: np.ndarray, slope_deg: np.ndarray,
                      mask: np.ndarray) -> dict:
    selected = mask & (slope_deg >= MIN_SLOPE_FOR_ASPECT_DEG) & ~np.isnan(aspect_deg)
    values = aspect_deg[selected]
    if values.size == 0:
        return {"dominant": None, "distribution": {s: 0.0 for s in _SECTORS}}

    # 8 sectors of 45 degrees centred on the compass points.
    indices = np.floor(((values + 22.5) % 360.0) / 45.0).astype(int)
    counts = np.bincount(indices, minlength=8)
    distribution = {sector: float(c) / values.size for sector, c in zip(_SECTORS, counts)}
    return {"dominant": _SECTORS[int(counts.argmax())], "distribution": distribution}
