"""Landform classification from Topographic Position Index.

TPI is elevation minus the mean elevation of a surrounding neighbourhood:
positive on crests, negative in hollows, near zero on uniform slopes. Combined
with slope it separates ridge / valley / bench / slope, which is what tells a
planner whether a zone is a single face or a set of separate compartments.
"""

import numpy as np

LANDFORM_ASSUMPTIONS = {
    # TPI magnitudes beyond this (in metres) count as crest or hollow.
    "tpi_threshold_m": 1.0,
    # Below this slope with a near-zero TPI, ground is a bench rather than a slope.
    "bench_slope_deg": 5.0,
    # Contiguous ground steeper than this is a cliff band, not merely steep.
    "cliff_slope_deg": 55.0,
    "tpi_radius_px": 15,
    "min_cliff_area_m2": 200.0,
}

_CLASSES = ("ridge", "valley", "bench", "slope")


def topographic_position_index(elevation: np.ndarray, radius_px: int) -> np.ndarray:
    """Elevation minus the local mean over a square neighbourhood."""
    from scipy import ndimage

    working = np.nan_to_num(elevation, nan=float(np.nanmean(elevation)))
    size = max(3, 2 * int(radius_px) + 1)
    local_mean = ndimage.uniform_filter(working, size=size, mode="nearest")
    return working - local_mean


def classify_landforms(elevation: np.ndarray, slope_deg: np.ndarray,
                       radius_px: int = LANDFORM_ASSUMPTIONS["tpi_radius_px"]
                       ) -> np.ndarray:
    """Per-pixel landform class as a string array."""
    tpi = topographic_position_index(elevation, radius_px)
    threshold = LANDFORM_ASSUMPTIONS["tpi_threshold_m"]
    bench_slope = LANDFORM_ASSUMPTIONS["bench_slope_deg"]

    classes = np.full(elevation.shape, "slope", dtype=object)
    classes[tpi >= threshold] = "ridge"
    classes[tpi <= -threshold] = "valley"
    flat = (np.abs(tpi) < threshold) & (np.nan_to_num(slope_deg) < bench_slope)
    classes[flat] = "bench"
    return classes


def landform_fractions(classes: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    inside = classes[mask]
    total = inside.size
    if total == 0:
        return {name: 0.0 for name in _CLASSES}
    return {
        name: float((inside == name).sum()) / total
        for name in _CLASSES
    }
