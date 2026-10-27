"""Vegetation density from the CalTopo CIR (false-colour IR) panel.

In a CIR render, NIR maps to the red channel and visible red to the green
channel, so (R - G) / (R + G) behaves like NDVI. It is a PROXY over a rendered
image, not radiometric NDVI — the method string says so, and thresholds here
are documented assumptions, not measurements.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Pixels above this proxy value count as vegetated canopy.
VEGETATION_PROXY_THRESHOLD = 0.15

# canopy_fraction -> class. Everything below "moderate" is "open".
DENSITY_THRESHOLDS = {"moderate": 0.30, "dense": 0.60}

_METHOD = "CIR proxy (R-G)/(R+G) on rendered NAIP false-colour IR"


def cir_proxy(panel_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """NDVI-like proxy in [-1, 1], resampled to the DEM grid shape if needed."""
    with Image.open(panel_path) as image:
        rgb = image.convert("RGB")
        rows, cols = expected_shape
        if rgb.size != (cols, rows):
            rgb = rgb.resize((cols, rows), Image.NEAREST)
        pixels = np.asarray(rgb, dtype="float64")

    nir = pixels[:, :, 0]
    red = pixels[:, :, 1]
    return ((nir - red) / (nir + red + 1e-9)).astype("float32")


def density_class(canopy_fraction: float) -> str:
    if canopy_fraction >= DENSITY_THRESHOLDS["dense"]:
        return "dense"
    if canopy_fraction >= DENSITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "open"


def vegetation_statistics(panel_path: Path, mask: np.ndarray,
                          expected_shape: tuple[int, int]) -> dict:
    proxy = cir_proxy(panel_path, expected_shape)
    inside = proxy[mask]
    if inside.size == 0:
        return {"canopy_fraction": None, "density_class": None, "method": _METHOD}

    canopy_fraction = float((inside > VEGETATION_PROXY_THRESHOLD).mean())
    return {
        "canopy_fraction": canopy_fraction,
        "density_class": density_class(canopy_fraction),
        "method": _METHOD,
    }
