import numpy as np
import pytest
from PIL import Image

from sar_geo_mcp.terrain.vegetation import (
    DENSITY_THRESHOLDS,
    VEGETATION_PROXY_THRESHOLD,
    cir_proxy,
    density_class,
    vegetation_statistics,
)


def cir_image(tmp_path, pixels):
    """Write a small RGB PNG; `pixels` is an (rows, cols, 3) uint8 array."""
    path = tmp_path / "vegetation.png"
    Image.fromarray(pixels.astype("uint8"), "RGB").save(path)
    return path

MAGENTA = (200, 60, 120)   # healthy vegetation in CIR: high NIR (R), low red (G)
TAN = (180, 170, 150)      # bare ground / dry grass
GRAY = (120, 120, 120)     # road


def test_cir_proxy_high_for_vegetation_low_for_bare(tmp_path):
    pixels = np.zeros((4, 4, 3), dtype="uint8")
    pixels[:, :2] = MAGENTA
    pixels[:, 2:] = TAN
    path = cir_image(tmp_path, pixels)

    proxy = cir_proxy(path, expected_shape=(4, 4))

    assert proxy.shape == (4, 4)
    assert float(proxy[0, 0]) > 0.4
    assert float(proxy[0, 3]) < 0.1


def test_cir_proxy_resizes_when_shapes_differ(tmp_path):
    pixels = np.zeros((8, 8, 3), dtype="uint8")
    pixels[:] = MAGENTA
    path = cir_image(tmp_path, pixels)

    proxy = cir_proxy(path, expected_shape=(4, 4))

    assert proxy.shape == (4, 4)
    assert float(proxy.mean()) > 0.4


def test_density_class_boundaries():
    assert density_class(0.1) == "open"
    assert density_class(DENSITY_THRESHOLDS["moderate"]) == "moderate"
    assert density_class(DENSITY_THRESHOLDS["dense"]) == "dense"
    assert density_class(0.95) == "dense"


def test_vegetation_statistics_mixed_scene(tmp_path):
    pixels = np.zeros((10, 10, 3), dtype="uint8")
    pixels[:, :6] = MAGENTA   # 60% vegetated
    pixels[:, 6:] = GRAY
    path = cir_image(tmp_path, pixels)
    mask = np.ones((10, 10), dtype=bool)

    stats = vegetation_statistics(path, mask, expected_shape=(10, 10))

    assert stats["canopy_fraction"] == pytest.approx(0.6, abs=0.05)
    assert stats["density_class"] == "dense"
    assert stats["method"].startswith("CIR proxy")


def test_vegetation_statistics_respects_mask(tmp_path):
    pixels = np.zeros((10, 10, 3), dtype="uint8")
    pixels[:, :5] = MAGENTA
    pixels[:, 5:] = GRAY
    path = cir_image(tmp_path, pixels)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True  # only the vegetated half

    stats = vegetation_statistics(path, mask, expected_shape=(10, 10))

    assert stats["canopy_fraction"] == pytest.approx(1.0, abs=0.05)
