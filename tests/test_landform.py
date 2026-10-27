import numpy as np
import pytest

from sar_geo_mcp.terrain.landform import (
    LANDFORM_ASSUMPTIONS,
    classify_landforms,
    landform_fractions,
    topographic_position_index,
)


def ridge_dem(rows=60, cols=60):
    """A north-south ridge down the centre column."""
    distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    return (30.0 - distance)[None, :].repeat(rows, axis=0)


def valley_dem(rows=60, cols=60):
    distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    return distance[None, :].repeat(rows, axis=0)


def test_tpi_is_positive_on_a_ridge_crest():
    tpi = topographic_position_index(ridge_dem(), radius_px=5)

    assert float(tpi[30, 30]) > 0


def test_tpi_is_negative_in_a_valley_floor():
    tpi = topographic_position_index(valley_dem(), radius_px=5)

    assert float(tpi[30, 30]) < 0


def test_tpi_is_zero_on_a_uniform_slope():
    plane = (np.arange(60) * 1.0)[None, :].repeat(60, axis=0)

    tpi = topographic_position_index(plane, radius_px=5)

    assert float(np.abs(tpi[30, 30])) < 0.5


def test_classify_labels_the_ridge_crest():
    dem = ridge_dem()
    slope_deg = np.full(dem.shape, 20.0)

    classes = classify_landforms(dem, slope_deg, radius_px=5)

    assert classes[30, 30] == "ridge"


def test_classify_labels_the_valley_floor():
    dem = valley_dem()
    slope_deg = np.full(dem.shape, 20.0)

    classes = classify_landforms(dem, slope_deg, radius_px=5)

    assert classes[30, 30] == "valley"


def test_classify_labels_flat_low_slope_ground_as_bench():
    dem = np.zeros((60, 60))
    slope_deg = np.zeros(dem.shape)

    classes = classify_landforms(dem, slope_deg, radius_px=5)

    assert classes[30, 30] == "bench"


def test_landform_fractions_sum_to_one():
    dem = ridge_dem()
    slope_deg = np.full(dem.shape, 20.0)
    classes = classify_landforms(dem, slope_deg, radius_px=5)
    mask = np.ones(dem.shape, dtype=bool)

    fractions = landform_fractions(classes, mask)

    assert sum(fractions.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in fractions.values())


def test_cliff_threshold_is_an_exported_assumption():
    assert "cliff_slope_deg" in LANDFORM_ASSUMPTIONS
