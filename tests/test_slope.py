import numpy as np
import pytest

from sar_geo_mcp.terrain.slope import (
    SLOPE_BANDS,
    aspect_statistics,
    compute_slope_aspect,
    slope_statistics,
)


def plane(rows=32, cols=32, dz_dx=0.0, dz_dy_north=0.0, pixel_m=1.0):
    """Elevation plane with the given ground gradients (east and north)."""
    col = np.arange(cols) * pixel_m
    row = np.arange(rows) * pixel_m
    # row index increases southward, so a +north gradient DECREASES with row.
    return (col[None, :] * dz_dx) + ((rows - 1) * pixel_m - row[:, None]) * dz_dy_north


def test_flat_ground_has_zero_slope():
    result = compute_slope_aspect(plane(), pixel_size_m=1.0)

    assert float(np.nanmax(result.slope_deg)) == pytest.approx(0.0, abs=1e-6)


def test_45_degree_plane():
    result = compute_slope_aspect(plane(dz_dx=1.0), pixel_size_m=1.0)

    interior = result.slope_deg[1:-1, 1:-1]
    assert float(np.nanmean(interior)) == pytest.approx(45.0, abs=0.1)


def test_aspect_east_descending_plane_reads_east():
    # Rises to the west (dz_dx negative) -> descends east -> aspect 90.
    result = compute_slope_aspect(plane(dz_dx=-0.5), pixel_size_m=1.0)

    interior = result.aspect_deg[1:-1, 1:-1]
    assert float(np.nanmean(interior)) == pytest.approx(90.0, abs=0.5)


def test_aspect_north_descending_plane_reads_north():
    # Rises to the south -> descends north -> aspect 0/360.
    result = compute_slope_aspect(plane(dz_dy_north=-0.5), pixel_size_m=1.0)

    interior = result.aspect_deg[1:-1, 1:-1]
    bearing = float(np.nanmean(np.where(interior > 180, interior - 360, interior)))
    assert bearing == pytest.approx(0.0, abs=0.5)


def test_slope_statistics_bands_sum_to_one():
    elevation = plane(dz_dx=0.6)  # ~31 degrees everywhere
    result = compute_slope_aspect(elevation, pixel_size_m=1.0)
    mask = np.ones(elevation.shape, dtype=bool)

    stats = slope_statistics(result.slope_deg, mask)

    assert set(stats["distribution"]) == set(SLOPE_BANDS)
    assert sum(stats["distribution"].values()) == pytest.approx(1.0)
    assert stats["distribution"]["27_35"] > 0.9
    assert stats["mean_deg"] == pytest.approx(31.0, abs=0.5)
    assert stats["max_deg"] >= stats["p90_deg"] >= stats["mean_deg"] - 0.1


def test_slope_statistics_respects_the_mask():
    """Values outside the mask must not contribute — even poisoned ones."""
    elevation = plane(dz_dx=1.0)
    result = compute_slope_aspect(elevation, pixel_size_m=1.0)
    mask = np.zeros(elevation.shape, dtype=bool)
    mask[:, :4] = True
    result.slope_deg[:, 4:] = np.nan  # poison everything outside the mask

    stats = slope_statistics(result.slope_deg, mask)

    assert stats["mean_deg"] == pytest.approx(45.0, abs=0.2)


def test_aspect_statistics_dominant_sector():
    result = compute_slope_aspect(plane(dz_dx=-0.5), pixel_size_m=1.0)  # descends east
    mask = np.ones(result.slope_deg.shape, dtype=bool)

    stats = aspect_statistics(result.aspect_deg, result.slope_deg, mask)

    assert stats["dominant"] == "E"
    assert stats["distribution"]["E"] > 0.9


def test_aspect_statistics_ignores_flat_ground():
    flat = compute_slope_aspect(plane(), pixel_size_m=1.0)
    mask = np.ones(flat.slope_deg.shape, dtype=bool)

    stats = aspect_statistics(flat.aspect_deg, flat.slope_deg, mask)

    assert stats["dominant"] is None
