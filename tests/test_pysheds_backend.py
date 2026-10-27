import numpy as np
import pytest
from rasterio.transform import from_origin

from sar_geo_mcp.terrain._pysheds_backend import flow_accumulation


def valley_dem(rows=60, cols=60):
    """A V-valley draining south: flow must concentrate in the centre column."""
    col_distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    across = col_distance[None, :].repeat(rows, axis=0) * 2.0
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return across + along


def test_flow_accumulation_concentrates_in_the_valley():
    dem = valley_dem()
    transform = from_origin(0, 60, 1, 1)

    result = flow_accumulation(dem, transform)

    column_totals = result.accumulation.sum(axis=0)
    assert int(column_totals.argmax()) == 30


def test_flow_accumulation_returns_finite_positive_values():
    result = flow_accumulation(valley_dem(), from_origin(0, 60, 1, 1))

    assert np.isfinite(result.accumulation).all()
    assert result.accumulation.min() >= 1


def test_flow_accumulation_shape_matches_input():
    dem = valley_dem(rows=40, cols=25)

    result = flow_accumulation(dem, from_origin(0, 40, 1, 1))

    assert result.accumulation.shape == (40, 25)
    assert result.flow_direction.shape == (40, 25)


def test_nan_elevation_is_filled_before_routing():
    """NaN nodata must not propagate into the accumulation grid."""
    dem = valley_dem()
    dem[0, 0] = np.nan

    result = flow_accumulation(dem, from_origin(0, 60, 1, 1))

    assert np.isfinite(result.accumulation).all()


def test_numpy2_shim_is_applied_by_importing_the_backend():
    """The shim must be a side effect of this module, not of the caller."""
    import numpy

    assert hasattr(numpy, "in1d")


def test_network_traces_the_valley_axis_not_a_blob():
    """A 60-cell valley must trace ~60 m, not thousands.

    Threshold-and-label produced a 4879 m zig-zag across the whole thresholded
    blob; following flow paths gives the actual centreline.
    """
    result = flow_accumulation(valley_dem(), from_origin(0, 60, 1, 1),
                               network_min_cells=50)

    assert result.network is not None
    assert len(result.network) >= 1
    longest = max(result.network, key=len)
    xs = [point[0] for point in longest]
    assert float(np.mean(xs)) == pytest.approx(30.0, abs=1.0)
    assert len(longest) < 100  # a centreline, not a blob traversal


def test_network_is_none_unless_requested():
    result = flow_accumulation(valley_dem(), from_origin(0, 60, 1, 1))

    assert result.network is None
