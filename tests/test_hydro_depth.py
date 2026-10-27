import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from sar_geo_mcp.terrain.hydro import (
    HYDRO_ASSUMPTIONS,
    bank_top_along_ray,
    measure_channel_depth,
)


def test_bank_top_stops_at_the_slope_break():
    """Ramp at 45 degrees for 20 samples, then flat: the break is at 20."""
    elevations = np.concatenate([np.arange(21) * 1.0, np.full(20, 20.0)])

    result = bank_top_along_ray(elevations, sample_spacing_m=1.0,
                                break_slope_deg=15.0, smoothing=1)

    assert result["distance_m"] == pytest.approx(20.0, abs=1.5)
    assert result["climb_m"] == pytest.approx(20.0, abs=1.0)
    assert result["slope_deg"] == pytest.approx(45.0, abs=2.0)
    assert result["truncated"] is False


def test_bank_top_reports_truncated_when_the_slope_never_breaks():
    elevations = np.arange(30) * 1.0  # 45 degrees all the way out

    result = bank_top_along_ray(elevations, sample_spacing_m=1.0,
                                break_slope_deg=15.0, smoothing=1)

    assert result["truncated"] is True
    assert result["distance_m"] == pytest.approx(29.0, abs=1.0)


def test_bank_top_ignores_a_single_noisy_sample():
    """One flat lidar pixel mid-wall must not end the walk."""
    elevations = np.arange(21) * 1.0
    elevations[10] = elevations[9]  # a one-sample flat spot

    result = bank_top_along_ray(elevations, sample_spacing_m=1.0,
                                break_slope_deg=15.0, smoothing=3)

    assert result["distance_m"] > 12.0


def test_bank_top_on_flat_ground_reports_zero_climb():
    """No wall is a complete answer, not a truncated one."""
    result = bank_top_along_ray(np.zeros(30), sample_spacing_m=1.0,
                                break_slope_deg=15.0, smoothing=1)

    assert result["climb_m"] == pytest.approx(0.0, abs=0.5)
    assert result["truncated"] is False


def v_valley(rows=80, cols=80, wall_gradient=1.0, bench_at=25):
    """V-valley along the centre column: walls rise at `wall_gradient` for
    `bench_at` cells each side, then flatten onto a bench."""
    distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    profile = np.where(distance <= bench_at,
                       distance * wall_gradient,
                       bench_at * wall_gradient)
    return profile[None, :].repeat(rows, axis=0)


def test_measure_channel_depth_on_a_known_v_valley():
    dem = v_valley(wall_gradient=1.0, bench_at=25)
    transform = from_origin(0, 80, 1, 1)
    # Channel runs down the centre column.
    channel = {
        "id": "drainage_0",
        "kind": "drainage",
        "geometry": {
            "type": "LineString",
            "coordinates": [[40.5, 70.0], [40.5, 40.0], [40.5, 10.0]],
        },
        "properties": {"length_m": 60.0, "contributing_area_m2": 5000.0},
    }

    result = measure_channel_depth(channel, dem, transform, pixel_size_m=1.0)

    # Walls rise 1 m per m for 25 m -> 25 m climb at 45 degrees, both sides.
    assert result["depth_m"]["left"] == pytest.approx(25.0, abs=4.0)
    assert result["depth_m"]["right"] == pytest.approx(25.0, abs=4.0)
    assert result["sidewall_slope_deg"]["left"] == pytest.approx(45.0, abs=6.0)
    assert result["sidewall_slope_deg"]["right"] == pytest.approx(45.0, abs=6.0)
    assert result["transect_count"] >= 2
    assert result["method"].startswith("bank-top")


def test_measure_channel_depth_reports_asymmetric_walls():
    """A steep wall on one side and a gentle one on the other must not average."""
    cols = 80
    distance = np.arange(cols) - cols // 2
    profile = np.where(distance < 0, np.abs(distance) * 1.0, distance * 0.2)
    profile = np.minimum(profile, 25.0)
    dem = profile[None, :].repeat(80, axis=0)

    channel = {
        "id": "drainage_0", "kind": "drainage",
        "geometry": {"type": "LineString",
                     "coordinates": [[40.5, 70.0], [40.5, 10.0]]},
        "properties": {"length_m": 60.0, "contributing_area_m2": 5000.0},
    }

    result = measure_channel_depth(channel, dem, transform=from_origin(0, 80, 1, 1),
                                   pixel_size_m=1.0)

    left = result["sidewall_slope_deg"]["left"]
    right = result["sidewall_slope_deg"]["right"]
    assert abs(left - right) > 10.0, f"walls should differ: {left} vs {right}"


def test_depth_assumptions_are_exported():
    for key in ("bank_top_slope_deg", "max_transect_m", "transect_interval_m"):
        assert key in HYDRO_ASSUMPTIONS
