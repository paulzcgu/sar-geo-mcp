"""Pins the Web Mercator -> ground-metres conversion across the coverage
pipeline at a real latitude.

This project has shipped the Web Mercator units bug twice before (once in
dem_resolution_m, once in hand-written prose) - Web Mercator units are
inflated by 1/cos(latitude), so treating them as ground metres directly
silently overstates every length and area away from the equator. Every
existing swath/tracks test passes ground_factor=1.0, under which dropping
the factor entirely is invisible: a mutation-testing pass on Phase 7 found
three surviving mutants (dropping ground_factor from track lengths, from
zone_area, and from the grid cell size) that the full suite did not catch.

37.36 deg N is a real assignment latitude (QC1202). cos(37.36 deg) is
approximately 0.7949, so a geometry whose Web Mercator measurement is 1000
units must report about 795 ground units, not 1000.
"""

import json

import pyproj
import pytest
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shapely_transform

from sar_geo_mcp.coverage.metrics import compute_coverage
from sar_geo_mcp.coverage.swath import build_grid
from sar_geo_mcp.coverage.tracks import select_tracks
from sar_geo_mcp.geo import ground_meters_per_mercator_unit

LATITUDE_DEG = 37.36


def _feature(title, coords, cls="Shape"):
    return {"properties": {"title": title, "class": cls},
            "geometry": {"type": "LineString", "coordinates": coords}}


def _write_terrain_fixture(data_dir, latitude_deg, sweep_width_m=10.0):
    from sar_geo_mcp.terrain.metrics import (
        AspectMetrics, DataQuality, ElevationMetrics, SlopeMetrics,
        TerrainMetrics, TraverseMetrics, VegetationMetrics,
    )

    terrain = TerrainMetrics(
        assignment_label="T1",
        elevation=ElevationMetrics(min_m=100.0, max_m=200.0, relief_m=100.0),
        slope=SlopeMetrics(mean_deg=10.0, p90_deg=20.0, max_deg=30.0, distribution={}),
        aspect=AspectMetrics(dominant="N", distribution={}),
        vegetation=VegetationMetrics(),
        traverse=TraverseMetrics(
            baseline_rate_kmh=3.0, vegetation_adjusted_rate_kmh=2.0,
            effective_sweep_width_m=sweep_width_m,
            sweep_time_hours_per_pass=1.0, method="test", assumptions={},
        ),
        shapes=[],
        data_quality=DataQuality(dem_resolution_m=2.5, latitude_deg=latitude_deg,
                                 area_m2=10000.0, warnings=[]),
    )
    (data_dir / "terrain_metrics.json").write_text(
        terrain.model_dump_json(), encoding="utf-8"
    )


def test_mercator_ground_conversions_at_a_real_latitude(tmp_path):
    factor = ground_meters_per_mercator_unit(LATITUDE_DEG)
    assert factor == pytest.approx(0.7949, abs=0.0005)

    # --- track length: tracks.select_tracks ---
    # A 1000-unit Mercator line, fully inside the zone, must report ~795 m,
    # not 1000. Mutant: `length_in_zone_m=clipped.length * ground_factor`
    # with the `* ground_factor` dropped would report 1000.0 here.
    zone_for_track = Polygon([(-10, -10), (2000, -10), (2000, 2000), (-10, 2000)])
    line = _feature("1.1 AA", [[0, 0, 0, 0], [1000, 0, 0, 0]])
    track = select_tracks([line], zone_for_track, ["1"], ground_factor=factor)[0]

    assert track.length_in_zone_m == pytest.approx(1000 * factor, rel=1e-9)
    assert track.length_total_m == pytest.approx(1000 * factor, rel=1e-9)
    assert track.length_in_zone_m == pytest.approx(794.9, abs=0.5)

    # --- grid cell size: swath.build_grid ---
    # cell = cell_ground_m / ground_factor, so a 2 m ground cell at this
    # latitude is ~2.516 Mercator units, not 2.0. Mutant: dropping
    # `/ ground_factor` would make the transform's cell size exactly 2.0.
    grid = build_grid(zone_for_track, cell_ground_m=2.0, ground_factor=factor)
    expected_cell = 2.0 / factor

    assert grid.transform.a == pytest.approx(expected_cell, rel=1e-9)
    assert -grid.transform.e == pytest.approx(expected_cell, rel=1e-9)
    assert grid.transform.a != pytest.approx(2.0, abs=0.01)

    # --- zone area: metrics.compute_coverage ---
    # zone_area_m2 = zone.area (Mercator units^2) * factor^2. Mutant:
    # `zone.area * factor * factor` with the multiplication dropped would
    # report zone.area, ~1/0.7949^2 = ~1.58x too large at this latitude.
    _write_terrain_fixture(tmp_path, latitude_deg=LATITUDE_DEG)
    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-121.9000, 37.3600], [-121.8990, 37.3600],
            [-121.8990, 37.3610], [-121.9000, 37.3610], [-121.9000, 37.3600],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )
    # A track inside the assignment so select_tracks succeeds (compute_coverage
    # would otherwise raise before reaching the zone_area line).
    in_zone_track = _feature("1.1 AA", [
        [-121.8998, 37.3602, 100.0, 0],
        [-121.8992, 37.3608, 105.0, 1000],
    ])

    result = compute_coverage(tmp_path, ["1"], features=[in_zone_track])

    to_mercator = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform
    mercator_zone = shapely_transform(to_mercator, shape(assignment_geometry))
    expected_area_m2 = mercator_zone.area * factor * factor

    assert result.metrics.zone_area_m2 == pytest.approx(expected_area_m2, rel=1e-6)
    # Sanity: catches the mutant even without recomputing the projection -
    # dropping factor**2 makes zone_area_m2 the raw (much larger) Mercator
    # area instead.
    assert result.metrics.zone_area_m2 < mercator_zone.area
