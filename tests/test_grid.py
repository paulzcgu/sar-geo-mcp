import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sar_geo_mcp.terrain.grid import (
    CORRIDOR_HALF_WIDTH_M,
    AssignmentGrid,
    load_assignment_grid,
    project_geometry_to_3857,
)

LATITUDE = 37.32


def write_fetch_dir(tmp_path, elevation, transform, geometry_wgs84, latitude=LATITUDE):
    """Create a minimal fetch output directory: dem.tif + manifest + assignment."""
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-999999.0,
    ) as dst:
        dst.write(elevation.astype("float32"), 1)

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "map_id": "M",
                "assignment": "T",
                "bbox_3857": [0, 0, 1, 1],
                "dem_bbox_3857": [0, 0, 1, 1],
                "dem_resolution_m": 1.0,
                "latitude_deg": latitude,
                "panels": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "assignment.json").write_text(
        json.dumps({"id": "a1", "letter": "T", "geometry": geometry_wgs84}),
        encoding="utf-8",
    )
    return tmp_path


def make_wgs84_square(min_lon, min_lat, max_lon, max_lat):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def test_project_geometry_produces_mercator_coordinates():
    square = make_wgs84_square(-122.34, 37.30, -122.30, 37.34)

    shape = project_geometry_to_3857(square)

    minx, miny, maxx, maxy = shape.bounds
    assert minx == pytest.approx(-13618826.50, abs=1.0)
    assert maxy == pytest.approx(4486604.96, abs=1.0)


def test_project_geometry_buffers_a_linestring_into_a_corridor():
    line = {"type": "LineString", "coordinates": [[-122.34, 37.30], [-122.30, 37.34]]}

    shape = project_geometry_to_3857(line)

    assert shape.geom_type == "Polygon"
    # Corridor half-width is a ground distance; in Mercator units it is inflated.
    expected_halfwidth = CORRIDOR_HALF_WIDTH_M / math.cos(math.radians(37.32))
    # The corridor must be at least that wide somewhere (bounds grew beyond the line).
    assert shape.area > 0
    assert shape.bounds[2] - shape.bounds[0] > expected_halfwidth


def test_load_assignment_grid_reads_dem_and_ground_pixel_size(tmp_path):
    # 20x20 DEM at 10 Mercator units/px anchored so the polygon below covers its centre.
    elevation = np.ones((20, 20), dtype="float32") * 100.0
    # Place the grid at the projection of (-122.34, 37.34) going east/south.
    transform = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
    # A WGS84 polygon fully inside the grid extent (roughly central).
    geometry = make_wgs84_square(-122.3395, 37.3390, -122.3390, 37.3395)
    write_fetch_dir(tmp_path, elevation, transform, geometry)

    grid = load_assignment_grid(tmp_path)

    assert isinstance(grid, AssignmentGrid)
    assert grid.elevation.shape == (20, 20)
    assert grid.pixel_size_m == pytest.approx(10.0 * math.cos(math.radians(LATITUDE)), rel=1e-6)
    assert grid.latitude_deg == LATITUDE
    assert grid.mask.dtype == bool
    assert grid.mask.any(), "mask must cover at least one pixel"
    assert not grid.mask.all(), "mask must not cover the whole grid"


def test_load_assignment_grid_masks_nodata_as_nan(tmp_path):
    elevation = np.ones((10, 10), dtype="float32") * 50.0
    elevation[0, 0] = -999999.0
    transform = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
    geometry = make_wgs84_square(-122.3399, 37.3395, -122.3396, 37.3398)
    write_fetch_dir(tmp_path, elevation, transform, geometry)

    grid = load_assignment_grid(tmp_path)

    assert np.isnan(grid.elevation[0, 0])
    assert grid.elevation[5, 5] == pytest.approx(50.0)


def test_load_assignment_grid_errors_helpfully_when_dem_missing(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="prepare_area or prepare_terrain"):
        load_assignment_grid(tmp_path)


# --- A staged directory's errors name the file, never the path ---------------


def _staged_for_errors(tmp_path, broken):
    """A good staged directory, then broken one way: its DEM missing, its
    subject missing, its manifest missing, or its DEM corrupt."""
    elevation = np.ones((20, 20), dtype="float32") * 100.0
    transform = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
    write_fetch_dir(tmp_path, elevation, transform,
                    make_wgs84_square(-122.3395, 37.3390, -122.3390, 37.3395))
    if broken == "dem missing":
        (tmp_path / "dem.tif").unlink()
    elif broken == "subject missing":
        (tmp_path / "assignment.json").unlink()
    elif broken == "manifest missing":
        (tmp_path / "manifest.json").unlink()
    elif broken == "dem corrupt":
        (tmp_path / "dem.tif").write_bytes(b"this is not a GeoTIFF")
    return tmp_path


def assert_path_free(error, directory):
    """No part of the local path, and nothing chained that could carry one."""
    message = str(error)
    assert str(directory) not in message and directory.name not in message, message
    assert "\\" not in message and "/tmp" not in message.lower(), message
    assert error.__cause__ is None and error.__context__ is None


BROKEN = ["dem missing", "subject missing", "manifest missing", "dem corrupt"]


@pytest.mark.parametrize("broken", BROKEN)
def test_a_broken_staged_directory_is_named_without_its_path(tmp_path, broken):
    directory = _staged_for_errors(tmp_path, broken)
    with pytest.raises(OSError) as refused:
        load_assignment_grid(directory)
    assert_path_free(refused.value, directory)
    assert "stage" in str(refused.value)
    expected = {"dem missing": "dem.tif", "subject missing": "subject.json",
                "manifest missing": "manifest.json", "dem corrupt": "dem.tif"}[broken]
    assert expected in str(refused.value)


def test_a_missing_dem_still_says_how_to_stage_it(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="prepare_area or prepare_terrain") as refused:
        load_assignment_grid(tmp_path)
    assert_path_free(refused.value, tmp_path)


@pytest.mark.parametrize("broken", ["dem missing", "subject missing", "dem corrupt"])
def test_lz_candidates_passes_the_path_free_error_on(tmp_path, broken):
    from sar_geo_mcp.planning.lz import lz_candidates

    directory = _staged_for_errors(tmp_path, broken)
    with pytest.raises(OSError) as refused:
        lz_candidates(directory, center_latitude=37.3392, center_longitude=-122.3392)
    assert_path_free(refused.value, directory)
