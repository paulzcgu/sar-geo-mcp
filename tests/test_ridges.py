"""Ridgelines: flow accumulation on the negated surface, the same pass the
anatomy report used. A ridge is what water would drain to if the hill were
turned upside down."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain.features import REGISTRY, identify, kinds
from sar_geo_mcp.terrain.features.context import TerrainContext
from sar_geo_mcp.terrain.features.ridges import NO_RIDGE_WARNING, RidgeDetector
from sar_geo_mcp.terrain.hydro import extract_channels

SQUARE = make_wgs84_square(-122.3398, 37.3388, -122.3392, 37.3396)
# Shifted west of the test_features.py TRANSFORM of the same name: the
# assignment polygon above projects to x in [-13618804, -13618737], and the
# crest/valley in `ridge_dem` below sits at the DEM's centre column (col 20).
# The origin here places that centre column inside the assignment instead of
# ~120 m east of it, so the clip in `extract_channels` (same clip_to every
# other detector honours) does not discard the very feature under test.
TRANSFORM = from_origin(-13618970.5, 4486605.0, 10.0, 10.0)


def ridge_dem(rows=40, cols=40):
    """A crest along the centre column, falling away east and west and
    sloping gently north so the crest has a downhill direction."""
    col_distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    across = -col_distance[None, :].repeat(rows, axis=0) * 2.0
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return 100.0 + across + along


def valley_dem(rows=40, cols=40):
    return 200.0 - ridge_dem(rows, cols)


def test_registry_keeps_ridge_in_place_as_kinds_are_appended():
    """The registry contract is append-only: `terrain_metrics` reports shapes
    in this order, so a reorder would silently rewrite that output. New kinds
    go on the end - `ridge` must stay where it is."""
    assert kinds()[:7] == ["steep_band", "clearing", "drainage", "cliff_band",
                           "route", "ingress", "ridge"]
    assert REGISTRY["ridge"] is RidgeDetector


def test_a_ridge_dem_yields_a_ridge_along_the_crest(tmp_path):
    data_dir = write_fetch_dir(tmp_path, ridge_dem(), TRANSFORM, SQUARE)
    result = identify(data_dir, ["ridge"])["ridge"]
    assert result.kind == "ridge"
    assert result.shapes, result.warnings
    crest = result.shapes[0]
    assert crest.id == "ridge_0" and crest.kind == "ridge"
    assert crest.geometry["type"] == "LineString"
    xs = [c[0] for c in crest.geometry["coordinates"]]
    centre_x = TRANSFORM.c + 20 * 10.0
    assert abs(float(np.mean(xs)) - centre_x) < 25.0
    assert set(crest.properties) >= {"length_m", "full_length_m", "contributing_area_m2"}
    assert "depth_m" not in crest.properties


def test_the_ridge_dem_has_no_drainage_and_the_valley_dem_has_no_ridge(tmp_path):
    """The two detectors look at opposite surfaces; each must be blind to the
    other's feature."""
    (tmp_path / "r").mkdir()
    (tmp_path / "v").mkdir()
    ridge_dir = write_fetch_dir(tmp_path / "r", ridge_dem(), TRANSFORM, SQUARE)
    valley_dir = write_fetch_dir(tmp_path / "v", valley_dem(), TRANSFORM, SQUARE)
    ridges_in_valley = identify(valley_dir, ["ridge"])["ridge"]
    drainages_in_ridge = identify(ridge_dir, ["drainage"])["drainage"]
    assert ridges_in_valley.shapes == [] and NO_RIDGE_WARNING in ridges_in_valley.warnings
    assert drainages_in_ridge.shapes == []


def test_inverted_flow_is_computed_once_and_independently_of_flow(tmp_path):
    data_dir = write_fetch_dir(tmp_path, ridge_dem(), TRANSFORM, SQUARE)
    ctx = TerrainContext(data_dir)
    first = ctx.inverted_flow
    assert ctx.inverted_flow is first
    assert not np.array_equal(first.accumulation, ctx.flow.accumulation)


def test_extract_channels_kind_and_prefix_are_parameters():
    from rasterio.transform import from_origin as origin
    from sar_geo_mcp.terrain._pysheds_backend import flow_accumulation

    dem = 200.0 - ridge_dem(60, 60)  # a valley, so channels exist
    flow = flow_accumulation(dem, origin(0, 60, 1, 1), network_min_cells=50)
    channels = extract_channels(flow.accumulation, flow.network, origin(0, 60, 1, 1),
                                pixel_size_m=1.0, min_area_m2=50.0, min_length_m=10.0,
                                kind="ridge", id_prefix="ridge")
    assert channels and channels[0]["id"] == "ridge_0" and channels[0]["kind"] == "ridge"


def test_terrain_metrics_reports_ridges_last_and_the_new_schema_version(tmp_path):
    from sar_geo_mcp.terrain.metrics import compute_terrain_metrics

    data_dir = write_fetch_dir(tmp_path, ridge_dem(), TRANSFORM, SQUARE)
    metrics = compute_terrain_metrics(data_dir)
    assert metrics.schema_version == "0.4.0"
    kinds_in_order = [shape.kind for shape in metrics.shapes]
    assert "ridge" in kinds_in_order
    assert kinds_in_order.index("ridge") > max(
        (i for i, k in enumerate(kinds_in_order) if k != "ridge"), default=-1)


def test_a_void_in_the_dem_is_warned_about_by_both_detectors(tmp_path):
    from sar_geo_mcp.terrain.features.base import DEM_VOID_WARNING

    dem = ridge_dem()
    dem[10:12, 5:7] = np.nan
    data_dir = write_fetch_dir(tmp_path, dem, TRANSFORM, SQUARE)
    results = identify(data_dir, ["ridge", "drainage"])
    assert DEM_VOID_WARNING in results["ridge"].warnings
    assert DEM_VOID_WARNING in results["drainage"].warnings


def test_a_dem_without_voids_does_not_warn(tmp_path):
    from sar_geo_mcp.terrain.features.base import DEM_VOID_WARNING

    data_dir = write_fetch_dir(tmp_path, ridge_dem(), TRANSFORM, SQUARE)
    assert DEM_VOID_WARNING not in identify(data_dir, ["ridge"])["ridge"].warnings


def ridge_and_valley_dem(rows=40, cols=40):
    """A crest at column 10 and a valley floor at column 30, both sloping
    gently north: one DEM, both features, as real ground has."""
    cols_idx = np.arange(cols).astype("float64")
    crest = -np.abs(cols_idx - 10) * 2.0
    valley = np.abs(cols_idx - 30) * 2.0
    across = np.where(cols_idx < 20, crest, valley - 20.0)
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return 100.0 + across[None, :].repeat(rows, axis=0) + along


def test_one_dem_yields_both_a_ridge_and_a_drainage_in_their_own_places(tmp_path):
    from sar_geo_mcp.terrain.features import identify

    wide = make_wgs84_square(-122.3420, 37.3380, -122.3380, 37.3400)   # covers the raster
    data_dir = write_fetch_dir(tmp_path, ridge_and_valley_dem(), TRANSFORM, wide)
    results = identify(data_dir, ["ridge", "drainage"])
    ridge_xs = [c[0] for c in results["ridge"].shapes[0].geometry["coordinates"]]
    drain_xs = [c[0] for c in results["drainage"].shapes[0].geometry["coordinates"]]
    assert abs(float(np.mean(ridge_xs)) - (TRANSFORM.c + 10 * 10.0)) < 25.0
    assert abs(float(np.mean(drain_xs)) - (TRANSFORM.c + 30 * 10.0)) < 25.0
