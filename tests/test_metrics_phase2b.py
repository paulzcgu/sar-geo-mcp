import json

import numpy as np
import pytest
from rasterio.transform import from_origin

from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain.metrics import TerrainMetrics, compute_terrain_metrics


def valley_dem(rows=60, cols=60):
    distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    across = distance[None, :].repeat(rows, axis=0) * 2.0
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return across + along


def prepared_dir(tmp_path):
    transform = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
    geometry = make_wgs84_square(-122.3400, 37.3380, -122.3380, 37.3400)
    write_fetch_dir(tmp_path, valley_dem(), transform, geometry)

    from PIL import Image

    pixels = np.zeros((60, 60, 3), dtype="uint8")
    pixels[:] = (200, 60, 120)
    panel_dir = tmp_path / "panels"
    panel_dir.mkdir()
    Image.fromarray(pixels, "RGB").save(panel_dir / "vegetation.png")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["panels"] = {"vegetation": "panels/vegetation.png"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_metrics_include_hydrology_and_landforms(tmp_path):
    metrics = compute_terrain_metrics(prepared_dir(tmp_path))

    assert metrics.hydrology is not None
    assert metrics.landforms is not None
    assert sum(metrics.landforms.fractions.values()) == pytest.approx(1.0)
    assert metrics.hydrology.assumptions


def test_hydrology_reports_truncation_status(tmp_path):
    metrics = compute_terrain_metrics(prepared_dir(tmp_path))

    assert isinstance(metrics.hydrology.upstream_truncated, bool)


def test_access_absent_without_a_cached_osm_payload(tmp_path):
    """OSM is fetched during `fetch`, so terrain must not reach the network."""
    metrics = compute_terrain_metrics(prepared_dir(tmp_path))

    assert metrics.access is None or metrics.access.routes == []
    assert any("osm" in w.lower() for w in metrics.data_quality.warnings)


def test_access_populated_when_osm_json_present(tmp_path):
    data_dir = prepared_dir(tmp_path)
    # A route crossing the assignment so ingress points exist.
    (data_dir / "osm.json").write_text(json.dumps({
        "routes": [
            {
                "id": "route_0", "kind": "route",
                "geometry": {"type": "LineString",
                             "coordinates": [[-13618900.0, 4486000.0],
                                             [-13618000.0, 4486000.0]]},
                "properties": {"route_type": "TRAIL", "name": "Test Trail"},
            }
        ]
    }), encoding="utf-8")

    metrics = compute_terrain_metrics(data_dir)

    assert metrics.access is not None
    assert len(metrics.access.routes) == 1
    assert metrics.access.source


def test_shape_ids_remain_unique_across_all_producers(tmp_path):
    metrics = compute_terrain_metrics(prepared_dir(tmp_path))

    ids = [s.id for s in metrics.shapes]
    assert len(ids) == len(set(ids))


def test_metrics_still_round_trip_through_json(tmp_path):
    metrics = compute_terrain_metrics(prepared_dir(tmp_path))

    restored = TerrainMetrics.model_validate_json(metrics.model_dump_json())

    assert restored.schema_version == metrics.schema_version
    assert restored.schema_version == "0.4.0"
