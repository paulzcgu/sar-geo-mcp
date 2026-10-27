import json

import numpy as np
import pytest
from rasterio.transform import from_origin

# tests/ is on sys.path during collection (pytest prepend mode), so sibling
# test modules import by bare name — there is no tests/__init__.py package.
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain.metrics import TerrainMetrics, compute_terrain_metrics


def ridge_dem(rows=40, cols=40):
    """A north-south ridge: west face descends west, east face descends east."""
    col = np.abs(np.arange(cols) - cols // 2).astype("float64")
    return (cols // 2 - col)[None, :].repeat(rows, axis=0) * 2.0


def prepared_dir(tmp_path):
    transform = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
    geometry = make_wgs84_square(-122.3398, 37.3388, -122.3392, 37.3396)
    write_fetch_dir(tmp_path, ridge_dem(), transform, geometry)

    # Vegetation panel: all-magenta (dense) at the DEM's shape.
    from PIL import Image

    pixels = np.zeros((40, 40, 3), dtype="uint8")
    pixels[:] = (200, 60, 120)
    panel_dir = tmp_path / "panels"
    panel_dir.mkdir()
    Image.fromarray(pixels, "RGB").save(panel_dir / "vegetation.png")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["panels"] = {"vegetation": "panels/vegetation.png"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_compute_terrain_metrics_end_to_end(tmp_path):
    data_dir = prepared_dir(tmp_path)

    metrics = compute_terrain_metrics(data_dir)

    assert isinstance(metrics, TerrainMetrics)
    assert metrics.elevation.relief_m == pytest.approx(
        metrics.elevation.max_m - metrics.elevation.min_m
    )
    assert metrics.slope.mean_deg is not None
    assert sum(metrics.slope.distribution.values()) == pytest.approx(1.0)
    assert metrics.vegetation.density_class == "dense"
    assert metrics.traverse.baseline_rate_kmh > 0
    assert metrics.traverse.assumptions  # recorded, not hidden
    assert metrics.data_quality.dem_resolution_m > 0


def test_metrics_json_round_trips(tmp_path):
    data_dir = prepared_dir(tmp_path)

    metrics = compute_terrain_metrics(data_dir)
    payload = metrics.model_dump_json(indent=2)
    restored = TerrainMetrics.model_validate_json(payload)

    assert restored.slope.mean_deg == metrics.slope.mean_deg


def test_missing_vegetation_panel_degrades_with_warning(tmp_path):
    data_dir = prepared_dir(tmp_path)
    (data_dir / "panels" / "vegetation.png").unlink()

    metrics = compute_terrain_metrics(data_dir)

    assert metrics.vegetation.canopy_fraction is None
    assert any("vegetation" in w for w in metrics.data_quality.warnings)


def test_missing_evt_degrades_with_warning(tmp_path):
    data_dir = prepared_dir(tmp_path)  # no evt.tif written

    metrics = compute_terrain_metrics(data_dir)

    assert metrics.vegetation.lifeform_fractions is None
    assert any("evt" in w for w in metrics.data_quality.warnings)


def test_shape_registry_ids_are_unique(tmp_path):
    data_dir = prepared_dir(tmp_path)

    metrics = compute_terrain_metrics(data_dir)

    ids = [s.id for s in metrics.shapes]
    assert len(ids) == len(set(ids))


def test_every_detector_warning_reaches_data_quality(tmp_path, monkeypatch):
    """The envelope exists so missing inputs "degrade into warnings, never
    into silence" - and compute_terrain_metrics used to hand-pick which
    detectors' warnings it kept, so the clearing detector's was computed
    and then dropped."""
    from sar_geo_mcp.terrain import metrics as metrics_module
    from sar_geo_mcp.terrain.features import REGISTRY
    from sar_geo_mcp.terrain.features.base import FeatureResult

    seen = {}

    class Loud:
        def __init__(self, kind, real):
            self.kind, self.real = kind, real

        def detect(self, ctx):
            envelope = self.real().detect(ctx)
            warning = f"{self.kind} says something"
            seen[self.kind] = warning
            return FeatureResult(
                kind=envelope.kind, shapes=envelope.shapes,
                summary=envelope.summary,
                warnings=[*envelope.warnings, warning],
                assumptions=envelope.assumptions,
            )

    patched = {kind: (lambda k=kind, r=cls: Loud(k, r))
               for kind, cls in REGISTRY.items()}
    monkeypatch.setattr(metrics_module, "REGISTRY", patched, raising=False)
    monkeypatch.setattr("sar_geo_mcp.terrain.features.REGISTRY", patched)

    result = metrics_module.compute_terrain_metrics(prepared_dir(tmp_path))

    for kind, warning in seen.items():
        assert warning in result.data_quality.warnings, (
            f"the {kind} detector's warning never reached the document")


def test_a_repeated_warning_is_reported_once(tmp_path):
    """Route and ingress both raise the same missing-OSM warning; a reader
    should see it once, not twice."""
    from sar_geo_mcp.terrain.metrics import compute_terrain_metrics

    warnings = compute_terrain_metrics(prepared_dir(tmp_path)).data_quality.warnings

    assert len(warnings) == len(set(warnings))


def _with_manifest_warnings(directory, warnings, osm_version=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["warnings"] = warnings
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if osm_version is not None:
        (directory / "osm.json").write_text(json.dumps(
            {"routes": [], "parking": [], "query_version": osm_version}))
    return directory


def test_a_stale_osm_outage_is_not_reported_once_osm_json_is_current(tmp_path):
    """The manifest's warnings are re-served here exactly as `prepare_area`
    re-serves them, so the same stale outage would repeat in every metrics
    document for ground whose `osm.json` has since been fetched whole."""
    from sar_geo_mcp.staging import is_osm_outage_warning
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

    directory = _with_manifest_warnings(
        prepared_dir(tmp_path),
        ["osm unavailable: ConnectTimeout: overpass-api.de", "evt unavailable: OSError"],
        osm_version=OSM_QUERY_VERSION)
    # A top-up wrote that file after the outage (see `manifest_warnings`).
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["osm_refreshed_versions"] = [OSM_QUERY_VERSION]
    (directory / "manifest.json").write_text(json.dumps(manifest))

    warnings = compute_terrain_metrics(directory).data_quality.warnings

    assert not any(is_osm_outage_warning(w) for w in warnings)
    assert "evt unavailable: OSError" in warnings


def test_an_osm_outage_is_still_reported_while_osm_json_is_missing(tmp_path):
    outage = "osm unavailable: ConnectTimeout: overpass-api.de"
    directory = _with_manifest_warnings(prepared_dir(tmp_path), [outage])

    assert outage in compute_terrain_metrics(directory).data_quality.warnings
