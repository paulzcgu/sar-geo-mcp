"""The feature registry: envelope contract, detectors, selection.

Detector behaviour is tested through real staged directories built with the
same helpers test_metrics uses, because a detector's input IS a staged
directory — faking the grid would test a code path production never runs.
"""

import json

import numpy as np
import pytest
from rasterio.transform import from_origin

# tests/ is on sys.path during collection (pytest prepend mode).
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain.features.base import (
    FeatureDetector,
    FeatureResult,
    FeatureShape,
)

# The square test_metrics uses, sitting inside a 40x40 10 m-pixel DEM whose
# origin is (-13618826.5, 4486605.0). Mask covers roughly cols 2-9, rows 5-17.
SQUARE = make_wgs84_square(-122.3398, 37.3388, -122.3392, 37.3396)
TRANSFORM = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)

MAGENTA = (200, 60, 120)   # healthy vegetation in CIR
TAN = (180, 170, 150)      # bare ground

# A route crossing the assignment's west and east edges: two ingress points.
ROUTE = {
    "id": "route_0", "kind": "route",
    "geometry": {"type": "LineString",
                 "coordinates": [[-13618826.0, 4486490.0],
                                 [-13618700.0, 4486490.0]]},
    "properties": {"route_type": "TRAIL", "name": "Test Trail",
                   "osm_id": 1, "surface": None},
}


def ramp_dem(rows=40, cols=40, step=16.0):
    """A uniform east-ascending ramp: ~63.6 degrees everywhere (16 m rise
    per ~7.95 m ground pixel), so steep bands AND cliff bands trigger."""
    return (np.arange(cols) * step)[None, :].repeat(rows, axis=0)


def bare_dir(tmp_path):
    """dem + manifest + assignment only: no panel, no osm.json."""
    return write_fetch_dir(tmp_path, ramp_dem(), TRANSFORM, SQUARE)


def add_panel(data_dir, color):
    from PIL import Image

    pixels = np.zeros((40, 40, 3), dtype="uint8")
    pixels[:] = color
    panel_dir = data_dir / "panels"
    panel_dir.mkdir(exist_ok=True)
    Image.fromarray(pixels, "RGB").save(panel_dir / "vegetation.png")


def add_osm(data_dir, routes=(ROUTE,)):
    (data_dir / "osm.json").write_text(
        json.dumps({"routes": list(routes)}), encoding="utf-8"
    )


class Dummy(FeatureDetector):
    kind = "dummy"
    description = "test detector"
    assumptions = {"x": 1}

    def detect(self, ctx):
        return self.envelope()


def test_envelope_fills_kind_count_and_assumptions():
    result = Dummy().detect(None)

    assert isinstance(result, FeatureResult)
    assert result.kind == "dummy"
    assert result.shapes == []
    assert result.summary == {"count": 0}
    assert result.warnings == []
    assert result.assumptions == {"x": 1}


def test_envelope_counts_shapes_and_merges_summary_and_warnings():
    shape = FeatureShape(id="d_0", kind="dummy",
                         geometry={"type": "Point", "coordinates": [0, 0]},
                         properties={})

    result = Dummy().envelope([shape], summary={"extra": True},
                              warnings=["w"])

    assert result.summary == {"count": 1, "extra": True}
    assert result.warnings == ["w"]
    assert result.shapes[0].id == "d_0"


from sar_geo_mcp.terrain.features.base import polygonize


def test_polygonize_stamps_ids_and_filters_small_areas():
    binary = np.zeros((40, 40), dtype=bool)
    binary[10:20, 10:20] = True      # 100 px patch
    binary[30, 30] = True            # 1 px fragment, below min area

    shapes = polygonize(binary, from_origin(0, 40, 1, 1), pixel_size_m=1.0,
                        kind="cliff_band", id_prefix="cliff", min_area_m2=10.0)

    assert len(shapes) == 1
    assert shapes[0].id == "cliff_0"
    assert shapes[0].kind == "cliff_band"
    assert shapes[0].geometry["type"] in ("Polygon", "MultiPolygon")
    assert shapes[0].properties["area_m2"] == pytest.approx(100.0, rel=0.3)


def test_polygonize_corrects_area_to_ground_square_metres():
    """Transform units are Web Mercator; pixel_size_m is ground. A 100-px
    patch at 0.8 ground-m pixels over 1.0-unit transform cells is 64 m2."""
    binary = np.zeros((20, 20), dtype=bool)
    binary[5:15, 5:15] = True

    shapes = polygonize(binary, from_origin(0, 20, 1, 1), pixel_size_m=0.8,
                        kind="steep_band", id_prefix="steep_45", min_area_m2=10.0)

    assert shapes[0].properties["area_m2"] == pytest.approx(64.0, rel=0.3)


def test_polygonize_merges_extra_properties_onto_every_shape():
    binary = np.zeros((20, 20), dtype=bool)
    binary[5:15, 5:15] = True

    shapes = polygonize(binary, from_origin(0, 20, 1, 1), pixel_size_m=1.0,
                        kind="steep_band", id_prefix="steep_45", min_area_m2=10.0,
                        extra_properties={"threshold_deg": 45.0})

    assert shapes[0].properties["threshold_deg"] == 45.0


def test_polygonize_returns_empty_for_an_empty_mask():
    binary = np.zeros((20, 20), dtype=bool)

    shapes = polygonize(binary, from_origin(0, 20, 1, 1), pixel_size_m=1.0,
                        kind="clearing", id_prefix="clearing", min_area_m2=10.0)

    assert shapes == []


from sar_geo_mcp.terrain.features.context import TerrainContext


def test_context_grid_is_computed_once(tmp_path):
    ctx = TerrainContext(bare_dir(tmp_path))

    assert ctx.grid is ctx.grid  # cached_property: same object


def test_context_returns_none_for_missing_optional_inputs(tmp_path):
    ctx = TerrainContext(bare_dir(tmp_path))

    assert ctx.vegetation_proxy is None
    assert ctx.osm_payload is None


def test_context_loads_optional_inputs_when_present(tmp_path):
    data_dir = bare_dir(tmp_path)
    add_panel(data_dir, TAN)
    add_osm(data_dir)
    ctx = TerrainContext(data_dir)

    assert ctx.vegetation_proxy.shape == (40, 40)
    assert ctx.osm_payload["routes"][0]["id"] == "route_0"


def test_context_slope_matches_the_ramp(tmp_path):
    ctx = TerrainContext(bare_dir(tmp_path))

    interior = ctx.slope_aspect.slope_deg[10:-10, 10:-10]
    # 16 m rise per ~7.95 m ground pixel -> ~63.6 degrees.
    assert float(np.nanmean(interior)) == pytest.approx(63.6, abs=1.0)


from sar_geo_mcp.terrain.features.cliff_bands import CliffBandDetector
from sar_geo_mcp.terrain.features.steep_bands import SteepBandDetector


def test_steep_band_detector_finds_both_bands_on_the_ramp(tmp_path):
    result = SteepBandDetector().detect(TerrainContext(bare_dir(tmp_path)))

    assert [s.id for s in result.shapes] == ["steep_35_0", "steep_45_0"]
    assert result.summary["count"] == 2
    for shape in result.shapes:
        assert shape.kind == "steep_band"
        assert shape.properties["area_m2"] > 500.0
    assert result.shapes[0].properties["threshold_deg"] == 35.0
    assert result.warnings == []


def test_cliff_band_detector_flags_the_ramp_as_cliff(tmp_path):
    result = CliffBandDetector().detect(TerrainContext(bare_dir(tmp_path)))

    assert [s.id for s in result.shapes] == ["cliff_0"]
    assert result.shapes[0].kind == "cliff_band"
    assert result.shapes[0].properties["threshold_deg"] == 55.0


def test_nothing_found_on_good_input_is_a_clean_zero(tmp_path):
    """Flat ground has no steep bands: empty shapes, count 0, NO warning —
    a real answer, not a degraded one."""
    data_dir = write_fetch_dir(tmp_path, np.zeros((40, 40)), TRANSFORM, SQUARE)

    result = SteepBandDetector().detect(TerrainContext(data_dir))

    assert result.shapes == []
    assert result.summary == {"count": 0}
    assert result.warnings == []


from sar_geo_mcp.terrain.features.clearings import ClearingDetector


def test_clearing_detector_finds_open_ground(tmp_path):
    data_dir = bare_dir(tmp_path)
    add_panel(data_dir, TAN)

    result = ClearingDetector().detect(TerrainContext(data_dir))

    assert [s.id for s in result.shapes] == ["clearing_0"]
    assert result.shapes[0].kind == "clearing"
    assert result.shapes[0].properties["area_m2"] > 300.0
    assert result.warnings == []


def test_clearing_detector_sees_no_clearing_under_canopy(tmp_path):
    data_dir = bare_dir(tmp_path)
    add_panel(data_dir, MAGENTA)

    result = ClearingDetector().detect(TerrainContext(data_dir))

    assert result.summary == {"count": 0}
    assert result.warnings == []


def test_clearing_detector_warns_when_the_panel_is_missing(tmp_path):
    result = ClearingDetector().detect(TerrainContext(bare_dir(tmp_path)))

    assert result.shapes == []
    assert result.summary["count"] == 0
    assert any("vegetation panel missing" in w for w in result.warnings)


from sar_geo_mcp.terrain.features.ingress import IngressDetector
from sar_geo_mcp.terrain.features.routes import RouteDetector


def test_route_detector_returns_the_staged_routes(tmp_path):
    data_dir = bare_dir(tmp_path)
    add_osm(data_dir)

    result = RouteDetector().detect(TerrainContext(data_dir))

    assert [s.id for s in result.shapes] == ["route_0"]
    assert result.shapes[0].properties["route_type"] == "TRAIL"


def test_ingress_detector_finds_both_boundary_crossings(tmp_path):
    data_dir = bare_dir(tmp_path)
    add_osm(data_dir)

    result = IngressDetector().detect(TerrainContext(data_dir))

    assert result.summary["count"] == 2
    assert all(s.kind == "ingress" for s in result.shapes)
    assert result.shapes[0].properties["route_id"] == "route_0"


@pytest.mark.parametrize("detector_cls", [RouteDetector, IngressDetector])
def test_osm_detectors_warn_when_the_cache_is_missing(tmp_path, detector_cls):
    result = detector_cls().detect(TerrainContext(bare_dir(tmp_path)))

    assert result.shapes == []
    assert any("osm.json missing" in w for w in result.warnings)


from sar_geo_mcp.terrain.features.drainages import DrainageDetector


def test_drainage_detector_on_a_dry_ramp_warns_about_the_threshold(tmp_path):
    result = DrainageDetector().detect(TerrainContext(bare_dir(tmp_path)))

    assert result.summary["count"] == 0
    assert result.summary["drainages"] == []
    assert result.summary["upstream_truncated"] is False
    assert any("no drainage above the contributing-area threshold" in w
               for w in result.warnings)


from sar_geo_mcp.terrain.features import (
    REGISTRY,
    identify,
    input_missing,
    kinds,
    list_kinds,
)


@pytest.fixture(scope="module")
def provisioned(tmp_path_factory):
    """A fully-provisioned staged directory: ramp DEM, panel, osm cache."""
    data_dir = bare_dir(tmp_path_factory.mktemp("provisioned"))
    add_panel(data_dir, TAN)
    add_osm(data_dir)
    return data_dir


def test_registry_order_is_the_terrain_metrics_shape_order():
    assert kinds() == ["steep_band", "clearing", "drainage",
                      "cliff_band", "route", "ingress", "ridge",
                      "cliff_pitch"]


def test_list_kinds_describes_every_detector():
    for entry in list_kinds():
        assert entry["kind"] in REGISTRY
        assert entry["description"]
        assert isinstance(entry["assumptions"], dict)


@pytest.mark.parametrize("kind", ["steep_band", "clearing", "drainage",
                                  "cliff_band", "route", "ingress"])
def test_every_detector_honours_the_envelope(provisioned, kind):
    result = REGISTRY[kind]().detect(TerrainContext(provisioned))

    assert result.kind == kind
    assert result.summary["count"] == len(result.shapes)
    assert result.assumptions == dict(REGISTRY[kind].assumptions)
    for shape in result.shapes:
        assert shape.kind == kind


#: The kinds that read an input beyond the elevation model: the vegetation
#: panel (clearing) and osm.json (route, ingress).
NEEDS_MORE_THAN_THE_DEM = {"clearing", "route", "ingress"}


@pytest.mark.parametrize("kind", list(REGISTRY))
def test_a_missing_input_is_told_from_none_by_the_detector_s_own_declaration(tmp_path, kind):
    """On ground with an elevation model and nothing else, a kind reading a
    panel or osm.json answers empty with a warning its detector declares
    for a missing input, so `input_missing` reads it as unknown. Every other
    kind's answer is a real one, empty or not and warned of or not - a dry
    ramp's "no drainage above the threshold" is a fact, not an unknown."""
    result = REGISTRY[kind]().detect(TerrainContext(bare_dir(tmp_path)))

    unknown = kind in NEEDS_MORE_THAN_THE_DEM
    assert input_missing(kind, result.warnings) is unknown
    assert bool(REGISTRY[kind].missing_input_warnings) is unknown
    if unknown:
        assert result.shapes == [] and result.summary["count"] == 0


@pytest.mark.parametrize("kind", sorted(NEEDS_MORE_THAN_THE_DEM))
def test_an_input_that_is_there_is_never_read_as_missing(provisioned, kind):
    result = REGISTRY[kind]().detect(TerrainContext(provisioned))

    assert not input_missing(kind, result.warnings)


def test_a_kind_the_registry_does_not_know_declares_nothing_missing():
    assert not input_missing("bogus", ["vegetation panel missing; clearings unknown"])
    assert "missing_input_warnings" not in list_kinds()[0]


def test_identify_returns_one_envelope_per_requested_kind(provisioned):
    results = identify(provisioned, ["steep_band", "clearing"])

    assert set(results) == {"steep_band", "clearing"}
    assert results["steep_band"].summary["count"] == 2


def test_identify_rejects_unknown_kinds_naming_what_exists(tmp_path):
    with pytest.raises(ValueError, match="steep_band"):
        identify(bare_dir(tmp_path), ["bogus"])


def test_identify_rejects_an_empty_request(tmp_path):
    """Selection is the point; an empty request is a caller error, and the
    answer names what could have been requested."""
    with pytest.raises(ValueError, match="steep_band"):
        identify(bare_dir(tmp_path), [])


def test_identify_computes_only_what_the_kinds_need(tmp_path, monkeypatch):
    """A slope-only request must never pay for flow routing."""
    data_dir = bare_dir(tmp_path)

    def boom(self):
        raise AssertionError("flow accumulation ran for a slope-only request")

    monkeypatch.setattr(TerrainContext, "flow", property(boom))

    results = identify(data_dir, ["steep_band", "cliff_band"])

    assert set(results) == {"steep_band", "cliff_band"}
