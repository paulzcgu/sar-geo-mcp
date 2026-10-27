"""list_data_sources, fetch_raster and query_features, offline."""

from __future__ import annotations

import functools
import json

import pytest

from sar_geo_mcp.components import sources as tools
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext

from test_source_contract import FAKE_KEY, FakeTransport

POINT = {"center_latitude": 37.4244, "center_longitude": -122.3151}


@pytest.fixture
def ctx(tmp_path):
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path))


def offline(monkeypatch, name, transport, **kwargs):
    cls = tools._CLASSES[name]
    monkeypatch.setitem(tools._CLASSES, name,
                        functools.partial(cls, transport=transport, **kwargs))


def test_every_source_is_listed_with_how_to_pull_it_and_no_key(ctx, monkeypatch):
    monkeypatch.setenv("ZA_MAPBOX_TOKEN", FAKE_KEY)
    listed = tools.list_data_sources(ctx)
    by_name = {source["name"]: source for source in listed["sources"]}
    assert set(by_name) == set(tools._CLASSES)
    assert by_name["usgs_3dep"]["pull_with"] == "fetch_raster"
    assert by_name["osm"]["pull_with"] == "query_features"
    assert by_name["osrm"]["pull_with"] is None
    assert by_name["google_satellite"]["pullable"] is False
    assert by_name["google_satellite"]["pull_with"] is None
    assert by_name["mapbox_satellite"]["ready"] is True
    assert FAKE_KEY not in json.dumps(listed)
    assert "Raw source data" in listed["note"]


def test_elevation_is_saved_and_summarised_never_returned_as_pixels(ctx, monkeypatch):
    from test_dem import synthetic_tile_bytes

    offline(monkeypatch, "usgs_3dep", FakeTransport(body=synthetic_tile_bytes(250.0)))
    result = tools.fetch_raster(ctx, "usgs_3dep", radius_m=300, pixel_m=10, **POINT)
    assert result["summary"]["band_1"] == {"min": 250.0, "max": 250.0, "mean": 250.0}
    assert result["path"].endswith(".tif") and "/usgs_3dep/" in result["path"]
    assert 55 <= result["width_px"] <= 65 and result["attribution"] == "USGS 3DEP"
    assert result["area"]["radius_m"] == 300


class TileAtItsPlace(FakeTransport):
    """Each LANDFIRE tile georeferenced where its request's box says."""

    def get_bytes(self, url):
        import math
        from urllib.parse import parse_qs, urlparse

        from sar_geo_mcp.sources.landfire import EVT_TILE_M
        from test_evt import fake_tile_bytes_at

        self.urls.append(url)
        west, south = (float(v) for v in
                       parse_qs(urlparse(url).query)["bbox"][0].split(",")[:2])
        return fake_tile_bytes_at(math.floor(west / EVT_TILE_M + 0.5),
                                  math.floor(south / EVT_TILE_M + 0.5), 7015)


def test_vegetation_is_summarised_by_class(ctx, monkeypatch):
    offline(monkeypatch, "landfire_evt", TileAtItsPlace())
    result = tools.fetch_raster(ctx, "landfire_evt", radius_m=300, **POINT)
    assert result["summary"]["classes"][0]["code"] == 7015
    assert result["summary"]["classes"][0]["share"] == 1.0


@pytest.mark.parametrize("arguments, message", [
    ({"source": "google_satellite", **POINT, "radius_m": 300}, "cannot be pulled|not a raster"),
    ({"source": "usgs_3dep", "radius_m": 300}, "give a handle"),
    ({"source": "usgs_3dep", **POINT, "radius_m": 4501}, "at most 4500"),
    ({"source": "usgs_3dep", **POINT, "radius_m": float("nan")}, "radius_m"),
    ({"source": "usgs_3dep", **POINT, "radius_m": 300, "handle": "aoi-0"}, "not both"),
    ({"source": "usgs_3dep", **POINT, "radius_m": 300, "layer": "sf"}, "caltopo_wms only"),
    ({"source": "caltopo_wms", **POINT, "radius_m": 300}, "not ready|needs a layer"),
    ({"source": "usgs_3dep", **POINT, "radius_m": 300, "pixel_m": 0}, "pixel_m"),
])
def test_a_bad_raster_request_is_refused_before_any_fetch(ctx, monkeypatch, arguments, message):
    monkeypatch.delenv("ZA_WMS_TOKEN", raising=False)
    offline(monkeypatch, "usgs_3dep", FakeTransport(body=b""))
    with pytest.raises(ValueError, match=message):
        tools.fetch_raster(ctx, **arguments)


def osm_answer():
    from sar_geo_mcp.sources.osm import OSM_TILE_MARKER, tiles_covering_osm
    from sar_geo_mcp.sources.base import Area

    area = Area.around(POINT["center_latitude"], POINT["center_longitude"], 300)
    elements = [{"type": OSM_TILE_MARKER, "tags": {"col": str(col), "row": str(row)}}
                for col, row in tiles_covering_osm(area.bbox_3857)]
    latitude, longitude = POINT["center_latitude"], POINT["center_longitude"]
    elements[1:1] = [
        {"type": "way", "id": 10 + i, "tags": {"highway": "track" if i % 2 else "path"},
         "geometry": [{"lat": latitude, "lon": longitude},
                      {"lat": latitude + 0.0005, "lon": longitude + 0.0001 * i}]}
        for i in range(5)]
    return {"elements": elements}


def test_osm_features_come_back_filtered_capped_and_in_a_file(ctx, monkeypatch):
    offline(monkeypatch, "osm", FakeTransport(json_body=osm_answer()))
    result = tools.query_features(ctx, "osm", radius_m=300, **POINT,
                                  filters={"highway": "track"}, limit=1)
    assert result["total"] == 2 and result["returned"] == 1
    assert result["features"][0]["properties"]["tags"] == {"highway": "track"}
    saved = json.loads(open(result["path"]).read())
    assert len(saved["features"]) == 2
    assert saved["attribution"] == "© OpenStreetMap contributors"


def test_a_bad_feature_request_is_refused(ctx, monkeypatch):
    offline(monkeypatch, "osm", FakeTransport(json_body=osm_answer()))
    with pytest.raises(ValueError, match="limit"):
        tools.query_features(ctx, "osm", radius_m=300, **POINT, limit=201)
    with pytest.raises(ValueError, match="filter"):
        tools.query_features(ctx, "osm", radius_m=300, **POINT, filters={"highway": 3})
    with pytest.raises(ValueError, match="not a feature source"):
        tools.query_features(ctx, "usgs_3dep", radius_m=300, **POINT)


def test_an_unavailable_source_is_an_error_not_an_empty_answer(ctx, monkeypatch):
    class Down(FakeTransport):
        def get_json(self, url, params=None):
            raise ConnectionError("down")

    offline(monkeypatch, "agency_ways", Down())
    with pytest.raises(ValueError, match="agency_ways unavailable"):
        tools.query_features(ctx, "agency_ways", radius_m=300, **POINT)
